//! Interval exhibits derived from the table's actual matching geometry.
use super::RatingTable;
use polars::prelude::*;
use std::collections::HashMap;

impl RatingTable {
    /// Label complete ordered grids only. General first-match tables can describe
    /// nonrectangular regions, so their thresholds alone do not prove lower bounds.
    pub(crate) fn review_data(&self) -> Result<DataFrame, PolarsError> {
        let mut data = self.data.clone();
        if self.metadata.spline.is_some() || self.get_numeric_columns().is_empty() {
            return Ok(data);
        }
        let mut numeric: Vec<_> = self.get_numeric_columns().keys().cloned().collect();
        numeric.sort();
        let mut categorical: Vec<_> = self.get_categorical_columns().keys().cloned().collect();
        categorical.sort();
        let mut axes = Vec::new();
        let mut coordinates = vec![Vec::new(); data.height()];
        let mut valid = data.height() > 0;
        for name in numeric.iter().chain(&categorical) {
            let values: Vec<Option<f64>> = if numeric.contains(name) {
                data.column(name)?.f64()?.into_iter().collect()
            } else {
                data.column(name)?
                    .i32()?
                    .into_iter()
                    .map(|v| v.filter(|v| *v != -999).map(f64::from))
                    .collect()
            };
            if values.iter().any(|v| v.is_none_or(f64::is_nan)) {
                valid = false;
                break;
            }
            let values: Vec<f64> = values.into_iter().flatten().collect();
            let mut axis = values.clone();
            axis.sort_by(f64::total_cmp);
            axis.dedup_by(|a, b| *a == *b);
            for (row, value) in values.iter().enumerate() {
                coordinates[row].push(axis.partition_point(|x| x < value));
            }
            axes.push(axis);
        }
        if valid {
            let size = axes
                .iter()
                .try_fold(1usize, |n, axis| n.checked_mul(axis.len()));
            let rows: HashMap<Vec<usize>, usize> = coordinates
                .iter()
                .cloned()
                .enumerate()
                .map(|(row, key)| (key, row))
                .collect();
            valid = size == Some(data.height()) && rows.len() == data.height();
            if valid {
                // Every immediate predecessor must occur first. Transitivity then
                // proves the minimal matching cell precedes all larger cells.
                for (row, key) in coordinates.iter().enumerate() {
                    for dimension in 0..numeric.len() {
                        if key[dimension] > 0 {
                            let mut previous = key.clone();
                            previous[dimension] -= 1;
                            if rows.get(&previous).is_none_or(|earlier| *earlier >= row) {
                                valid = false;
                            }
                        }
                    }
                }
            }
        }
        data.with_column(Series::new(
            "Band_Interval_Status".into(),
            vec![
                if valid {
                    "ordered_grid"
                } else {
                    "requires_match_review"
                };
                data.height()
            ],
        ))?;
        if valid {
            for (dimension, name) in numeric.iter().enumerate() {
                for suffix in ["Lower", "Upper", "Lower_Inclusive", "Upper_Inclusive"] {
                    let label = format!("{name}_{suffix}");
                    if self.data.column(&label).is_ok() {
                        return Err(PolarsError::ComputeError(format!(
                            "Review interval label '{label}' conflicts with an existing table column"
                        ).into()));
                    }
                }
                let axis = &axes[dimension];
                let lower: Vec<f64> = coordinates
                    .iter()
                    .map(|key| {
                        if key[dimension] == 0 {
                            f64::NEG_INFINITY
                        } else {
                            axis[key[dimension] - 1]
                        }
                    })
                    .collect();
                let upper: Vec<f64> = coordinates.iter().map(|key| axis[key[dimension]]).collect();
                // Infinite endpoints express unboundedness. Flags apply to finite
                // endpoints; matching of finite quotes is (lower, upper].
                let inclusive: Vec<bool> = upper.iter().map(|v| v.is_finite()).collect();
                data.with_column(Series::new(format!("{name}_Lower").into(), lower))?;
                data.with_column(Series::new(format!("{name}_Upper").into(), upper))?;
                data.with_column(Series::new(
                    format!("{name}_Lower_Inclusive").into(),
                    vec![false; data.height()],
                ))?;
                data.with_column(Series::new(
                    format!("{name}_Upper_Inclusive").into(),
                    inclusive,
                ))?;
            }
        }
        Ok(data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rating_model::FeatureValue;

    #[test]
    fn displayed_grid_intervals_match_lookup_at_boundaries_and_tails() {
        let mut a = vec![];
        let mut b = vec![];
        let mut category = vec![];
        for level in [1i32, 2] {
            for x in [0., f64::INFINITY] {
                for z in [5., f64::INFINITY] {
                    a.push(x);
                    b.push(z);
                    category.push(level);
                }
            }
        }
        let table = RatingTable::new(
            df!("a" => a, "b" => b, "c" => category,
            "Rating_Factor" => vec![0.; 8])
            .unwrap(),
            None,
        );
        let review = table.review_data().unwrap();
        for c in [1, 2] {
            for x in [-100., 0., f64::EPSILON, 100.] {
                for z in [-100., 5., 5. + 1e-12, 100.] {
                    let input = HashMap::from([
                        ("a".into(), FeatureValue::Numeric(x)),
                        ("b".into(), FeatureValue::Numeric(z)),
                        ("c".into(), FeatureValue::Categorical(c)),
                    ]);
                    let matches: Vec<_> = (0..8)
                        .filter(|&r| {
                            review.column("c").unwrap().i32().unwrap().get(r) == Some(c)
                                && [("a", x), ("b", z)].iter().all(|(name, value)| {
                                    let lower = review
                                        .column(&format!("{name}_Lower"))
                                        .unwrap()
                                        .f64()
                                        .unwrap()
                                        .get(r)
                                        .unwrap();
                                    let upper = review
                                        .column(&format!("{name}_Upper"))
                                        .unwrap()
                                        .f64()
                                        .unwrap()
                                        .get(r)
                                        .unwrap();
                                    *value > lower && *value <= upper
                                })
                        })
                        .collect();
                    assert_eq!(matches, vec![table.find_row_match(&input).unwrap()]);
                }
            }
        }
    }

    #[test]
    fn irregular_and_missing_route_tables_are_not_assigned_guessed_bounds() {
        for data in [
            df!("x" => [0., f64::INFINITY, f64::INFINITY],
                "z" => [0., 0., f64::INFINITY], "Rating_Factor" => [0.; 3])
            .unwrap(),
            df!("x" => [0., f64::INFINITY], "c" => [1i32, -999],
                "Rating_Factor" => [0.; 2])
            .unwrap(),
        ] {
            let review = RatingTable::new(data, None).review_data().unwrap();
            assert!(review.column("x_Lower").is_err());
        }
        for bounds in [
            vec![Some(f64::INFINITY), Some(0.)],
            vec![Some(0.), None],
            vec![Some(0.), Some(f64::NAN)],
        ] {
            let table = RatingTable::new(
                df!("x" => bounds, "Rating_Factor" => [0., 0.]).unwrap(),
                None,
            );
            let review = table.review_data().unwrap();
            assert!(review.column("x_Lower").is_err());
            assert_eq!(
                review
                    .column("Band_Interval_Status")
                    .unwrap()
                    .str()
                    .unwrap()
                    .get(0),
                Some("requires_match_review")
            );
        }
    }

    #[test]
    fn interval_labels_cannot_replace_an_existing_predictor() {
        let table = RatingTable::new(
            df!("x" => [0., 0., f64::INFINITY, f64::INFINITY],
            "x_Lower" => [1., f64::INFINITY, 1., f64::INFINITY],
            "Rating_Factor" => [0.; 4])
            .unwrap(),
            None,
        );
        assert!(table
            .review_data()
            .unwrap_err()
            .to_string()
            .contains("conflicts"));
    }
}
