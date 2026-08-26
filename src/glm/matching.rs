use polars::prelude::*;
use crate::rating_model::{RatingModel, RatingTable, FeatureValue};
use std::collections::HashMap;
use rayon::prelude::*;

/// Marks an observation that fell in no row of a table.
///
/// A sentinel rather than `Option<u32>`, which Rust widens to 8 bytes for the
/// discriminant. The match arrays are re-read for every table on every sweep, so their
/// width is felt directly as memory bandwidth in the hot loop: at 5M observations over
/// six tables this is 120 MB rather than the 480 MB that `Vec<Option<usize>>` cost.
///
/// Table row counts are validated against this, so a real row index can never collide
/// with it.
pub const NO_MATCH: u32 = u32::MAX;

/// How a table's rows will be looked up.
///
/// Chosen once per table, then applied to every observation. The general scan is
/// correct for any table; the other two are shortcuts for shapes common enough to be
/// worth recognising, and each must reproduce the scan's answer exactly.
enum MatchPlan {
    /// The table constrains nothing — no numeric or categorical feature columns — so
    /// a scan would stop at the first row. Intercept tables take this path.
    Constant(u32),
    /// One numeric column, no categoricals, thresholds non-null and non-decreasing.
    ///
    /// The scan looks for the first row whose threshold is not below the observation's
    /// value, which is a lower bound: binary search finds it in O(log rows) without
    /// touching the rest of the table.
    SortedNumeric { column: String, thresholds: Vec<f64> },
    /// Anything else: multiple columns, categoricals, wildcards, unsorted or null
    /// thresholds. Falls back to the row-by-row scan.
    General,
}

/// Pre-computes, for every table, the row each observation falls in.
///
/// `result[t][i]` is the row of table `t` matched by observation `i`, or [`NO_MATCH`].
pub fn precompute_all_matches(
    model: &RatingModel,
    df: &DataFrame,
) -> Result<Vec<Vec<u32>>, PolarsError> {
    let n_rows = df.height();

    // A row index has to survive the round trip through a u32 sentinel.
    for (t, table) in model.tables.iter().enumerate() {
        if table.data.height() >= NO_MATCH as usize {
            return Err(PolarsError::ComputeError(
                format!(
                    "Table {} has {} rows, which exceeds the {} supported",
                    t,
                    table.data.height(),
                    NO_MATCH - 1
                )
                .into(),
            ));
        }
    }

    Ok(model
        .tables
        .par_iter()
        .map(|table| precompute_table_matches(table, df, n_rows))
        .collect())
}

/// Pre-computes matches for a single table.
fn precompute_table_matches(table: &RatingTable, df: &DataFrame, n_rows: usize) -> Vec<u32> {
    match plan_for(table, df) {
        MatchPlan::Constant(row) => vec![row; n_rows],
        MatchPlan::SortedNumeric { column, thresholds } => {
            sorted_numeric_matches(df, &column, &thresholds, n_rows)
        }
        MatchPlan::General => general_matches(table, df, n_rows),
    }
}

/// Picks the cheapest lookup that is exactly equivalent to the general scan.
fn plan_for(table: &RatingTable, df: &DataFrame) -> MatchPlan {
    let numeric = table.get_numeric_columns();
    let categorical = table.get_categorical_columns();

    if numeric.is_empty() && categorical.is_empty() {
        return if table.data.height() > 0 {
            MatchPlan::Constant(0)
        } else {
            MatchPlan::Constant(NO_MATCH)
        };
    }

    if !categorical.is_empty() || numeric.len() != 1 {
        return MatchPlan::General;
    }

    let (name, &col_idx) = numeric.iter().next().unwrap();

    // The observation column has to be present and f64, or the scan would find no
    // match for every row and the fast path must say the same thing.
    match df.column(name) {
        Ok(obs) if obs.dtype() == &DataType::Float64 => {}
        _ => return MatchPlan::General,
    }

    let Ok(thresholds_ca) = table.data.get_columns()[col_idx].f64() else {
        return MatchPlan::General;
    };

    // A null threshold means "this column does not constrain this row", and a NaN
    // makes the ordering meaningless. Either way the scan and a binary search can
    // disagree, so hand those tables back to the scan.
    let mut thresholds = Vec::with_capacity(thresholds_ca.len());
    for i in 0..thresholds_ca.len() {
        match thresholds_ca.get(i) {
            Some(v) if !v.is_nan() => thresholds.push(v),
            _ => return MatchPlan::General,
        }
    }

    if thresholds.windows(2).any(|w| w[0] > w[1]) {
        return MatchPlan::General;
    }

    MatchPlan::SortedNumeric {
        column: name.clone(),
        thresholds,
    }
}

/// The row an observation of `value` falls in, for non-decreasing `thresholds`.
#[inline]
fn lower_bound(thresholds: &[f64], value: f64) -> u32 {
    // The scan skips a row when `value > threshold`, which is false for NaN, so a NaN
    // observation stops at the first row. Reproduce that rather than letting it reach
    // the comparison-based search, where it has no defined position.
    if value.is_nan() {
        return if thresholds.is_empty() { NO_MATCH } else { 0 };
    }
    let idx = thresholds.partition_point(|t| *t < value);
    if idx < thresholds.len() {
        idx as u32
    } else {
        NO_MATCH
    }
}

fn sorted_numeric_matches(
    df: &DataFrame,
    column: &str,
    thresholds: &[f64],
    n_rows: usize,
) -> Vec<u32> {
    let ca = df.column(column).unwrap().f64().unwrap();

    // One contiguous chunk with no nulls is the common case and lets the search run
    // straight off a slice.
    if let Ok(values) = ca.cont_slice() {
        return values
            .par_iter()
            .map(|v| lower_bound(thresholds, *v))
            .collect();
    }

    (0..n_rows)
        .into_par_iter()
        .map(|i| match ca.get(i) {
            // A null feature value is a missing feature, which the scan treats as
            // matching nothing.
            Some(v) => lower_bound(thresholds, v),
            None => NO_MATCH,
        })
        .collect()
}

/// The general row-by-row scan, for tables no shortcut covers.
fn general_matches(table: &RatingTable, df: &DataFrame, n_rows: usize) -> Vec<u32> {
    const PARALLEL_THRESHOLD: usize = 1000;

    let match_row = |row_idx: usize| -> u32 {
        match extract_row_features(df, row_idx) {
            Ok(features) => table
                .find_row_match(&features)
                .map_or(NO_MATCH, |r| r as u32),
            Err(_) => NO_MATCH,
        }
    };

    if n_rows > PARALLEL_THRESHOLD {
        (0..n_rows).into_par_iter().map(match_row).collect()
    } else {
        (0..n_rows).map(match_row).collect()
    }
}

/// Extracts feature values from a DataFrame row
fn extract_row_features(
    df: &DataFrame,
    row_idx: usize,
) -> Result<HashMap<String, FeatureValue>, PolarsError> {
    let mut features = HashMap::new();

    for col_name in df.get_column_names() {
        let col = df.column(col_name)?;
        match col.dtype() {
            DataType::Float64 => {
                if let Some(val) = col.f64()?.get(row_idx) {
                    features.insert(col_name.to_string(), FeatureValue::Numeric(val));
                }
            }
            DataType::Int32 => {
                if let Some(val) = col.i32()?.get(row_idx) {
                    features.insert(col_name.to_string(), FeatureValue::Categorical(val));
                }
            }
            _ => continue,
        }
    }

    Ok(features)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn table(col: &str, bounds: &[f64]) -> RatingTable {
        RatingTable::new(
            DataFrame::new(vec![
                Series::new(col.into(), bounds.to_vec()).into(),
                Series::new("Rating_Factor".into(), vec![0.0; bounds.len()]).into(),
            ])
            .unwrap(),
            None,
        )
    }

    /// The shortcuts exist only to be faster. Any disagreement with the scan they
    /// replace is a bug, so check them against it directly.
    fn agrees_with_scan(table: &RatingTable, df: &DataFrame) {
        let fast = precompute_table_matches(table, df, df.height());
        let scan = general_matches(table, df, df.height());
        assert_eq!(fast, scan, "fast path disagreed with the general scan");
    }

    #[test]
    fn binary_search_agrees_with_the_scan() {
        let t = table("x", &[10.0, 20.0, 30.0, f64::INFINITY]);
        let df = DataFrame::new(vec![Series::new(
            "x".into(),
            // Below, on, between and above every boundary.
            vec![-1.0, 0.0, 9.9, 10.0, 10.1, 20.0, 25.0, 30.0, 30.1, 1e300],
        )
        .into()])
        .unwrap();
        agrees_with_scan(&t, &df);
    }

    #[test]
    fn binary_search_agrees_on_a_table_with_no_open_top_band() {
        // Highest threshold is finite, so values above it match nothing.
        let t = table("x", &[10.0, 20.0]);
        let df = DataFrame::new(vec![
            Series::new("x".into(), vec![5.0, 20.0, 20.001, 1e9]).into()
        ])
        .unwrap();
        agrees_with_scan(&t, &df);

        let m = precompute_table_matches(&t, &df, df.height());
        assert_eq!(m, vec![0, 1, NO_MATCH, NO_MATCH]);
    }

    #[test]
    fn repeated_thresholds_pick_the_first_row_like_the_scan() {
        let t = table("x", &[10.0, 10.0, 20.0]);
        let df =
            DataFrame::new(vec![Series::new("x".into(), vec![5.0, 10.0, 15.0]).into()]).unwrap();
        agrees_with_scan(&t, &df);
    }

    #[test]
    fn descending_thresholds_fall_back_to_the_scan() {
        let t = table("x", &[30.0, 10.0, 20.0]);
        assert!(matches!(
            plan_for(
                &t,
                &DataFrame::new(vec![Series::new("x".into(), vec![15.0]).into()]).unwrap()
            ),
            MatchPlan::General
        ));
    }

    #[test]
    fn a_nan_observation_lands_where_the_scan_puts_it() {
        let t = table("x", &[10.0, 20.0, f64::INFINITY]);
        let df = DataFrame::new(vec![
            Series::new("x".into(), vec![f64::NAN, 5.0]).into()
        ])
        .unwrap();
        agrees_with_scan(&t, &df);
    }

    #[test]
    fn a_table_with_no_features_matches_row_zero() {
        let t = RatingTable::new(
            DataFrame::new(vec![Series::new("Rating_Factor".into(), vec![0.0]).into()]).unwrap(),
            None,
        );
        let df = DataFrame::new(vec![Series::new("x".into(), vec![1.0, 2.0, 3.0]).into()]).unwrap();
        agrees_with_scan(&t, &df);
        assert_eq!(precompute_table_matches(&t, &df, 3), vec![0, 0, 0]);
    }

    #[test]
    fn a_missing_feature_column_matches_nothing() {
        let t = table("x", &[10.0, f64::INFINITY]);
        let df = DataFrame::new(vec![Series::new("other".into(), vec![1.0]).into()]).unwrap();
        agrees_with_scan(&t, &df);
        assert_eq!(precompute_table_matches(&t, &df, 1), vec![NO_MATCH]);
    }
}
