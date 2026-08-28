use crate::rating_model::{FeatureValue, RatingModel, RatingTable};
use polars::prelude::*;
use rayon::prelude::*;
use std::collections::HashMap;

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
    SortedNumeric {
        column: String,
        thresholds: Vec<f64>,
    },
    /// One categorical column, no numerics, no null table values.
    ///
    /// The scan's answer here is decided entirely by two things: the first row carrying
    /// the observation's code, and the first wildcard row. An exact match beats a
    /// wildcard wherever both exist, and among equals the earlier row wins — so both can
    /// be resolved once per table into a lookup, rather than rediscovered by scanning
    /// every row for every observation.
    Categorical {
        column: String,
        /// Code to the first row carrying it. Wildcard rows are excluded.
        rows: HashMap<i32, u32>,
        /// The first `-999` row, or [`NO_MATCH`] if the table has none.
        wildcard: u32,
    },
    /// Anything else: multiple columns, mixed types, unsorted or null thresholds.
    /// Falls back to the row-by-row scan.
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

    for (t, table) in model.tables.iter().enumerate() {
        reject_unreadable_columns(table, t)?;
    }

    Ok(model
        .tables
        .par_iter()
        .map(|table| precompute_table_matches(table, df, n_rows))
        .collect())
}

/// Refuses a table holding a feature column whose dtype nothing can read.
///
/// `RatingTable::new` classifies a `Float64` column as a numeric band and an `Int32` one
/// as a category code, and **drops anything else**. A dropped column constrains no row,
/// so the table matches every observation to row 0 and fits a single factor for the whole
/// factor — a wrong model, produced silently, with no unmatched observation to trip the
/// check that exists for that.
///
/// It is an easy mistake to make: `Int64` is numpy's default integer dtype and what
/// `pandas.Categorical(...).codes` widens to. Measured on a four-level factor with a real
/// effect, the `Int32` table fits factors of `[0, 0.51, 1.01, 1.50]` at a deviance of
/// 22,091 and the `Int64` one fits `[0, 0, 0, 0]` at 35,995.
///
/// Only numeric dtypes are rejected, since those are unambiguously meant as features.
fn reject_unreadable_columns(table: &RatingTable, index: usize) -> Result<(), PolarsError> {
    for name in table.data.get_column_names() {
        if name == "Rating_Factor"
            || table.get_numeric_columns().contains_key(name.as_str())
            || table.get_categorical_columns().contains_key(name.as_str())
        {
            continue;
        }
        let dtype = table.data.column(name).unwrap().dtype();
        if dtype.is_primitive_numeric() {
            return Err(PolarsError::ComputeError(
                format!(
                    "Table {} column '{}' is {:?}, which the matcher cannot read, so the \
                     column would be ignored and every observation matched to row 0. \
                     Cast it to Float64 for a numeric band or Int32 for a category code.",
                    index, name, dtype
                )
                .into(),
            ));
        }
    }
    Ok(())
}

/// Pre-computes matches for a single table.
fn precompute_table_matches(table: &RatingTable, df: &DataFrame, n_rows: usize) -> Vec<u32> {
    match plan_for(table, df) {
        MatchPlan::Constant(row) => vec![row; n_rows],
        MatchPlan::SortedNumeric { column, thresholds } => {
            sorted_numeric_matches(df, &column, &thresholds, n_rows)
        }
        MatchPlan::Categorical {
            column,
            rows,
            wildcard,
        } => categorical_matches(df, &column, &rows, wildcard, n_rows),
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

    if numeric.is_empty() && categorical.len() == 1 {
        let (name, &col_idx) = categorical.iter().next().unwrap();
        return categorical_plan(table, df, name, col_idx);
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

/// Builds the categorical lookup, or hands the table back to the scan.
///
/// The scan reaches its answer through `has_all_required_features` and a first-match
/// rule, and both have corners the lookup cannot express. Each bail-out below is one of
/// them, so the fast path is only taken where it provably agrees.
fn categorical_plan(table: &RatingTable, df: &DataFrame, name: &str, col_idx: usize) -> MatchPlan {
    // `extract_row_features` reads an `Int32` column as `Categorical` and a `Float64`
    // one as `Numeric`. Against a categorical table column a `Numeric` input leaves the
    // column unconstrained rather than unmatched, so every row matches and the scan
    // returns row 0. Only `Int32` behaves the way this lookup assumes.
    match df.column(name) {
        Ok(obs) if obs.dtype() == &DataType::Int32 => {}
        _ => return MatchPlan::General,
    }

    let Ok(values) = table.data.get_columns()[col_idx].i32() else {
        return MatchPlan::General;
    };

    let mut rows: HashMap<i32, u32> = HashMap::with_capacity(values.len());
    let mut wildcard = NO_MATCH;

    for i in 0..values.len() {
        match values.get(i) {
            // A null table value constrains nothing, so the scan matches every
            // observation against this row and counts it as an exact match — a
            // catch-all that outranks later rows. Rare, and not worth encoding.
            None => return MatchPlan::General,
            Some(-999) => {
                if wildcard == NO_MATCH {
                    wildcard = i as u32;
                }
            }
            // First row wins, which is what the scan's `best_row.is_none()` test does.
            Some(code) => {
                rows.entry(code).or_insert(i as u32);
            }
        }
    }

    MatchPlan::Categorical {
        column: name.to_string(),
        rows,
        wildcard,
    }
}

fn categorical_matches(
    df: &DataFrame,
    column: &str,
    rows: &HashMap<i32, u32>,
    wildcard: u32,
    n_rows: usize,
) -> Vec<u32> {
    let ca = df.column(column).unwrap().i32().unwrap();

    // An exact row beats the wildcard, which is the scan's rule that a later
    // non-wildcard match displaces an earlier wildcard one.
    let lookup = |code: i32| rows.get(&code).copied().unwrap_or(wildcard);

    if let Ok(values) = ca.cont_slice() {
        return values.par_iter().map(|c| lookup(*c)).collect();
    }

    (0..n_rows)
        .into_par_iter()
        .map(|i| match ca.get(i) {
            // A null feature value is a missing feature, which the scan treats as
            // matching nothing.
            Some(code) => lookup(code),
            None => NO_MATCH,
        })
        .collect()
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

/// For tables no shortcut covers: the pre-resolved scan where it applies, else the
/// row-by-row one.
fn general_matches(table: &RatingTable, df: &DataFrame, n_rows: usize) -> Vec<u32> {
    pre_resolved_scan(table, df, n_rows).unwrap_or_else(|| scan_matches(table, df, n_rows))
}

/// The reference implementation: build each observation's features, then ask the table.
///
/// Every other path in this module has to reproduce this one exactly, and the tests
/// check them against it directly. Slow — a `HashMap` and its keys are allocated per
/// observation — so it runs only where nothing else applies.
fn scan_matches(table: &RatingTable, df: &DataFrame, n_rows: usize) -> Vec<u32> {
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

/// The same scan, with the column lookups hoisted out of the per-observation work.
///
/// [`extract_row_features`] builds a `HashMap<String, FeatureValue>` for every
/// observation, over every column of the frame rather than the two or three the table
/// reads, and [`RatingTable::find_row_match`] then re-resolves the table's own columns on
/// each call. None of that depends on the observation. Hoisting it costs a two-way
/// interaction table about 70 ns per observation, which on a 200k-row fit with four such
/// tables is more time than every sweep put together.
///
/// Returns `None` for anything this cannot reproduce exactly, leaving the scan to it.
fn pre_resolved_scan(table: &RatingTable, df: &DataFrame, n_rows: usize) -> Option<Vec<u32>> {
    // The table's own values are few and may be null, which means "this column does not
    // constrain this row". Materialised once, so the inner loop is over plain slices.
    let mut categorical: Vec<(Vec<Option<i32>>, &[i32])> = Vec::new();
    for (name, &col_idx) in table.get_categorical_columns() {
        let obs = df.column(name).ok()?;
        // A dtype other than the expected one does not mean "no match": it changes
        // which `FeatureValue` variant the scan sees, and so whether the column
        // constrains at all. Too subtle to reproduce — hand those back.
        if obs.dtype() != &DataType::Int32 {
            return None;
        }
        // `cont_slice` refuses nulls and chunked columns, which is the guard we want:
        // a null observation is a missing feature, and the scan matches it to nothing.
        let values = obs.i32().ok()?.cont_slice().ok()?;
        let thresholds = table.data.get_columns()[col_idx]
            .i32()
            .ok()?
            .iter()
            .collect();
        categorical.push((thresholds, values));
    }

    let mut numeric: Vec<(Vec<Option<f64>>, &[f64])> = Vec::new();
    for (name, &col_idx) in table.get_numeric_columns() {
        let obs = df.column(name).ok()?;
        if obs.dtype() != &DataType::Float64 {
            return None;
        }
        let values = obs.f64().ok()?.cont_slice().ok()?;
        let thresholds = table.data.get_columns()[col_idx]
            .f64()
            .ok()?
            .iter()
            .collect();
        numeric.push((thresholds, values));
    }

    let table_rows = table.data.height();

    let match_row = |i: usize| -> u32 {
        let mut best = NO_MATCH;
        let mut best_used_wildcard = false;

        'row: for r in 0..table_rows {
            let mut used_wildcard = false;

            for (thresholds, values) in &categorical {
                if let Some(table_cat) = thresholds[r] {
                    if table_cat == -999 {
                        used_wildcard = true;
                    } else if table_cat != values[i] {
                        continue 'row;
                    }
                }
            }

            for (thresholds, values) in &numeric {
                if let Some(threshold) = thresholds[r] {
                    if values[i] > threshold {
                        continue 'row;
                    }
                }
            }

            // First match wins, except that an exact match displaces a wildcard one.
            if best == NO_MATCH || (best_used_wildcard && !used_wildcard) {
                best = r as u32;
                best_used_wildcard = used_wildcard;
            }
        }

        best
    };

    Some(if n_rows > 1000 {
        (0..n_rows).into_par_iter().map(match_row).collect()
    } else {
        (0..n_rows).map(match_row).collect()
    })
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
    ///
    /// Deliberately against [`scan_matches`], the reference, rather than against
    /// `general_matches` — that now dispatches to a shortcut of its own, and comparing
    /// two shortcuts to each other would prove nothing about either.
    fn agrees_with_scan(table: &RatingTable, df: &DataFrame) {
        let reference = scan_matches(table, df, df.height());
        assert_eq!(
            precompute_table_matches(table, df, df.height()),
            reference,
            "the chosen plan disagreed with the reference scan"
        );
        assert_eq!(
            general_matches(table, df, df.height()),
            reference,
            "the pre-resolved scan disagreed with the reference scan"
        );
    }

    #[test]
    fn a_feature_column_the_matcher_cannot_read_is_rejected() {
        // Int64 is numpy's default integer dtype, so this is the likely way to hit it.
        for values in [
            Series::new("g".into(), vec![0i64, 1]),
            Series::new("g".into(), vec![0u32, 1]),
            Series::new("g".into(), vec![0.0f32, 1.0]),
        ] {
            let table = RatingTable::new(
                DataFrame::new(vec![
                    values.clone().into(),
                    Series::new("Rating_Factor".into(), vec![0.0, 0.0]).into(),
                ])
                .unwrap(),
                None,
            );
            let err = reject_unreadable_columns(&table, 3)
                .unwrap_err()
                .to_string();
            assert!(
                err.contains("Table 3 column 'g'") && err.contains("Cast it to Float64"),
                "unhelpful message for {:?}: {err}",
                values.dtype()
            );
        }
    }

    #[test]
    fn the_readable_dtypes_and_the_factor_column_are_accepted() {
        let table = RatingTable::new(
            DataFrame::new(vec![
                Series::new("band".into(), vec![1.0f64, 2.0]).into(),
                Series::new("code".into(), vec![1i32, 2]).into(),
                Series::new("Rating_Factor".into(), vec![0.0, 0.0]).into(),
            ])
            .unwrap(),
            None,
        );
        assert!(reject_unreadable_columns(&table, 0).is_ok());
    }

    /// An Int32 feature column, which `RatingTable` classifies as categorical.
    fn cat_table(col: &str, codes: &[i32]) -> RatingTable {
        RatingTable::new(
            DataFrame::new(vec![
                Series::new(col.into(), codes.to_vec()).into(),
                Series::new("Rating_Factor".into(), vec![0.0; codes.len()]).into(),
            ])
            .unwrap(),
            None,
        )
    }

    fn cat_obs(col: &str, codes: &[i32]) -> DataFrame {
        DataFrame::new(vec![Series::new(col.into(), codes.to_vec()).into()]).unwrap()
    }

    #[test]
    fn the_categorical_lookup_agrees_with_the_scan() {
        // Out of order and with gaps, so position cannot stand in for the code.
        let t = cat_table("g", &[7, 3, 11, 3]);
        let df = cat_obs("g", &[3, 7, 11, 0, -1, 3]);
        agrees_with_scan(&t, &df);
        // 3 appears twice; the scan's first-match rule takes row 1, not row 3.
        assert_eq!(
            precompute_table_matches(&t, &df, df.height()),
            vec![1, 0, 2, NO_MATCH, NO_MATCH, 1]
        );
    }

    #[test]
    fn an_exact_row_beats_an_earlier_wildcard() {
        // The scan reaches this by displacing a wildcard match with a later exact one,
        // so a lookup that simply took the first matching row would disagree.
        let t = cat_table("g", &[-999, 4, 9]);
        let df = cat_obs("g", &[9, 4, 77]);
        agrees_with_scan(&t, &df);
        assert_eq!(
            precompute_table_matches(&t, &df, df.height()),
            vec![2, 1, 0]
        );
    }

    #[test]
    fn a_second_wildcard_row_is_never_reached() {
        let t = cat_table("g", &[5, -999, -999]);
        let df = cat_obs("g", &[5, 6]);
        agrees_with_scan(&t, &df);
        assert_eq!(precompute_table_matches(&t, &df, df.height()), vec![0, 1]);
    }

    #[test]
    fn a_null_categorical_observation_matches_nothing() {
        let t = cat_table("g", &[1, 2]);
        let df = DataFrame::new(vec![Series::new(
            "g".into(),
            vec![Some(1i32), None, Some(2i32)],
        )
        .into()])
        .unwrap();
        agrees_with_scan(&t, &df);
        assert_eq!(
            precompute_table_matches(&t, &df, df.height()),
            vec![0, NO_MATCH, 1]
        );
    }

    #[test]
    fn a_null_table_value_falls_back_to_the_scan() {
        // A null table value constrains nothing, so that row is a catch-all the lookup
        // cannot express. The scan has to keep it.
        let t = RatingTable::new(
            DataFrame::new(vec![
                Series::new("g".into(), vec![Some(1i32), None]).into(),
                Series::new("Rating_Factor".into(), vec![0.0, 0.0]).into(),
            ])
            .unwrap(),
            None,
        );
        let df = cat_obs("g", &[1, 2]);
        assert!(matches!(plan_for(&t, &df), MatchPlan::General));
        agrees_with_scan(&t, &df);
    }

    #[test]
    fn a_float_observation_column_falls_back_to_the_scan() {
        // Read as `Numeric`, which leaves a categorical table column unconstrained
        // rather than unmatched — the scan then returns row 0 for everything.
        let t = cat_table("g", &[1, 2]);
        let df = DataFrame::new(vec![Series::new("g".into(), vec![1.0, 2.0]).into()]).unwrap();
        assert!(matches!(plan_for(&t, &df), MatchPlan::General));
        agrees_with_scan(&t, &df);
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
        let df = DataFrame::new(vec![Series::new("x".into(), vec![f64::NAN, 5.0]).into()]).unwrap();
        agrees_with_scan(&t, &df);
    }

    /// A two-column table: no shortcut plan covers it, so this exercises the
    /// pre-resolved scan against the reference.
    #[test]
    fn a_two_way_numeric_table_agrees_with_the_scan() {
        let t = RatingTable::new(
            DataFrame::new(vec![
                Series::new("x".into(), vec![10.0, 10.0, f64::INFINITY, f64::INFINITY]).into(),
                Series::new("y".into(), vec![5.0, f64::INFINITY, 5.0, f64::INFINITY]).into(),
                Series::new("Rating_Factor".into(), vec![0.0; 4]).into(),
            ])
            .unwrap(),
            None,
        );
        let df = DataFrame::new(vec![
            Series::new("x".into(), vec![1.0, 1.0, 50.0, 50.0, 10.0]).into(),
            Series::new("y".into(), vec![1.0, 50.0, 1.0, 50.0, 5.0]).into(),
        ])
        .unwrap();
        agrees_with_scan(&t, &df);
        assert_eq!(precompute_table_matches(&t, &df, 5), vec![0, 1, 2, 3, 0]);
    }

    #[test]
    fn a_mixed_table_agrees_with_the_scan() {
        // One categorical and one numeric column, plus a wildcard row that only the
        // observations missing an exact pairing should reach.
        let t = RatingTable::new(
            DataFrame::new(vec![
                Series::new("g".into(), vec![1i32, 1, 2, -999]).into(),
                Series::new("x".into(), vec![10.0, f64::INFINITY, 10.0, f64::INFINITY]).into(),
                Series::new("Rating_Factor".into(), vec![0.0; 4]).into(),
            ])
            .unwrap(),
            None,
        );
        let df = DataFrame::new(vec![
            Series::new("g".into(), vec![1i32, 1, 2, 2, 9]).into(),
            Series::new("x".into(), vec![5.0, 50.0, 5.0, 50.0, 5.0]).into(),
        ])
        .unwrap();
        agrees_with_scan(&t, &df);
        //                        g=2,x=50 finds no exact row and falls to the wildcard.
        assert_eq!(precompute_table_matches(&t, &df, 5), vec![0, 1, 2, 3, 3]);
    }

    #[test]
    fn a_null_in_a_two_way_table_still_agrees_with_the_scan() {
        // A null table value constrains nothing, so row 1 is a catch-all on `y`.
        let t = RatingTable::new(
            DataFrame::new(vec![
                Series::new("x".into(), vec![Some(10.0), Some(f64::INFINITY)]).into(),
                Series::new("y".into(), vec![Some(5.0), None]).into(),
                Series::new("Rating_Factor".into(), vec![0.0, 0.0]).into(),
            ])
            .unwrap(),
            None,
        );
        let df = DataFrame::new(vec![
            Series::new("x".into(), vec![1.0, 1.0, 50.0]).into(),
            Series::new("y".into(), vec![1.0, 50.0, 1.0]).into(),
        ])
        .unwrap();
        agrees_with_scan(&t, &df);
    }

    #[test]
    fn a_null_observation_in_a_two_way_table_matches_nothing() {
        let t = RatingTable::new(
            DataFrame::new(vec![
                Series::new("x".into(), vec![f64::INFINITY]).into(),
                Series::new("y".into(), vec![f64::INFINITY]).into(),
                Series::new("Rating_Factor".into(), vec![0.0]).into(),
            ])
            .unwrap(),
            None,
        );
        let df = DataFrame::new(vec![
            Series::new("x".into(), vec![Some(1.0), None]).into(),
            Series::new("y".into(), vec![Some(1.0), Some(1.0)]).into(),
        ])
        .unwrap();
        agrees_with_scan(&t, &df);
        assert_eq!(precompute_table_matches(&t, &df, 2), vec![0, NO_MATCH]);
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
