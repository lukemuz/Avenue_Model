use polars::prelude::*;
use crate::rating_model::{RatingModel, RatingTable, FeatureValue};
use std::collections::HashMap;
use rayon::prelude::*;

/// Pre-computes all observation-to-table-row matches for all tables
/// Returns a Vec where each element is the match indices for a table
pub fn precompute_all_matches(
    model: &RatingModel,
    df: &DataFrame,
) -> Result<Vec<Vec<Option<usize>>>, PolarsError> {
    let n_rows = df.height();
    let n_tables = model.tables.len();

    // Parallelize across tables
    let all_matches: Vec<Vec<Option<usize>>> = (0..n_tables)
        .into_par_iter()
        .map(|table_idx| {
            precompute_table_matches(&model.tables[table_idx], df, n_rows)
        })
        .collect();

    Ok(all_matches)
}

/// Pre-computes matches for a single table
fn precompute_table_matches(
    table: &RatingTable,
    df: &DataFrame,
    n_rows: usize,
) -> Vec<Option<usize>> {
    // Parallelize row matching if dataset is large enough
    const PARALLEL_THRESHOLD: usize = 1000;

    if n_rows > PARALLEL_THRESHOLD {
        // Parallel version
        (0..n_rows)
            .into_par_iter()
            .map(|row_idx| {
                match extract_row_features(df, row_idx) {
                    Ok(features) => table.find_row_match(&features),
                    Err(_) => None,
                }
            })
            .collect()
    } else {
        // Sequential version for small datasets
        (0..n_rows)
            .map(|row_idx| {
                match extract_row_features(df, row_idx) {
                    Ok(features) => table.find_row_match(&features),
                    Err(_) => None,
                }
            })
            .collect()
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
