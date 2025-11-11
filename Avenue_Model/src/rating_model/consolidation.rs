use polars::prelude::*;
use std::collections::{HashMap, HashSet};
use rayon::prelude::*;

use super::{RatingTable, FeatureValue};

pub fn expand_and_combine_tables(table1: &RatingTable, table2: &RatingTable) -> RatingTable {
    // Get unique feature values for each feature from both tables
    let mut numeric_values: HashMap<String, Vec<f64>> = HashMap::new();
    let mut categorical_values: HashMap<String, Vec<i32>> = HashMap::new();

    // Collect values from both tables
    for table in [table1, table2] {
        for col_name in table.data.get_column_names() {
            if col_name == "Rating_Factor" || col_name == "row_number"  { continue; }

            let col = table.data.column(col_name).unwrap();
            match col.dtype() {
                DataType::Float64 => {
                    let values = numeric_values.entry(col_name.to_string())
                        .or_insert_with(Vec::new);
                    col.f64()
                        .unwrap()
                        .into_iter()
                        .filter_map(|v| v)
                        .for_each(|v| values.push(v));
                },
                DataType::Int32 => {
                    let values = categorical_values.entry(col_name.to_string())
                        .or_insert_with(Vec::new);
                    col.i32()
                        .unwrap()
                        .into_iter()
                        .filter_map(|v| v)
                        .for_each(|v| values.push(v));
                },
                _ => continue,
            }
        }
    }

    // Dedupe and sort values
    for values in numeric_values.values_mut() {
        values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        values.dedup();
    }

    for values in categorical_values.values_mut() {
        values.sort_unstable();
        values.dedup();
    }

    // Calculate the total size needed for combinations
    let total_combinations = numeric_values.values()
        .map(|v| v.len())
        .chain(categorical_values.values().map(|v| v.len()))
        .product::<usize>();

    // Pre-allocate combinations with calculated size
    let mut combinations = Vec::with_capacity(total_combinations);

    // Initialize combinations with first feature (either numeric or categorical)
    let mut processed_features = HashSet::new();

    if let Some((first_feature, first_values)) = numeric_values.iter().next() {
        for &value in first_values {
            let mut combo = HashMap::new();
            combo.insert(first_feature.clone(), FeatureValue::Numeric(value));
            combinations.push(combo);
        }
        processed_features.insert(first_feature.clone());
    } else if let Some((first_feature, first_values)) = categorical_values.iter().next() {
        for &value in first_values {
            let mut combo = HashMap::new();
            combo.insert(first_feature.clone(), FeatureValue::Categorical(value));
            combinations.push(combo);
        }
        processed_features.insert(first_feature.clone());
    }

    // Add remaining numeric features
    for (feature, values) in numeric_values.iter() {
        if processed_features.contains(feature) {
            continue;
        }
        let mut new_combinations = Vec::new();
        for combo in combinations {
            for &value in values {
                let mut new_combo = combo.clone();
                new_combo.insert(feature.clone(), FeatureValue::Numeric(value));
                new_combinations.push(new_combo);
            }
        }
        combinations = new_combinations;
        processed_features.insert(feature.clone());
    }

    // Add remaining categorical features
    for (feature, values) in categorical_values.iter() {
        if processed_features.contains(feature) {
            continue;
        }
        let mut new_combinations = Vec::new();
        for combo in combinations {
            for &value in values {
                let mut new_combo = combo.clone();
                new_combo.insert(feature.clone(), FeatureValue::Categorical(value));
                new_combinations.push(new_combo);
            }
        }
        combinations = new_combinations;
        processed_features.insert(feature.clone());
    }

    // Create the result table
    let mut table_data: Vec<polars::prelude::Column> = Vec::new();

    // Add feature columns
    for (feature, _values) in &numeric_values {
        let column_values: Vec<f64> = combinations.iter()
            .map(|combo| {
                if let Some(FeatureValue::Numeric(v)) = combo.get(feature) {
                    *v
                } else {
                    panic!("Missing numeric value for feature {}", feature);
                }
            })
            .collect();
        table_data.push(Series::new(feature.into(), column_values).into());
    }

    for (feature, _) in &categorical_values {
        let column_values: Vec<i32> = combinations.iter()
            .map(|combo| {
                if let Some(FeatureValue::Categorical(v)) = combo.get(feature) {
                    *v
                } else {
                    panic!("Missing categorical value for feature {}", feature);
                }
            })
            .collect();
        table_data.push(Series::new(feature.into(), column_values).into());
    }

    // Calculate rating factors in parallel
    let rating_factors: Vec<f64> = combinations.par_iter()
        .map(|combo| {
            let rf1 = table1.predict(combo);
            let rf2 = table2.predict(combo);
            rf1 + rf2
        })
        .collect();

    table_data.push(Series::new("Rating_Factor".into(), rating_factors).into());

    // Before creating DataFrame, verify all series have same length
    let series_len = if !table_data.is_empty() {
        table_data[0].len()
    } else {
        0
    };

    if !table_data.iter().all(|s| s.len() == series_len) {
        panic!("Mismatched series lengths");
    }

    let result = RatingTable::new(
        DataFrame::new(table_data).unwrap(),
        None
    );

    result
}

/// Combines tables by merging those with overlapping features
pub fn combine_all_tables(mut tables: Vec<RatingTable>) -> Vec<RatingTable> {
    let mut made_changes = true;

    while made_changes {
        made_changes = false;
        let mut i = 0;

        while i < tables.len() {
            // Create parallel iterator for remaining tables
            let combinations: Vec<_> = ((i + 1)..tables.len())
                .into_par_iter()
                .filter_map(|j| {
                    let columns_i: HashSet<_> = tables[i].data.get_column_names().into_iter().collect();
                    let columns_j: HashSet<_> = tables[j].data.get_column_names().into_iter().collect();

                    if columns_i.is_subset(&columns_j) || columns_j.is_subset(&columns_i) {
                        // Return index and combined table
                        Some((j, if columns_i.len() > columns_j.len() {
                            expand_and_combine_tables(&tables[i], &tables[j])
                        } else {
                            expand_and_combine_tables(&tables[j], &tables[i])
                        }))
                    } else {
                        None
                    }
                })
                .collect();

            // Apply combinations if any were found
            if let Some((j, combined)) = combinations.first() {
                // Remove the table at index j and replace table i with combined version
                tables.remove(*j);
                tables[i] = combined.clone();
                made_changes = true;
            }

            i += 1;
        }
    }

    tables
}

pub fn combine_all_tables_exact(mut tables: Vec<RatingTable>) -> Vec<RatingTable> {
    let mut made_changes = true;
    while made_changes {
        made_changes = false;
        let mut i = 0;
        while i < tables.len() {
            let columns_i: HashSet<String> = tables[i]
                .data
                .get_column_names()
                .iter()
                .filter(|&n| *n != "Rating_Factor" && *n != "row_number")
                .map(|s| s.to_string())
                .collect();
            let mut found_index = None;
            for j in (i + 1)..tables.len() {
                let columns_j: HashSet<String> = tables[j]
                    .data
                    .get_column_names()
                    .iter()
                    .filter(|&n| *n != "Rating_Factor" && *n != "row_number")
                    .map(|s| s.to_string())
                    .collect();
                if columns_i == columns_j {
                    found_index = Some(j);
                    break;
                }
            }
            if let Some(j) = found_index {
                // Here we combine the two tables (assumed to have an identical feature layout)
                // using your existing 'expand_and_combine_tables' helper function.
                let new_table = expand_and_combine_tables(&tables[i], &tables[j]);
                tables.remove(j);
                tables[i] = new_table;
                made_changes = true;
            } else {
                i += 1;
            }
        }
    }
    tables
}
