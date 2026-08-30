use polars::frame::DataFrame;
use polars::prelude::*;
use polars::series::IntoSeries;
use rayon::prelude::*;
use serde_json::Value;
use std::collections::HashMap;

use super::{LinkFunction, RatingTable};

#[derive(Debug, Clone)]
pub(super) struct SplitNodeInfo {
    pub feature_name: String,
    pub threshold: f64,
    pub decision_type: String,
    pub is_categorical: bool,
    pub categories: Vec<i32>,
}

#[derive(Debug, Clone)]
pub(super) struct PathInfo {
    pub path: Vec<SplitNodeInfo>,
    pub is_in_first_tree: bool,
    pub mean_adjustment: f64,
}

impl PathInfo {
    pub fn new(path: Vec<SplitNodeInfo>, is_in_first_tree: bool, mean_adjustment: f64) -> Self {
        Self {
            path,
            is_in_first_tree,
            mean_adjustment,
        }
    }

    pub fn create_df(&self) -> Result<DataFrame, PolarsError> {
        // Initialize maps to collect values for each feature
        let mut numeric_values: HashMap<String, Vec<f64>> = HashMap::new();
        let mut categorical_values: HashMap<String, Vec<i32>> = HashMap::new();

        // Process each node in the path to collect feature values
        for node in &self.path {
            if node.is_categorical {
                let values = categorical_values
                    .entry(node.feature_name.clone())
                    .or_insert_with(Vec::new);
                values.push(-999); // Always include wildcard
                values.extend(&node.categories);
            } else {
                let values = numeric_values
                    .entry(node.feature_name.clone())
                    .or_insert_with(Vec::new);
                values.push(node.threshold);
                // Only add infinity if this is the last threshold for this feature
                if !self
                    .path
                    .iter()
                    .any(|n| n.feature_name == node.feature_name && n.threshold > node.threshold)
                {
                    values.push(f64::INFINITY);
                }
            }
        }

        // Sort and dedupe values
        for values in numeric_values.values_mut() {
            values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Less));
            values.dedup_by(|a, b| {
                ((*a) - (*b)).abs() < 1e-10 || ((*a).is_infinite() && (*b).is_infinite())
            });
        }
        for values in categorical_values.values_mut() {
            values.sort_unstable();
            values.dedup();
        }

        // Convert categorical to feature_values map
        let mut feature_values: HashMap<String, Vec<f64>> = numeric_values;
        for (feature, values) in categorical_values {
            feature_values.insert(feature, values.into_iter().map(|x| x as f64).collect());
        }

        // Generate all combinations
        let mut combinations = vec![HashMap::new()];
        for (feature, values) in &feature_values {
            let mut new_combinations = Vec::new();
            for combo in combinations {
                for &value in values {
                    let mut new_combo = combo.clone();
                    new_combo.insert(feature.clone(), value);
                    new_combinations.push(new_combo);
                }
            }
            combinations = new_combinations;
        }

        // Convert to DataFrame
        let mut series_vec = Vec::new();
        if !combinations.is_empty() {
            // Create series for each feature
            for (feature, _) in &feature_values {
                let values: Vec<f64> = combinations
                    .iter()
                    .map(|combo| *combo.get(feature).unwrap())
                    .collect();
                // Check if this feature is categorical by looking at path nodes
                let is_categorical = self
                    .path
                    .iter()
                    .any(|node| node.feature_name == *feature && node.is_categorical);

                if is_categorical {
                    // Convert to i32 for categorical columns
                    let cat_values: Vec<i32> = values.iter().map(|&x| x as i32).collect();
                    series_vec.push(Series::new(feature.into(), cat_values).into());
                } else {
                    series_vec.push(Series::new(feature.into(), values).into());
                }
            }

            // Add Rating_Factor column (initialized to 0.0, will be updated later)
            series_vec
                .push(Series::new("Rating_Factor".into(), vec![0.0; combinations.len()]).into());
        }

        DataFrame::new(series_vec)
    }
}

struct LeafNodeInfo {
    leaf_value: f64,
    path_info: PathInfo,
}

impl LeafNodeInfo {
    fn new(leaf_value: f64, path_info: PathInfo) -> Self {
        Self {
            leaf_value,
            path_info,
        }
    }

    fn create_rating_table(&self) -> Result<RatingTable, PolarsError> {
        let mut df = self.path_info.create_df()?;
        let mut mask = Series::new("mask".into(), vec![true; df.height()]);

        for node in &self.path_info.path {
            let col = df.column(&node.feature_name)?;

            let node_mask = if node.is_categorical {
                let values = col.cast(&DataType::Int32)?;
                match node.decision_type.as_str() {
                    "==" => {
                        // Handle empty categories case
                        if node.categories.is_empty() {
                            // If no categories specified, nothing matches (all false)
                            Series::new("empty_mask".into(), vec![false; df.height()])
                        } else {
                            // For left branch, match when the value is among the categories
                            node.categories
                                .iter()
                                .fold(None, |acc, &cat| {
                                    let cat_series =
                                        Series::new("cat".into(), vec![cat; values.len()]);
                                    let eq = values.equal(&cat_series.into()).unwrap();
                                    Some(match acc {
                                        Some(a) => a | eq,
                                        None => eq,
                                    })
                                })
                                .unwrap_or_else(|| {
                                    // Should never reach here if we checked for empty above,
                                    // but providing a fallback just in case
                                    Series::new("fallback_mask".into(), vec![false; df.height()])
                                        .bool()
                                        .unwrap()
                                        .clone()
                                })
                                .into_series()
                        }
                    }
                    "!=" => {
                        // Handle empty categories case
                        if node.categories.is_empty() {
                            // If no categories specified, everything matches (all true)
                            Series::new("empty_mask".into(), vec![true; df.height()])
                        } else {
                            // For right branch, match when the value is NOT among the categories
                            let not_in_categories = node
                                .categories
                                .iter()
                                .fold(None, |acc, &cat| {
                                    let cat_series =
                                        Series::new("cat".into(), vec![cat; values.len()]);
                                    let eq = values.equal(&cat_series.into()).unwrap();
                                    Some(match acc {
                                        Some(a) => a & !eq,
                                        None => !eq,
                                    })
                                })
                                .unwrap_or_else(|| {
                                    // Fallback - should not be needed but safer
                                    Series::new("fallback_mask".into(), vec![true; df.height()])
                                        .bool()
                                        .unwrap()
                                        .clone()
                                });

                            // Also match -999 for wildcard
                            let wildcard_series =
                                Series::new("wildcard".into(), vec![-999; values.len()]);
                            let wildcard = values.equal(&wildcard_series.into())?;
                            (not_in_categories | wildcard).into_series()
                        }
                    }
                    _ => {
                        return Err(PolarsError::ComputeError(
                            format!("Invalid categorical decision type: {}", node.decision_type)
                                .into(),
                        ))
                    }
                }
            } else {
                let values = col.cast(&DataType::Float64)?;
                let threshold_series =
                    Series::new("threshold".into(), vec![node.threshold; values.len()]);
                match node.decision_type.as_str() {
                    "<=" => values.lt_eq(&threshold_series.into())?.into_series(),
                    ">" => values.gt(&threshold_series.into())?.into_series(),
                    _ => {
                        return Err(PolarsError::ComputeError(
                            format!("Invalid decision type: {}", node.decision_type).into(),
                        ))
                    }
                }
            };

            mask = (mask.bool()? & node_mask.bool()?).into_series();
        }

        let rating_factors: Vec<f64> = mask
            .bool()?
            .into_iter()
            .map(|v| match v {
                Some(true) => {
                    if self.path_info.is_in_first_tree {
                        self.leaf_value - self.path_info.mean_adjustment
                    } else {
                        self.leaf_value
                    }
                }
                _ => 0.0,
            })
            .collect();

        df.with_column(Series::new("Rating_Factor".into(), rating_factors))?;
        Ok(RatingTable::new(df, None))
    }
}

#[derive(Debug, Clone)]
struct NodeInfo {
    effect_value: f64,
    path_info: PathInfo,
}

impl NodeInfo {
    fn create_rating_table(&self) -> Result<RatingTable, PolarsError> {
        let mut df = self.path_info.create_df()?;
        let mut mask = Series::new("mask".into(), vec![true; df.height()]);

        for node in &self.path_info.path {
            let col = df.column(&node.feature_name)?;
            let node_mask = if node.is_categorical {
                let values = col.cast(&DataType::Int32)?;
                match node.decision_type.as_str() {
                    "==" => {
                        // Handle empty categories case
                        if node.categories.is_empty() {
                            // If no categories specified, nothing matches (all false)
                            Series::new("empty_mask".into(), vec![false; df.height()])
                        } else {
                            // For left branch, match when the value is among the categories
                            node.categories
                                .iter()
                                .fold(None, |acc, &cat| {
                                    let cat_series =
                                        Series::new("cat".into(), vec![cat; values.len()]);
                                    let eq = values.equal(&cat_series.into()).unwrap();
                                    Some(match acc {
                                        Some(a) => a | eq,
                                        None => eq,
                                    })
                                })
                                .unwrap_or_else(|| {
                                    // Should never reach here if we checked for empty above,
                                    // but providing a fallback just in case
                                    Series::new("fallback_mask".into(), vec![false; df.height()])
                                        .bool()
                                        .unwrap()
                                        .clone()
                                })
                                .into_series()
                        }
                    }
                    "!=" => {
                        // Handle empty categories case
                        if node.categories.is_empty() {
                            // If no categories specified, everything matches (all true)
                            Series::new("empty_mask".into(), vec![true; df.height()])
                        } else {
                            // For right branch, match when the value is NOT among the categories
                            let not_in_categories = node
                                .categories
                                .iter()
                                .fold(None, |acc, &cat| {
                                    let cat_series =
                                        Series::new("cat".into(), vec![cat; values.len()]);
                                    let eq = values.equal(&cat_series.into()).unwrap();
                                    Some(match acc {
                                        Some(a) => a & !eq,
                                        None => !eq,
                                    })
                                })
                                .unwrap_or_else(|| {
                                    // Fallback - should not be needed but safer
                                    Series::new("fallback_mask".into(), vec![true; df.height()])
                                        .bool()
                                        .unwrap()
                                        .clone()
                                });

                            // Also match -999 for wildcard
                            let wildcard_series =
                                Series::new("wildcard".into(), vec![-999; values.len()]);
                            let wildcard = values.equal(&wildcard_series.into())?;
                            (not_in_categories | wildcard).into_series()
                        }
                    }
                    _ => {
                        return Err(PolarsError::ComputeError(
                            format!("Invalid categorical decision type: {}", node.decision_type)
                                .into(),
                        ))
                    }
                }
            } else {
                // For numeric columns, cast to Float64 and compare.
                let values = col.cast(&DataType::Float64)?;
                let threshold_series =
                    Series::new("threshold".into(), vec![node.threshold; values.len()]);
                match node.decision_type.as_str() {
                    "<=" => values.lt_eq(&threshold_series.into())?.into_series(),
                    ">" => values.gt(&threshold_series.into())?.into_series(),
                    _ => {
                        return Err(PolarsError::ComputeError(
                            format!("Invalid decision type: {}", node.decision_type).into(),
                        ))
                    }
                }
            };
            mask = (mask.bool()? & node_mask.bool()?).into_series();
        }

        let rating_factors: Vec<f64> = mask
            .bool()?
            .into_iter()
            .map(|v| match v {
                Some(true) => self.effect_value,
                _ => 0.0,
            })
            .collect();
        df.with_column(Series::new("Rating_Factor".into(), rating_factors))?;
        Ok(RatingTable::new(df, None))
    }
}

// Return RatingTables instead of modifying a vector
fn process_tree(
    node: &Value,
    is_first_tree: bool,
    mean_adjustment: f64,
    model: &Value,
) -> Result<Vec<RatingTable>, PolarsError> {
    let mut tables = Vec::new();
    let mut stack = vec![(node, Vec::new(), true)];

    while let Some((current_node, path, is_left)) = stack.pop() {
        if current_node.get("leaf_index").is_some() {
            // Process leaf node
            let leaf_value = current_node["leaf_value"]
                .as_f64()
                .ok_or_else(|| PolarsError::ComputeError("Missing leaf value".into()))?;

            // Create PathInfo for this leaf
            let path_info = PathInfo::new(path.clone(), is_first_tree, mean_adjustment);

            // Create LeafNodeInfo
            let leaf_info = LeafNodeInfo::new(leaf_value, path_info);

            // Create rating table and collect it
            let rating_table = leaf_info.create_rating_table()?;
            tables.push(rating_table);
        } else {
            // Process internal node (split node)
            let feature_idx = current_node["split_feature"]
                .as_i64()
                .ok_or_else(|| {
                    // Reached whenever a node carries no `split_feature`, which in
                    // practice means the booster is a bare leaf: LightGBM rejected every
                    // candidate split and returned a constant. A heavy
                    // `interaction_penalty` is the usual way to arrive here, so the
                    // message names that rather than the JSON key.
                    PolarsError::ComputeError(
                        "this booster has no splits - every candidate was rejected, so \
                         it predicts a constant and has no rating tables to convert. \
                         Lower interaction_penalty / interaction_complexity, or relax \
                         min_gain_to_split and min_data_in_leaf."
                            .into(),
                    )
                })?
                as usize;

            let feature_name = model["feature_names"][feature_idx]
                .as_str()
                .ok_or_else(|| PolarsError::ComputeError("Missing feature name".into()))?
                .to_string();

            let decision_type = current_node["decision_type"]
                .as_str()
                .ok_or_else(|| PolarsError::ComputeError("Missing decision type".into()))?;

            let is_categorical = decision_type == "==";

            // Handle threshold/categories based on split type
            let (threshold, categories) = if is_categorical {
                let cats = match &current_node["threshold"] {
                    Value::String(s) => s
                        .split("||")
                        .filter_map(|v| v.parse::<i32>().ok())
                        .collect::<Vec<i32>>(),
                    Value::Number(n) => vec![n.as_i64().unwrap() as i32],
                    _ => {
                        return Err(PolarsError::ComputeError(
                            "Invalid categorical threshold".into(),
                        ))
                    }
                };
                (0.0, cats)
            } else {
                let thresh = match &current_node["threshold"] {
                    Value::String(s) => s.parse::<f64>().map_err(|e| {
                        PolarsError::ComputeError(
                            format!("Invalid numeric threshold: {}", e).into(),
                        )
                    })?,
                    Value::Number(n) => n.as_f64().unwrap(),
                    _ => return Err(PolarsError::ComputeError("Missing threshold".into())),
                };
                (thresh, Vec::new())
            };

            // Create SplitNodeInfo for this node
            let split_info = SplitNodeInfo {
                feature_name: feature_name.clone(),
                threshold,
                decision_type: if is_categorical { "==" } else { "<=" }.to_string(),
                is_categorical,
                categories: categories.clone(),
            };

            // Process children
            if let Some(left_child) = current_node.get("left_child") {
                let mut left_path = path.clone();
                left_path.push(split_info.clone());
                stack.push((left_child, left_path, true));
            }

            if let Some(right_child) = current_node.get("right_child") {
                let mut right_path = path.clone();
                // For right path, adjust the decision type
                let mut right_split_info = split_info.clone();
                right_split_info.decision_type =
                    if is_categorical { "!=" } else { ">" }.to_string();
                right_path.push(right_split_info);
                stack.push((right_child, right_path, false));
            }
        }
    }

    Ok(tables)
}

pub fn process_lgbm_trees(model_json: &str) -> Result<Vec<RatingTable>, PolarsError> {
    let model: Value = serde_json::from_str(model_json)
        .map_err(|e| PolarsError::ComputeError(format!("JSON parsing error: {}", e).into()))?;

    let mut tables = Vec::new();

    // Extract overall mean from first tree's root node
    let mean_adjustment =
        if let Some(first_tree) = model["tree_info"][0]["tree_structure"].as_object() {
            // `.get`, not `[..]`: indexing a serde_json::Map panics with "no entry
            // found for key" on a missing key, and a tree that is a bare leaf has no
            // `internal_value` at all. LightGBM emits exactly that when every candidate
            // split is rejected, which a heavy `interaction_penalty` does routinely - so
            // a Pareto search over that parameter walks straight into it.
            if let Some(mean) = first_tree
                .get("internal_value")
                .and_then(|value| value.as_f64())
            {
                let mean_df =
                    DataFrame::new(vec![Series::new("Rating_Factor".into(), vec![mean]).into()])?;
                tables.push(RatingTable::new(mean_df, None));
                mean
            } else {
                0.0
            }
        } else {
            0.0
        };

    // Process each tree
    let tree_tables: Result<Vec<_>, _> = model["tree_info"]
        .as_array()
        .ok_or_else(|| PolarsError::ComputeError("Missing tree_info array".into()))?
        .par_iter() // Use parallel iterator
        .enumerate()
        .map(|(tree_idx, tree_info)| {
            process_tree(
                &tree_info["tree_structure"],
                tree_idx == 0,
                mean_adjustment,
                &model,
            )
        })
        .collect();

    // Combine all tables
    tables.extend(tree_tables?.into_iter().flatten());

    Ok(tables)
}

/// Traverses a LightGBM tree (provided as a JSON Value) and creates a collection
/// of RatingTables from internal nodes (using `internal_value`) and leaf nodes (using `leaf_value`).
/// The `is_first_tree` flag and `mean_adjustment` are passed along so that the first tree's
/// nodes can be treated specially (subtracting the overall mean, for example).
fn process_tree_analysis(
    node: &Value,
    is_first_tree: bool,
    mean_adjustment: f64,
    model: &Value,
    parent_value: Option<f64>,
) -> Result<Vec<RatingTable>, PolarsError> {
    let mut tables = Vec::new();

    // Extract the internal value for the root node
    let root_internal_value = node["internal_value"]
        .as_f64()
        .ok_or_else(|| PolarsError::ComputeError("Missing internal value in root node".into()))?;

    // Stack holds (node, path, parent_value, tree_level)
    // tree_level helps track which level of the tree we're in (0 = root)
    let mut stack = vec![(node, Vec::new(), parent_value, 0usize)];

    while let Some((current_node, path, parent_val, level)) = stack.pop() {
        // Get the current node's internal value if it exists
        let current_internal_value = current_node
            .get("internal_value")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0);

        // Only create tables for non-root nodes
        if !path.is_empty() && current_node.get("internal_value").is_some() {
            // Calculate the effect value - difference from parent
            let effect_value = match parent_val {
                Some(parent_value) => current_internal_value - parent_value,
                None => current_internal_value,
            };

            // Create a rating table for this internal node's effect
            let path_info = PathInfo::new(path.clone(), is_first_tree, mean_adjustment);
            let node_info = NodeInfo {
                effect_value,
                path_info,
            };
            tables.push(node_info.create_rating_table()?);
        }

        // Process leaf nodes
        if current_node.get("leaf_index").is_some() {
            let leaf_value = current_node["leaf_value"]
                .as_f64()
                .ok_or_else(|| PolarsError::ComputeError("Missing leaf value".into()))?;

            // For leaf nodes, the effect is the deviation from its parent internal node
            let effect_value = match parent_val {
                Some(parent_value) => leaf_value - parent_value,
                None => leaf_value,
            };

            // Create a rating table for the leaf effect
            let path_info = PathInfo::new(path.clone(), is_first_tree, mean_adjustment);
            let node_info = NodeInfo {
                effect_value,
                path_info,
            };
            tables.push(node_info.create_rating_table()?);
        }
        // Process internal nodes with splits
        else if current_node.get("split_feature").is_some() {
            // Extract split feature information
            let feature_idx = current_node["split_feature"]
                .as_i64()
                .ok_or_else(|| {
                    // Reached whenever a node carries no `split_feature`, which in
                    // practice means the booster is a bare leaf: LightGBM rejected every
                    // candidate split and returned a constant. A heavy
                    // `interaction_penalty` is the usual way to arrive here, so the
                    // message names that rather than the JSON key.
                    PolarsError::ComputeError(
                        "this booster has no splits - every candidate was rejected, so \
                         it predicts a constant and has no rating tables to convert. \
                         Lower interaction_penalty / interaction_complexity, or relax \
                         min_gain_to_split and min_data_in_leaf."
                            .into(),
                    )
                })?
                as usize;

            let feature_name = model["feature_names"][feature_idx]
                .as_str()
                .ok_or_else(|| PolarsError::ComputeError("Missing feature name".into()))?
                .to_string();

            let decision_type = current_node["decision_type"]
                .as_str()
                .ok_or_else(|| PolarsError::ComputeError("Missing decision type".into()))?;

            let is_categorical = decision_type == "==";

            // Extract threshold/categories
            let (threshold, categories) = if is_categorical {
                // Handle categorical features
                let cats = match &current_node["threshold"] {
                    Value::String(s) => s
                        .split("||")
                        .filter_map(|v| v.parse::<i32>().ok())
                        .collect(),
                    Value::Number(n) => vec![n.as_i64().unwrap() as i32],
                    _ => {
                        return Err(PolarsError::ComputeError(
                            "Invalid categorical threshold".into(),
                        ))
                    }
                };
                (0.0, cats)
            } else {
                // Handle numeric features
                let thresh = match &current_node["threshold"] {
                    Value::String(s) => s.parse::<f64>().map_err(|e| {
                        PolarsError::ComputeError(
                            format!("Invalid numeric threshold: {}", e).into(),
                        )
                    })?,
                    Value::Number(n) => n.as_f64().unwrap(),
                    _ => return Err(PolarsError::ComputeError("Missing threshold".into())),
                };
                (thresh, Vec::new())
            };

            // Create split info for left branch
            let left_split_info = SplitNodeInfo {
                feature_name: feature_name.clone(),
                threshold,
                // Left branch decision type
                decision_type: if is_categorical { "==" } else { "<=" }.to_string(),
                is_categorical,
                categories: categories.clone(),
            };

            // Create split info for right branch with proper decision type
            let right_split_info = SplitNodeInfo {
                feature_name,
                threshold,
                // Right branch decision type - critical for correct path traversal
                decision_type: if is_categorical { "!=" } else { ">" }.to_string(),
                is_categorical,
                categories,
            };

            // Process left child
            if let Some(left_child) = current_node.get("left_child") {
                let mut left_path = path.clone();
                // Add left split info to path
                left_path.push(left_split_info);
                // Pass current internal value as parent for the child
                stack.push((
                    left_child,
                    left_path,
                    Some(current_internal_value),
                    level + 1,
                ));
            }

            // Process right child
            if let Some(right_child) = current_node.get("right_child") {
                let mut right_path = path.clone();
                right_path.push(right_split_info);
                // Pass current internal value as parent for the child
                stack.push((
                    right_child,
                    right_path,
                    Some(current_internal_value),
                    level + 1,
                ));
            }
        }
    }

    Ok(tables)
}

pub fn build_consolidated_tablemodel(
    tables: Vec<RatingTable>,
    link_function: LinkFunction,
) -> super::RatingModel {
    use super::consolidation::combine_all_tables;
    let mut combined_tables = vec![tables[0].clone()];
    let consolidated = combine_all_tables(tables[1..].to_vec());
    combined_tables.extend(consolidated);
    super::RatingModel::new(combined_tables, link_function)
}

/// Revised build_analysis_tablemodel that uses internal node and leaf values
/// from the LightGBM JSON to construct lower‐level (analysis) tables.
pub fn build_analysis_tablemodel(
    model_json: &str,
    link_function: LinkFunction,
) -> Result<super::RatingModel, PolarsError> {
    use super::consolidation::combine_all_tables_exact;

    // Parse the model JSON.
    let model: Value = serde_json::from_str(model_json)
        .map_err(|e| PolarsError::ComputeError(format!("JSON parsing error: {}", e).into()))?;

    let mut tables: Vec<RatingTable> = Vec::new();

    // Extract the overall mean from the first tree's root internal value.
    // (This will become the mean table.)
    let mean_adjustment =
        if let Some(first_tree) = model["tree_info"][0]["tree_structure"].as_object() {
            // `.get`, not `[..]`: indexing a serde_json::Map panics with "no entry
            // found for key" on a missing key, and a tree that is a bare leaf has no
            // `internal_value` at all. LightGBM emits exactly that when every candidate
            // split is rejected, which a heavy `interaction_penalty` does routinely - so
            // a Pareto search over that parameter walks straight into it.
            if let Some(mean) = first_tree
                .get("internal_value")
                .and_then(|value| value.as_f64())
            {
                let mean_df =
                    DataFrame::new(vec![Series::new("Rating_Factor".into(), vec![mean]).into()])?;
                let mean_table = RatingTable::new(mean_df, None);
                tables.push(mean_table);
                mean
            } else {
                0.0
            }
        } else {
            0.0
        };

    // Process each tree using internal node/leaf values for analysis.
    if let Some(tree_info_array) = model["tree_info"].as_array() {
        for (tree_idx, tree_info) in tree_info_array.iter().enumerate() {
            // process_tree_analysis traverses a tree and builds RatingTables
            // from both internal and leaf nodes.
            let node_tables = process_tree_analysis(
                &tree_info["tree_structure"],
                tree_idx == 0,
                mean_adjustment,
                &model,
                None,
            )?;
            tables.extend(node_tables);
        }
    } else {
        return Err(PolarsError::ComputeError("Missing tree_info array".into()));
    }

    // Combine tables that have overlapping feature sets if needed.
    let consolidated_tables = combine_all_tables_exact(tables);

    Ok(super::RatingModel::new(consolidated_tables, link_function))
}
