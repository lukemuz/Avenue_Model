use polars::error::PolarsError;
use std::collections::HashSet;
use serde_json::Value;


fn find_feature_sets(model_json: &str) -> Result<HashSet<Vec<String>>, PolarsError> {
    let model: Value = serde_json::from_str(model_json)
        .map_err(|e| PolarsError::ComputeError(format!("JSON parsing error: {}", e).into()))?;
    
    // Validate model structure
    if !model.is_object() {
        return Err(PolarsError::ComputeError("Model JSON is not an object".into()));
    }

    let tree_info = model.get("tree_info")
        .ok_or_else(|| PolarsError::ComputeError("Missing tree_info field".into()))?
        .as_array()
        .ok_or_else(|| PolarsError::ComputeError("tree_info is not an array".into()))?;

    if tree_info.is_empty() {
        return Err(PolarsError::ComputeError("tree_info array is empty".into()));
    }

    let feature_names = model.get("feature_names")
        .ok_or_else(|| PolarsError::ComputeError("Missing feature_names field".into()))?
        .as_array()
        .ok_or_else(|| PolarsError::ComputeError("feature_names is not an array".into()))?;

    if feature_names.is_empty() {
        return Err(PolarsError::ComputeError("feature_names array is empty".into()));
    }
    
    let mut unique_feature_sets: HashSet<Vec<String>> = HashSet::new();
    
    // Process each tree
    for (tree_idx, tree_info) in tree_info.iter().enumerate() {
        let tree = tree_info.get("tree_structure")
            .ok_or_else(|| PolarsError::ComputeError(
                format!("Missing tree_structure in tree {}", tree_idx).into()
            ))?;

        collect_feature_sets(tree, &model, Vec::new(), &mut unique_feature_sets)
            .map_err(|e| PolarsError::ComputeError(
                format!("Error processing tree {}: {}", tree_idx, e).into()
            ))?;
    }
    
    if unique_feature_sets.is_empty() {
        return Err(PolarsError::ComputeError("No feature sets found in any tree".into()));
    }
    
    // Consolidate feature sets by removing subsets and deduplicating features
    let consolidated: HashSet<Vec<String>> = unique_feature_sets.iter()
        .map(|set| {
            // Deduplicate features within the set
            let deduped: HashSet<_> = set.iter().cloned().collect();
            let mut sorted_vec: Vec<_> = deduped.into_iter().collect();
            sorted_vec.sort();
            sorted_vec
        })
        .filter(|set| {
            // Keep this set only if it's not a subset of any other deduped set
            !unique_feature_sets.iter()
                .any(|other| {
                    if set == other {
                        return false;  // Don't compare a set with itself
                    }
                    let other_deduped: HashSet<_> = other.iter().cloned().collect();
                    let set_hash: HashSet<_> = set.iter().cloned().collect();
                    
                    // Only filter out if other_deduped is a proper superset
                    other_deduped.is_superset(&set_hash) && 
                    other_deduped.len() > set_hash.len()
                })
        })
        .collect();
    if consolidated.is_empty() {
        return Err(PolarsError::ComputeError("No feature sets remained after consolidation".into()));
    }

    Ok(consolidated)
}

fn collect_feature_sets(
    node: &Value,
    model: &Value,
    mut current_features: Vec<String>,
    unique_sets: &mut HashSet<Vec<String>>
) -> Result<(), PolarsError> {
    // Validate node structure
    if !node.is_object() {
        return Err(PolarsError::ComputeError("Node is not an object".into()));
    }

    // If this is a leaf node
    if node.get("leaf_index").is_some() || node.get("leaf_value").is_some() {
        if !current_features.is_empty() {
            let mut sorted_features: Vec<_> = current_features.into_iter().collect();
            sorted_features.sort();
            unique_sets.insert(sorted_features);
        }
        return Ok(());
    }
    
    // Get feature index with detailed error checking
    let feature_idx = node.get("split_feature")
        .ok_or_else(|| {
            let node_str = serde_json::to_string_pretty(node)
                .unwrap_or_else(|_| "[Failed to serialize node]".to_string());
            PolarsError::ComputeError(
                format!("Missing split_feature field in node: {}", node_str).into()
            )
        })?
        .as_i64()
        .ok_or_else(|| PolarsError::ComputeError("split_feature is not an integer".into()))? as usize;
    
    // Validate feature_names access
    let feature_names = model.get("feature_names")
        .ok_or_else(|| PolarsError::ComputeError("Missing feature_names field".into()))?
        .as_array()
        .ok_or_else(|| PolarsError::ComputeError("feature_names is not an array".into()))?;

    if feature_idx >= feature_names.len() {
        return Err(PolarsError::ComputeError(
            format!("Feature index {} out of bounds (max {})", feature_idx, feature_names.len() - 1).into()
        ));
    }
    
    let feature_name = feature_names[feature_idx]
        .as_str()
        .ok_or_else(|| PolarsError::ComputeError(
            format!("Feature name at index {} is not a string", feature_idx).into()
        ))?
        .to_string();
    
    current_features.push(feature_name);
    
    // Process children with detailed error checking
    match (node.get("left_child"), node.get("right_child")) {
        (Some(left), Some(right)) => {
            collect_feature_sets(left, model, current_features.clone(), unique_sets)?;
            collect_feature_sets(right, model, current_features, unique_sets)?;
        }
        (None, None) => {
            if !node.get("leaf_index").is_some() {
                return Err(PolarsError::ComputeError(
                    "Internal node has no children and is not a leaf".into()
                ));
            }
        }
        _ => {
            return Err(PolarsError::ComputeError(
                "Internal node must have both left and right children".into()
            ));
        }
    }
    
    Ok(())
}

pub fn estimate_number_of_tables(model_json: &str) -> Result<usize, PolarsError> {

    // Add validation for empty or malformed JSON
    if model_json.trim().is_empty() {
        return Err(PolarsError::ComputeError("Empty model JSON".into()));
    }

    let feature_sets = find_feature_sets(model_json)?;
    Ok(feature_sets.len() + 1)  // +1 for the mean table
}
