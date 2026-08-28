//! Fixtures shared across test modules.

/// A minimal LightGBM model: one tree splitting on a numeric and a categorical
/// feature. Enough to exercise conversion without pinning any particular ensemble.
#[allow(dead_code)]
pub fn simple_lgbm_json() -> String {
    r#"{
        "objective": "poisson",
        "feature_names": ["numeric_feat", "categorical_feat"],
        "tree_info": [{
            "tree_structure": {
                "internal_value": 0.5,
                "split_feature": 0,
                "threshold": "1.0",
                "decision_type": "<=",
                "left_child": {
                    "split_feature": 1,
                    "threshold": "1||2",
                    "decision_type": "==",
                    "left_child": { "leaf_value": 0.3, "leaf_index": 0 },
                    "right_child": { "leaf_value": 0.4, "leaf_index": 1 }
                },
                "right_child": { "leaf_value": 0.7, "leaf_index": 2 }
            }
        }]
    }"#
    .to_string()
}
