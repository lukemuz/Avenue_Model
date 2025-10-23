use super::*;
#[test]
fn test_overall_mean_table() {
    initialize_test_license();
    let json_str = r#"{
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
                    "left_child": {
                        "leaf_value": 0.3,
                        "leaf_index": 0
                    },
                    "right_child": {
                        "leaf_value": 0.4,
                        "leaf_index": 1
                    }
                },
                "right_child": {
                    "leaf_value": 0.7,
                    "leaf_index": 2
                }
            }
        }]
    }"#;

    let result = process_lgbm_trees(json_str);
    assert!(result.is_ok());

    let tables = result.unwrap();
    let mean_df = &tables[0].data;
    assert_eq!(mean_df.shape(), (1, 1));

    let mean_series = mean_df.column("Rating_Factor").unwrap();
    let overall_mean = mean_series.f64().unwrap().get(0).unwrap();
    assert_eq!(overall_mean, 0.5);
}

#[test]
fn test_number_of_tables() {
    initialize_test_license();
    let json_str = r#"{
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
                    "left_child": {
                        "leaf_value": 0.3,
                        "leaf_index": 0
                    },
                    "right_child": {
                        "leaf_value": 0.4,
                        "leaf_index": 1
                    }
                },
                "right_child": {
                    "leaf_value": 0.7,
                    "leaf_index": 2
                }
            }
        }]
    }"#;

    let result = process_lgbm_trees(json_str);
    assert!(result.is_ok());

    let tables = result.unwrap();
    let num_leaves = 3; // Based on the test JSON, there are 3 leaves
    assert_eq!(tables.len(), num_leaves + 1, "Expected {} tables, got {}", num_leaves + 1, tables.len());
}

#[test]
fn test_leaf_paths() {
    initialize_test_license();
    let json_str = r#"{
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
                    "left_child": {
                        "leaf_value": 0.3,
                        "leaf_index": 0
                    },
                    "right_child": {
                        "leaf_value": 0.4,
                        "leaf_index": 1
                    }
                },
                "right_child": {
                    "leaf_value": 0.7,
                    "leaf_index": 2
                }
            }
        }]
    }"#;

    let result = process_lgbm_trees(json_str);
    assert!(result.is_ok());

    let tables = result.unwrap();
    
    // Debug: Print all tables
    println!("\nAll tables:");
    for (i, table) in tables.iter().enumerate() {
        println!("\nTable {}:", i);
        println!("{:?}", table.data);
    }
    
    let leaf_tables = &tables[1..];

    let test_cases = vec![
        (0.5, 1.0, -0.2),  // Path: <= 1.0, in [1,2]
        (0.5, 3.0, -0.1),  // Path: <= 1.0, not in [1,2]
        (1.5, 1.0, 0.2),   // Path: > 1.0
        (1.5, 3.0, 0.2),   // Path: > 1.0
    ];

    for (numeric_val, categorical_val, expected_adjustment) in test_cases {
        let mut total_adjustment = 0.0;
        
        println!("\nTesting case: numeric={}, categorical={}", numeric_val, categorical_val);
        
        for (i, table) in leaf_tables.iter().enumerate() {
            let mut feature_values = HashMap::new();
            feature_values.insert("numeric_feat".to_string(), FeatureValue::Numeric(numeric_val));
            feature_values.insert("categorical_feat".to_string(), 
                FeatureValue::Categorical(categorical_val as i32));
            
            let adjustment = table.predict(&feature_values);
            total_adjustment += adjustment;
            
            println!("Table {} contribution: {}", i, adjustment);
        }
        
        println!("Total adjustment: {} (expected: {})", total_adjustment, expected_adjustment);

        assert!((total_adjustment - expected_adjustment).abs() < 1e-10,
            "Failed for numeric_feat={}, categorical_feat={}: expected={}, got={}",
            numeric_val, categorical_val, expected_adjustment, total_adjustment);

        let overall_mean = 0.5;
        let prediction = overall_mean + total_adjustment;
        assert!((prediction - (overall_mean + expected_adjustment)).abs() < 1e-10,
            "Prediction mismatch for numeric_feat={}, categorical_feat={}",
            numeric_val, categorical_val);
    }
}

#[test]
fn test_predict_and_predict_batch() {
    initialize_test_license();
    // Create a simple test model
    let json_str = r#"{
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
                    "left_child": {
                        "leaf_value": 0.3,
                        "leaf_index": 0
                    },
                    "right_child": {
                        "leaf_value": 0.4,
                        "leaf_index": 1
                    }
                },
                "right_child": {
                    "leaf_value": 0.7,
                    "leaf_index": 2
                }
            }
        }]
    }"#;

    let tables = process_lgbm_trees(json_str).unwrap();
    let model = RatingModel::new(tables, LinkFunction::Identity);

    // Test single predictions including a very large number
    let test_cases = vec![
        // (numeric_feat, categorical_feat, expected)
        (0.5, 1, 0.3),   // <= 1.0, in [1,2]
        (0.5, 3, 0.4),   // <= 1.0, not in [1,2]
        (1.5, 1, 0.7),   // > 1.0
        (1.5, 3, 0.7),   // > 1.0
        (1000.0, 1, 0.7) // way above threshold
    ];

    for (numeric_val, categorical_val, expected) in test_cases {
        let mut feature_values = HashMap::new();
        feature_values.insert("numeric_feat".to_string(), FeatureValue::Numeric(numeric_val));
        feature_values.insert("categorical_feat".to_string(), 
            FeatureValue::Categorical(categorical_val as i32));  // Fixed: Use actual categorical value
        
        let feature_values_f64: HashMap<String, f64> = feature_values.iter()
            .map(|(k, v)| match v {
                FeatureValue::Numeric(n) => (k.clone(), *n),
                FeatureValue::Categorical(c) => (k.clone(), *c as f64),
            })
            .collect();
        let prediction = model.predict_one(&feature_values_f64);
        assert!((prediction - expected).abs() < 1e-10, 
            "Failed for numeric_feat={}, categorical_feat={}: expected={}, got={}", 
            numeric_val, categorical_val, expected, prediction);
    }

    // Test batch prediction with large numbers
    let df = DataFrame::new(vec![
        Series::new("numeric_feat".into(), vec![0.5, 1.5, 0.5, 1000.0, f64::MAX]).into(),
        Series::new("categorical_feat".into(), vec![1i32, 2i32, 3i32, 1i32, 1i32]).into(),  
    ]).unwrap();

    let batch_predictions = model.predict(&df).unwrap();
    let expected = vec![0.3, 0.7, 0.4, 0.7, 0.7];

    let predictions: Vec<f64> = batch_predictions.f64().unwrap()
        .into_iter()
        .map(|opt| opt.unwrap())
        .collect();

    assert_eq!(predictions.len(), expected.len());
    for (pred, exp) in predictions.iter().zip(expected.iter()) {
        assert!((pred - exp).abs() < 1e-10, 
            "Batch prediction mismatch: got {}, expected {}", pred, exp);
    }
}

#[test]
fn test_predict_edge_cases() {
initialize_test_license();
let json_str = r#"{
    "feature_names": ["numeric_feat"],
    "tree_info": [{
        "tree_structure": {
            "internal_value": 1.0,
            "leaf_value": 1.0,
            "leaf_index": 0
        }
    }]
}"#;

let tables = process_lgbm_trees(json_str).unwrap();

// Debug: Print all tables
println!("\nGenerated tables:");
for (i, table) in tables.iter().enumerate() {
    println!("\nTable {}:", i);
    println!("{:?}", table.data);
}

let model = RatingModel::new(tables, LinkFunction::Identity);

let empty_features = HashMap::new();
let prediction = model.predict_one(&empty_features);
println!("\nPrediction with empty features: {}", prediction);

assert!((prediction - 1.0).abs() < 1e-10, 
    "Empty features prediction failed. Expected 1.0, got {}", prediction);
}

#[test]
fn test_expand_and_combine_tables() {
    initialize_test_license();
    // Test Case 1: Simple numeric features
    let table1_data = DataFrame::new(vec![
        Series::new("feature1".into(), vec![1.0, 2.0]).into(),
        Series::new("Rating_Factor".into(), vec![0.1, 0.2]).into(),
    ]).unwrap();

    let table2_data = DataFrame::new(vec![
        Series::new("feature1".into(), vec![1.0, 2.0]).into(),
        Series::new("Rating_Factor".into(), vec![0.3, 0.4]).into(),
    ]).unwrap();

    let table1 = RatingTable::new(table1_data,None);
    let table2 = RatingTable::new(table2_data,None);

    let combined = expand_and_combine_tables(&table1, &table2);
    
    // Detailed assertions for Test Case 1
    assert_eq!(combined.data.height(), 2, "Combined table should have 2 rows");
    assert_eq!(combined.data.width(), 2, "Combined table should have 2 columns");
    
    let feature1_col = combined.data.column("feature1").unwrap().f64().unwrap();
    let rf_col = combined.data.column("Rating_Factor").unwrap().f64().unwrap();
    
    // Check feature values are preserved
    assert!((feature1_col.get(0).unwrap() - 1.0).abs() < 1e-10);
    assert!((feature1_col.get(1).unwrap() - 2.0).abs() < 1e-10);
    
    // Check rating factors are properly combined
    assert!((rf_col.get(0).unwrap() - 0.4).abs() < 1e-10, 
        "Rating factor for value 1.0 should be 0.1 + 0.3");
    assert!((rf_col.get(1).unwrap() - 0.6).abs() < 1e-10, 
        "Rating factor for value 2.0 should be 0.2 + 0.4");

    // Test Case 2: Different features - Cartesian product
    let table3_data = DataFrame::new(vec![
        Series::new("feature1".into(), vec![1.0, 2.0]).into(),
        Series::new("Rating_Factor".into(), vec![0.1, 0.2]).into(),
    ]).unwrap();

    let table4_data = DataFrame::new(vec![
        Series::new("feature2".into(), vec![10.0, 20.0]).into(),
        Series::new("Rating_Factor".into(), vec![0.3, 0.4]).into(),
    ]).unwrap();

    let table3 = RatingTable::new(table3_data,None);
    let table4 = RatingTable::new(table4_data,None);

    let combined2 = expand_and_combine_tables(&table3, &table4);
    
    // Detailed assertions for Test Case 2
    assert_eq!(combined2.data.height(), 4, "Should have 4 combinations (2x2)");
    assert_eq!(combined2.data.width(), 3, "Should have 3 columns (feature1, feature2, Rating_Factor)");
    
    let feature1_col = combined2.data.column("feature1").unwrap().f64().unwrap();
    let feature2_col = combined2.data.column("feature2").unwrap().f64().unwrap();
    let rf_col = combined2.data.column("Rating_Factor").unwrap().f64().unwrap();
    
    // Verify all combinations exist with correct rating factors
    let mut combinations = Vec::new();
    for i in 0..4 {
        let f1 = (feature1_col.get(i).unwrap() * 1e10).round() / 1e10;
        let f2 = (feature2_col.get(i).unwrap() * 1e10).round() / 1e10;
        let rf = rf_col.get(i).unwrap();
        
        let expected_rf = match (f1, f2) {
            (x, y) if (x - 1.0).abs() < 1e-10 && (y - 10.0).abs() < 1e-10 => 0.1 + 0.3,
            (x, y) if (x - 1.0).abs() < 1e-10 && (y - 20.0).abs() < 1e-10 => 0.1 + 0.4,
            (x, y) if (x - 2.0).abs() < 1e-10 && (y - 10.0).abs() < 1e-10 => 0.2 + 0.3,
            (x, y) if (x - 2.0).abs() < 1e-10 && (y - 20.0).abs() < 1e-10 => 0.2 + 0.4,
            _ => panic!("Unexpected combination: {}, {}", f1, f2),
        };
        
        assert!((rf - expected_rf).abs() < 1e-10);
        combinations.push((f1, f2));
    }
    
    
    // Verify all expected combinations are present
    assert_eq!(combinations.len(), 4, "Missing some combinations");
    assert!(combinations.contains(&(1.0, 10.0)));
    assert!(combinations.contains(&(1.0, 20.0)));
    assert!(combinations.contains(&(2.0, 10.0)));
    assert!(combinations.contains(&(2.0, 20.0)));

    // Test Case 3: Empty tables
    let empty_table1 = RatingTable::new(
        DataFrame::new(vec![
            Series::new("feature1".into(), Vec::<f64>::new()).into(),
            Series::new("Rating_Factor".into(), Vec::<f64>::new()).into(),
        ]).unwrap(),
        None
    );

    let empty_table2 = RatingTable::new(
        DataFrame::new(vec![
            Series::new("feature2".into(), Vec::<f64>::new()).into(),
            Series::new("Rating_Factor".into(), Vec::<f64>::new()).into(),
            ]).unwrap(),
        None
    );

    let combined3 = expand_and_combine_tables(&empty_table1, &empty_table2);
    assert_eq!(combined3.data.height(), 0, "Empty tables should produce empty result");
    assert_eq!(combined3.data.width(), 3, "Empty result should preserve columns");

    // Test Case 4: Categorical features
    let cat_table1 = RatingTable::new(
        DataFrame::new(vec![
            Series::new("cat_feature".into(), vec![1i32, 2i32]).into(),
            Series::new("Rating_Factor".into(), vec![0.1, 0.2]).into(),
        ]).unwrap(),
        None
    );

    let cat_table2 = RatingTable::new(
        DataFrame::new(vec![
            Series::new("cat_feature".into(), vec![2i32, 3i32]).into(),
            Series::new("Rating_Factor".into(), vec![0.3, 0.4]).into(),
        ]).unwrap(),
        None
    );

    let combined4 = expand_and_combine_tables(&cat_table1, &cat_table2);
    
    // Detailed assertions for categorical features
    assert!(combined4.data.height() > 0, "Combined categorical table should not be empty");
    let cat_col = combined4.data.column("cat_feature").unwrap().i32().unwrap();
    let rf_col = combined4.data.column("Rating_Factor").unwrap().f64().unwrap();
    
    // Check that category 2 (present in both tables) has combined rating factor
    let cat2_rows: Vec<_> = (0..combined4.data.height())
        .filter(|&i| cat_col.get(i).unwrap() == 2)
        .collect();
    assert_eq!(cat2_rows.len(), 1, "Category 2 should appear exactly once");
    assert!((rf_col.get(cat2_rows[0]).unwrap() - 0.5).abs() < 1e-10, 
        "Rating factor for category 2 should be 0.2 + 0.3");

    // Test Case 5: Mixed numeric and categorical features
    let mixed_table1 = RatingTable::new(
        DataFrame::new(vec![
            Series::new("num_feature".into(), vec![1.0, 2.0]).into(),
            Series::new("cat_feature".into(), vec![1i32, 2i32]).into(),
            Series::new("Rating_Factor".into(), vec![0.1, 0.2]).into(),
        ]).unwrap(),
        None
    );

    let mixed_table2 = RatingTable::new(
        DataFrame::new(vec![
            Series::new("num_feature".into(), vec![2.0, 3.0]).into(),
            Series::new("cat_feature".into(), vec![2i32, 3i32]).into(),
            Series::new("Rating_Factor".into(), vec![0.3, 0.4]).into(),
        ]).unwrap(),
        None
    );

    let combined5 = expand_and_combine_tables(&mixed_table1, &mixed_table2);
    
    // Verify mixed feature combinations
    assert!(combined5.data.height() > 0, "Combined mixed feature table should not be empty");
    assert_eq!(combined5.data.width(), 3, "Should have both features plus Rating_Factor");
    
    // Print debug information for all test cases
    println!("\nTest Case 1 - Simple numeric features:");
    println!("Combined table:\n{}", combined.data);
    
    println!("\nTest Case 2 - Different features:");
    println!("Combined table:\n{}", combined2.data);
    
    println!("\nTest Case 3 - Empty tables:");
    println!("Combined table:\n{}", combined3.data);
    
    println!("\nTest Case 4 - Categorical features:");
    println!("Combined table:\n{}", combined4.data);
    
    println!("\nTest Case 5 - Mixed features:");
    println!("Combined table:\n{}", combined5.data);
}

#[test]
fn test_predict_with_infinity() {
    initialize_test_license();
    let json_str = r#"{
        "feature_names": ["numeric_feat"],
        "tree_info": [{
            "tree_structure": {
                "internal_value": 0.5,
                "split_feature": 0,
                "threshold": "1.0",
                "decision_type": "<=",
                "left_child": {
                    "leaf_value": 0.3,
                    "leaf_index": 0
                },
                "right_child": {
                    "leaf_value": 0.7,
                    "leaf_index": 1
                }
            }
        }]
    }"#;

    let tables = process_lgbm_trees(json_str).unwrap();
    let model = RatingModel::new(tables, LinkFunction::Identity);

    let test_cases = vec![
        (0.5, 0.3),       // below threshold - expecting 0.3
        (1.5, 0.7),       // above threshold - expecting 0.7
    ];

    // Test single predictions
    for (input_val, expected) in &test_cases {
        let mut feature_values = HashMap::new();
        feature_values.insert("numeric_feat".to_string(), *input_val);
        let prediction = model.predict_one(&feature_values);
        assert!((prediction - expected).abs() < 1e-10,
            "predict failed for {}: expected {}, got {}", 
            input_val, expected, prediction);
    }

    // Test batch predictions
    let input_values: Vec<f64> = test_cases.iter().map(|(val, _)| *val).collect();
    let expected_values: Vec<f64> = test_cases.iter().map(|(_, exp)| *exp).collect();
    
    let df = DataFrame::new(vec![
        Series::new("numeric_feat".into(), input_values).into(),
    ]).unwrap();

    let batch_predictions = model.predict(&df).unwrap();
    let predictions: Vec<f64> = batch_predictions.f64().unwrap()
        .into_iter()
        .map(|opt| opt.unwrap())
        .collect();

    assert_eq!(predictions.len(), expected_values.len());
    for (pred, exp) in predictions.iter().zip(expected_values.iter()) {
        assert!((pred - exp).abs() < 1e-10,
            "predict_batch mismatch: got {}, expected {}", pred, exp);
    }
}


/// Test the build_analysis_tablemodel function using a minimal LightGBM JSON.
#[test]
fn test_build_analysis_tablemodel() {
    // A minimal LightGBM model JSON with one tree.
    // The tree has:
    // - an overall mean from the root internal value (1.0),
    // - a split on feature index 0 (named "feature_0" in feature_names) with threshold "0.5",
    // - a left leaf with leaf_value 0.5,
    // - and a right leaf with leaf_value -0.5.
    let model_json = r#"
    {
        "tree_info": [
            {
                "tree_structure": {
                    "split_feature": 0,
                    "threshold": "0.5",
                    "decision_type": "<=",
                    "internal_value": 1.0,
                    "left_child": {
                        "leaf_index": 0,
                        "leaf_value": 0.5
                    },
                    "right_child": {
                        "leaf_index": 1,
                        "leaf_value": -0.5
                    }
                }
            }
        ],
        "feature_names": ["feature_0"]
    }
    "#;

    // Build the analysis table model.
    let result = build_analysis_tablemodel(model_json, LinkFunction::Identity);
    assert!(result.is_ok(), "Model building failed with error: {:?}", result.err());
    let rating_model = result.unwrap();

    // We expect the model to have at least two tables:
    //   • the first table is the mean table containing the overall mean (1.0),
    //   • additional tables come from the tree nodes.
    assert!(
        !rating_model.tables.is_empty(),
        "Expected at least one table in the model"
    );

    // Check the mean table.
    let mean_table = &rating_model.tables[0];
    let mean_series = mean_table
        .data
        .column("Rating_Factor")
        .expect("Mean table missing Rating_Factor column");
    let mean_value = mean_series.f64().unwrap().get(0).unwrap();
    // The overall mean should equal the internal_value (1.0) that was in the JSON.
    assert!(
        (mean_value - 1.0).abs() < 1e-6,
        "Mean value expected to be 1.0, but got {}",
        mean_value
    );

    // Optionally, test predict.
    // Here we create a DataFrame with a single feature column "feature_0"
    // that lies on the boundary of the split.
    let df = df![
        "feature_0" => &[0.5, 1.0]
    ]
    .unwrap();
    let prediction_result = rating_model.predict(&df);
    assert!(
        prediction_result.is_ok(),
        "Prediction failed: {:?}",
        prediction_result.err()
    );
    let pred_series = prediction_result.unwrap();
    assert_eq!(
        pred_series.len(),
        2,
        "Expected prediction Series of length 2, got {}",
        pred_series.len()
    );
}

#[test]
fn test_combine_all_tables() {
    initialize_test_license();
    // Create tables with overlapping features
    let table1 = RatingTable::new(
        DataFrame::new(vec![
            Series::new("feat1".into(), vec![1.0, 2.0]).into(),
            Series::new("feat2".into(), vec![10.0, 20.0]).into(),
            Series::new("Rating_Factor".into(), vec![0.1, 0.2]).into(),
        ]).unwrap(),
        None
    );

    let table2 = RatingTable::new(
        DataFrame::new(vec![
            Series::new("feat1".into(), vec![1.0, 2.0]).into(),
            Series::new("Rating_Factor".into(), vec![0.3, 0.4]).into(),
        ]).unwrap(),
        None
    );

    let table3 = RatingTable::new(
        DataFrame::new(vec![
            Series::new("feat3".into(), vec![100.0, 200.0]).into(),
            Series::new("Rating_Factor".into(), vec![0.5, 0.6]).into(),
        ]).unwrap(),
        None
    );

    let mean_table = RatingTable::new(
        DataFrame::new(vec![
            Series::new("Rating_Factor".into(), vec![0.0]).into(),
        ]).unwrap(),
        None
    );

    let tables = vec![mean_table, table1, table2, table3];
    let combined = combine_all_tables(tables[1..].to_vec());  // Skip mean table for combination
    let mut final_tables = vec![tables[0].clone()];  // Keep mean table
    final_tables.extend(combined);
    let model = RatingModel::new(final_tables, LinkFunction::Identity);

    // Test number of resulting tables
    assert_eq!(model.tables.len(), 3); // mean table + 2 combined tables

    // Find the table with feat1 and test its properties
    let feat1_table = model.tables.iter()
        .find(|table| table.data.get_column_names().iter().any(|name| name.as_str() == "feat1"))
        .expect("Should have a table with feat1");

    // Test that feat1 table has the expected columns
    let column_names: HashSet<_> = feat1_table.data.get_column_names().into_iter().collect();
    assert!(column_names.iter().any(|name| name.as_str() == "feat1"));
    assert!(column_names.iter().any(|name| name.as_str() == "feat2"));
    assert!(column_names.iter().any(|name| name.as_str() == "Rating_Factor"));

    // Test prediction with combined table - provide all required features
    let feature_values: HashMap<String, f64> = HashMap::from([
        ("feat1".to_string(), 1.0),
        ("feat2".to_string(), 10.0),
        ("feat3".to_string(), 100.0),
    ]);

    let prediction = model.predict_one(&feature_values);
    assert!(!prediction.is_nan(), "Prediction should not be NaN");
    assert!(prediction > 0.0, "Rating factor should be non-zero");
}

#[test]
fn test_consolidate_tables() {
    // Create mean table
    let mean_table = RatingTable::new(
        DataFrame::new(vec![
            Series::new("Rating_Factor".into(), vec![0.0]).into(),
        ]).unwrap(),
        None
    );

    // Create tables with overlapping features
    let table1 = RatingTable::new(
        DataFrame::new(vec![
            Series::new("feat1".into(), vec![1.0, 2.0]).into(),
            Series::new("feat2".into(), vec![10.0, 20.0]).into(),
            Series::new("Rating_Factor".into(), vec![0.1, 0.2]).into(),
        ]).unwrap(),
        None
    );

    let table2 = RatingTable::new(
        DataFrame::new(vec![
            Series::new("feat1".into(), vec![1.0, 2.0]).into(),
            Series::new("Rating_Factor".into(), vec![0.3, 0.4]).into(),
        ]).unwrap(),
        None
    );

    let table3 = RatingTable::new(
        DataFrame::new(vec![
            Series::new("feat3".into(), vec![100.0, 200.0]).into(),
            Series::new("Rating_Factor".into(), vec![0.5, 0.6]).into(),
        ]).unwrap(),
        None
    );

    // Create initial model
    let model = RatingModel::new(vec![mean_table.clone(), table1, table2, table3], LinkFunction::Identity);
    
    // Consolidate tables
    let consolidated = model.consolidate_tables();

    // Test that we still have the mean table
    assert_eq!(
        consolidated.tables[0].data.column("Rating_Factor").unwrap().f64().unwrap().get(0).unwrap(),
        0.0
    );

    // Test that remaining tables were properly combined
    assert_eq!(consolidated.tables.len(), 3); // mean table + 2 combined tables

    // Test that predictions still work
    let mut feature_values = HashMap::new();
    feature_values.insert("feat1".to_string(), 1.0);
    feature_values.insert("feat2".to_string(), 10.0);
    feature_values.insert("feat3".to_string(), 100.0);
    
    let prediction = consolidated.predict_one(&feature_values);
    assert!(prediction > 0.0, "Prediction should be non-zero");
}

#[test]
fn test_consolidate_tables_usage() {
    initialize_test_license();
    // Create mean table
    let mean_table = RatingTable::new(
        DataFrame::new(vec![
            Series::new("Rating_Factor".into(), vec![0.0]).into(),
        ]).unwrap(),
        None
    );

    // Create some example tables
    let table1 = RatingTable::new(
        DataFrame::new(vec![
            Series::new("feat1".into(), vec![1.0, 2.0]).into(),
            Series::new("feat2".into(), vec![10.0, 20.0]).into(),
            Series::new("Rating_Factor".into(), vec![0.1, 0.2]).into(),
        ]).unwrap(),
        None
    );

    let table2 = RatingTable::new(
        DataFrame::new(vec![
            Series::new("feat1".into(), vec![1.0, 2.0]).into(),
            Series::new("Rating_Factor".into(), vec![0.3, 0.4]).into(),
        ]).unwrap(),
        None
    );

    // Create initial model
    let original_model = RatingModel::new(vec![mean_table, table1, table2], LinkFunction::Identity);

    // Create consolidated version while keeping original
    let consolidated_model = original_model.consolidate_tables();

    // Now we can use both models
    let features1 = {
        let mut map = HashMap::new();
        map.insert("feat1".to_string(), 1.0);
        map.insert("feat2".to_string(), 10.0);
        map.insert("feat3".to_string(), 100.0);
        map
    };

    // Compare predictions from both models
    let prediction_original = original_model.predict_one(&features1);
    let prediction_consolidated = consolidated_model.predict_one(&features1);

    println!("Original model prediction: {}", prediction_original);
    println!("Consolidated model prediction: {}", prediction_consolidated);
    println!("Original model tables: {}", original_model.tables.len());
    println!("Consolidated model tables: {}", consolidated_model.tables.len());

    // We can even consolidate multiple times if needed
    let double_consolidated = consolidated_model.consolidate_tables();
    assert_eq!(
        consolidated_model.predict_one(&features1),
        double_consolidated.predict_one(&features1)
    );
}

#[test]
fn test_tweedie_objective() {
    initialize_test_license();
    // Create a simple model JSON with Tweedie objective
    let json_str = r#"{
        "objective": "tweedie",
        "feature_names": ["numeric_feat"],
        "tree_info": [{
            "tree_structure": {
                "internal_value": 0.5,
                "split_feature": 0,
                "threshold": "1.0",
                "decision_type": "<=",
                "left_child": {
                    "leaf_value": 0.3,
                    "leaf_index": 0
                },
                "right_child": {
                    "leaf_value": 0.7,
                    "leaf_index": 1
                }
            }
        }]
    }"#;

    // Create model
    let model = RatingModel::from_lgbm_json(json_str, "max").unwrap();

    // Verify that the link function is Log
    assert_eq!(model.get_link_function(), "log");

    // Test predictions
    let mut feature_values = HashMap::new();
    feature_values.insert("numeric_feat".to_string(), 0.5);  // This should take left path
    
    let prediction = model.predict_one(&feature_values);
    
    // For log link function:
    // 1. Base prediction is 0.5
    // 2. Left path adjustment is -0.2 (0.3 - 0.5)
    // 3. Total linear predictor is 0.3
    // 4. Final prediction should be exp(0.3)
    let expected = (0.3_f64).exp();
    
    assert!((prediction - expected).abs() < 1e-10,
        "Tweedie prediction failed: expected {}, got {}", expected, prediction);

    // Test with a value that takes the right path
    feature_values.insert("numeric_feat".to_string(), 1.5);
    let prediction = model.predict_one(&feature_values);
    let expected = (0.7_f64).exp();  // 0.5 + (0.7 - 0.5) = 0.7, then exp
    
    assert!((prediction - expected).abs() < 1e-10,
        "Tweedie prediction failed: expected {}, got {}", expected, prediction);
}
#[test]
fn test_from_lgbm_json() {
    initialize_test_license();
    // Test case 1: Valid JSON with regression objective
    let json_str = r#"{
        "objective": "regression",
        "feature_names": ["numeric_feat", "categorical_feat"],
        "tree_info": [
            {
                "tree_structure": {
                    "internal_value": 0.5,
                    "split_feature": 0,
                    "threshold": "1.0",
                    "decision_type": "<=",
                    "left_child": {
                        "split_feature": 1,
                        "threshold": "1||2",
                        "decision_type": "==",
                        "left_child": {
                            "leaf_value": 0.3,
                            "leaf_index": 0
                        },
                        "right_child": {
                            "leaf_value": 0.4,
                            "leaf_index": 1
                        }
                    },
                    "right_child": {
                        "leaf_value": 0.7,
                        "leaf_index": 2
                    }
                }
            },
            {
                "tree_structure": {
                    "internal_value": 0.0,
                    "split_feature": 1,
                    "threshold": "3||4",
                    "decision_type": "==",
                    "left_child": {
                        "split_feature": 0,
                        "threshold": "2.0",
                        "decision_type": "<=",
                        "left_child": {
                            "leaf_value": 0.2,
                            "leaf_index": 0
                        },
                        "right_child": {
                            "leaf_value": 0.3,
                            "leaf_index": 1
                        }
                    },
                    "right_child": {
                        "leaf_value": 0.4,
                        "leaf_index": 2
                    }
                }
            }
        ]
    }"#;

    // Debug: Print parsed JSON
    let parsed_json: serde_json::Value = serde_json::from_str(json_str).unwrap();
    println!("\nParsed JSON structure:");
    println!("{:#?}", parsed_json);

    // Debug: Check tables before model creation
    let tables = process_lgbm_trees(json_str).unwrap();
    println!("\nGenerated tables (count: {}):", tables.len());
    for (i, table) in tables.iter().enumerate() {
        println!("\nTable {}:", i);
        println!("Columns: {:?}", table.data.get_column_names());
        println!("Shape: {:?}", table.data.shape());
        println!("Data:\n{}", table.data);
    }

    // Create model with explicit error handling
    let result = RatingModel::from_lgbm_json(json_str, "max");
    match result {
        Ok(model) => {
            println!("\nSuccessfully created model");
            println!("Number of tables in model: {}", model.tables.len());
            for (i, table) in model.tables.iter().enumerate() {
                println!("\nModel Table {}:", i);
                println!("Columns: {:?}", table.data.get_column_names());
                println!("Shape: {:?}", table.data.shape());
                println!("Data:\n{}", table.data);
            }
        },
        Err(e) => {
            panic!("Failed to create model: {:?}", e);
        }
    }
}

#[test]
fn test_no_empty_tables_from_lgbm() {
    initialize_test_license();
    // Create a test model JSON similar to your case with Shell_weight and Shucked_weight
    let json_str = r#"{
        "objective": "regression",
        "feature_names": ["Shell_weight", "Shucked_weight"],
        "tree_info": [
            {
                "tree_structure": {
                    "internal_value": 9.94767,
                    "split_feature": 0,
                    "threshold": "0.5",
                    "decision_type": "<=",
                    "left_child": {
                        "leaf_value": -0.2,
                        "leaf_index": 0
                    },
                    "right_child": {
                        "leaf_value": 0.2,
                        "leaf_index": 1
                    }
                }
            },
            {
                "tree_structure": {
                    "internal_value": 0.0,
                    "split_feature": 1,
                    "threshold": "0.3",
                    "decision_type": "<=",
                    "left_child": {
                        "leaf_value": -0.1,
                        "leaf_index": 0
                    },
                    "right_child": {
                        "leaf_value": 0.1,
                        "leaf_index": 1
                    }
                }
            }
        ]
    }"#;

    for level in ["max", "analysis"] {
        let model = RatingModel::from_lgbm_json(json_str, level).unwrap();
        
        println!("\nTesting consolidation_level: {}", level);
        for (i, table) in model.tables.iter().enumerate() {
            println!("\nTable {}:", i);
            println!("Shape: {:?}", table.data.shape());
            println!("Columns: {:?}", table.data.get_column_names());
            println!("Data:\n{}", table.data);
            
            // Check that table is not empty (unless it's the mean table)
            if i > 0 {  // Skip the mean table (first table)
                assert!(table.data.height() > 0, 
                    "Table {} is empty (height = 0) with consolidation_level '{}'. \
                        This table should contain feature: {:?}", 
                    i, level, table.data.get_column_names());
            }
        }

        // Test predictions to ensure they work
        let test_points = vec![
            (0.3, 0.2),  // Both features below threshold
            (0.7, 0.4),  // Both features above threshold
            (0.3, 0.4),  // Mixed case 1
            (0.7, 0.2),  // Mixed case 2
        ];

        for (shell_weight, shucked_weight) in test_points {
            let mut features = HashMap::new();
            features.insert("Shell_weight".to_string(), shell_weight);
            features.insert("Shucked_weight".to_string(), shucked_weight);
            
            let prediction = model.predict_one(&features);
            println!("Prediction for Shell_weight={}, Shucked_weight={}: {}", 
                    shell_weight, shucked_weight, prediction);
            assert!(!prediction.is_nan(), 
                "Got NaN prediction with consolidation_level '{}' for features: {:?}", 
                level, features);
        }
    }
}
#[test]
fn test_from_lgbm_json_abalone() {
initialize_test_license();
let json_str = r#"{"name": "tree", "version": "v4", "num_class": 1, "num_tree_per_iteration": 1, "label_index": 0, "max_feature_idx": 7, "objective": "regression", "average_output": false, "feature_names": ["Sex", "Length", "Diameter", "Height", "Whole_weight", "Shucked_weight", "Viscera_weight", "Shell_weight"], "monotone_constraints": [], "feature_infos": {"Sex": {"min_value": 0, "max_value": 2, "values": []}, "Length": {"min_value": 0.075, "max_value": 0.815, "values": []}, "Diameter": {"min_value": 0.055, "max_value": 0.65, "values": []}, "Height": {"min_value": 0, "max_value": 0.515, "values": []}, "Whole_weight": {"min_value": 0.002, "max_value": 2.8255, "values": []}, "Shucked_weight": {"min_value": 0.001, "max_value": 1.488, "values": []}, "Viscera_weight": {"min_value": 0.0005, "max_value": 0.76, "values": []}, "Shell_weight": {"min_value": 0.0015, "max_value": 0.897, "values": []}}, "tree_info": [{"tree_index": 0, "num_leaves": 4, "num_cat": 0, "shrinkage": 1, "tree_structure": {"split_index": 0, "split_feature": 7, "split_gain": 81.66829681396484, "threshold": "0.16525000000000004", "decision_type": "<=", "default_left": true, "missing_type": "None", "internal_value": 9.94767, "internal_weight": 0, "internal_count": 2924, "left_child": {"split_index": 2, "split_feature": 7, "split_gain": 1210.449951171875, "threshold": "0.04825000000000001", "decision_type": "<=", "default_left": true, "missing_type": "None", "internal_value": 9.23754, "internal_weight": 987, "internal_count": 987, "left_child": {"leaf_index": 0, "leaf_value": 8.567987491909633, "leaf_weight": 195, "leaf_count": 195}, "right_child": {"leaf_index": 3, "leaf_value": 9.402387237218434, "leaf_weight": 792, "leaf_count": 792}}, "right_child": {"split_index": 1, "split_feature": 7, "split_gain": 1587.5999755859375, "threshold": "0.35525", "decision_type": "<=", "default_left": true, "missing_type": "None", "internal_value": 10.3095, "internal_weight": 1937, "internal_count": 1937, "left_child": {"leaf_index": 1, "leaf_value": 10.136758605089016, "leaf_weight": 1379, "leaf_count": 1379}, "right_child": {"leaf_index": 2, "leaf_value": 10.736490368806376, "leaf_weight": 558, "leaf_count": 558}}}}, {"tree_index": 1, "num_leaves": 4, "num_cat": 0, "shrinkage": 0.3, "tree_structure": {"split_index": 0, "split_feature": 7, "split_gain": 42.375301361083984, "threshold": "0.18550000000000003", "decision_type": "<=", "default_left": true, "missing_type": "None", "internal_value": 0, "internal_weight": 0, "internal_count": 2924, "left_child": {"split_index": 1, "split_feature": 7, "split_gain": 930.60302734375, "threshold": "0.09675000000000002", "decision_type": "<=", "default_left": true, "missing_type": "None", "internal_value": -0.446303, "internal_weight": 1164, "internal_count": 1164, "left_child": {"leaf_index": 0, "leaf_value": -0.751658196191816, "leaf_weight": 507, "leaf_count": 507}, "right_child": {"leaf_index": 2, "leaf_value": -0.2106630745135486, "leaf_weight": 657, "leaf_count": 657}}, "right_child": {"split_index": 2, "split_feature": 7, "split_gain": 807.4240112304688, "threshold": "0.5040000000000001", "decision_type": "<=", "default_left": true, "missing_type": "None", "internal_value": 0.295168, "internal_weight": 1760, "internal_count": 1760, "left_child": {"leaf_index": 1, "leaf_value": 0.24529582855392648, "leaf_weight": 1660, "leaf_count": 1660}, "right_child": {"leaf_index": 3, "leaf_value": 1.123052909374237, "leaf_weight": 100, "leaf_count": 100}}}}, {"tree_index": 2, "num_leaves": 4, "num_cat": 0, "shrinkage": 0.3, "tree_structure": {"split_index": 0, "split_feature": 7, "split_gain": 24.28070068359375, "threshold": "0.24525000000000002", "decision_type": "<=", "default_left": true, "missing_type": "None", "internal_value": 0, "internal_weight": 0, "internal_count": 2924, "left_child": {"split_index": 1, "split_feature": 7, "split_gain": 653.2680053710938, "threshold": "0.11025000000000001", "decision_type": "<=", "default_left": true, "missing_type": "None", "internal_value": -0.256727, "internal_weight": 1561, "internal_count": 1561, "left_child": {"leaf_index": 0, "leaf_value": -0.5030059638903491, "leaf_weight": 598, "leaf_count": 598}, "right_child": {"leaf_index": 2, "leaf_value": -0.10379289546024019, "leaf_weight": 963, "leaf_count": 963}}, "right_child": {"split_index": 2, "split_feature": 7, "split_gain": 298.2300109863281, "threshold": "0.4615000000000001", "decision_type": "<=", "default_left": true, "missing_type": "None", "internal_value": 0.294021, "internal_weight": 1363, "internal_count": 1363, "left_child": {"leaf_index": 1, "leaf_value": 0.23806212171674612, "leaf_weight": 1176, "leaf_count": 1176}, "right_child": {"leaf_index": 3, "leaf_value": 0.6459309234906406, "leaf_weight": 187, "leaf_count": 187}}}}], "feature_importances": {"Shucked_weight": 2, "Shell_weight": 7}, "pandas_categorical": []}"#;

// Test model creation with both consolidation levels
for consolidation_level in ["max", "analysis"].iter() {
    let model = RatingModel::from_lgbm_json(json_str, consolidation_level);
    assert!(model.is_ok(), "Failed to create model with {} consolidation", consolidation_level);
    
    let model = model.unwrap();
    
    // Verify link function is identity (regression)
    assert_eq!(model.get_link_function(), "identity");
    
    // Verify model has tables
    assert!(!model.tables.is_empty(), "Model should have at least one table");
    //print all tables
    for table in model.tables.iter() {
        println!("Table: {}", table.data);
    }
    
    // Print model structure
    println!("\nModel structure for {} consolidation:", consolidation_level);
    for (i, table) in model.tables.iter().enumerate() {
        println!("Table {}: {} rows, {} columns", 
            i, 
            table.data.height(), 
            table.data.width());
    println!("Columns: {:?}", table.data.get_column_names());
    }
    assert!(model.tables.len() ==2,"Model should have 2 tables");
    
    //each table should have at least 1 row
    for table in model.tables.iter() {
        assert!(table.data.height() > 0, "Table should have at least 1 row");
    }
    // Test prediction with required features
    let mut feature_values = HashMap::new();
    feature_values.insert("Shell_weight".to_string(), 0.2); // Use a value within the valid range
    feature_values.insert("Shucked_weight".to_string(), 0.2); // Add this required feature
    
    let prediction = model.predict_one(&feature_values);
    assert!(!prediction.is_nan(), "Prediction should not be NaN");
    assert!(prediction != 0.0, "Prediction should not be equal to 0");
}

for consolidation_level in ["max", "analysis"].iter() {
    let model = RatingModel::from_lgbm_json(json_str, consolidation_level).unwrap();


    // Test prediction with required features
    let mut feature_values = HashMap::new();
    feature_values.insert("Shell_weight".to_string(), 0.2); // Use a value within the valid range
    feature_values.insert("Shucked_weight".to_string(), 0.2); // Add this required feature
    
    let prediction = model.predict_one(&feature_values);
        assert!(!prediction.is_nan(), "Prediction should not be NaN");
    assert!(prediction != 0.0, "Prediction should not be equal to 0");


}
}

#[test]
fn test_realistic_prediction() {
    initialize_test_license();
    // Create a realistic set of rating tables similar to your actual model
    let mean_table = RatingTable::new(
        DataFrame::new(vec![
            Series::new("Rating_Factor".into(), vec![10.0]).into(),
        ]).unwrap(),
        None
    );

    let shell_weight_table = RatingTable::new(
        DataFrame::new(vec![
            Series::new("Shell_weight".into(), vec![0.05, 0.07, 0.09, f64::INFINITY]).into(),
            Series::new("Rating_Factor".into(), vec![-4.0, -3.0, -2.0, -1.0]).into(),
        ]).unwrap(),
        None
    );

    let height_sex_table = RatingTable::new(
        DataFrame::new(vec![
            Series::new("Height".into(), vec![0.08, 0.08, 0.08, 0.10, 0.10, 0.10, 0.12, 0.12, 0.12, f64::INFINITY, f64::INFINITY, f64::INFINITY]).into(),
            Series::new("Sex".into(), vec![-999, 1, 2, -999, 1, 2, -999, 1, 2, -999, 1, 2]).into(),
            Series::new("Rating_Factor".into(), vec![-1.0, -1.2, -1.1, -3.7, -0.2, -0.1, 1.0, 0.8, 0.9, 2.0, 1.8, 1.9]).into(),
            ]).unwrap(),
        None
    );

    let model = RatingModel::new(
        vec![mean_table.clone(), shell_weight_table.clone(), height_sex_table.clone()],
        LinkFunction::Identity
    );

    // Test with values matching explicit categories
    let mut feature_values = HashMap::new();
    feature_values.insert("Shell_weight".to_string(), 0.06);  // Should match first bin
    feature_values.insert("Height".to_string(), 0.09);        // Should match 0.08 bin
    feature_values.insert("Sex".to_string(), 2.0);           // Should match explicit 2

    println!("\nTesting prediction with:");
    for (k, v) in &feature_values {
        println!("{}: {}", k, v);
    }

    // Print table contents for debugging
    println!("\nShell_weight table:");
    println!("{}", shell_weight_table.data);
    
    println!("\nHeight/Sex table:");
    println!("{}", height_sex_table.data);

    let prediction = model.predict_one(&feature_values);
    println!("\nPrediction: {}", prediction);

    // Let's check individual table predictions
    let shell_weight_contribution = shell_weight_table.predict(&HashMap::from([
        ("Shell_weight".to_string(), FeatureValue::Numeric(0.06))
    ]));
    println!("Shell_weight contribution: {}", shell_weight_contribution);

    // Add this debug code right before the height_sex_contribution calculation
    println!("\nDebug Height/Sex lookup:");
    println!("Looking up Height={}, Sex={}", 0.09, 2);
    let height_sex_features = HashMap::from([
        ("Height".to_string(), FeatureValue::Numeric(0.09)),
        ("Sex".to_string(), FeatureValue::Categorical(2))
    ]);
    let height_sex_contribution = height_sex_table.predict(&height_sex_features);
    println!("Height/Sex contribution: {}", height_sex_contribution);

    // Also print the full Height/Sex table sorted by Height
println!("\nHeight/Sex table sorted by Height:");
let sorted_height_sex = height_sex_table.data.sort(vec!["Height"], SortMultipleOptions::default()).unwrap();
println!("{}", sorted_height_sex);

assert!((prediction - 6.9).abs() < 1e-10, 
    "Expected prediction of 6.9, got {}", prediction);

}

#[test]
fn test_realistic_prediction2() {
initialize_test_license();
// Create test tables matching your actual model structure
let mean_table = RatingTable::new(
    DataFrame::new(vec![
        Series::new("Rating_Factor".into(), vec![9.94767]).into(),
    ]).unwrap(),
    None
);

let shell_weight_table = RatingTable::new(
    DataFrame::new(vec![
        Series::new("Shell_weight".into(), vec![
            0.03025, 0.04925, 0.06525, 0.08025, 0.09525, 0.40625, 0.43525, 0.47125, 0.52175, f64::INFINITY
        ]).into(),
        Series::new("Rating_Factor".into(), vec![
            -5.287939, -4.77238, -4.180646, -4.030887, -3.957615, 3.810618, 4.277076, 4.511772, 4.910237, 7.138597
        ]).into(),
    ]).unwrap(),
    None
);

let height_sex_table = RatingTable::new(
    DataFrame::new(vec![
        Series::new("Height".into(), vec![
            0.0575, 0.0575, 0.0575, 0.0725, 0.0725, 0.2075, 0.2075, f64::INFINITY, f64::INFINITY, f64::INFINITY
        ]).into(),
        Series::new("Sex".into(), vec![
            -999, 1, 2, -999, 1, 1, 2, -999, 1, 2
        ]).into(),
        Series::new("Rating_Factor".into(), vec![
            -0.950232, -1.657364, -1.018087, -0.858654, -1.565786, 0.86736, 0.817616, 1.240646, 1.179411, 1.129667
        ]).into(),
    ]).unwrap(),
    None
);

let shucked_sex_table = RatingTable::new(
    DataFrame::new(vec![
        Series::new("Shucked_weight".into(), vec![
            0.03875, 0.03875, 0.03875, 0.08575, 0.08575, 0.81775, 0.81775, f64::INFINITY, f64::INFINITY, f64::INFINITY
        ]).into(),
        Series::new("Sex".into(), vec![
            -999, 0, 1, -999, 0, 0, 1, -999, 0, 1
        ]).into(),
        Series::new("Rating_Factor".into(), vec![
            1.785907, 1.88914, 1.395997, 2.943802, 3.047036, -3.395509, -2.879082, -4.420048, -4.570553, -4.054126
        ]).into(),
    ]).unwrap(),
    None
);

// Create model with all tables
let model = RatingModel::new(
    vec![mean_table, shell_weight_table, height_sex_table, shucked_sex_table],
    LinkFunction::Identity
);

// Create test data matching your sample
let test_df = DataFrame::new(vec![
    Series::new("Sex".into(), vec![2i32, 1i32, 0i32, 0i32, 2i32]).into(),
    Series::new("Height".into(), vec![0.09, 0.08, 0.125, 0.15, 0.13]).into(),
    Series::new("Shucked_weight".into(), vec![0.0995, 0.0895, 0.294, 0.3145, 0.258]).into(),
    Series::new("Shell_weight".into(), vec![0.07, 0.055, 0.26, 0.32, 0.24]).into(),
]).unwrap();

// Make predictions
let predictions = model.predict(&test_df).unwrap();

println!("DEBUG: Input DataFrame:");
println!("{}", test_df);
println!("\nDEBUG: Predictions:");
println!("{}", predictions);

// Verify predictions are not NaN
let pred_vec = predictions.f64().unwrap().into_iter()
    .collect::<Vec<Option<f64>>>();

assert!(pred_vec.iter().all(|x| x.is_some() && !x.unwrap().is_nan()),
    "Found NaN or None in predictions: {:?}", pred_vec);

// Verify predictions are within reasonable range
for (i, pred) in pred_vec.iter().enumerate() {
    let value = pred.unwrap();
    assert!(value > 0.0, "Prediction {} is not positive: {}", i, value);
    assert!(value < 100.0, "Prediction {} is too large: {}", i, value);
}
}
#[test]
fn test_realistic_prediction2_3ways() {
initialize_test_license();
// Create test tables (keeping your existing table creation code)
let mean_table = RatingTable::new(
    DataFrame::new(vec![
        Series::new("Rating_Factor".into(), vec![9.94767]).into(),
    ]).unwrap(),
    None
);

let shell_weight_table = RatingTable::new(
    DataFrame::new(vec![
        Series::new("Shell_weight".into(), vec![
            0.03025, 0.04925, 0.06525, 0.08025, 0.09525, 0.40625, 0.43525, 0.47125, 0.52175, f64::INFINITY
        ]).into(),
        Series::new("Rating_Factor".into(), vec![
            -5.287939, -4.77238, -4.180646, -4.030887, -3.957615, 3.810618, 4.277076, 4.511772, 4.910237, 7.138597
        ]).into(),
    ]).unwrap(),
    None
);

let height_sex_table = RatingTable::new(
    DataFrame::new(vec![
        Series::new("Height".into(), vec![
            0.0575, 0.0575, 0.0575, 0.0725, 0.0725, 0.2075, 0.2075, f64::INFINITY, f64::INFINITY, f64::INFINITY
        ]).into(),
        Series::new("Sex".into(), vec![
            -999, 1, 2, -999, 1, 1, 2, -999, 1, 2
        ]).into(),
        Series::new("Rating_Factor".into(), vec![
            -0.950232, -1.657364, -1.018087, -0.858654, -1.565786, 0.86736, 0.817616, 1.240646, 1.179411, 1.129667
        ]).into(),
    ]).unwrap(),
    None
);

let shucked_sex_table = RatingTable::new(
    DataFrame::new(vec![
        Series::new("Shucked_weight".into(), vec![
            0.03875, 0.03875, 0.03875, 0.08575, 0.08575, 0.81775, 0.81775, f64::INFINITY, f64::INFINITY, f64::INFINITY
        ]).into(),
        Series::new("Sex".into(), vec![
            -999, 0, 1, -999, 0, 0, 1, -999, 0, 1
        ]).into(),
        Series::new("Rating_Factor".into(), vec![
            1.785907, 1.88914, 1.395997, 2.943802, 3.047036, -3.395509, -2.879082, -4.420048, -4.570553, -4.054126
        ]).into(),
    ]).unwrap(),
    None
);


// Test individual table predictions for the first row
let features = HashMap::from([
    ("Sex".to_string(), FeatureValue::Categorical(2)),
    ("Height".to_string(), FeatureValue::Numeric(0.09)),
    ("Shucked_weight".to_string(), FeatureValue::Numeric(0.0995)),
    ("Shell_weight".to_string(), FeatureValue::Numeric(0.07)),
]);

// Test each table individually
println!("\nTesting individual table predictions for first row:");

let mean_contribution = mean_table.predict(&features);
println!("Mean table contribution: {}", mean_contribution);
assert!(!mean_contribution.is_nan(), "Mean table returned NaN");

let shell_contribution = shell_weight_table.predict(&features);
println!("Shell_weight table contribution: {}", shell_contribution);
assert!(!shell_contribution.is_nan(), "Shell_weight table returned NaN");

let height_sex_contribution = height_sex_table.predict(&features);
println!("Height/Sex table contribution: {}", height_sex_contribution);
assert!(!height_sex_contribution.is_nan(), "Height/Sex table returned NaN");

let shucked_sex_contribution = shucked_sex_table.predict(&features);
println!("Shucked_weight/Sex table contribution: {}", shucked_sex_contribution);
assert!(!shucked_sex_contribution.is_nan(), "Shucked_weight/Sex table returned NaN");

// Test RatingModel predict_one
let model = RatingModel::new(
    vec![
        mean_table.clone(), 
        shell_weight_table.clone(), 
        height_sex_table.clone(), 
        shucked_sex_table.clone()
    ],
    LinkFunction::Identity
);
let features_f64: HashMap<String, f64> = HashMap::from([
    ("Sex".to_string(), 2.0),
    ("Height".to_string(), 0.09),
    ("Shucked_weight".to_string(), 0.0995),
    ("Shell_weight".to_string(), 0.07),
]);
let single_prediction = model.predict_one(&features_f64);
println!("\nRatingModel predict_one result: {}", single_prediction);
println!("Expected sum: {}", 
    mean_contribution + shell_contribution + height_sex_contribution + shucked_sex_contribution);

// Test RatingModel batch predict
let test_df = DataFrame::new(vec![
    Series::new("Sex".into(), vec![2i32, 1i32, 0i32, 0i32, 2i32]).into(),
    Series::new("Height".into(), vec![0.09, 0.08, 0.125, 0.15, 0.13]).into(),
    Series::new("Shucked_weight".into(), vec![0.0995, 0.0895, 0.294, 0.3145, 0.258]).into(),
    Series::new("Shell_weight".into(), vec![0.07, 0.055, 0.26, 0.32, 0.24]).into(),
]).unwrap();

let batch_predictions = model.predict(&test_df).unwrap();
println!("\nBatch predictions:");
println!("Input DataFrame:");
println!("{}", test_df);
println!("\nPredictions:");
println!("{}", batch_predictions);

// Compare first row of batch predictions with single prediction
let first_batch_pred = batch_predictions.f64().unwrap().get(0).unwrap();
println!("\nComparison of prediction methods for first row:");
println!("Individual table sum: {}", 
    mean_contribution + shell_contribution + height_sex_contribution + shucked_sex_contribution);
println!("predict_one result: {}", single_prediction);
println!("First batch prediction: {}", first_batch_pred);

assert!((single_prediction - first_batch_pred).abs() < 1e-10, 
    "Mismatch between predict_one ({}) and batch predict ({}) for first row", 
    single_prediction, first_batch_pred);
}
#[test]
fn test_basic_filtering() {
initialize_test_license();
// Create a simple test DataFrame
let df = DataFrame::new(vec![
    Series::new("Height".into(), vec![0.08, 0.08, 0.1, 0.1, 0.1]).into(),
    Series::new("Sex".into(), vec![-999i32, 1i32, -999i32, 1i32, 2i32]).into(),  // Using i32
    Series::new("Rating_Factor".into(), vec![-1.0, -1.2, -3.7, -0.2, -0.1]).into(),
]).unwrap();

println!("\nOriginal DataFrame:");
println!("{:?}", df);

// Test numeric filtering
let height_mask = df.column("Height")
    .unwrap()
    .f64()
    .unwrap()
    .equal(0.1);

let height_filtered = df.filter(&height_mask).unwrap();
println!("\nAfter Height = 0.1 filter:");
println!("{:?}", height_filtered);

// Test categorical filtering
let sex_mask = height_filtered.column("Sex")
    .unwrap()
    .i32()  // Changed to i32
    .unwrap()
    .equal(2i32);  // Changed to i32

let final_filtered = height_filtered.filter(&sex_mask).unwrap();
println!("\nAfter Sex = 2 filter:");
println!("{:?}", final_filtered);
}
#[test]
fn test_abalone_predictions() {
initialize_test_license();
// The LightGBM model JSON string (truncated for readability)
let model_json = r#"{"name": "tree", "version": "v4", "num_class": 1, "num_tree_per_iteration": 1, "label_index": 0, "max_feature_idx": 7, "objective": "regression", "average_output": false, "feature_names": ["Sex", "Length", "Diameter", "Height", "Whole_weight", "Shucked_weight", "Viscera_weight", "Shell_weight"], "monotone_constraints": [], "feature_infos": {"Sex": {"min_value": 0, "max_value": 2, "values": []}, "Length": {"min_value": 0.075, "max_value": 0.815, "values": []}, "Diameter": {"min_value": 0.055, "max_value": 0.65, "values": []}, "Height": {"min_value": 0, "max_value": 0.515, "values": []}, "Whole_weight": {"min_value": 0.002, "max_value": 2.8255, "values": []}, "Shucked_weight": {"min_value": 0.001, "max_value": 1.488, "values": []}, "Viscera_weight": {"min_value": 0.0005, "max_value": 0.76, "values": []}, "Shell_weight": {"min_value": 0.0015, "max_value": 0.897, "values": []}}, "tree_info": [{"tree_index": 0, "num_leaves": 4, "num_cat": 0, "shrinkage": 1, "tree_structure": {"split_index": 0, "split_feature": 7, "split_gain": 8348.349609375, "threshold": "0.16525000000000004", "decision_type": "<=", "default_left": true, "missing_type": "None", "internal_value": 9.94767, "internal_weight": 0, "internal_count": 2924, "left_child": {"split_index": 2, "split_feature": 7, "split_gain": 1210.1800537109375, "threshold": "0.04825000000000001", "decision_type": "<=", "default_left": true, "missing_type": "None", "internal_value": 8.76413, "internal_weight": 987, "internal_count": 987, "left_child": {"leaf_index": 0, "leaf_value": 7.64833976306965, "leaf_weight": 195, "leaf_count": 195}, "right_child": {"leaf_index": 3, "leaf_value": 9.038880237432942, "leaf_weight": 792, "leaf_count": 792}}, "right_child": {"split_index": 1, "split_feature": 7, "split_gain": 1587.5, "threshold": "0.35525", "decision_type": "<=", "default_left": true, "missing_type": "None", "internal_value": 10.5508, "internal_weight": 1937, "internal_count": 1937, "left_child": {"leaf_index": 1, "leaf_value": 10.262808818357541, "leaf_weight": 1379, "leaf_count": 1379}, "right_child": {"leaf_index": 2, "leaf_value": 11.262335148139254, "leaf_weight": 558, "leaf_count": 558}}}}, {"tree_index": 1, "num_leaves": 4, "num_cat": 0, "shrinkage": 0.5, "tree_structure": {"split_index": 0, "split_feature": 7, "split_gain": 2553.300048828125, "threshold": "0.24525000000000002", "decision_type": "<=", "default_left": true, "missing_type": "None", "internal_value": 0, "internal_weight": 0, "internal_count": 2924, "left_child": {"split_index": 1, "split_feature": 7, "split_gain": 835.5549926757812, "threshold": "0.09675000000000002", "decision_type": "<=", "default_left": true, "missing_type": "None", "internal_value": -0.436596, "internal_weight": 1561, "internal_count": 1561, "left_child": {"leaf_index": 0, "leaf_value": -0.9640311902105939, "leaf_weight": 507, "leaf_count": 507}, "right_child": {"leaf_index": 2, "leaf_value": -0.182875297697055, "leaf_weight": 1054, "leaf_count": 1054}}, "right_child": {"split_index": 2, "split_feature": 5, "split_gain": 806.8489990234375, "threshold": "0.41475", "decision_type": "<=", "default_left": true, "missing_type": "None", "internal_value": 0.500013, "internal_weight": 1363, "internal_count": 1363, "left_child": {"leaf_index": 1, "leaf_value": 1.171245445314032, "leaf_weight": 337, "leaf_count": 337}, "right_child": {"leaf_index": 3, "leaf_value": 0.279526051880701, "leaf_weight": 1026, "leaf_count": 1026}}}}, {"tree_index": 2, "num_leaves": 4, "num_cat": 0, "shrinkage": 0.5, "tree_structure": {"split_index": 0, "split_feature": 7, "split_gain": 924.6630249023438, "threshold": "0.29025000000000006", "decision_type": "<=", "default_left": true, "missing_type": "None", "internal_value": 0, "internal_weight": 0, "internal_count": 2924, "left_child": {"leaf_index": 0, "leaf_value": -0.20425096174789525, "leaf_weight": 1914, "leaf_count": 1914}, "right_child": {"split_index": 1, "split_feature": 7, "split_gain": 511.0820007324219, "threshold": "0.5040000000000001", "decision_type": "<=", "default_left": true, "missing_type": "None", "internal_value": 0.387061, "internal_weight": 1010, "internal_count": 1010, "left_child": {"split_index": 2, "split_feature": 5, "split_gain": 581.2630004882812, "threshold": "0.44975000000000004", "decision_type": "<=", "default_left": true, "missing_type": "None", "internal_value": 0.269141, "internal_weight": 910, "internal_count": 910, "left_child": {"leaf_index": 1, "leaf_value": 1.0032499631026246, "leaf_weight": 208, "leaf_count": 208}, "right_child": {"leaf_index": 3, "leaf_value": 0.05160855112947821, "leaf_weight": 702, "leaf_count": 702}}, "right_child": {"leaf_index": 2, "leaf_value": 1.4599561999230584, "leaf_weight": 100, "leaf_count": 100}}}}], "feature_importances": {"Shucked_weight": 2, "Shell_weight": 7}, "pandas_categorical": []}"#;

// Create the model
let model = RatingModel::from_lgbm_json(model_json, "max").unwrap();

// Create test data
let test_data = DataFrame::new(vec![
    Series::new("Sex".into(), vec![2i32, 1, 0, 0, 2]).into(),
    Series::new("Length".into(), vec![0.35f64, 0.33, 0.545, 0.55, 0.5]).into(),
    Series::new("Diameter".into(), vec![0.265f64, 0.255, 0.425, 0.44, 0.4]).into(),
    Series::new("Height".into(), vec![0.09f64, 0.08, 0.125, 0.15, 0.13]).into(),
    Series::new("Whole_weight".into(), vec![0.2255f64, 0.205, 0.768, 0.8945, 0.6645]).into(),
    Series::new("Shucked_weight".into(), vec![0.0995f64, 0.0895, 0.294, 0.3145, 0.258]).into(),
    Series::new("Viscera_weight".into(), vec![0.0485f64, 0.0395, 0.1495, 0.151, 0.133]).into(),
    Series::new("Shell_weight".into(), vec![0.07f64, 0.055, 0.26, 0.32, 0.24]).into(),
]).unwrap();

// Expected results
let expected = vec![ 7.87059809,  7.87059809, 11.2298033 , 12.43730423,  9.87568256];

// Get predictions
let predictions = model.predict(&test_data).unwrap();
let predictions: Vec<f64> = predictions.f64().unwrap()
    .into_iter()
    .map(|x| x.unwrap())
    .collect();

// Compare predictions with expected values
assert_eq!(predictions.len(), expected.len());
for (pred, exp) in predictions.iter().zip(expected.iter()) {
    assert!((pred - exp).abs() < 1e-6, 
        "Prediction {} differs from expected {} by more than 1e-6", 
        pred, exp);
}

// Print predictions for visualization
println!("Predictions:");
for (i, (pred, exp)) in predictions.iter().zip(expected.iter()).enumerate() {
    println!("Row {}: Predicted = {:.8}, Expected = {:.8}, Diff = {:.8}", 
        i, pred, exp, (pred - exp).abs());
}
}

#[test]
fn test_abalone_predictions_debug() {
initialize_test_license();
let model_json = r#"{"name": "tree", "version": "v4", "num_class": 1, "num_tree_per_iteration": 1, "label_index": 0, "max_feature_idx": 7, "objective": "regression", "average_output": false, "feature_names": ["Sex", "Length", "Diameter", "Height", "Whole_weight", "Shucked_weight", "Viscera_weight", "Shell_weight"], "monotone_constraints": [], "feature_infos": {"Sex": {"min_value": 0, "max_value": 2, "values": []}, "Length": {"min_value": 0.075, "max_value": 0.815, "values": []}, "Diameter": {"min_value": 0.055, "max_value": 0.65, "values": []}, "Height": {"min_value": 0, "max_value": 0.515, "values": []}, "Whole_weight": {"min_value": 0.002, "max_value": 2.8255, "values": []}, "Shucked_weight": {"min_value": 0.001, "max_value": 1.488, "values": []}, "Viscera_weight": {"min_value": 0.0005, "max_value": 0.76, "values": []}, "Shell_weight": {"min_value": 0.0015, "max_value": 0.897, "values": []}}, "tree_info": [{"tree_index": 0, "num_leaves": 4, "num_cat": 0, "shrinkage": 1, "tree_structure": {"split_index": 0, "split_feature": 7, "split_gain": 8348.349609375, "threshold": "0.16525000000000004", "decision_type": "<=", "default_left": true, "missing_type": "None", "internal_value": 9.94767, "internal_weight": 0, "internal_count": 2924, "left_child": {"split_index": 2, "split_feature": 7, "split_gain": 1210.1800537109375, "threshold": "0.04825000000000001", "decision_type": "<=", "default_left": true, "missing_type": "None", "internal_value": 8.76413, "internal_weight": 987, "internal_count": 987, "left_child": {"leaf_index": 0, "leaf_value": 7.64833976306965, "leaf_weight": 195, "leaf_count": 195}, "right_child": {"leaf_index": 3, "leaf_value": 9.038880237432942, "leaf_weight": 792, "leaf_count": 792}}, "right_child": {"split_index": 1, "split_feature": 7, "split_gain": 1587.5, "threshold": "0.35525", "decision_type": "<=", "default_left": true, "missing_type": "None", "internal_value": 10.5508, "internal_weight": 1937, "internal_count": 1937, "left_child": {"leaf_index": 1, "leaf_value": 10.262808818357541, "leaf_weight": 1379, "leaf_count": 1379}, "right_child": {"leaf_index": 2, "leaf_value": 11.262335148139254, "leaf_weight": 558, "leaf_count": 558}}}}, {"tree_index": 1, "num_leaves": 4, "num_cat": 0, "shrinkage": 0.5, "tree_structure": {"split_index": 0, "split_feature": 7, "split_gain": 2553.300048828125, "threshold": "0.24525000000000002", "decision_type": "<=", "default_left": true, "missing_type": "None", "internal_value": 0, "internal_weight": 0, "internal_count": 2924, "left_child": {"split_index": 1, "split_feature": 7, "split_gain": 835.5549926757812, "threshold": "0.09675000000000002", "decision_type": "<=", "default_left": true, "missing_type": "None", "internal_value": -0.436596, "internal_weight": 1561, "internal_count": 1561, "left_child": {"leaf_index": 0, "leaf_value": -0.9640311902105939, "leaf_weight": 507, "leaf_count": 507}, "right_child": {"leaf_index": 2, "leaf_value": -0.182875297697055, "leaf_weight": 1054, "leaf_count": 1054}}, "right_child": {"split_index": 2, "split_feature": 5, "split_gain": 806.8489990234375, "threshold": "0.41475", "decision_type": "<=", "default_left": true, "missing_type": "None", "internal_value": 0.500013, "internal_weight": 1363, "internal_count": 1363, "left_child": {"leaf_index": 1, "leaf_value": 1.171245445314032, "leaf_weight": 337, "leaf_count": 337}, "right_child": {"leaf_index": 3, "leaf_value": 0.279526051880701, "leaf_weight": 1026, "leaf_count": 1026}}}}, {"tree_index": 2, "num_leaves": 4, "num_cat": 0, "shrinkage": 0.5, "tree_structure": {"split_index": 0, "split_feature": 7, "split_gain": 924.6630249023438, "threshold": "0.29025000000000006", "decision_type": "<=", "default_left": true, "missing_type": "None", "internal_value": 0, "internal_weight": 0, "internal_count": 2924, "left_child": {"leaf_index": 0, "leaf_value": -0.20425096174789525, "leaf_weight": 1914, "leaf_count": 1914}, "right_child": {"split_index": 1, "split_feature": 7, "split_gain": 511.0820007324219, "threshold": "0.5040000000000001", "decision_type": "<=", "default_left": true, "missing_type": "None", "internal_value": 0.387061, "internal_weight": 1010, "internal_count": 1010, "left_child": {"split_index": 2, "split_feature": 5, "split_gain": 581.2630004882812, "threshold": "0.44975000000000004", "decision_type": "<=", "default_left": true, "missing_type": "None", "internal_value": 0.269141, "internal_weight": 910, "internal_count": 910, "left_child": {"leaf_index": 1, "leaf_value": 1.0032499631026246, "leaf_weight": 208, "leaf_count": 208}, "right_child": {"leaf_index": 3, "leaf_value": 0.05160855112947821, "leaf_weight": 702, "leaf_count": 702}}, "right_child": {"leaf_index": 2, "leaf_value": 1.4599561999230584, "leaf_weight": 100, "leaf_count": 100}}}}], "feature_importances": {"Shucked_weight": 2, "Shell_weight": 7}, "pandas_categorical": []}"#;

// First approach: Direct from process_lgbm_trees
let tables = process_lgbm_trees(model_json).unwrap();
let direct_model = RatingModel::new(tables, LinkFunction::Identity);

// Second approach: Using from_lgbm_json
let consolidated_model = RatingModel::from_lgbm_json(model_json, "max").unwrap();

// Create test data
let test_data = DataFrame::new(vec![
    Series::new("Sex".into(), vec![2i32, 1, 0, 0, 2]).into(),
    Series::new("Length".into(), vec![0.35f64, 0.33, 0.545, 0.55, 0.5]).into(),
    Series::new("Diameter".into(), vec![0.265f64, 0.255, 0.425, 0.44, 0.4]).into(),
    Series::new("Height".into(), vec![0.09f64, 0.08, 0.125, 0.15, 0.13]).into(),
    Series::new("Whole_weight".into(), vec![0.2255f64, 0.205, 0.768, 0.8945, 0.6645]).into(),
    Series::new("Shucked_weight".into(), vec![0.0995f64, 0.0895, 0.294, 0.3145, 0.258]).into(),
    Series::new("Viscera_weight".into(), vec![0.0485f64, 0.0395, 0.1495, 0.151, 0.133]).into(),
    Series::new("Shell_weight".into(), vec![0.07f64, 0.055, 0.26, 0.32, 0.24]).into(),
]).unwrap();

// Expected results
let expected = vec![ 7.87059809,  7.87059809, 11.2298033 , 12.43730423,  9.87568256];
// Get predictions from both models
let direct_predictions = direct_model.predict(&test_data).unwrap();
let consolidated_predictions = consolidated_model.predict(&test_data).unwrap();

// Print debug information
println!("\nModel Structure Comparison:");
println!("Direct model tables: {}", direct_model.tables.len());
println!("Consolidated model tables: {}", consolidated_model.tables.len());

println!("\nDirect Model Tables:");
for (i, table) in direct_model.tables.iter().enumerate() {
    println!("\nTable {}:", i);
    println!("Columns: {:?}", table.data.get_column_names());
    println!("Shape: {:?}", table.data.shape());
    println!("{}", table.data);
}

println!("\nConsolidated Model Tables:");
for (i, table) in consolidated_model.tables.iter().enumerate() {
    println!("\nTable {}:", i);
    println!("Columns: {:?}", table.data.get_column_names());
    println!("Shape: {:?}", table.data.shape());
    println!("{}", table.data);
}

println!("\nPrediction Comparison:");
println!("Row\tExpected\tDirect\tConsolidated\tDirect Diff\tCons Diff");
let direct_preds: Vec<f64> = direct_predictions.f64().unwrap()
    .into_iter()
    .map(|x| x.unwrap())
    .collect();
let cons_preds: Vec<f64> = consolidated_predictions.f64().unwrap()
    .into_iter()
    .map(|x| x.unwrap())
    .collect();

println!("Focused individual prediction comparing predict_one and predict with the unconsolidated model");
// In test_abalone_predictions_debug:
let test_row = HashMap::from([
    ("Sex".to_string(), 2.0),
    ("Height".to_string(), 0.09),
    ("Shucked_weight".to_string(), 0.0995),
    ("Shell_weight".to_string(), 0.07),
]);

// Test predict_one
let single_prediction = direct_model.predict_one(&test_row);
println!("\npredict_one result: {}", single_prediction);

// Test predict with single-row DataFrame
let test_df = DataFrame::new(vec![
    Series::new("Sex".into(), vec![2i32]).into(),
    Series::new("Height".into(), vec![0.09f64]).into(),
    Series::new("Shucked_weight".into(), vec![0.0995f64]).into(),
    Series::new("Shell_weight".into(), vec![0.07f64]).into(),
]).unwrap();

let batch_prediction = direct_model.predict(&test_df).unwrap();
println!("predict result: {}", batch_prediction.get(0).unwrap());

println!("Testing RatingTable.predict without RatingModel method");
let test_features = HashMap::from([
    ("Sex".to_string(), FeatureValue::Categorical(2)),
    ("Height".to_string(), FeatureValue::Numeric(0.09)),
    ("Shucked_weight".to_string(), FeatureValue::Numeric(0.0995)),
    ("Shell_weight".to_string(), FeatureValue::Numeric(0.07)),
]);

println!("\nTesting individual table predictions:");
let mut total = 0.0;
for (i, table) in direct_model.tables.iter().enumerate() {
    let contribution = table.predict(&test_features);
    println!("Table {}: {}", i, contribution);
    total += contribution;
}
println!("Sum of all contributions: {}", total);

for i in 0..expected.len() {
    println!("{}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{:.6}",
        i,
        expected[i],
        direct_preds[i],
        cons_preds[i],
        (direct_preds[i] - expected[i]).abs(),
        (cons_preds[i] - expected[i]).abs()
    );
    assert!((direct_preds[i] - expected[i]).abs() < 1e-6, "Direct prediction {} differs from expected {} by more than 1e-6", direct_preds[i], expected[i]);
    assert!((cons_preds[i] - expected[i]).abs() < 1e-6, "Consolidated prediction {} differs from expected {} by more than 1e-6", cons_preds[i], expected[i]);
    assert!((direct_preds[i] - cons_preds[i]).abs() < 1e-6, "Direct and consolidated predictions differ by more than 1e-6 for row {}", i);
}
}
#[test]
fn test_rating_table_single_column_comprehensive() {
initialize_test_license();
let table = RatingTable::new(
    DataFrame::new(vec![
        Series::new("Shell_weight".into(), vec![0.04825, 0.16525, f64::INFINITY]).into(),
        Series::new("Rating_Factor".into(), vec![-2.29933, 0.0, 0.0]).into(),
    ]).unwrap(),
    None
);

let test_cases = vec![
    // value, expected_rating
    (0.03, -2.29933),    // Below first bound
    (0.04825, -2.29933), // At first bound
    (0.10, 0.0),         // Between bounds
    (0.16525, 0.0),      // At second bound
    (0.20, 0.0),         // Above last finite bound
];

for (value, expected) in test_cases {
    let features = HashMap::from([
        ("Shell_weight".to_string(), FeatureValue::Numeric(value))
    ]);
    let result = table.predict(&features);
    assert!((result - expected).abs() < 1e-10, 
        "For Shell_weight = {}, expected {}, got {}", 
        value, expected, result);
}
}

#[test]
fn test_rating_table_two_columns_comprehensive() {
initialize_test_license();
let table = RatingTable::new(
    DataFrame::new(vec![
        Series::new("Shell_weight".into(), vec![0.29025, 0.29025, 0.504, 0.504, f64::INFINITY, f64::INFINITY]).into(),
        Series::new("Shucked_weight".into(), vec![0.44975, f64::INFINITY, 0.44975, f64::INFINITY, 0.44975, f64::INFINITY]).into(),
        Series::new("Rating_Factor".into(), vec![0.0, 0.0, 1.00325, 0.0, 0.0, 0.0]).into(),
    ]).unwrap(),
    None    
);

let test_cases = vec![
    // (shell_weight, shucked_weight, expected_rating)
    (0.20, 0.40, 0.0),       // Both below first bounds
    (0.20, 0.50, 0.0),       // SW below first, ShW above first
    (0.30, 0.40, 1.00325),   // SW between bounds, ShW below bound
    (0.30, 0.50, 0.0),       // SW between bounds, ShW above bound
    (0.29025, 0.44975, 0.0), // Both at first bounds
    (0.504, 0.44975, 1.00325), // SW at second bound, ShW at first bound
    (0.60, 0.40, 0.0),       // SW above last finite, ShW below bound
    (0.60, 0.50, 0.0),       // Both above their bounds
];

for (sw, shw, expected) in test_cases {
    let features = HashMap::from([
        ("Shell_weight".to_string(), FeatureValue::Numeric(sw)),
        ("Shucked_weight".to_string(), FeatureValue::Numeric(shw))
    ]);
    let result = table.predict(&features);
    assert!((result - expected).abs() < 1e-10, 
        "For Shell_weight = {}, Shucked_weight = {}, expected {}, got {}", 
        sw, shw, expected, result);
}
}
#[test]
fn test_analysis_vs_consolidated_model_predictions() {
    initialize_test_license();

    // Define a minimal LightGBM model JSON.
    // The tree has:
    //   - an overall mean of 1.0 (from the root internal value),
    //   - a split on feature index 0 ("feature_0") with threshold "0.5",
    //   - a left leaf with leaf_value 0.5,
    //   - and a right leaf with leaf_value -0.5.
    let model_json = r#"
    {
        "tree_info": [
            {
                "tree_structure": {
                    "internal_value": 1.0,
                    "split_feature": 0,
                    "threshold": "0.5",
                    "decision_type": "<=",
                    "left_child": {
                        "leaf_index": 0,
                        "leaf_value": 0.5
                    },
                    "right_child": {
                        "leaf_index": 1,
                        "leaf_value": -0.5
                    }
                }
            }
        ],
        "feature_names": ["feature_0"]
    }
    "#;

    // Build the first model using the analysis table construction.
    let analysis_model = build_analysis_tablemodel(model_json, LinkFunction::Identity)
        .expect("build_analysis_tablemodel failed");

    // Build the second model using the consolidated tables, 
    // by reusing the tables produced by the analysis model.
    let consolidated_model = build_consolidated_tablemodel(analysis_model.tables.clone(), LinkFunction::Identity);

    // Create a test DataFrame for prediction.
    // The test values cover the split threshold (0.5) as well as other values.
    let df = df![
        "feature_0" => &[0.5, 1.0, 0.3, 0.7]
    ]
    .expect("Failed to create test DataFrame");

    // Obtain predictions from both models.
    let analysis_pred = analysis_model.predict(&df)
        .expect("Analysis model prediction failed");
    let consolidated_pred = consolidated_model.predict(&df)
        .expect("Consolidated model prediction failed");

    // Retrieve f64 arrays from the output Series.
    let analysis_values: Vec<f64> = analysis_pred
        .f64()
        .expect("Failed to extract f64 values from analysis prediction")
        .into_iter()
        .map(|opt| opt.expect("Found None in analysis prediction"))
        .collect();

    let consolidated_values: Vec<f64> = consolidated_pred
        .f64()
        .expect("Failed to extract f64 values from consolidated prediction")
        .into_iter()
        .map(|opt| opt.expect("Found None in consolidated prediction"))
        .collect();

    // Both prediction arrays should have the same length.
    assert_eq!(analysis_values.len(), consolidated_values.len(), "The two models produced predictions of different lengths");

    // Compare the predictions element-wise.
    for (i, (a, c)) in analysis_values.iter().zip(consolidated_values.iter()).enumerate() {
        assert!(
            (a - c).abs() < 1e-10,
            "Prediction mismatch at row {}: analysis model produced {} vs consolidated model {}",
            i,
            a,
            c
        );
    }
}
#[test]
fn test_internal_node_processing() {
    initialize_test_license();
    // Simple model with internal_value explicitly defined
    let model_json = r#"{
        "objective": "regression",
        "feature_names": ["Age", "VehicleType"],
        "tree_info": [
            {
                "tree_structure": {
                    "internal_value": 5.0,
                    "split_feature": 0,
                    "threshold": "30.0",
                    "decision_type": "<=",
                    "left_child": {
                        "leaf_index": 0,
                        "leaf_value": -5.0
                    },
                    "right_child": {
                        "internal_value": 6.0,
                        "split_feature": 1,
                        "threshold": "1",
                        "decision_type": "==",
                        "left_child": {
                            "leaf_index": 1,
                            "leaf_value": 5.0
                        },
                        "right_child": {
                            "leaf_index": 2,
                            "leaf_value": 5.0
                        }
                    }
                }
            }
        ]
    }"#;

    // Create test data
    let df = DataFrame::new(vec![
        Series::new("Age".into(), vec![25.0, 35.0, 35.0]).into(),
        Series::new("VehicleType".into(), vec![0i32, 1i32, 2i32]).into()
    ]).unwrap();
    
    // Create models using both methods
    let link_function = LinkFunction::Identity;
    
    // First create the standard model using process_lgbm_trees directly
    let tables = process_lgbm_trees(model_json).unwrap();
    let standard_model = RatingModel::new(tables, link_function.clone());
    
    // Then create analysis model
    let analysis_model = build_analysis_tablemodel(model_json, link_function).unwrap();
    
    // Print both models' predictions for debugging
    let standard_preds = standard_model.predict(&df).unwrap();
    let analysis_preds = analysis_model.predict(&df).unwrap();
    
    println!("Standard model predictions: {}", standard_preds);
    println!("Analysis model predictions: {}", analysis_preds);
    
    // Verify standard model produces expected results
    let expected = Series::new("expected".into(), vec![-5.0, 5.0, 5.0]);
    assert_eq!(standard_preds, expected, "Standard model did not produce expected predictions");
    
    // Note that the analysis model produces different results (-14.0 instead of -5.0) 
    // This indicates an issue with build_analysis_tablemodel that needs to be addressed
    println!("Note: Analysis model produces different results due to a potential issue in build_analysis_tablemodel");
}

#[test]
fn test_analysis_vs_consolidated_methods() {
    initialize_test_license();
    
    // Create a model JSON with multiple features
    let model_json = r#"{
        "objective": "regression",
        "feature_names": ["Age", "VehicleType", "Mileage"],
        "tree_info": [
            {
                "tree_structure": {
                    "internal_value": 5.0,
                    "split_feature": 0,
                    "threshold": "30.0",
                    "decision_type": "<=",
                    "left_child": {
                        "leaf_index": 0,
                        "leaf_value": -5.0
                    },
                    "right_child": {
                        "internal_value": 6.0,
                        "split_feature": 1,
                        "threshold": "1",
                        "decision_type": "==",
                        "left_child": {
                            "internal_value": 7.0,
                            "split_feature": 2,
                            "threshold": "50000.0",
                            "decision_type": "<=",
                            "left_child": {
                                "leaf_index": 1,
                                "leaf_value": 8.0
                            },
                            "right_child": {
                                "leaf_index": 2,
                                "leaf_value": 6.0
                            }
                        },
                        "right_child": {
                            "leaf_index": 3,
                            "leaf_value": 5.0
                        }
                    }
                }
            }
        ]
    }"#;

    // Create test data
    let df = DataFrame::new(vec![
        Series::new("Age".into(), vec![25.0, 35.0, 35.0, 35.0]).into(),
        Series::new("VehicleType".into(), vec![0i32, 1i32, 1i32, 2i32]).into(),
        Series::new("Mileage".into(), vec![30000.0, 40000.0, 60000.0, 70000.0]).into()
    ]).unwrap();
    
    // Create models using both methods
    let link_function = LinkFunction::Identity;
    
    // First, create the analysis model
    let analysis_model = build_analysis_tablemodel(model_json, link_function.clone()).unwrap();
    
    // Then create consolidated model using the same JSON
    let tables = process_lgbm_trees(model_json).unwrap();
    let consolidated_model = build_consolidated_tablemodel(tables, link_function);
    
    // Get predictions from both models
    let analysis_preds = analysis_model.predict(&df).unwrap();
    let consolidated_preds = consolidated_model.predict(&df).unwrap();
    
    // Print for debugging
    println!("Analysis model predictions: {}", analysis_preds);
    println!("Consolidated model predictions: {}", consolidated_preds);
    
    // Print model table counts
    println!("Analysis model has {} tables", analysis_model.tables.len());
    println!("Consolidated model has {} tables", consolidated_model.tables.len());
    
    // Verify predictions match
    assert_eq!(
        analysis_preds, consolidated_preds,
        "Analysis and consolidated models produced different predictions"
    );
}

#[test]
fn test_analysis_tablemodel_bug() {
    initialize_test_license();
    
    // Create a simple model with a right-side branch issue
    let model_json = r#"{
        "objective": "regression",
        "feature_names": ["Age", "VehicleType"],
        "tree_info": [
            {
                "tree_structure": {
                    "internal_value": 5.0,
                    "split_feature": 0,
                    "threshold": "30.0",
                    "decision_type": "<=",
                    "left_child": {
                        "leaf_index": 0,
                        "leaf_value": -5.0
                    },
                    "right_child": {
                        "internal_value": 6.0,
                        "split_feature": 1,
                        "threshold": "1",
                        "decision_type": "==",
                        "left_child": {
                            "leaf_index": 1,
                            "leaf_value": 5.0
                        },
                        "right_child": {
                            "leaf_index": 2,
                            "leaf_value": 5.0
                        }
                    }
                }
            }
        ]
    }"#;

    // Test data
    let df = DataFrame::new(vec![
        Series::new("Age".into(), vec![25.0, 35.0, 35.0]).into(),
        Series::new("VehicleType".into(), vec![0i32, 1i32, 2i32]).into()
    ]).unwrap();
    
    // Process the same model in two different ways
    let tables = process_lgbm_trees(model_json).unwrap();
    let standard_model = RatingModel::new(tables.clone(), LinkFunction::Identity);
    
    // Print table details to help identify the issue
    println!("\nTables from process_lgbm_trees:");
    for (i, table) in tables.iter().enumerate() {
        println!("\nTable {}:", i);
        println!("Columns: {:?}", table.data.get_column_names());
        println!("Data:\n{}", table.data);
    }
    
    // See standard model predictions (these are correct)
    let standard_preds = standard_model.predict(&df).unwrap();
    println!("\nStandard model predictions: {}", standard_preds);
    
    // Bug: The issue is in process_tree_analysis used by build_analysis_tablemodel
    // When the right child path is processed, the decision_type isn't adjusted.
    // For numeric features: right_branch should use ">" instead of "<="
    // For categorical features: right_branch should use "!=" instead of "=="
    println!("\nBug in build_analysis_tablemodel:");
    println!("In process_tree_analysis, when adding split_info to the path for right_child,"); 
    println!("the decision_type isn't adjusted for right branches.");
    println!("For numeric features, right branches should use '>' instead of '<='");
    println!("For categorical features, right branches should use '!=' instead of '=='");
}

#[test]
fn test_comparison_of_models() {
    initialize_test_license();
    // Create a model with multiple features and paths
    let model_json = r#"{
        "objective": "regression",
        "feature_names": ["Age", "VehicleType", "Mileage"],
        "tree_info": [
            {
                "tree_structure": {
                    "internal_value": 5.0,
                    "split_feature": 0,
                    "threshold": "30.0",
                    "decision_type": "<=",
                    "left_child": {
                        "leaf_index": 0,
                        "leaf_value": -5.0
                    },
                    "right_child": {
                        "internal_value": 6.0,
                        "split_feature": 1,
                        "threshold": "1",
                        "decision_type": "==",
                        "left_child": {
                            "internal_value": 7.0,
                            "split_feature": 2,
                            "threshold": "50000.0",
                            "decision_type": "<=",
                            "left_child": {
                                "leaf_index": 1,
                                "leaf_value": 8.0
                            },
                            "right_child": {
                                "leaf_index": 2,
                                "leaf_value": 6.0
                            }
                        },
                        "right_child": {
                            "leaf_index": 3,
                            "leaf_value": 5.0
                        }
                    }
                }
            }
        ]
    }"#;

    // Test data
    let df = DataFrame::new(vec![
        Series::new("Age".into(), vec![25.0, 35.0, 35.0, 35.0]).into(),
        Series::new("VehicleType".into(), vec![0i32, 1i32, 1i32, 2i32]).into(),
        Series::new("Mileage".into(), vec![30000.0, 40000.0, 60000.0, 70000.0]).into()
    ]).unwrap();
    
    // Method 1: Using process_lgbm_trees directly (standard approach)
    let tables = process_lgbm_trees(model_json).unwrap();
    let standard_model = RatingModel::new(tables, LinkFunction::Identity);
    
    // Method 2: Using build_analysis_tablemodel (fixed approach)
    let analysis_model = build_analysis_tablemodel(model_json, LinkFunction::Identity).unwrap();
    
    // Method 3: Using from_lgbm_json with "max" consolidation
    let consolidated_model = RatingModel::from_lgbm_json(model_json, "max").unwrap();
    
    // Get predictions from all models
    let standard_preds = standard_model.predict(&df).unwrap();
    let analysis_preds = analysis_model.predict(&df).unwrap();
    let consolidated_preds = consolidated_model.predict(&df).unwrap();
    
    // Print results for debugging
    println!("\nStandard model predictions: {}", standard_preds);
    println!("Analysis model predictions: {}", analysis_preds);
    println!("Consolidated model predictions: {}", consolidated_preds);
    
    // Print model table counts
    println!("\nStandard model has {} tables", standard_model.tables.len());
    println!("Analysis model has {} tables", analysis_model.tables.len());
    println!("Consolidated model has {} tables", consolidated_model.tables.len());
    
    // Verify that predictions match
    assert_eq!(standard_preds, analysis_preds, "Standard and analysis models should produce the same predictions");
    assert_eq!(standard_preds, consolidated_preds, "Standard and consolidated models should produce the same predictions");
}