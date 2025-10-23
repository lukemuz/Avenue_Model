use crate::rating_model::{RatingTable, FeatureValue};
use crate::analysis::one_way_analysis_table;
use crate::tests::testing_utils::initialize_test_license;
use polars::prelude::*;
use std::collections::HashMap;

#[cfg(test)]
mod weight_distribution_tests {
    use super::*;

    #[test]
    fn test_numeric_feature_weight_distribution() {
        initialize_test_license();
        
        // Create test data matching the user's scenario
        let test_data = DataFrame::new(vec![
            Series::new("GrLivArea".into(), vec![800.0, 900.0, 1200.0, 1400.0, 1600.0, 1800.0, 2200.0, 3000.0]).into(),
            Series::new("target".into(), vec![0.8, 0.9, 0.85, 0.75, 1.1, 1.2, 1.5, 2.0]).into(),
            Series::new("weight".into(), vec![100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0]).into()
        ]).unwrap();
        
        // Create rating table with the same thresholds as user's screenshot
        let rating_table_data = DataFrame::new(vec![
            Series::new("GrLivArea".into(), vec![865.0, 987.5, 1296.5, 1475.5, 1653.0, 1853.5, 2338.5, f64::INFINITY]).into(),
            Series::new("Rating_Factor".into(), vec![-16298.8782, -12735.8992, -11614.9516, -10092.1940, 3048.4988, 4824.2280, 16593.4732, 39781.3432]).into()
        ]).unwrap();
        
        let rating_table = RatingTable::new(rating_table_data, None);
        
        println!("\n=== Test Weight Distribution ===");
        println!("Test data shape: {:?}", test_data.shape());
        println!("Rating table shape: {:?}", rating_table.data.shape());
        
        // Test individual row mapping first
        println!("\nTesting individual row mappings:");
        for i in 0..test_data.height() {
            let grlivarea = test_data.column("GrLivArea").unwrap().f64().unwrap().get(i).unwrap();
            let weight = test_data.column("weight").unwrap().f64().unwrap().get(i).unwrap();
            
            let mut feature_values = HashMap::new();
            feature_values.insert("GrLivArea".to_string(), FeatureValue::Numeric(grlivarea));
            
            let matched_row = rating_table.find_row_match(&feature_values);
            println!("  Data {}: GrLivArea={}, weight={} -> Bucket {:?}", i, grlivarea, weight, matched_row);
        }
        
        // Run one-way analysis
        let result = one_way_analysis_table(&rating_table, &test_data, "target", Some("weight"));
        
        match result {
            Ok(analysis_result) => {
                println!("\nOne-way analysis result:");
                println!("{}", analysis_result);
                
                // Check weight distribution
                if let Ok(weight_col) = analysis_result.column("weight") {
                    let weights: Vec<f64> = weight_col.f64().unwrap().into_iter().flatten().collect();
                    let total_weight: f64 = weights.iter().sum();
                    let first_bucket_weight = weights[0];
                    let concentration = (first_bucket_weight / total_weight) * 100.0;
                    
                    println!("\nWeight Distribution Analysis:");
                    println!("Total weight: {}", total_weight);
                    println!("First bucket weight: {}", first_bucket_weight);
                    println!("First bucket concentration: {:.1}%", concentration);
                    
                    for (i, weight) in weights.iter().enumerate() {
                        println!("  Bucket {}: {}", i, weight);
                    }
                    
                    // The test should pass if weights are distributed properly
                    assert!(concentration < 50.0, "Too much weight concentrated in first bucket: {:.1}%", concentration);
                    assert!((total_weight - 3600.0).abs() < 0.001, "Total weight should be preserved");
                } else {
                    panic!("Weight column not found in analysis result");
                }
            },
            Err(e) => {
                panic!("One-way analysis failed: {}", e);
            }
        }
    }
    
    #[test]
    fn test_find_row_match_individual_cases() {
        initialize_test_license();
        
        // Test the specific cases from the user's data
        let rating_table_data = DataFrame::new(vec![
            Series::new("GrLivArea".into(), vec![865.0, 987.5, 1296.5, 1475.5, 1653.0, 1853.5, 2338.5, f64::INFINITY]).into(),
            Series::new("Rating_Factor".into(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).into()
        ]).unwrap();
        
        let rating_table = RatingTable::new(rating_table_data, None);
        
        let test_cases = vec![
            (800.0, 0),   // Should go to bucket 0 (≤ 865.0)
            (900.0, 1),   // Should go to bucket 1 (≤ 987.5)
            (1200.0, 2),  // Should go to bucket 2 (≤ 1296.5)
            (1400.0, 3),  // Should go to bucket 3 (≤ 1475.5)
            (1600.0, 4),  // Should go to bucket 4 (≤ 1653.0)
            (1800.0, 5),  // Should go to bucket 5 (≤ 1853.5)
            (2200.0, 6),  // Should go to bucket 6 (≤ 2338.5)
            (3000.0, 7),  // Should go to bucket 7 (≤ inf)
        ];
        
        for (value, expected_bucket) in test_cases {
            let mut feature_values = HashMap::new();
            feature_values.insert("GrLivArea".to_string(), FeatureValue::Numeric(value));
            
            let matched_row = rating_table.find_row_match(&feature_values);
            println!("Value {} -> Expected bucket {}, Got {:?}", value, expected_bucket, matched_row);
            
            assert_eq!(matched_row, Some(expected_bucket), 
                "Value {} should map to bucket {}, but got {:?}", value, expected_bucket, matched_row);
        }
    }
} 