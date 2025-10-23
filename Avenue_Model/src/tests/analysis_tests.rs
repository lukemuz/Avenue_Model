use crate::rating_model::{RatingTable, RatingModel, LinkFunction, FeatureValue};
use crate::analysis::{one_way_analysis_table, one_way_analysis};
use crate::tests::testing_utils::initialize_test_license;
use polars::prelude::*;
use std::time::Instant;
use std::collections::HashMap;

#[cfg(test)]
mod analysis_tests {
    use super::*;

    /// Helper function to create test data with known patterns
    fn create_test_data() -> DataFrame {
        DataFrame::new(vec![
            Series::new("feature_a".into(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).into(),
            Series::new("feature_b".into(), vec![10, 20, 30, 40, 50, 60, 70, 80]).into(),
            Series::new("target".into(), vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]).into(),
            Series::new("weight".into(), vec![100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0]).into()
        ]).unwrap()
    }

    /// Helper function to create a simple rating table
    fn create_test_rating_table_a() -> RatingTable {
        let table_data = DataFrame::new(vec![
            Series::new("feature_a".into(), vec![2.5, 5.0, f64::INFINITY]).into(),
            Series::new("Rating_Factor".into(), vec![1.0, 2.0, 3.0]).into()
        ]).unwrap();
        RatingTable::new(table_data, None)
    }

    #[test]
    fn test_one_way_analysis_single_table() {
        initialize_test_license();
        
        let test_data = create_test_data();
        let rating_table = create_test_rating_table_a();
        
        let result = one_way_analysis_table(
            &rating_table,  // ⭐ OPTIMIZED: Reference
            &test_data,     // ⭐ OPTIMIZED: Reference
            "target", 
            Some("weight")
        );
        
        assert!(result.is_ok(), "Single table analysis should succeed");
        
        let analysis_result = result.unwrap();
        
        // Check that result has expected columns
        assert!(analysis_result.column("feature_a").is_ok(), "Should have feature_a column");
        assert!(analysis_result.column("Rating_Factor").is_ok(), "Should have Rating_Factor column");
        assert!(analysis_result.column("weight").is_ok(), "Should have weight column");
        assert!(analysis_result.column("target_avg").is_ok(), "Should have target_avg column");
        
        // Check that total weight is preserved
        let total_weight: f64 = analysis_result.column("weight").unwrap()
            .f64().unwrap().into_iter().flatten().sum();
        assert!((total_weight - 3600.0).abs() < 0.001, "Total weight should be preserved");
    }

    #[test]
    fn test_one_way_analysis_multiple_tables() {
        initialize_test_license();
        
        let test_data = create_test_data();
        let table_1 = create_test_rating_table_a();
        let table_2 = create_test_rating_table_a();
        
        let model = RatingModel::new(vec![table_1, table_2], LinkFunction::Identity);
        
        let result = one_way_analysis(
            &model,        // ⭐ OPTIMIZED: Reference
            &test_data,    // ⭐ OPTIMIZED: Reference
            "target",      // ⭐ OPTIMIZED: Reference
            Some("weight") // ⭐ OPTIMIZED: Reference
        );
        
        assert!(result.is_ok(), "Multi-table analysis should succeed: {:?}", result.as_ref().err());
        
        let analysis_results = result.unwrap();
        assert_eq!(analysis_results.len(), 2, "Should have results for both tables");
        
        // Each table should preserve total weight
        for (i, table_result) in analysis_results.iter().enumerate() {
            let total_weight: f64 = table_result.column("weight").unwrap()
                .f64().unwrap().into_iter().flatten().sum();
            assert!((total_weight - 3600.0).abs() < 0.001, "Table {} should preserve total weight", i);
        }
    }

    #[test]
    fn test_one_way_analysis_without_weights() {
        initialize_test_license();
        
        let test_data = create_test_data();
        let rating_table = create_test_rating_table_a();
        
        let result = one_way_analysis_table(
            &rating_table,  // ⭐ OPTIMIZED: Reference
            &test_data,     // ⭐ OPTIMIZED: Reference
            "target", 
            None  // No weights
        );
        
        assert!(result.is_ok(), "Analysis without weights should succeed");
        
        let analysis_result = result.unwrap();
        
        // Check that result has expected columns
        assert!(analysis_result.column("feature_a").is_ok(), "Should have feature_a column");
        assert!(analysis_result.column("Rating_Factor").is_ok(), "Should have Rating_Factor column");
        assert!(analysis_result.column("weight").is_ok(), "Should have weight column (defaulted to 1s)");
        assert!(analysis_result.column("target_avg").is_ok(), "Should have target_avg column");
        
        // With no weights, total weight should equal number of rows = 8
        let total_weight: f64 = analysis_result.column("weight").unwrap()
            .f64().unwrap().into_iter().flatten().sum();
        assert!((total_weight - 8.0).abs() < 0.001, "Total weight should equal number of input rows");
    }

    /// New test: baseline-only table (no feature columns) should not require join predicates
    #[test]
    fn test_one_way_analysis_baseline_only_table() {
        initialize_test_license();

        // Input data: only target and weight columns
        let input_df = DataFrame::new(vec![
            Series::new("target".into(), vec![1.0, 2.0, 3.0, 4.0]).into(),
            Series::new("weight".into(), vec![1.0, 1.0, 2.0, 2.0]).into(),
        ]).unwrap();

        // Rating table with NO feature columns (baseline only)
        // Multiple rows are allowed; all should receive the same aggregated stats
        let baseline_table_df = DataFrame::new(vec![
            Series::new("Rating_Factor".into(), vec![1.0, 1.2, 0.8]).into(),
        ]).unwrap();
        let baseline_table = RatingTable::new(baseline_table_df, None);

        let result = one_way_analysis_table(&baseline_table, &input_df, "target", Some("weight"));
        assert!(result.is_ok(), "Baseline-only table analysis should succeed: {:?}", result.as_ref().err());

        let output = result.unwrap();

        // Expect aggregated average column present
        assert!(output.column("target_avg").is_ok(), "Should have target_avg column");
        assert!(output.column("weight").is_ok(), "Should have weight column");

        // Weighted average of target: (1*1 + 2*1 + 3*2 + 4*2) / (1+1+2+2) = (1+2+6+8)/6 = 17/6
        let expected_avg = 17.0 / 6.0;
        let got_avg: f64 = output.column("target_avg").unwrap().f64().unwrap().mean().unwrap();
        assert!((got_avg - expected_avg).abs() < 1e-9, "Expected avg {}, got {}", expected_avg, got_avg);

        // Total weight in output should equal input weight sum for each table row aggregated once
        // Since we attach the aggregate to every rating row, we cannot sum across rows; check per-row weight equals total input weight
        let input_wsum: f64 = input_df.column("weight").unwrap().f64().unwrap().sum().unwrap();
        let out_w_col = output.column("weight").unwrap().f64().unwrap();
        for v in out_w_col.into_iter().flatten() {
            assert!((v - input_wsum).abs() < 1e-9, "Per-row weight should equal total input weight");
        }
    }

    // Benchmark helper functions
    fn create_benchmark_data(num_rows: usize) -> DataFrame {
        let feature_a: Vec<f64> = (0..num_rows).map(|i| (i % 8) as f64 + 1.0).collect();
        let target: Vec<f64> = (0..num_rows).map(|i| (i as f64 * 0.001) % 1.0).collect();
        let weight: Vec<f64> = (0..num_rows).map(|i| (i % 5) as f64 + 1.0).collect();
        
        DataFrame::new(vec![
            Series::new("feature_a".into(), feature_a).into(),
            Series::new("target".into(), target).into(),
            Series::new("weight".into(), weight).into()
        ]).unwrap()
    }

    fn benchmark_function<F, T>(name: &str, mut f: F, iterations: usize) -> f64 
    where 
        F: FnMut() -> T,
    {
        println!("{}: ", name);
        let mut times = Vec::with_capacity(iterations);
        
        for i in 0..iterations {
            let start = Instant::now();
            let _ = f();
            let duration = start.elapsed();
            let ms = duration.as_secs_f64() * 1000.0;
            times.push(ms);
            println!("  Iteration {}: {:.2} ms", i + 1, ms);
        }
        
        let avg_time = times.iter().sum::<f64>() / times.len() as f64;
        println!("  Average: {:.2} ms", avg_time);
        avg_time
    }

    #[cfg(feature = "benchmarks")]
    #[test]
    fn benchmark_original_vs_v2() {
        initialize_test_license();
        let sizes = vec![1_000, 10_000, 100_000, 1_000_000];

        println!("\n🏁 Benchmark: original one_way_analysis_table vs v2");
        println!("{}", "=".repeat(60));

        for &size in &sizes {
            println!("\n📊 Dataset size: {} rows", size);
            let test_data = create_benchmark_data(size);
            let rating_table = create_test_rating_table_a();

            let iterations = if size <= 10_000 { 5 } else if size < 1_000_000 { 3 } else { 1 };

            let time_original = benchmark_function(&format!("original ({})", size), || {
                one_way_analysis_table(
                    &rating_table,
                    &test_data,
                    "target",
                    Some("weight"),
                )
            }, iterations);

            let time_v2 = benchmark_function(&format!("v2 ({})", size), || {
                one_way_analysis_table_v2(
                    &rating_table,
                    &test_data,
                    "target",
                    Some("weight"),
                )
            }, iterations);

            let speedup = time_original / time_v2;
            let throughput_orig = size as f64 / (time_original / 1000.0);
            let throughput_v2 = size as f64 / (time_v2 / 1000.0);

            println!("📈 Performance Summary (size {}):", size);
            println!("  original: {:.2} ms ({:.0} rows/sec)", time_original, throughput_orig);
            println!("  v2:       {:.2} ms ({:.0} rows/sec)", time_v2, throughput_v2);
            println!("  Speedup (v2 vs original): {:.2}x", speedup);
        }
    }

    #[cfg(feature = "benchmarks")]
    #[test]
    fn benchmark_optimized_analysis() {
        initialize_test_license();
        
        let num_rows = 100_000; // Large dataset
        let iterations = 3;
        
        println!("\n🚀 Benchmarking OPTIMIZED Zero-Clone Analysis ({} rows, {} iterations)", num_rows, iterations);
        
        let test_data = create_benchmark_data(num_rows);
        
        // Create model with multiple tables to test multi-table performance
        let model = RatingModel::new(vec![
            create_test_rating_table_a(),
            create_test_rating_table_a(),
            create_test_rating_table_a(),
            create_test_rating_table_a(),
            create_test_rating_table_a(),
        ], LinkFunction::Identity);
        
        println!("📊 Dataset size: {:.1} MB", (num_rows * 3 * 8) as f64 / (1024.0 * 1024.0));
        println!("🔄 Testing with {} tables", model.tables.len());
        println!("⚡ Total operations: {} table lookups", num_rows * model.tables.len());
        
        let analysis_time = benchmark_function("OPTIMIZED Multi-table Analysis", || {
            one_way_analysis(
                &model,       // ⭐ ZERO CLONE!
                &test_data,   // ⭐ ZERO CLONE!
                "target",
                Some("weight")
            )
        }, iterations);
        
        // Performance analysis
        let operations_per_sec = (num_rows as f64 * model.tables.len() as f64) / (analysis_time / 1000.0);
        println!("⚡ Throughput: {:.0} table lookups/sec", operations_per_sec);
        
        // Memory savings analysis
        let estimated_df_size_mb = (num_rows * 3 * 8) as f64 / (1024.0 * 1024.0);
        let memory_saved_mb = estimated_df_size_mb * (model.tables.len() - 1) as f64;
        println!("💾 Memory savings: {:.1} MB (eliminated {} DataFrame clones)", memory_saved_mb, model.tables.len() - 1);
        println!("🧠 Memory efficiency: Uses ~{:.1}x less memory than original", model.tables.len() as f64);
        
        if operations_per_sec > 100_000.0 {
            println!("✅ Excellent performance achieved! 🎉");
        }
    }

    /// Reference implementation using the original row-by-row approach for comparison
    fn one_way_analysis_table_original(
        table: &RatingTable,
        df: &DataFrame,
        target_column: &str,
        weight_column: Option<&str>
    ) -> Result<DataFrame, PolarsError> {
        use crate::license_handler::validate_current_license;
        
        // Validate DataFrame columns
        let required_features: HashMap<String, DataType> = table.get_feature_info();

        // Check for missing columns
        let missing_cols: Vec<_> = required_features.keys()
            .filter(|col| !df.get_column_names().iter().any(|c| c.as_str() == col.as_str()))
            .collect();
            
        if !missing_cols.is_empty() {
            return Err(PolarsError::ComputeError(
                format!("Missing required columns: {}", missing_cols.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(", ")).into()
            ));
        }

        // Check column types (read-only validation)
        for (col, expected_type) in &required_features {
            let df_col = df.column(col)?;
            if df_col.dtype() != expected_type {
                return Err(PolarsError::ComputeError(
                    format!("Column '{}' has type {:?}, expected {:?}", 
                        col, df_col.dtype(), expected_type).into()
                ));
            }
        }
        
        // Check license
        if !validate_current_license() {
            panic!("License not valid");
        }

        // Check if target column exists
        if !df.get_column_names().iter().any(|c| c.as_str() == target_column) {
            return Err(PolarsError::ComputeError(format!("Target column '{}' does not exist", target_column).into()));
        }

        // Validate weight column if provided
        if let Some(weight_col) = weight_column {
            if *df.column(weight_col)?.dtype() != DataType::Float64 {
                return Err(PolarsError::ComputeError(
                    format!("Weight column must be of type Float64, got {:?}", 
                        df.column(weight_col)?.dtype()).into()
                ));
            }
        }

        // ⭐ ORIGINAL ROW-BY-ROW TABLE MATCHING - Direct read from original DataFrame
        let table_row_numbers: Vec<u32> = (0..df.height())
            .map(|row_idx| {
                let row_values: HashMap<_, _> = required_features.iter()
                    .map(|(feature, expected_dtype)| {
                        let value = df.column(feature)?.get(row_idx)?;  // Direct read from original!
                        let feature_value = match (value.clone(), expected_dtype) {
                            (AnyValue::Float64(v), DataType::Float64) => FeatureValue::Numeric(v),
                            (AnyValue::Int32(v), DataType::Int32) => FeatureValue::Categorical(v),
                            (AnyValue::Float64(v), DataType::Int32) => FeatureValue::Categorical(v as i32),
                            (AnyValue::Int32(v), DataType::Float64) => FeatureValue::Numeric(v as f64),
                            _ => return Err(PolarsError::ComputeError(
                                format!("Unsupported value type for feature '{}': got {:?}, expected {:?}", 
                                    feature, value, expected_dtype).into()
                            )),
                        };
                        Ok((feature.clone(), feature_value))
                    })
                    .collect::<Result<HashMap<_, _>, PolarsError>>()?;
                
                match table.find_row_match(&row_values) {
                    Some(row_idx) => Ok(row_idx as u32),
                    None => Err(PolarsError::ComputeError(
                        format!("Could not find matching row for values: {:?}", row_values).into()
                    ))
                }
            })
            .collect::<Result<Vec<_>, PolarsError>>()?;

        // ⭐ Create result table - only clone the small table data (not the large input DataFrame!)
        let mut result_table = table.data.clone(); // Small table clone (~3-10 rows vs 100K+ input rows)
        
        // Add table row index for joining
        let table_row_indices: Vec<u32> = (0..result_table.height() as u32).collect();
        let table_row_series = Series::new("table_row_number".into(), table_row_indices);
        let mut result_columns = result_table.get_columns().to_vec();
        result_columns.push(table_row_series.into());
        result_table = DataFrame::new(result_columns)?;

        // ⭐ Build aggregation DataFrame using ONLY necessary columns (no full clone!)
        let original_row_indices: Vec<u64> = (0..df.height() as u64).collect();
        let weight_values: Vec<f64> = if let Some(weight_col) = weight_column {
            // Extract weights directly from original DataFrame
            df.column(weight_col)?
                .f64()?
                .into_iter()
                .map(|opt_val| opt_val.unwrap_or(0.0))
                .collect()
        } else {
            vec![1.0; df.height()]
        };
        
        let target_values: Vec<f64> = df.column(target_column)?
            .f64()?
            .into_iter()
            .map(|opt_val| opt_val.unwrap_or(0.0))
            .collect();

        // ⭐ Create minimal aggregation DataFrame (only what we need!)
        let agg_df = DataFrame::new(vec![
            Series::new("original_row_nr".into(), original_row_indices).into(),
            Series::new("table_row_number".into(), table_row_numbers).into(),
            Series::new("target".into(), target_values).into(),
            Series::new("weight".into(), weight_values).into(),
        ])?;

        // ⭐ Use lazy evaluation for aggregation and join
        result_table = result_table.lazy()
            .join(
                agg_df.lazy()
                    .group_by(["table_row_number"])
                    .agg([
                        (col("target") * col("weight")).sum().alias("weighted_sum"),
                        col("weight").sum().alias("weight_sum")
                    ])
                    .with_column((col("weighted_sum") / col("weight_sum")).alias(&format!("{}_avg", target_column))),
                [col("table_row_number")],
                [col("table_row_number")],
                JoinArgs::new(JoinType::Left)
            )
            .select([
                col("*").exclude(["table_row_number", "weighted_sum", "weight_sum"]),
                col("weight_sum").alias("weight"),
            ])
            .collect()?;
        
        Ok(result_table)
    }

    #[test]
    fn test_join_where_vs_original() {
        initialize_test_license();
        
        println!("\n🧪 Testing join_where() vs Original Implementation");
        
        // Create test data with various edge cases
        let test_data = DataFrame::new(vec![
            Series::new("feature_a".into(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 1.5, 2.5]).into(),
            Series::new("feature_b".into(), vec![10, 20, 30, 40, 50, 60, 70, 80, 15, 25]).into(),
            Series::new("target".into(), vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.15, 0.25]).into(),
            Series::new("weight".into(), vec![100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0, 150.0, 250.0]).into()
        ]).unwrap();
        
        // Create a more complex rating table with both categorical and numeric features
        let rating_table_data = DataFrame::new(vec![
            Series::new("feature_a".into(), vec![2.5, 5.0, f64::INFINITY]).into(),
            Series::new("feature_b".into(), vec![25, 50, -999]).into(),  // Mix of thresholds and wildcards
            Series::new("Rating_Factor".into(), vec![1.0, 2.0, 3.0]).into()
        ]).unwrap();
        let rating_table = RatingTable::new(rating_table_data, None);
        
        // Test the original implementation
        println!("📊 Running original implementation...");
        let start = Instant::now();
        let result_original = one_way_analysis_table_original(
            &rating_table,
            &test_data,
            "target",
            Some("weight")
        ).expect("Original implementation should succeed");
        let time_original = start.elapsed();
        
        // Test the new join_where implementation  
        println!("⚡ Running join_where implementation...");
        let start = Instant::now();
        let result_new = one_way_analysis_table(
            &rating_table,
            &test_data,
            "target",
            Some("weight")
        ).expect("New implementation should succeed");
        let time_new = start.elapsed();
        
        // Compare results
        println!("🔍 Comparing results...");
        
        // Check that both results have the same shape
        assert_eq!(result_original.height(), result_new.height(), 
            "Results should have same number of rows");
        assert_eq!(result_original.width(), result_new.width(), 
            "Results should have same number of columns");
        
        // Check that column names match
        let orig_cols = result_original.get_column_names();
        let new_cols = result_new.get_column_names();
        assert_eq!(orig_cols, new_cols, "Column names should match");
        
        // Check that values are approximately equal (allowing for floating point precision)
        for col_name in result_original.get_column_names() {
            let orig_col = result_original.column(col_name).unwrap();
            let new_col = result_new.column(col_name).unwrap();
            
            match orig_col.dtype() {
                DataType::Float64 => {
                    let orig_values = orig_col.f64().unwrap();
                    let new_values = new_col.f64().unwrap();
                    
                    for (i, (orig, new)) in orig_values.into_iter().zip(new_values.into_iter()).enumerate() {
                        match (orig, new) {
                            (Some(o), Some(n)) => {
                                // Handle infinity and NaN values specially
                                if o.is_infinite() && n.is_infinite() && o.signum() == n.signum() {
                                    // Both are the same type of infinity (positive or negative)
                                    continue;
                                } else if o.is_nan() && n.is_nan() {
                                    // Both are NaN
                                    continue;
                                } else {
                                    assert!((o - n).abs() < 1e-10, 
                                        "Values in column '{}' row {} differ: original={}, new={}", 
                                        col_name, i, o, n);
                                }
                            },
                            (None, None) => {},
                            _ => panic!("Null value mismatch in column '{}' row {}", col_name, i),
                        }
                    }
                },
                DataType::Int32 => {
                    let orig_values = orig_col.i32().unwrap();
                    let new_values = new_col.i32().unwrap();
                    
                    for (i, (orig, new)) in orig_values.into_iter().zip(new_values.into_iter()).enumerate() {
                        assert_eq!(orig, new, 
                            "Values in column '{}' row {} differ: original={:?}, new={:?}", 
                            col_name, i, orig, new);
                    }
                },
                _ => {
                    // For other types, just check equality
                    assert_eq!(orig_col, new_col, "Column '{}' values should match", col_name);
                }
            }
        }
        
        // Performance comparison
        let speedup = time_original.as_secs_f64() / time_new.as_secs_f64();
        
        println!("✅ Results match perfectly!");
        println!("⏱️  Original implementation: {:.2} ms", time_original.as_secs_f64() * 1000.0);
        println!("⚡ join_where implementation: {:.2} ms", time_new.as_secs_f64() * 1000.0);
        println!("🚀 Speedup: {:.2}x", speedup);
        
        if speedup > 1.0 {
            println!("🎉 New implementation is faster!");
        } else {
            println!("⚠️  Performance regression detected");
        }
    }

    /// Test specifically for categorical features with -999 wildcard behavior
    #[test]
    fn test_categorical_features_with_wildcard() {
        initialize_test_license();
        
        println!("\n🧪 Testing categorical features with -999 wildcard");
        
        // Create test data with categorical features
        let test_data = DataFrame::new(vec![
            Series::new("cat_1".into(), vec![0i32, 1, 2, 3, 4]).into(),
            Series::new("cat_2".into(), vec![0i32, 1, 2, 3, 4]).into(),
            Series::new("target".into(), vec![0.1, 0.2, 0.3, 0.4, 0.5]).into(),
            Series::new("weight".into(), vec![100.0, 200.0, 300.0, 400.0, 500.0]).into()
        ]).unwrap();
        
        // Create rating table with specific categories and wildcards
        let rating_table_data = DataFrame::new(vec![
            Series::new("cat_1".into(), vec![0i32, 1, 2, -999]).into(),  // Exact matches for 0,1,2 and wildcard for rest
            Series::new("cat_2".into(), vec![0i32, -999, -999, -999]).into(),  // Exact match for 0, wildcard for rest
            Series::new("Rating_Factor".into(), vec![1.0, 2.0, 3.0, 4.0]).into()
        ]).unwrap();
        let rating_table = RatingTable::new(rating_table_data, None);
        
        println!("📊 Test data:");
        println!("  Input cat_1 values: [0, 1, 2, 3, 4]");
        println!("  Input cat_2 values: [0, 1, 2, 3, 4]");
        println!("  Rating table cat_1: [0, 1, 2, -999]");
        println!("  Rating table cat_2: [0, -999, -999, -999]");
        
        let result = one_way_analysis_table(
            &rating_table,
            &test_data,
            "target",
            Some("weight")
        );
        
        assert!(result.is_ok(), "Categorical analysis should succeed: {:?}", result.as_ref().err());
        
        let analysis_result = result.unwrap();
        
        println!("📊 Analysis result shape: {} rows, {} cols", analysis_result.height(), analysis_result.width());
        println!("📊 Columns: {:?}", analysis_result.get_column_names());
        
        // Check that result has expected columns
        assert!(analysis_result.column("cat_1").is_ok(), "Should have cat_1 column");
        assert!(analysis_result.column("cat_2").is_ok(), "Should have cat_2 column");
        assert!(analysis_result.column("Rating_Factor").is_ok(), "Should have Rating_Factor column");
        assert!(analysis_result.column("weight").is_ok(), "Should have weight column");
        assert!(analysis_result.column("target_avg").is_ok(), "Should have target_avg column");
        
        // Check that total weight is preserved - this should NOT be zero!
        let total_weight: f64 = analysis_result.column("weight").unwrap()
            .f64().unwrap().into_iter().flatten().sum();
        
        println!("📊 Total weight in result: {}", total_weight);
        println!("📊 Expected total weight: 1500.0");
        
        // Print detailed results for debugging
        for i in 0..analysis_result.height() {
            let cat1 = analysis_result.column("cat_1").unwrap().get(i).unwrap();
            let cat2 = analysis_result.column("cat_2").unwrap().get(i).unwrap();
            let weight = analysis_result.column("weight").unwrap().get(i).unwrap();
            let avg = analysis_result.column("target_avg").unwrap().get(i).unwrap();
            println!("  Row {}: cat_1={:?}, cat_2={:?}, weight={:?}, target_avg={:?}", i, cat1, cat2, weight, avg);
        }
        
        // This is the key assertion that should catch the bug
        assert!(total_weight > 0.0, "Total weight should not be zero - this indicates the categorical matching is broken!");
        assert!((total_weight - 1500.0).abs() < 0.001, "Total weight should be preserved: expected 1500.0, got {}", total_weight);
    }

    /// Test mixed categorical and numeric features
    #[test]
    fn test_mixed_categorical_numeric_features() {
        initialize_test_license();
        
        println!("\n🧪 Testing mixed categorical and numeric features");
        
        // Create test data with both categorical and numeric features
        let test_data = DataFrame::new(vec![
            Series::new("cat_feature".into(), vec![0i32, 1, 2, 3, 4]).into(),
            Series::new("num_feature".into(), vec![1.0, 2.0, 3.0, 4.0, 5.0]).into(),
            Series::new("target".into(), vec![0.1, 0.2, 0.3, 0.4, 0.5]).into(),
            Series::new("weight".into(), vec![100.0, 200.0, 300.0, 400.0, 500.0]).into()
        ]).unwrap();
        
        // Create rating table with mixed features
        let rating_table_data = DataFrame::new(vec![
            Series::new("cat_feature".into(), vec![0i32, 1, -999]).into(),  // Specific values and wildcard
            Series::new("num_feature".into(), vec![2.5, 4.5, f64::INFINITY]).into(),  // Numeric thresholds
            Series::new("Rating_Factor".into(), vec![1.0, 2.0, 3.0]).into()
        ]).unwrap();
        let rating_table = RatingTable::new(rating_table_data, None);
        
        let result = one_way_analysis_table(
            &rating_table,
            &test_data,
            "target",
            Some("weight")
        );
        
        assert!(result.is_ok(), "Mixed feature analysis should succeed: {:?}", result.as_ref().err());
        
        let analysis_result = result.unwrap();
        
        // Check that total weight is preserved
        let total_weight: f64 = analysis_result.column("weight").unwrap()
            .f64().unwrap().into_iter().flatten().sum();
        
        println!("📊 Mixed features total weight: {}", total_weight);
        
        assert!(total_weight > 0.0, "Total weight should not be zero for mixed features!");
        assert!((total_weight - 1500.0).abs() < 0.001, "Total weight should be preserved for mixed features");
    }

    /// Test edge case: all categorical values should match wildcard
    #[test]
    fn test_all_wildcard_categorical() {
        initialize_test_license();
        
        println!("\n🧪 Testing all-wildcard categorical table");
        
        // Create test data
        let test_data = DataFrame::new(vec![
            Series::new("cat_feature".into(), vec![10i32, 20, 30]).into(),  // Values not in rating table
            Series::new("target".into(), vec![0.1, 0.2, 0.3]).into(),
            Series::new("weight".into(), vec![100.0, 200.0, 300.0]).into()
        ]).unwrap();
        
        // Create rating table with only wildcard rows
        let rating_table_data = DataFrame::new(vec![
            Series::new("cat_feature".into(), vec![-999i32, -999]).into(),  // All wildcards
            Series::new("Rating_Factor".into(), vec![1.0, 2.0]).into()
        ]).unwrap();
        let rating_table = RatingTable::new(rating_table_data, None);
        
        let result = one_way_analysis_table(
            &rating_table,
            &test_data,
            "target",
            Some("weight")
        );
        
        assert!(result.is_ok(), "All-wildcard analysis should succeed: {:?}", result.as_ref().err());
        
        let analysis_result = result.unwrap();
        let total_weight: f64 = analysis_result.column("weight").unwrap()
            .f64().unwrap().into_iter().flatten().sum();
        
        println!("📊 All-wildcard total weight: {}", total_weight);
        
        // Key test: wildcard should catch everything, so weight should not be zero
        assert!(total_weight > 0.0, "All-wildcard table should have non-zero weight!");
    }

    /// Test the specific error case from the logs
    #[test]
    fn test_error_reproduction() {
        initialize_test_license();
        
        println!("\n🧪 Reproducing the specific error from logs");
        
        // Simulate the exact scenario from the error logs
        let test_data = DataFrame::new(vec![
            Series::new("cat_1".into(), vec![0i32, 1, 2, 3, 4, 0, 1, 2]).into(),
            Series::new("cat_2".into(), vec![0i32, 1, 2, 3, 4, 1, 2, 3]).into(),
            Series::new("cat_3".into(), vec![0i32, 1, 2, 3, 4, 2, 3, 4]).into(),
            Series::new("target".into(), vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]).into(),
            Series::new("weight".into(), vec![100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0]).into()
        ]).unwrap();
        
        // Create a categorical-only rating table (like tables 6, 7, 9 from the error)
        let rating_table_data = DataFrame::new(vec![
            Series::new("cat_1".into(), vec![-999i32, 0, 1, 2, 3, 4]).into(),  // Wildcard + specific values
            Series::new("Rating_Factor".into(), vec![1.0, 1.1, 1.2, 1.3, 1.4, 1.5]).into()
        ]).unwrap();
        let rating_table = RatingTable::new(rating_table_data, None);
        
        println!("📊 Error reproduction test:");
        println!("  Input data has {} rows with cat_1 values: [0,1,2,3,4,0,1,2]", test_data.height());
        println!("  Rating table has cat_1 values: [-999,0,1,2,3,4]");
        
        let result = one_way_analysis_table(
            &rating_table,
            &test_data,
            "target",
            Some("weight")
        );
        
        assert!(result.is_ok(), "Error reproduction test should succeed: {:?}", result.as_ref().err());
        
        let analysis_result = result.unwrap();
        let total_weight: f64 = analysis_result.column("weight").unwrap()
            .f64().unwrap().into_iter().flatten().sum();
        
        println!("📊 Error reproduction total weight: {}", total_weight);
        println!("📊 Expected total weight: 3600.0");
        
        // Print detailed weight breakdown
        for i in 0..analysis_result.height() {
            let cat1 = analysis_result.column("cat_1").unwrap().get(i).unwrap();
            let weight = analysis_result.column("weight").unwrap().get(i).unwrap();
            println!("  Row {}: cat_1={:?}, weight={:?}", i, cat1, weight);
        }
        
        // This should fail if the bug exists
        assert!(total_weight > 0.0, "CRITICAL BUG: Categorical table has zero weight! This matches the error in the logs.");
        assert!((total_weight - 3600.0).abs() < 0.001, "Total weight should be preserved in error reproduction test");
    }

    #[cfg(feature = "benchmarks")]
    #[test]
    fn benchmark_join_where_vs_original() {
        initialize_test_license();
        
        let sizes = vec![1_000, 10_000, 100_000];
        
        println!("\n🏆 COMPREHENSIVE BENCHMARK: join_where() vs Original");
        println!("{}", "=".repeat(60));
        
        for &size in &sizes {
            println!("\n📊 Dataset size: {} rows", size);
            
            // Create benchmark data
            let test_data = create_benchmark_data(size);
            let rating_table = create_test_rating_table_a();
            
            // Benchmark original implementation
            let iterations = if size <= 10_000 { 5 } else { 3 };
            
            let time_original = benchmark_function(&format!("Original ({})", size), || {
                one_way_analysis_table_original(
                    &rating_table,
                    &test_data,
                    "target",
                    Some("weight")
                )
            }, iterations);
            
            // Benchmark new implementation
            let time_new = benchmark_function(&format!("join_where ({})", size), || {
                one_way_analysis_table(
                    &rating_table,
                    &test_data,
                    "target",
                    Some("weight")
                )
            }, iterations);
            
            let speedup = time_original / time_new;
            let throughput_orig = size as f64 / (time_original / 1000.0);
            let throughput_new = size as f64 / (time_new / 1000.0);
            
            println!("📈 Performance Summary:");
            println!("  Original: {:.2} ms ({:.0} rows/sec)", time_original, throughput_orig);
            println!("  join_where: {:.2} ms ({:.0} rows/sec)", time_new, throughput_new);
            println!("  Speedup: {:.2}x", speedup);
            
            if speedup > 1.0 {
                println!("  ✅ join_where is {:.1}% faster!", (speedup - 1.0) * 100.0);
            } else {
                println!("  ⚠️  join_where is {:.1}% slower", (1.0 - speedup) * 100.0);
            }
            
            println!("  {}", "-".repeat(40));
        }
        
        println!("\n🎯 Test completed - check results above!");
    }
}