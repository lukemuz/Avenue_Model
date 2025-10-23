#[cfg(test)]
mod glm_benchmarks {
    use crate::rating_model::RatingModel;
    use crate::glm::{fit_glm, GLMOptions};
    use crate::tests::testing_utils::initialize_test_license;
    use polars::prelude::*;
    use std::time::Instant;

    fn benchmark_report(name: &str, duration: std::time::Duration, n_rows: usize, n_iterations: usize) {
        let total_ms = duration.as_secs_f64() * 1000.0;
        let per_iter_ms = total_ms / n_iterations as f64;
        let per_row_us = (duration.as_micros() as f64) / (n_rows * n_iterations) as f64;

        println!("\n{}", "=".repeat(60));
        println!("BENCHMARK: {}", name);
        println!("{}", "=".repeat(60));
        println!("Total time:        {:.3} ms", total_ms);
        println!("Time per iter:     {:.3} ms", per_iter_ms);
        println!("Time per row/iter: {:.3} µs", per_row_us);
        println!("Throughput:        {:.0} rows/sec", (n_rows as f64 * n_iterations as f64) / duration.as_secs_f64());
    }

    #[test]
    #[ignore] // Run with: cargo test --test glm_benchmarks -- --ignored --nocapture
    fn benchmark_small_dataset_gaussian() {
        initialize_test_license();

        let n = 1000;
        let x_values: Vec<f64> = (0..n).map(|i| (i % 100) as f64).collect();
        let y_values: Vec<f64> = x_values.iter().map(|&x| 5.0 + 0.5 * x).collect();

        let train_df = DataFrame::new(vec![
            Series::new("x".into(), x_values).into(),
            Series::new("y".into(), y_values).into(),
        ]).unwrap();

        let mean_table = DataFrame::new(vec![
            Series::new("Rating_Factor".into(), vec![0.0]).into(),
        ]).unwrap();

        let x_table = DataFrame::new(vec![
            Series::new("x".into(), vec![25.0, 50.0, 75.0, f64::INFINITY]).into(),
            Series::new("Rating_Factor".into(), vec![0.0, 0.0, 0.0, 0.0]).into(),
        ]).unwrap();

        let model = RatingModel::from_dataframes(
            vec![mean_table, x_table],
            "regression",
            None,
            None,
        ).unwrap();

        let options = GLMOptions {
            max_iterations: 50,
            tolerance: 1e-8,
            verbose: false,
            ..Default::default()
        };

        let start = Instant::now();
        let fitted_model = fit_glm(&model, &train_df, "y", None, options).unwrap();
        let duration = start.elapsed();

        benchmark_report("Small Dataset - Gaussian (1k rows, 4 bins)", duration, n, 50);

        // Prediction benchmark
        let pred_start = Instant::now();
        for _ in 0..10 {
            let _ = fitted_model.predict(&train_df).unwrap();
        }
        let pred_duration = pred_start.elapsed();
        println!("\nPrediction (10x):  {:.3} ms", pred_duration.as_secs_f64() * 1000.0);
        println!("Per prediction:    {:.3} µs", pred_duration.as_micros() as f64 / (n * 10) as f64);
    }

    #[test]
    #[ignore]
    fn benchmark_medium_dataset_poisson() {
        initialize_test_license();

        let n = 10_000;
        let x_values: Vec<f64> = (0..n).map(|i| (i % 100) as f64).collect();
        let y_values: Vec<f64> = x_values.iter().map(|&x| (1.0 + 0.05 * x).exp()).collect();
        let exposure = vec![1.0; n];

        let train_df = DataFrame::new(vec![
            Series::new("x".into(), x_values).into(),
            Series::new("y".into(), y_values).into(),
            Series::new("exposure".into(), exposure).into(),
        ]).unwrap();

        let mean_table = DataFrame::new(vec![
            Series::new("Rating_Factor".into(), vec![0.0]).into(),
        ]).unwrap();

        let x_bins: Vec<f64> = (0..10).map(|i| (i * 10) as f64).chain(std::iter::once(f64::INFINITY)).collect();
        let x_table = DataFrame::new(vec![
            Series::new("x".into(), x_bins.clone()).into(),
            Series::new("Rating_Factor".into(), vec![0.0; x_bins.len()]).into(),
        ]).unwrap();

        let model = RatingModel::from_dataframes(
            vec![mean_table, x_table],
            "poisson",
            None,
            None,
        ).unwrap();

        let options = GLMOptions {
            objective: "poisson".to_string(),
            max_iterations: 50,
            tolerance: 1e-8,
            verbose: false,
            tweedie_power: 1.5,
        };

        let start = Instant::now();
        let fitted_model = fit_glm(&model, &train_df, "y", Some("exposure"), options).unwrap();
        let duration = start.elapsed();

        benchmark_report("Medium Dataset - Poisson (10k rows, 11 bins)", duration, n, 50);

        // Prediction benchmark
        let pred_start = Instant::now();
        let _ = fitted_model.predict(&train_df).unwrap();
        let pred_duration = pred_start.elapsed();
        println!("\nPrediction:        {:.3} ms", pred_duration.as_secs_f64() * 1000.0);
        println!("Per prediction:    {:.3} µs", pred_duration.as_micros() as f64 / n as f64);
    }

    #[test]
    #[ignore]
    fn benchmark_large_dataset_gamma() {
        initialize_test_license();

        let n = 100_000;
        let x_values: Vec<f64> = (0..n).map(|i| (i % 200) as f64).collect();
        let y_values: Vec<f64> = x_values.iter().map(|&x| (2.0 + 0.02 * x).exp()).collect();

        let train_df = DataFrame::new(vec![
            Series::new("x".into(), x_values).into(),
            Series::new("y".into(), y_values).into(),
        ]).unwrap();

        let mean_table = DataFrame::new(vec![
            Series::new("Rating_Factor".into(), vec![0.0]).into(),
        ]).unwrap();

        let x_bins: Vec<f64> = (0..20).map(|i| (i * 10) as f64).chain(std::iter::once(f64::INFINITY)).collect();
        let x_table = DataFrame::new(vec![
            Series::new("x".into(), x_bins.clone()).into(),
            Series::new("Rating_Factor".into(), vec![0.0; x_bins.len()]).into(),
        ]).unwrap();

        let model = RatingModel::from_dataframes(
            vec![mean_table, x_table],
            "gamma",
            None,
            None,
        ).unwrap();

        let options = GLMOptions {
            objective: "gamma".to_string(),
            max_iterations: 30,
            tolerance: 1e-8,
            verbose: false,
            tweedie_power: 1.5,
        };

        let start = Instant::now();
        let fitted_model = fit_glm(&model, &train_df, "y", None, options).unwrap();
        let duration = start.elapsed();

        benchmark_report("Large Dataset - Gamma (100k rows, 21 bins)", duration, n, 30);

        // Prediction benchmark
        let pred_start = Instant::now();
        let _ = fitted_model.predict(&train_df).unwrap();
        let pred_duration = pred_start.elapsed();
        println!("\nPrediction:        {:.3} ms", pred_duration.as_secs_f64() * 1000.0);
        println!("Per prediction:    {:.3} µs", pred_duration.as_micros() as f64 / n as f64);
    }

    #[test]
    #[ignore]
    fn benchmark_multitable_model() {
        initialize_test_license();

        let n = 50_000;
        let x1_values: Vec<f64> = (0..n).map(|i| (i % 50) as f64).collect();
        let x2_values: Vec<f64> = (0..n).map(|i| ((i / 50) % 30) as f64).collect();
        let x3_values: Vec<i32> = (0..n).map(|i| ((i / 100) % 5) as i32).collect();
        let y_values: Vec<f64> = x1_values.iter().zip(&x2_values).zip(&x3_values)
            .map(|((&x1, &x2), &x3)| 10.0 + 0.3 * x1 + 0.2 * x2 + x3 as f64 * 2.0)
            .collect();

        let train_df = DataFrame::new(vec![
            Series::new("x1".into(), x1_values).into(),
            Series::new("x2".into(), x2_values).into(),
            Series::new("x3".into(), x3_values).into(),
            Series::new("y".into(), y_values).into(),
        ]).unwrap();

        let mean_table = DataFrame::new(vec![
            Series::new("Rating_Factor".into(), vec![0.0]).into(),
        ]).unwrap();

        let x1_table = DataFrame::new(vec![
            Series::new("x1".into(), vec![10.0, 20.0, 30.0, 40.0, f64::INFINITY]).into(),
            Series::new("Rating_Factor".into(), vec![0.0; 5]).into(),
        ]).unwrap();

        let x2_table = DataFrame::new(vec![
            Series::new("x2".into(), vec![7.5, 15.0, 22.5, f64::INFINITY]).into(),
            Series::new("Rating_Factor".into(), vec![0.0; 4]).into(),
        ]).unwrap();

        let x3_table = DataFrame::new(vec![
            Series::new("x3".into(), vec![-999, 0, 1, 2, 3, 4]).into(),
            Series::new("Rating_Factor".into(), vec![0.0; 6]).into(),
        ]).unwrap();

        let model = RatingModel::from_dataframes(
            vec![mean_table, x1_table, x2_table, x3_table],
            "regression",
            None,
            None,
        ).unwrap();

        let options = GLMOptions {
            max_iterations: 50,
            tolerance: 1e-8,
            verbose: false,
            ..Default::default()
        };

        let start = Instant::now();
        let fitted_model = fit_glm(&model, &train_df, "y", None, options).unwrap();
        let duration = start.elapsed();

        benchmark_report("Multi-Table Model (50k rows, 3 features, 15 bins)", duration, n, 50);

        // Prediction benchmark
        let pred_start = Instant::now();
        let _ = fitted_model.predict(&train_df).unwrap();
        let pred_duration = pred_start.elapsed();
        println!("\nPrediction:        {:.3} ms", pred_duration.as_secs_f64() * 1000.0);
        println!("Per prediction:    {:.3} µs", pred_duration.as_micros() as f64 / n as f64);
    }

    #[test]
    #[ignore]
    fn benchmark_logistic_regression() {
        initialize_test_license();

        let n = 20_000;
        let x_values: Vec<f64> = (0..n).map(|i| (i as f64 / 100.0)).collect();
        let y_values: Vec<f64> = x_values.iter()
            .map(|&x| if (1.0 / (1.0 + (-0.5 * (x - 100.0)).exp())) > 0.5 { 1.0 } else { 0.0 })
            .collect();

        let train_df = DataFrame::new(vec![
            Series::new("x".into(), x_values).into(),
            Series::new("y".into(), y_values).into(),
        ]).unwrap();

        let mean_table = DataFrame::new(vec![
            Series::new("Rating_Factor".into(), vec![0.0]).into(),
        ]).unwrap();

        let x_bins: Vec<f64> = (0..10).map(|i| (i * 20) as f64).chain(std::iter::once(f64::INFINITY)).collect();
        let x_table = DataFrame::new(vec![
            Series::new("x".into(), x_bins.clone()).into(),
            Series::new("Rating_Factor".into(), vec![0.0; x_bins.len()]).into(),
        ]).unwrap();

        let model = RatingModel::from_dataframes(
            vec![mean_table, x_table],
            "binary",
            None,
            None,
        ).unwrap();

        let options = GLMOptions {
            objective: "binary".to_string(),
            max_iterations: 50,
            tolerance: 1e-8,
            verbose: false,
            tweedie_power: 1.5,
        };

        let start = Instant::now();
        let fitted_model = fit_glm(&model, &train_df, "y", None, options).unwrap();
        let duration = start.elapsed();

        benchmark_report("Logistic Regression (20k rows, 11 bins)", duration, n, 50);

        // Prediction benchmark
        let pred_start = Instant::now();
        let _ = fitted_model.predict(&train_df).unwrap();
        let pred_duration = pred_start.elapsed();
        println!("\nPrediction:        {:.3} ms", pred_duration.as_secs_f64() * 1000.0);
        println!("Per prediction:    {:.3} µs", pred_duration.as_micros() as f64 / n as f64);
    }

    #[test]
    #[ignore]
    fn benchmark_comparison_all_distributions() {
        initialize_test_license();

        let n = 10_000;
        let x_values: Vec<f64> = (0..n).map(|i| (i % 100) as f64).collect();
        let y_values_positive: Vec<f64> = x_values.iter().map(|&x| 5.0 + 0.5 * x + 1.0).collect();
        let y_values_binary: Vec<f64> = x_values.iter().map(|&x| if x > 50.0 { 1.0 } else { 0.0 }).collect();

        let distributions = vec![
            ("Gaussian", "regression", y_values_positive.clone()),
            ("Poisson", "poisson", y_values_positive.clone()),
            ("Gamma", "gamma", y_values_positive.clone()),
            ("Tweedie", "tweedie", y_values_positive.clone()),
            ("Binary", "binary", y_values_binary.clone()),
        ];

        println!("\n{}", "=".repeat(70));
        println!("COMPARATIVE BENCHMARK - All Distributions (10k rows)");
        println!("{}", "=".repeat(70));

        for (name, objective, y_values) in distributions {
            let train_df = DataFrame::new(vec![
                Series::new("x".into(), x_values.clone()).into(),
                Series::new("y".into(), y_values).into(),
            ]).unwrap();

            let mean_table = DataFrame::new(vec![
                Series::new("Rating_Factor".into(), vec![0.0]).into(),
            ]).unwrap();

            let x_table = DataFrame::new(vec![
                Series::new("x".into(), vec![25.0, 50.0, 75.0, f64::INFINITY]).into(),
                Series::new("Rating_Factor".into(), vec![0.0, 0.0, 0.0, 0.0]).into(),
            ]).unwrap();

            let model = RatingModel::from_dataframes(
                vec![mean_table, x_table],
                objective,
                None,
                None,
            ).unwrap();

            let options = GLMOptions {
                objective: objective.to_string(),
                max_iterations: 30,
                tolerance: 1e-8,
                verbose: false,
                tweedie_power: 1.5,
            };

            let start = Instant::now();
            let _ = fit_glm(&model, &train_df, "y", None, options).unwrap();
            let duration = start.elapsed();

            println!("\n{:12} | Fit: {:7.2} ms | Per iter: {:6.2} ms | Per row/iter: {:5.2} µs",
                name,
                duration.as_secs_f64() * 1000.0,
                duration.as_secs_f64() * 1000.0 / 30.0,
                duration.as_micros() as f64 / (n * 30) as f64
            );
        }
        println!("\n{}", "=".repeat(70));
    }
}
