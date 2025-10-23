#[cfg(test)]
mod realistic_benchmarks {
    use crate::rating_model::RatingModel;
    use crate::glm::{fit_glm, GLMOptions};
    use polars::prelude::*;
    use std::time::Instant;

    fn create_realistic_dataset(n: usize) -> (DataFrame, RatingModel) {
        // Create a realistic insurance dataset with deterministic pseudo-random data

        // Features similar to actual insurance data (using simple deterministic generation)
        let ages: Vec<f64> = (0..n).map(|i| 18.0 + ((i * 7919) % 62) as f64).collect();
        let vehicles: Vec<i32> = (0..n).map(|i| 1 + ((i * 3) % 4) as i32).collect();
        let regions: Vec<i32> = (0..n).map(|i| 1 + ((i * 17) % 9) as i32).collect();
        let claim_history: Vec<i32> = (0..n).map(|i| ((i * 11) % 5) as i32).collect();

        // Simulate target variable (e.g., claim frequency)
        let exposure: Vec<f64> = (0..n).map(|i| 0.1 + ((i * 13) % 90) as f64 / 100.0).collect();
        let claims: Vec<f64> = ages.iter().zip(&vehicles).zip(&regions).zip(&claim_history).zip(&exposure)
            .enumerate()
            .map(|(i, ((((age, veh), reg), hist), exp))| {
                let base_rate = 0.05;
                let age_effect = if *age < 25.0 { 1.5 } else if *age > 65.0 { 1.2 } else { 1.0 };
                let veh_effect = 1.0 + (*veh as f64 - 2.0) * 0.1;
                let reg_effect = 1.0 + (*reg as f64 - 5.0) * 0.05;
                let hist_effect = 1.0 + *hist as f64 * 0.3;

                let lambda = base_rate * age_effect * veh_effect * reg_effect * hist_effect * exp;
                // Simulate Poisson-like counts deterministically
                let rand_val = ((i * 6151) % 1000) as f64 / 1000.0;
                if rand_val < lambda {
                    1.0 + ((i * 23) % 2) as f64
                } else {
                    0.0
                }
            })
            .collect();

        let df = DataFrame::new(vec![
            Series::new("age".into(), ages).into(),
            Series::new("vehicles".into(), vehicles).into(),
            Series::new("region".into(), regions).into(),
            Series::new("claim_history".into(), claim_history).into(),
            Series::new("exposure".into(), exposure).into(),
            Series::new("claims".into(), claims).into(),
        ]).unwrap();

        // Create realistic model structure with multiple tables
        let mean_table = DataFrame::new(vec![
            Series::new("Rating_Factor".into(), vec![0.0]).into(),
        ]).unwrap();

        // Age table (common in insurance)
        let age_bins = vec![25.0, 35.0, 45.0, 55.0, 65.0, f64::INFINITY];
        let age_table = DataFrame::new(vec![
            Series::new("age".into(), age_bins.clone()).into(),
            Series::new("Rating_Factor".into(), vec![0.0; age_bins.len()]).into(),
        ]).unwrap();

        // Vehicle count table
        let veh_table = DataFrame::new(vec![
            Series::new("vehicles".into(), vec![-999, 1, 2, 3, 4]).into(),
            Series::new("Rating_Factor".into(), vec![0.0; 5]).into(),
        ]).unwrap();

        // Region table
        let reg_table = DataFrame::new(vec![
            Series::new("region".into(), vec![-999, 1, 2, 3, 4, 5, 6, 7, 8, 9]).into(),
            Series::new("Rating_Factor".into(), vec![0.0; 10]).into(),
        ]).unwrap();

        // Claim history table
        let hist_table = DataFrame::new(vec![
            Series::new("claim_history".into(), vec![-999, 0, 1, 2, 3, 4]).into(),
            Series::new("Rating_Factor".into(), vec![0.0; 6]).into(),
        ]).unwrap();

        let model = RatingModel::from_dataframes(
            vec![mean_table, age_table, veh_table, reg_table, hist_table],
            "poisson",
            None,
            None,
        ).unwrap();

        (df, model)
    }

    #[test]
    fn bench_realistic_1k() {
        // License check skipped for pure Rust benchmarks
        let (df, model) = create_realistic_dataset(1_000);

        let options = GLMOptions {
            objective: "poisson".to_string(),
            max_iterations: 50,
            tolerance: 1e-6,
            verbose: false,
            tweedie_power: 1.5,
        };

        let start = Instant::now();
        let fitted = fit_glm(&model, &df, "claims", Some("exposure"), None, options).unwrap();
        let fit_time = start.elapsed();

        let pred_start = Instant::now();
        let _ = fitted.predict(&df).unwrap();
        let pred_time = pred_start.elapsed();

        println!("\n📊 Realistic Dataset: 1,000 records, 4 features, 27 total bins");
        println!("   Fit time:   {:.2} ms", fit_time.as_secs_f64() * 1000.0);
        println!("   Pred time:  {:.2} ms", pred_time.as_secs_f64() * 1000.0);
        println!("   Per pred:   {:.2} µs", pred_time.as_micros() as f64 / 1_000.0);
    }

    #[test]
    fn bench_realistic_5k() {
        // License check skipped for pure Rust benchmarks
        let (df, model) = create_realistic_dataset(5_000);

        let options = GLMOptions {
            objective: "poisson".to_string(),
            max_iterations: 50,
            tolerance: 1e-6,
            verbose: false,
            tweedie_power: 1.5,
        };

        let start = Instant::now();
        let fitted = fit_glm(&model, &df, "claims", Some("exposure"), None, options).unwrap();
        let fit_time = start.elapsed();

        let pred_start = Instant::now();
        let _ = fitted.predict(&df).unwrap();
        let pred_time = pred_start.elapsed();

        println!("\n📊 Realistic Dataset: 5,000 records, 4 features, 27 total bins");
        println!("   Fit time:   {:.2} ms", fit_time.as_secs_f64() * 1000.0);
        println!("   Pred time:  {:.2} ms", pred_time.as_secs_f64() * 1000.0);
        println!("   Per pred:   {:.2} µs", pred_time.as_micros() as f64 / 5_000.0);
    }

    #[test]
    fn bench_realistic_10k() {
        // License check skipped for pure Rust benchmarks
        let (df, model) = create_realistic_dataset(10_000);

        let options = GLMOptions {
            objective: "poisson".to_string(),
            max_iterations: 50,
            tolerance: 1e-6,
            verbose: false,
            tweedie_power: 1.5,
        };

        let start = Instant::now();
        let fitted = fit_glm(&model, &df, "claims", Some("exposure"), None, options).unwrap();
        let fit_time = start.elapsed();

        let pred_start = Instant::now();
        let _ = fitted.predict(&df).unwrap();
        let pred_time = pred_start.elapsed();

        println!("\n📊 Realistic Dataset: 10,000 records, 4 features, 27 total bins");
        println!("   Fit time:   {:.2} ms", fit_time.as_secs_f64() * 1000.0);
        println!("   Pred time:  {:.2} ms", pred_time.as_secs_f64() * 1000.0);
        println!("   Per pred:   {:.2} µs", pred_time.as_micros() as f64 / 10_000.0);
    }

    #[test]
    fn bench_realistic_50k() {
        // License check skipped for pure Rust benchmarks
        let (df, model) = create_realistic_dataset(50_000);

        let options = GLMOptions {
            objective: "poisson".to_string(),
            max_iterations: 50,
            tolerance: 1e-6,
            verbose: false,
            tweedie_power: 1.5,
        };

        let start = Instant::now();
        let fitted = fit_glm(&model, &df, "claims", Some("exposure"), None, options).unwrap();
        let fit_time = start.elapsed();

        let pred_start = Instant::now();
        let _ = fitted.predict(&df).unwrap();
        let pred_time = pred_start.elapsed();

        println!("\n📊 Realistic Dataset: 50,000 records, 4 features, 27 total bins");
        println!("   Fit time:   {:.2} ms ({:.2} sec)", fit_time.as_secs_f64() * 1000.0, fit_time.as_secs_f64());
        println!("   Pred time:  {:.2} ms", pred_time.as_secs_f64() * 1000.0);
        println!("   Per pred:   {:.2} µs", pred_time.as_micros() as f64 / 50_000.0);
    }

    #[test]
    fn bench_realistic_100k() {
        // License check skipped for pure Rust benchmarks
        let (df, model) = create_realistic_dataset(100_000);

        let options = GLMOptions {
            objective: "poisson".to_string(),
            max_iterations: 50,
            tolerance: 1e-6,
            verbose: false,
            tweedie_power: 1.5,
        };

        let start = Instant::now();
        let fitted = fit_glm(&model, &df, "claims", Some("exposure"), None, options).unwrap();
        let fit_time = start.elapsed();

        let pred_start = Instant::now();
        let _ = fitted.predict(&df).unwrap();
        let pred_time = pred_start.elapsed();

        println!("\n📊 Realistic Dataset: 100,000 records, 4 features, 27 total bins");
        println!("   Fit time:   {:.2} ms ({:.2} sec)", fit_time.as_secs_f64() * 1000.0, fit_time.as_secs_f64());
        println!("   Pred time:  {:.2} ms", pred_time.as_secs_f64() * 1000.0);
        println!("   Per pred:   {:.2} µs", pred_time.as_micros() as f64 / 100_000.0);
    }

    #[test]
    fn bench_realistic_1m() {
        // License check skipped for pure Rust benchmarks
        let (df, model) = create_realistic_dataset(1_000_000);

        let options = GLMOptions {
            objective: "poisson".to_string(),
            max_iterations: 50,
            tolerance: 1e-6,
            verbose: false,
            tweedie_power: 1.5,
        };

        let start = Instant::now();
        let fitted = fit_glm(&model, &df, "claims", Some("exposure"), None, options).unwrap();
        let fit_time = start.elapsed();

        let pred_start = Instant::now();
        let _ = fitted.predict(&df).unwrap();
        let pred_time = pred_start.elapsed();

        println!("\n📊 Realistic Dataset: 1,000,000 records, 4 features, 27 total bins");
        println!("   Fit time:   {:.2} ms ({:.2} sec)", fit_time.as_secs_f64() * 1000.0, fit_time.as_secs_f64());
        println!("   Pred time:  {:.2} ms ({:.2} sec)", pred_time.as_secs_f64() * 1000.0, pred_time.as_secs_f64());
        println!("   Per pred:   {:.2} µs", pred_time.as_micros() as f64 / 1_000_000.0);
    }

    #[test]
    fn bench_all_sizes_summary() {
        // License check skipped for pure Rust benchmarks

        println!("\n{}", "=".repeat(80));
        println!("REALISTIC PERFORMANCE BENCHMARKS - Multi-feature Insurance Model");
        println!("Features: age, vehicles, region, claim_history (27 total bins)");
        println!("{}\n", "=".repeat(80));

        for &n in &[1_000, 5_000, 10_000, 50_000, 100_000, 1_000_000] {
            let (df, model) = create_realistic_dataset(n);

            let options = GLMOptions {
                objective: "poisson".to_string(),
                max_iterations: 50,
                tolerance: 1e-6,
                verbose: false,
                tweedie_power: 1.5,
            };

            let start = Instant::now();
            let fitted = fit_glm(&model, &df, "claims", Some("exposure"), None, options).unwrap();
            let fit_time = start.elapsed();

            let pred_start = Instant::now();
            let _ = fitted.predict(&df).unwrap();
            let pred_time = pred_start.elapsed();

            println!("{:>7} records | Fit: {:>8.2} ms | Predict: {:>7.2} ms | Per-pred: {:>6.2} µs",
                n,
                fit_time.as_secs_f64() * 1000.0,
                pred_time.as_secs_f64() * 1000.0,
                pred_time.as_micros() as f64 / n as f64
            );
        }

        println!("\n{}\n", "=".repeat(80));
    }
}
