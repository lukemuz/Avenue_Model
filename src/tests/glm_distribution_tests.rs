#[cfg(test)]
mod distribution_tests {
    use crate::glm::{fit_glm, GLMOptions};
    use crate::rating_model::RatingModel;
    use polars::prelude::*;
    use std::time::Instant;

    fn create_simple_model(objective: &str) -> RatingModel {
        // Mean table
        let mean_table =
            DataFrame::new(vec![Series::new("Rating_Factor".into(), vec![0.0]).into()]).unwrap();

        // Feature table for x
        let x_table = DataFrame::new(vec![
            Series::new("x".into(), vec![2.0, 4.0, f64::INFINITY]).into(),
            Series::new("Rating_Factor".into(), vec![0.0, 0.0, 0.0]).into(),
        ])
        .unwrap();

        RatingModel::from_dataframes(vec![mean_table, x_table], objective, None, None).unwrap()
    }

    #[test]
    fn test_poisson_glm() {
        // Create Poisson data: log(E[Y]) = 1.0 + 0.5*x
        let x_values = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        // Expected means: exp(1.0 + 0.5*x)
        let y_values = vec![3.0, 4.0, 5.0, 7.0, 10.0, 12.0, 18.0, 24.0, 36.0, 49.0]; // Approx Poisson counts
        let exposure = vec![1.0; 10];

        let train_df = DataFrame::new(vec![
            Series::new("x".into(), x_values).into(),
            Series::new("y".into(), y_values).into(),
            Series::new("exposure".into(), exposure).into(),
        ])
        .unwrap();

        let model = create_simple_model("poisson");

        let options = GLMOptions {
            objective: "poisson".to_string(),
            max_iterations: 100,
            tolerance: 1e-6,
            verbose: true,
            tweedie_power: 1.5,
            ..Default::default()
        };

        let start = Instant::now();
        let fitted_model =
            fit_glm(&model, &train_df, "y", Some("exposure"), None, options).unwrap();
        let duration = start.elapsed();

        println!("\n=== POISSON GLM ===");
        println!("Fitting time: {:?}", duration);
        println!("Fitted model tables:");
        for (i, table) in fitted_model.model_tables().iter().enumerate() {
            println!("Table {}:\n{}", i, table);
        }

        // Make predictions
        let predictions = fitted_model.predict(&train_df).unwrap();
        println!("\nPredictions: {:?}", predictions);

        assert!(fitted_model.tables.len() == 2);
    }

    #[test]
    fn test_gamma_glm() {
        // Create Gamma data: log(E[Y]) = 2.0 + 0.3*x
        let x_values = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        // Expected means: exp(2.0 + 0.3*x)
        let y_values = vec![7.5, 9.8, 13.0, 17.2, 22.8, 30.2, 40.0, 52.9, 70.0, 92.6]; // Simulated Gamma
        let weights = vec![1.0; 10];

        let train_df = DataFrame::new(vec![
            Series::new("x".into(), x_values).into(),
            Series::new("y".into(), y_values).into(),
            Series::new("weight".into(), weights).into(),
        ])
        .unwrap();

        let model = create_simple_model("gamma");

        let options = GLMOptions {
            objective: "gamma".to_string(),
            max_iterations: 100,
            tolerance: 1e-6,
            verbose: true,
            tweedie_power: 1.5,
            ..Default::default()
        };

        let start = Instant::now();
        let fitted_model = fit_glm(&model, &train_df, "y", Some("weight"), None, options).unwrap();
        let duration = start.elapsed();

        println!("\n=== GAMMA GLM ===");
        println!("Fitting time: {:?}", duration);
        println!("Fitted model tables:");
        for (i, table) in fitted_model.model_tables().iter().enumerate() {
            println!("Table {}:\n{}", i, table);
        }

        // Make predictions
        let predictions = fitted_model.predict(&train_df).unwrap();
        println!("\nPredictions: {:?}", predictions);

        assert!(fitted_model.tables.len() == 2);
    }

    #[test]
    fn test_tweedie_glm() {
        // Create Tweedie data with p=1.5
        let x_values = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let y_values = vec![0.0, 2.0, 0.0, 5.0, 8.0, 0.0, 12.0, 18.0, 25.0, 30.0]; // Mix of zeros and continuous
        let weights = vec![1.0; 10];

        let train_df = DataFrame::new(vec![
            Series::new("x".into(), x_values).into(),
            Series::new("y".into(), y_values).into(),
            Series::new("weight".into(), weights).into(),
        ])
        .unwrap();

        let model = create_simple_model("tweedie");

        let options = GLMOptions {
            objective: "tweedie".to_string(),
            max_iterations: 100,
            tolerance: 1e-6,
            verbose: true,
            tweedie_power: 1.5,
            ..Default::default()
        };

        let start = Instant::now();
        let fitted_model = fit_glm(&model, &train_df, "y", Some("weight"), None, options).unwrap();
        let duration = start.elapsed();

        println!("\n=== TWEEDIE GLM (p=1.5) ===");
        println!("Fitting time: {:?}", duration);
        println!("Fitted model tables:");
        for (i, table) in fitted_model.model_tables().iter().enumerate() {
            println!("Table {}:\n{}", i, table);
        }

        // Make predictions
        let predictions = fitted_model.predict(&train_df).unwrap();
        println!("\nPredictions: {:?}", predictions);

        assert!(fitted_model.tables.len() == 2);
    }

    #[test]
    fn test_logistic_regression() {
        // Create binary classification data
        let x_values = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let y_values = vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0]; // Binary outcomes
        let weights = vec![1.0; 10];

        let train_df = DataFrame::new(vec![
            Series::new("x".into(), x_values).into(),
            Series::new("y".into(), y_values).into(),
            Series::new("weight".into(), weights).into(),
        ])
        .unwrap();

        let model = create_simple_model("binary");

        let options = GLMOptions {
            objective: "binary".to_string(),
            max_iterations: 100,
            tolerance: 1e-6,
            verbose: true,
            tweedie_power: 1.5,
            ..Default::default()
        };

        let start = Instant::now();
        let fitted_model = fit_glm(&model, &train_df, "y", Some("weight"), None, options).unwrap();
        let duration = start.elapsed();

        println!("\n=== LOGISTIC REGRESSION ===");
        println!("Fitting time: {:?}", duration);
        println!("Fitted model tables:");
        for (i, table) in fitted_model.model_tables().iter().enumerate() {
            println!("Table {}:\n{}", i, table);
        }

        // Make predictions
        let predictions = fitted_model.predict(&train_df).unwrap();
        println!("\nPredictions (probabilities): {:?}", predictions);

        // Check predictions are probabilities (between 0 and 1)
        let pred_f64 = predictions.f64().unwrap();
        for i in 0..pred_f64.len() {
            if let Some(pred) = pred_f64.get(i) {
                assert!(
                    pred >= 0.0 && pred <= 1.0,
                    "Prediction {} is not a valid probability: {}",
                    i,
                    pred
                );
            }
        }

        assert!(fitted_model.tables.len() == 2);
    }

    #[test]
    fn test_large_dataset_performance() {
        // Create a larger dataset for performance testing
        let n = 10000;
        let x_values: Vec<f64> = (0..n).map(|i| (i % 100) as f64).collect();
        let y_values: Vec<f64> = x_values
            .iter()
            .map(|&x| 5.0 + 0.5 * x + (x * 0.1).sin() * 2.0)
            .collect();
        let weights = vec![1.0; n];

        let train_df = DataFrame::new(vec![
            Series::new("x".into(), x_values).into(),
            Series::new("y".into(), y_values).into(),
            Series::new("weight".into(), weights).into(),
        ])
        .unwrap();

        // Create model with more granular bins
        let mean_table =
            DataFrame::new(vec![Series::new("Rating_Factor".into(), vec![0.0]).into()]).unwrap();

        let x_bins: Vec<f64> = (0..20)
            .map(|i| (i * 5) as f64)
            .chain(std::iter::once(f64::INFINITY))
            .collect();
        let x_table = DataFrame::new(vec![
            Series::new("x".into(), x_bins.clone()).into(),
            Series::new("Rating_Factor".into(), vec![0.0; x_bins.len()]).into(),
        ])
        .unwrap();

        let model =
            RatingModel::from_dataframes(vec![mean_table, x_table], "regression", None, None)
                .unwrap();

        let options = GLMOptions {
            objective: "gaussian".to_string(),
            max_iterations: 50,
            tolerance: 1e-6,
            verbose: true,
            tweedie_power: 1.5,
            ..Default::default()
        };

        println!("\n=== LARGE DATASET PERFORMANCE TEST ===");
        println!("Dataset size: {} rows", n);
        println!("Number of bins: {}", x_bins.len());

        let start = Instant::now();
        let fitted_model = fit_glm(&model, &train_df, "y", Some("weight"), None, options).unwrap();
        let duration = start.elapsed();

        println!("Total fitting time: {:?}", duration);
        println!("Time per iteration: {:?}", duration / 50);

        // Make predictions
        let pred_start = Instant::now();
        let predictions = fitted_model.predict(&train_df).unwrap();
        let pred_duration = pred_start.elapsed();

        println!("Prediction time: {:?}", pred_duration);
        println!("Time per prediction: {:?}", pred_duration / n as u32);

        assert_eq!(predictions.len(), n);
    }
}
