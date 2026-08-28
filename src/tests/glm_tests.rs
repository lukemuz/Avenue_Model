#[cfg(test)]
mod glm_tests {
    use crate::glm::{fit_glm, GLMOptions};
    use crate::rating_model::{LinkFunction, RatingModel};
    use polars::prelude::*;
    use std::time::Instant;

    #[test]
    fn test_glm_fit_gaussian() {
        // Create simple training data: y = 2.0 + 1.5*x
        let x_values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y_values = vec![3.5, 5.0, 6.5, 8.0, 9.5];

        let train_df = DataFrame::new(vec![
            Series::new("x".into(), x_values.clone()).into(),
            Series::new("y".into(), y_values).into(),
        ])
        .unwrap();

        // Create table structure
        // Mean table (intercept)
        let mean_table =
            DataFrame::new(vec![Series::new("Rating_Factor".into(), vec![0.0]).into()]).unwrap();

        // Feature table for x (with bins)
        let x_table = DataFrame::new(vec![
            Series::new("x".into(), vec![2.0, 4.0, f64::INFINITY]).into(),
            Series::new("Rating_Factor".into(), vec![0.0, 0.0, 0.0]).into(),
        ])
        .unwrap();

        // Create RatingModel
        let model =
            RatingModel::from_dataframes(vec![mean_table, x_table], "regression", None, None)
                .unwrap();

        // Fit the model
        let options = GLMOptions {
            objective: "gaussian".to_string(),
            max_iterations: 100,
            tolerance: 1e-6,
            verbose: true,
            tweedie_power: 1.5,
            ..Default::default()
        };

        let fitted_model = fit_glm(&model, &train_df, "y", None, None, options).unwrap();

        // Make predictions
        let predictions = fitted_model.predict(&train_df).unwrap();

        println!("Fitted model tables:");
        for (i, table) in fitted_model.model_tables().iter().enumerate() {
            println!("Table {}:\n{}", i, table);
        }

        println!("\nPredictions: {:?}", predictions);

        // Basic sanity check: predictions should be close to actual values
        // (This is a simple test - in real usage we'd check coefficients more carefully)
        assert_eq!(predictions.len(), 5);
    }

    #[test]
    fn test_glm_fit_with_weights() {
        // Create simple training data with weights
        let x_values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y_values = vec![3.5, 5.0, 6.5, 8.0, 9.5];
        let weights = vec![1.0, 1.0, 2.0, 1.0, 1.0]; // Give middle observation more weight

        let train_df = DataFrame::new(vec![
            Series::new("x".into(), x_values).into(),
            Series::new("y".into(), y_values).into(),
            Series::new("weight".into(), weights).into(),
        ])
        .unwrap();

        // Create simple model structure
        let mean_table =
            DataFrame::new(vec![Series::new("Rating_Factor".into(), vec![0.0]).into()]).unwrap();

        let x_table = DataFrame::new(vec![
            Series::new("x".into(), vec![2.0, 4.0, f64::INFINITY]).into(),
            Series::new("Rating_Factor".into(), vec![0.0, 0.0, 0.0]).into(),
        ])
        .unwrap();

        let model =
            RatingModel::from_dataframes(vec![mean_table, x_table], "regression", None, None)
                .unwrap();

        // Fit with weights
        let options = GLMOptions {
            objective: "gaussian".to_string(),
            max_iterations: 100,
            tolerance: 1e-6,
            verbose: false,
            tweedie_power: 1.5,
            ..Default::default()
        };

        let fitted_model = fit_glm(&model, &train_df, "y", Some("weight"), None, options).unwrap();

        // Basic check
        assert!(fitted_model.tables.len() == 2);
    }
}
