use polars::prelude::*;
use crate::rating_model::{RatingModel, RatingTable};
use super::loss::LossFunction;
use super::matching::precompute_all_matches;

/// Options for GLM fitting
#[derive(Debug, Clone)]
pub struct GLMOptions {
    pub max_iterations: usize,
    pub tolerance: f64,
    pub verbose: bool,
    pub objective: String, // Required: loss function (e.g., "poisson", "gamma", "tweedie", "gaussian", "binary")
    pub tweedie_power: f64, // Power parameter for Tweedie (default 1.5)
}

impl Default for GLMOptions {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            tolerance: 1e-6,
            verbose: false,
            objective: "gaussian".to_string(), // Default to Gaussian/regression
            tweedie_power: 1.5,
        }
    }
}

/// Fits a GLM by updating the rating factors in the provided RatingModel
///
/// # Arguments
/// * `model` - The RatingModel whose factors will be updated (structure is preserved)
/// * `df` - Training data
/// * `target_col` - Name of the target column
/// * `weight_col` - Optional name of the weight column
/// * `options` - GLM fitting options
///
/// # Returns
/// A new RatingModel with updated rating factors
pub fn fit_glm(
    model: &RatingModel,
    df: &DataFrame,
    target_col: &str,
    weight_col: Option<&str>,
    options: GLMOptions,
) -> Result<RatingModel, PolarsError> {
    // Validate inputs
    validate_inputs(model, df, target_col, weight_col)?;

    // Initialize loss function from required objective
    let mut loss_fn = LossFunction::from_objective(&options.objective);
    // Override Tweedie power if specified
    if let LossFunction::Tweedie(_) = loss_fn {
        loss_fn = LossFunction::Tweedie(options.tweedie_power);
    }

    // Extract target and weights
    let target = df.column(target_col)?.f64()?;
    let weights_owned;
    let weights: &ChunkedArray<Float64Type> = match weight_col {
        Some(col) => df.column(col)?.f64()?,
        None => {
            weights_owned = ChunkedArray::from_vec("weights".into(), vec![1.0; df.height()]);
            &weights_owned
        }
    };

    // Clone the model to create our working copy
    let mut working_model = model.clone();

    // 🚀 OPTIMIZATION: Pre-compute observation-to-table matches ONCE
    // This avoids re-matching every iteration
    if options.verbose {
        println!("Pre-computing observation matches...");
    }
    let precomputed_matches = precompute_all_matches(&working_model, df)?;
    if options.verbose {
        println!("Matches computed. Starting iterations...");
    }

    // Main coordinate descent loop
    for iteration in 0..options.max_iterations {
        let mut max_change: f64 = 0.0;

        // Iterate over each table (skip the mean table at index 0)
        for table_idx in 1..working_model.tables.len() {
            let change = update_table_factors_precomputed(
                &mut working_model,
                table_idx,
                &precomputed_matches[table_idx],
                &precomputed_matches,  // Pass all matches for prediction
                &target,
                &weights,
                &loss_fn,
                &options,
            )?;

            max_change = max_change.max(change);
        }

        if options.verbose {
            println!("Iteration {}: max_change = {:.6e}", iteration + 1, max_change);
        }

        // Check convergence
        if max_change < options.tolerance {
            if options.verbose {
                println!("Converged after {} iterations", iteration + 1);
            }
            break;
        }
    }

    Ok(working_model)
}

/// Updates the rating factors for a single table using precomputed matches
/// Returns the maximum absolute change in rating factors
fn update_table_factors_precomputed(
    model: &mut RatingModel,
    table_idx: usize,
    table_matches: &[Option<usize>],
    all_matches: &[Vec<Option<usize>>],
    target: &ChunkedArray<Float64Type>,
    weights: &ChunkedArray<Float64Type>,
    loss_fn: &LossFunction,
    options: &GLMOptions,
) -> Result<f64, PolarsError> {
    // 1. Compute predictions WITHOUT this table using precomputed matches
    let partial_preds = predict_without_table_precomputed(model, table_idx, all_matches)?;

    // 2. Compute working residuals (depends on link function)
    let working_residuals = loss_fn.compute_working_residuals(target, &partial_preds)?;

    // 3. Aggregate by matched row to compute optimal factors (matches already precomputed!)
    let new_factors = compute_optimal_factors(
        table_matches,
        &working_residuals,
        weights,
        &model.tables[table_idx],
    )?;

    // 4. Update the table and compute max change
    let max_change = update_table_with_factors(&mut model.tables[table_idx], new_factors)?;

    Ok(max_change)
}

/// Predicts using all tables except the one at table_idx, using precomputed matches
fn predict_without_table_precomputed(
    model: &RatingModel,
    exclude_idx: usize,
    all_matches: &[Vec<Option<usize>>],
) -> Result<Vec<f64>, PolarsError> {
    let n_rows = all_matches[0].len();
    let mut predictions = vec![0.0; n_rows];

    for (idx, table) in model.tables.iter().enumerate() {
        if idx == exclude_idx {
            continue;
        }

        // Use precomputed matches for this table
        let matches = &all_matches[idx];
        for (obs_idx, match_idx_opt) in matches.iter().enumerate() {
            if let Some(match_idx) = match_idx_opt {
                let factor = table.get_rating_factor(*match_idx);
                predictions[obs_idx] += factor;
            }
        }
    }

    Ok(predictions)
}

/// Computes optimal rating factors for each row in the table
/// by aggregating working residuals for observations that match that row
/// 🚀 OPTIMIZED: Pre-allocated vector aggregation for zero-overhead performance
fn compute_optimal_factors(
    match_indices: &[Option<usize>],
    working_residuals: &[f64],
    weights: &ChunkedArray<Float64Type>,
    table: &RatingTable,
) -> Result<Vec<f64>, PolarsError> {
    let n_table_rows = table.data.height();

    // Pre-allocate accumulators (indexed by table row)
    let mut sum_weighted = vec![0.0; n_table_rows];
    let mut sum_weight = vec![0.0; n_table_rows];

    // Single pass: direct vector indexing (cache-friendly, auto-vectorizable)
    for (obs_idx, &match_idx_opt) in match_indices.iter().enumerate() {
        if let Some(match_idx) = match_idx_opt {
            let w = weights.get(obs_idx).unwrap_or(1.0);
            sum_weighted[match_idx] += working_residuals[obs_idx] * w;
            sum_weight[match_idx] += w;
        }
    }

    // Compute factors as weighted mean (IRLS update)
    let new_factors: Vec<f64> = (0..n_table_rows)
        .map(|i| {
            if sum_weight[i] > 0.0 {
                sum_weighted[i] / sum_weight[i]
            } else {
                table.get_rating_factor(i)
            }
        })
        .collect();

    Ok(new_factors)
}

/// Updates a table's Rating_Factor column with new factors
/// Returns the maximum absolute change
fn update_table_with_factors(
    table: &mut RatingTable,
    new_factors: Vec<f64>,
) -> Result<f64, PolarsError> {
    let mut max_change: f64 = 0.0;
    let n_rows = table.data.height();

    // Compute max change
    for row_idx in 0..n_rows {
        let old_factor = table.get_rating_factor(row_idx);
        let new_factor = new_factors[row_idx];
        let change = (new_factor - old_factor).abs();
        max_change = max_change.max(change);
    }

    // Update the Rating_Factor column and recreate RatingTable to update metadata
    let new_factor_series = Series::new("Rating_Factor".into(), new_factors);

    // Clone existing data and update Rating_Factor column
    let mut updated_data = table.data.clone();
    updated_data.with_column(new_factor_series)?;

    // Recreate RatingTable with updated data
    *table = RatingTable::new(updated_data, None);

    Ok(max_change)
}

/// Validates inputs for GLM fitting
fn validate_inputs(
    model: &RatingModel,
    df: &DataFrame,
    target_col: &str,
    weight_col: Option<&str>,
) -> Result<(), PolarsError> {
    // Check target column exists
    if df.column(target_col).is_err() {
        return Err(PolarsError::ColumnNotFound(
            format!("Target column '{}' not found", target_col).into()
        ));
    }

    // Check weight column exists if specified
    if let Some(wcol) = weight_col {
        if df.column(wcol).is_err() {
            return Err(PolarsError::ColumnNotFound(
                format!("Weight column '{}' not found", wcol).into()
            ));
        }
    }

    // Check that model has at least 2 tables (mean + at least one feature table)
    if model.tables.len() < 2 {
        return Err(PolarsError::ComputeError(
            "Model must have at least 2 tables (mean + feature tables)".into()
        ));
    }

    Ok(())
}
