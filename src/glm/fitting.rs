use polars::prelude::*;
use crate::rating_model::{RatingModel, RatingTable};
use super::inference::{compute_inference, GLMInference};
use super::loss::{LossFunction, ETA_CLAMP};
use super::matching::precompute_all_matches;

/// How the fitted tables are anchored once the fit has converged.
///
/// A model carrying an intercept table *and* a free factor for every level of every
/// table is over-parameterised: you can add a constant to one table and subtract it
/// from the intercept without changing a single prediction. Backfitting will happily
/// settle on any point along that flat direction, which makes the tables — the actual
/// deliverable — depend on table order and starting values rather than on the data.
///
/// Normalising after every sweep pins the solution down, and as a side effect removes
/// the null-space drift that otherwise slows convergence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Normalization {
    /// Leave factors exactly where the fit put them. Predictions are still correct,
    /// but the split between the intercept and the feature tables is arbitrary.
    None,
    /// Anchor each feature table's first row at zero, moving it into the intercept.
    /// Every other level then reads directly as a relativity against that base level,
    /// which is how rating tables are normally presented.
    BaseLevel,
    /// Anchor each feature table at its exposure-weighted mean, so the intercept
    /// carries the overall average level and the factors sum to zero across exposure.
    WeightedMean,
}

impl Default for Normalization {
    fn default() -> Self {
        Normalization::BaseLevel
    }
}

/// Options for GLM fitting
#[derive(Debug, Clone)]
pub struct GLMOptions {
    pub max_iterations: usize,
    pub tolerance: f64,
    pub verbose: bool,
    /// Required: loss function ("poisson", "gamma", "tweedie", "gaussian", "binary")
    pub objective: String,
    /// Power parameter for Tweedie (default 1.5)
    pub tweedie_power: f64,
    /// How to anchor the over-parameterised tables. See [`Normalization`].
    pub normalization: Normalization,
    /// Compute standard errors and fit statistics after converging.
    ///
    /// Costs one pass over the data plus the inversion of a `p x p` matrix, where `p`
    /// is the number of free parameters. Negligible for ordinary rating models; turn
    /// it off for models with thousands of levels.
    pub compute_standard_errors: bool,
}

impl Default for GLMOptions {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            tolerance: 1e-8,
            verbose: false,
            objective: "gaussian".to_string(), // Default to Gaussian/regression
            tweedie_power: 1.5,
            normalization: Normalization::default(),
            compute_standard_errors: true,
        }
    }
}

/// What happened during a fit, alongside the fitted model.
#[derive(Debug, Clone)]
pub struct GLMDiagnostics {
    /// Sweeps performed over the full set of tables.
    pub iterations: usize,
    /// Whether the relative deviance change fell below `tolerance`.
    pub converged: bool,
    /// Weighted deviance of the final fit.
    pub deviance: f64,
    /// Weighted deviance of the intercept-only fit, for reference.
    pub null_deviance: f64,
    /// Deviance after each sweep, in order.
    pub deviance_history: Vec<f64>,
    /// Table rows that received no exposure and so kept their starting factor,
    /// as `(table_index, row_index)`.
    pub unfitted_rows: Vec<(usize, usize)>,
    /// Standard errors and fit statistics, when
    /// [`GLMOptions::compute_standard_errors`] is set.
    pub inference: Option<GLMInference>,
    /// Why `inference` is absent despite being requested. The fit itself is unaffected.
    pub inference_error: Option<String>,
}

impl GLMDiagnostics {
    /// Fraction of the null deviance explained by the fit.
    pub fn pseudo_r2(&self) -> f64 {
        if self.null_deviance > 0.0 {
            1.0 - self.deviance / self.null_deviance
        } else {
            0.0
        }
    }
}

/// Largest change permitted in a single rating factor in one sweep, on the link scale.
///
/// Only binds for the logit link, whose IRLS denominator collapses toward zero under
/// separation. Large enough never to interfere with an ordinary fit.
const MAX_STEP: f64 = 10.0;

/// Fits a GLM by updating the rating factors in the provided RatingModel
///
/// # Arguments
/// * `model` - The RatingModel whose factors will be updated (structure is preserved)
/// * `df` - Training data
/// * `target_col` - Name of the target column
/// * `weight_col` - Optional name of the weight column
/// * `offset_col` - Optional name of the offset column (added to linear predictor)
/// * `options` - GLM fitting options
///
/// # Returns
/// A new RatingModel with updated rating factors
pub fn fit_glm(
    model: &RatingModel,
    df: &DataFrame,
    target_col: &str,
    weight_col: Option<&str>,
    offset_col: Option<&str>,
    options: GLMOptions,
) -> Result<RatingModel, PolarsError> {
    fit_glm_with_diagnostics(model, df, target_col, weight_col, offset_col, options)
        .map(|(m, _)| m)
}

/// As [`fit_glm`], but also returns convergence and deviance information.
pub fn fit_glm_with_diagnostics(
    model: &RatingModel,
    df: &DataFrame,
    target_col: &str,
    weight_col: Option<&str>,
    offset_col: Option<&str>,
    options: GLMOptions,
) -> Result<(RatingModel, GLMDiagnostics), PolarsError> {
    validate_inputs(model, df, target_col, weight_col, offset_col)?;

    // Initialize loss function from required objective
    let mut loss_fn = LossFunction::from_objective(&options.objective);
    // Override Tweedie power if specified
    if let LossFunction::Tweedie(_) = loss_fn {
        loss_fn = LossFunction::Tweedie(options.tweedie_power);
    }

    let n = df.height();
    let target = read_f64_column(df, target_col, "target")?;
    let weights = match weight_col {
        Some(col) => {
            let w = read_f64_column(df, col, "weight")?;
            if let Some(bad) = w.iter().position(|v| *v < 0.0) {
                return Err(PolarsError::ComputeError(
                    format!("Weight column '{}' has a negative value at row {}", col, bad).into(),
                ));
            }
            w
        }
        None => vec![1.0; n],
    };
    let offset = match offset_col {
        Some(col) => read_f64_column(df, col, "offset")?,
        None => vec![0.0; n],
    };

    let mut working_model = model.clone();

    if options.verbose {
        println!("Pre-computing observation matches...");
    }
    let matches = precompute_all_matches(&working_model, df)?;
    validate_matches(&working_model, &matches, n)?;
    if options.verbose {
        println!("Matches computed. Starting iterations...");
    }

    // Work on plain Vecs during fitting; the DataFrames are rewritten once at the end.
    // Keeps the hot loop away from Polars column lookups and DataFrame clones.
    let n_tables = working_model.tables.len();
    let mut factors: Vec<Vec<f64>> = working_model
        .tables
        .iter()
        .map(|t| {
            let ca = t.data.column("Rating_Factor").unwrap().f64().unwrap();
            (0..ca.len()).map(|i| ca.get(i).unwrap_or(0.0)).collect()
        })
        .collect();

    // Which tables the sweep is allowed to touch.
    let updatable: Vec<bool> = working_model
        .tables
        .iter()
        .map(|t| !t.metadata.is_offset)
        .collect();

    // Exposure behind each table row, fixed for the whole fit. Used for the
    // weighted-mean anchor and to report rows that never got any data.
    let row_exposure: Vec<Vec<f64>> = (0..n_tables)
        .map(|t| {
            let mut e = vec![0.0; factors[t].len()];
            for (i, m) in matches[t].iter().enumerate() {
                if let Some(r) = m {
                    e[*r] += weights[i];
                }
            }
            e
        })
        .collect();

    // Running linear predictor from the tables only; `offset` is added where used.
    let mut eta = vec![0.0; n];
    for t in 0..n_tables {
        for (i, m) in matches[t].iter().enumerate() {
            if let Some(r) = m {
                eta[i] += factors[t][*r];
            }
        }
    }

    // Scratch buffers reused across every table and every sweep.
    let mut means = vec![0.0; n];
    let mut numer = Vec::new();
    let mut denom = Vec::new();

    let null_deviance = null_deviance(&loss_fn, &target, &weights, &offset);

    let mut deviance_history: Vec<f64> = Vec::with_capacity(options.max_iterations);
    let mut prev_deviance = f64::INFINITY;
    let mut converged = false;
    let mut iterations = 0usize;

    for iteration in 0..options.max_iterations {
        iterations = iteration + 1;

        for t in 0..n_tables {
            if !updatable[t] {
                continue;
            }
            update_table(
                t,
                &mut factors,
                &mut eta,
                &matches[t],
                &target,
                &weights,
                &offset,
                &row_exposure[t],
                &working_model.tables[t],
                &loss_fn,
                &mut numer,
                &mut denom,
            );
        }

        if options.normalization != Normalization::None {
            normalize(
                &mut factors,
                &row_exposure,
                &working_model.tables,
                &updatable,
                options.normalization,
            );
        }

        // Deviance of the current fit.
        for i in 0..n {
            means[i] = loss_fn.inverse_link(eta[i] + offset[i]);
        }
        let deviance = loss_fn.total_deviance(&target, &means, &weights);
        deviance_history.push(deviance);

        let rel_change = if prev_deviance.is_finite() && prev_deviance.abs() > 0.0 {
            (prev_deviance - deviance).abs() / (prev_deviance.abs() + 1e-12)
        } else {
            f64::INFINITY
        };

        if options.verbose {
            println!(
                "Iteration {}: deviance = {:.10e}, rel change = {:.3e}",
                iterations, deviance, rel_change
            );
        }

        if rel_change < options.tolerance {
            converged = true;
            if options.verbose {
                println!("Converged after {} iterations", iterations);
            }
            break;
        }
        prev_deviance = deviance;
    }

    if options.verbose && !converged {
        println!(
            "WARNING: did not converge in {} iterations (tolerance {:.1e})",
            options.max_iterations, options.tolerance
        );
    }

    // `means` holds the final fit only if the loop ran at least one sweep; recompute
    // so inference never reads a stale buffer.
    for i in 0..n {
        means[i] = loss_fn.inverse_link(eta[i] + offset[i]);
    }

    // Inference is a report on the fit, not part of it. A model whose tables are
    // collinear still has perfectly good predictions, so a failure here is recorded
    // rather than allowed to discard the fit the caller asked for.
    let mut inference_error: Option<String> = None;
    let inference = if options.compute_standard_errors {
        match compute_inference(
            &loss_fn,
            &target,
            &weights,
            &means,
            &matches,
            &factors,
            &row_exposure,
            &updatable,
            options.normalization,
        ) {
            Ok(inf) => Some(inf),
            Err(e) => {
                if options.verbose {
                    println!("WARNING: standard errors unavailable: {}", e);
                }
                inference_error = Some(e.to_string());
                None
            }
        }
    } else {
        None
    };

    // Write the fitted factors back into the model's DataFrames.
    for t in 0..n_tables {
        write_back_factors(&mut working_model.tables[t], &factors[t])?;
    }

    let mut unfitted_rows = Vec::new();
    for t in 0..n_tables {
        if !updatable[t] {
            continue;
        }
        for (r, e) in row_exposure[t].iter().enumerate() {
            if *e <= 0.0 {
                unfitted_rows.push((t, r));
            }
        }
    }

    let diagnostics = GLMDiagnostics {
        iterations,
        converged,
        deviance: *deviance_history.last().unwrap_or(&f64::NAN),
        null_deviance,
        deviance_history,
        unfitted_rows,
        inference,
        inference_error,
    };

    Ok((working_model, diagnostics))
}

/// Updates every row of one table, holding all other tables fixed, and folds the
/// change straight into the running linear predictor.
///
/// Two update rules, both exact minimisers of the deviance for this table given the
/// others, except for the logit case which takes a single IRLS step:
///
/// * **Log link.** With `mu_i = c_i * exp(beta_r)` the score equation solves in closed
///   form to `beta_r <- beta_r + ln(A / E)`, where `A = sum(a * mu^(1-p) * y)` and
///   `E = sum(a * mu^(2-p))`. For Poisson (`p = 1`) that is literally
///   `ln(actual / expected)`. Writing it as an increment on the current factor keeps
///   every quantity scaled by `mu`, which tracks `y` — far better conditioned than
///   solving for the level from scratch.
///
/// * **Otherwise.** The IRLS step `beta_r <- beta_r + sum(a*w*r) / sum(a*w)`, with the
///   weight and link-scale residual supplied by the family.
fn update_table(
    t: usize,
    factors: &mut [Vec<f64>],
    eta: &mut [f64],
    table_matches: &[Option<usize>],
    target: &[f64],
    weights: &[f64],
    offset: &[f64],
    row_exposure: &[f64],
    table: &RatingTable,
    loss_fn: &LossFunction,
    numer: &mut Vec<f64>,
    denom: &mut Vec<f64>,
) {
    let n_rows = factors[t].len();
    numer.clear();
    numer.resize(n_rows, 0.0);
    denom.clear();
    denom.resize(n_rows, 0.0);

    let power = loss_fn.log_link_variance_power();

    for (i, m) in table_matches.iter().enumerate() {
        let Some(r) = *m else { continue };
        let a = weights[i];
        if a == 0.0 {
            continue;
        }
        let mu = loss_fn.inverse_link(eta[i] + offset[i]);
        match power {
            // A = sum(a * mu^(1-p) * y), E = sum(a * mu^(2-p))
            Some(p) => {
                let base = mu.powf(1.0 - p);
                numer[r] += a * base * target[i];
                denom[r] += a * base * mu;
            }
            // sum(a * w * r) and sum(a * w)
            None => {
                numer[r] += a * loss_fn.weighted_link_residual(target[i], mu);
                denom[r] += a * loss_fn.irls_weight(mu);
            }
        }
    }

    for r in 0..n_rows {
        // Rows with no exposure, locked rows, and degenerate denominators keep
        // whatever factor they started with.
        if row_exposure[r] <= 0.0 || table.is_row_offset(r) {
            continue;
        }
        if !(denom[r] > 0.0) || !denom[r].is_finite() {
            continue;
        }

        let step = match power {
            Some(_) => {
                // ln(A / E); A <= 0 means the level has no positive response at all,
                // whose MLE is -inf. Fall back to the largest downward step allowed.
                if numer[r] > 0.0 {
                    (numer[r] / denom[r]).ln()
                } else {
                    -MAX_STEP
                }
            }
            None => numer[r] / denom[r],
        };

        if !step.is_finite() {
            continue;
        }

        let old = factors[t][r];
        let new = (old + step.clamp(-MAX_STEP, MAX_STEP)).clamp(-ETA_CLAMP, ETA_CLAMP);
        factors[t][r] = new;
        numer[r] = new - old; // reuse as the delta to apply to eta
    }

    // Fold the changes into the running linear predictor. Reusing `numer` as the
    // per-row delta keeps this to a single pass with no extra allocation.
    for r in 0..n_rows {
        if row_exposure[r] <= 0.0 || table.is_row_offset(r) || !(denom[r] > 0.0) || !denom[r].is_finite() {
            numer[r] = 0.0;
        }
    }
    for (i, m) in table_matches.iter().enumerate() {
        if let Some(r) = m {
            eta[i] += numer[*r];
        }
    }
}

/// Shifts a constant out of each feature table and into the intercept table, leaving
/// every prediction unchanged. See [`Normalization`].
fn normalize(
    factors: &mut [Vec<f64>],
    row_exposure: &[Vec<f64>],
    tables: &[RatingTable],
    updatable: &[bool],
    mode: Normalization,
) {
    // Nothing to anchor against if the intercept itself is locked.
    if factors.is_empty() || !updatable[0] || factors[0].len() != 1 || tables[0].is_row_offset(0) {
        return;
    }

    let mut shift_into_intercept = 0.0;

    for t in 1..factors.len() {
        if !updatable[t] {
            continue;
        }
        // A table with locked rows cannot be shifted wholesale — moving the free rows
        // alone would change predictions.
        if (0..factors[t].len()).any(|r| tables[t].is_row_offset(r)) {
            continue;
        }

        let anchor = match mode {
            Normalization::None => continue,
            Normalization::BaseLevel => factors[t][0],
            Normalization::WeightedMean => {
                let total: f64 = row_exposure[t].iter().sum();
                if !(total > 0.0) {
                    continue;
                }
                factors[t]
                    .iter()
                    .zip(row_exposure[t].iter())
                    .map(|(f, e)| f * e)
                    .sum::<f64>()
                    / total
            }
        };

        if anchor == 0.0 || !anchor.is_finite() {
            continue;
        }
        for f in factors[t].iter_mut() {
            *f -= anchor;
        }
        shift_into_intercept += anchor;
    }

    factors[0][0] = (factors[0][0] + shift_into_intercept).clamp(-ETA_CLAMP, ETA_CLAMP);
}

/// Deviance of the best intercept-only fit, used as the reference for `pseudo_r2`.
fn null_deviance(loss_fn: &LossFunction, target: &[f64], weights: &[f64], offset: &[f64]) -> f64 {
    let mut beta = 0.0f64;
    // Same coordinate update as the fitter, applied to a single global level.
    for _ in 0..200 {
        let mut numer = 0.0;
        let mut denom = 0.0;
        for i in 0..target.len() {
            let a = weights[i];
            if a == 0.0 {
                continue;
            }
            let mu = loss_fn.inverse_link(beta + offset[i]);
            match loss_fn.log_link_variance_power() {
                Some(p) => {
                    let base = mu.powf(1.0 - p);
                    numer += a * base * target[i];
                    denom += a * base * mu;
                }
                None => {
                    numer += a * loss_fn.weighted_link_residual(target[i], mu);
                    denom += a * loss_fn.irls_weight(mu);
                }
            }
        }
        if !(denom > 0.0) || !denom.is_finite() {
            break;
        }
        let step = match loss_fn.log_link_variance_power() {
            Some(_) => {
                if numer > 0.0 {
                    (numer / denom).ln()
                } else {
                    -MAX_STEP
                }
            }
            None => numer / denom,
        };
        if !step.is_finite() {
            break;
        }
        let next = (beta + step.clamp(-MAX_STEP, MAX_STEP)).clamp(-ETA_CLAMP, ETA_CLAMP);
        if (next - beta).abs() < 1e-14 {
            beta = next;
            break;
        }
        beta = next;
    }

    let means: Vec<f64> = offset.iter().map(|o| loss_fn.inverse_link(beta + o)).collect();
    loss_fn.total_deviance(target, &means, weights)
}

/// Writes fitted factors back into a table's DataFrame, preserving its metadata.
fn write_back_factors(table: &mut RatingTable, factors: &[f64]) -> Result<(), PolarsError> {
    let metadata = table.metadata.clone();
    let row_metadata = table.row_metadata.clone();

    let mut data = table.data.clone();
    data.with_column(Series::new("Rating_Factor".into(), factors.to_vec()))?;

    let mut rebuilt = RatingTable::new(data, None);
    rebuilt.metadata = metadata;
    rebuilt.row_metadata = row_metadata;
    *table = rebuilt;
    Ok(())
}

/// Reads a column as `f64`, rejecting nulls and non-finite values.
fn read_f64_column(df: &DataFrame, name: &str, role: &str) -> Result<Vec<f64>, PolarsError> {
    let ca = df.column(name)?.f64().map_err(|_| {
        PolarsError::ComputeError(
            format!("{} column '{}' must be Float64, found {:?}", role, name, df.column(name).unwrap().dtype()).into(),
        )
    })?;

    let mut out = Vec::with_capacity(ca.len());
    for i in 0..ca.len() {
        match ca.get(i) {
            Some(v) if v.is_finite() => out.push(v),
            Some(v) => {
                return Err(PolarsError::ComputeError(
                    format!("{} column '{}' has a non-finite value ({}) at row {}", role, name, v, i).into(),
                ))
            }
            None => {
                return Err(PolarsError::ComputeError(
                    format!("{} column '{}' has a null at row {}", role, name, i).into(),
                ))
            }
        }
    }
    Ok(out)
}

/// Every observation must land on a row of every table.
///
/// An unmatched observation silently contributes nothing to that table's linear
/// predictor and is silently excluded from the table's update, so a missing feature
/// column or an uncovered value would otherwise produce a plausible-looking fit from
/// a model that quietly dropped a term.
fn validate_matches(
    model: &RatingModel,
    matches: &[Vec<Option<usize>>],
    n_rows: usize,
) -> Result<(), PolarsError> {
    for (t, table_matches) in matches.iter().enumerate() {
        let unmatched = table_matches.iter().filter(|m| m.is_none()).count();
        if unmatched == 0 {
            continue;
        }
        let first = table_matches.iter().position(|m| m.is_none()).unwrap();
        let features: Vec<String> = model.tables[t]
            .get_feature_info()
            .keys()
            .cloned()
            .collect();
        return Err(PolarsError::ComputeError(
            format!(
                "Table {} matched no row for {} of {} observations (first at row {}). \
                 Table features: [{}]. Every observation must fall in some row: check that \
                 those columns are present with the expected dtype, that numeric tables have \
                 a final unbounded (inf) row, and that categorical tables cover every level \
                 or carry a -999 wildcard.",
                t, unmatched, n_rows, first, features.join(", ")
            )
            .into(),
        ));
    }
    Ok(())
}

/// Validates inputs for GLM fitting
fn validate_inputs(
    model: &RatingModel,
    df: &DataFrame,
    target_col: &str,
    weight_col: Option<&str>,
    offset_col: Option<&str>,
) -> Result<(), PolarsError> {
    if df.height() == 0 {
        return Err(PolarsError::ComputeError("Training data has no rows".into()));
    }

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

    // Check offset column exists if specified
    if let Some(ocol) = offset_col {
        if df.column(ocol).is_err() {
            return Err(PolarsError::ColumnNotFound(
                format!("Offset column '{}' not found", ocol).into()
            ));
        }
    }

    // Check that model has at least 2 tables (mean + at least one feature table)
    if model.tables.len() < 2 {
        return Err(PolarsError::ComputeError(
            "Model must have at least 2 tables (mean + feature tables)".into()
        ));
    }

    // The intercept table must be a single row for normalization to be meaningful.
    if model.tables[0].data.height() != 1 {
        return Err(PolarsError::ComputeError(
            format!(
                "Table 0 is the intercept and must have exactly 1 row, found {}",
                model.tables[0].data.height()
            )
            .into(),
        ));
    }

    Ok(())
}
