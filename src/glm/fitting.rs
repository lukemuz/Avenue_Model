use polars::prelude::*;
use rayon::prelude::*;
use crate::rating_model::{variate_basis_params, RatingModel, RatingTable, TableSemantics};
use super::inference::{compute_inference, solve_spd, GLMInference};
use super::loss::{pow_special, LossFunction, MAX_STEP};
use super::matching::{precompute_all_matches, NO_MATCH};

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
    /// Convergence threshold on the largest absolute score component, scaled by the
    /// total prior weight.
    ///
    /// At the optimum every free parameter's score is zero, so this measures how far
    /// the fitted factors still have to move. It is the same criterion glum applies
    /// (`gradient_tol`), on the same scale, so the two are comparable.
    ///
    /// This replaced a test on the relative change in deviance, which was far weaker
    /// than it appeared: deviance is quadratic in the parameter error near the
    /// optimum, so `1e-t` on deviance bought only about `1e-(t/2)` on the factors.
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
            // On the score scale. Tighter than glum's 1e-4 default because IRLS
            // converges quadratically and can afford a loose threshold - one more
            // iteration takes it to machine precision - whereas coordinate descent
            // converges linearly and genuinely stops where it is told to.
            tolerance: 1e-9,
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
    /// Whether the largest absolute score fell to `tolerance`.
    ///
    /// False means the returned factors are not at the optimum. Check
    /// [`max_gradient`](Self::max_gradient) to see how far off they are.
    pub converged: bool,
    /// Largest absolute score component at the final iterate, on the same scale as
    /// [`GLMOptions::tolerance`].
    pub max_gradient: f64,
    /// Largest absolute score after each sweep, in order. A sequence that falls
    /// steeply and then crawls is the signature of two near-aliased tables.
    pub gradient_history: Vec<f64>,
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

    // Per-row values and degree for variate tables, None for step tables.
    let variate_values: Vec<Option<(Vec<f64>, usize)>> = working_model
        .tables
        .iter()
        .map(|t| match (t.variate_values(), t.variate_degree()) {
            (Some(v), Some(d)) => Some((v.to_vec(), d)),
            _ => None,
        })
        .collect();

    // Exposure behind each table row, fixed for the whole fit. Used for the
    // weighted-mean anchor and to report rows that never got any data.
    let row_exposure: Vec<Vec<f64>> = (0..n_tables)
        .map(|t| {
            let mut e = vec![0.0; factors[t].len()];
            for (i, m) in matches[t].iter().enumerate() {
                if *m != NO_MATCH {
                    e[*m as usize] += weights[i];
                }
            }
            e
        })
        .collect();

    // Running linear predictor from the tables only; `offset` is added where used.
    let mut eta = vec![0.0; n];
    for t in 0..n_tables {
        for (i, m) in matches[t].iter().enumerate() {
            if *m != NO_MATCH {
                eta[i] += factors[t][*m as usize];
            }
        }
    }

    // The fitted means, carried alongside `eta` rather than re-derived from it inside
    // every table update — see [`apply_row_deltas`]. Seeded here so the invariant
    // `means[i] == inverse_link(eta[i] + offset[i])` holds before the first sweep.
    let mut means: Vec<f64> = (0..n)
        .map(|i| loss_fn.inverse_link(eta[i] + offset[i]))
        .collect();

    // Scratch buffers reused across every table and every sweep.
    let mut numer = Vec::new();
    let mut denom = Vec::new();

    let null_deviance = null_deviance(&loss_fn, &target, &weights, &offset);

    let mut deviance_history: Vec<f64> = Vec::with_capacity(options.max_iterations);
    let mut gradient_history: Vec<f64> = Vec::with_capacity(options.max_iterations);
    let mut converged = false;
    let mut iterations = 0usize;
    let mut max_gradient = f64::INFINITY;
    let mut best_gradient = f64::INFINITY;
    let mut sweeps_without_progress = 0usize;

    // Reused by the convergence test; one slot per table row.
    let mut score_scratch: Vec<Vec<f64>> = factors.iter().map(|f| vec![0.0; f.len()]).collect();

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
                &mut means,
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
                loss_fn.eta_limit(),
            );
        }

        // Re-derive the means exactly. `apply_row_deltas` has been carrying them
        // forward multiplicatively through the sweep, which is worth one pass per sweep
        // to reset: it costs what a single table update used to, and it bounds the
        // rounding those increments can accumulate to a single sweep's worth.
        for i in 0..n {
            means[i] = loss_fn.inverse_link(eta[i] + offset[i]);
        }
        let deviance = loss_fn.total_deviance(&target, &means, &weights);
        deviance_history.push(deviance);

        max_gradient = max_abs_score(
            &loss_fn,
            &target,
            &weights,
            &means,
            &matches,
            &working_model.tables,
            &updatable,
            &row_exposure,
            &variate_values,
            &mut score_scratch,
        );
        gradient_history.push(max_gradient);

        if options.verbose {
            println!(
                "Iteration {}: deviance = {:.10e}, max |score| = {:.3e}",
                iterations, deviance, max_gradient
            );
        }

        if max_gradient <= options.tolerance {
            converged = true;
            if options.verbose {
                println!("Converged after {} iterations", iterations);
            }
            break;
        }

        // A fit can run out of reachable precision above the tolerance - two
        // near-aliased tables trading a constant back and forth, or a threshold set
        // below the noise floor of the sums involved. Continuing cannot help, so stop;
        // but the test is on the score itself, not on the deviance. Stalling the
        // deviance means only that the parameters are within about sqrt(eps) of the
        // optimum, which is exactly the weak signal this criterion exists to replace.
        if max_gradient < best_gradient * (1.0 - STALL_IMPROVEMENT) {
            best_gradient = max_gradient;
            sweeps_without_progress = 0;
        } else {
            sweeps_without_progress += 1;
            if sweeps_without_progress >= STALL_SWEEPS {
                if options.verbose {
                    println!(
                        "Stopping after {} iterations: max |score| = {:.3e} has not \
                         improved in {} sweeps and is above the tolerance of {:.1e}",
                        iterations, max_gradient, STALL_SWEEPS, options.tolerance
                    );
                }
                break;
            }
        }
    }

    if options.verbose && !converged {
        println!(
            "WARNING: did not converge in {} iterations (max |score| = {:.3e}, \
             tolerance {:.1e})",
            iterations, max_gradient, options.tolerance
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
            &variate_values,
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

    // A step row with no exposure keeps whatever factor it started with, so callers
    // need to know. A variate row with no exposure is still fitted — it reads its
    // factor off the table's slope — so it is not listed here.
    let mut unfitted_rows = Vec::new();
    for t in 0..n_tables {
        if !updatable[t] || working_model.tables[t].variate_values().is_some() {
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
        max_gradient,
        gradient_history,
        deviance: *deviance_history.last().unwrap_or(&f64::NAN),
        null_deviance,
        deviance_history,
        unfitted_rows,
        inference,
        inference_error,
    };

    Ok((working_model, diagnostics))
}

/// Updates one table, holding all others fixed, and folds the change straight into
/// the running linear predictor.
///
/// For a [`TableSemantics::Variate`] table the rows share a single slope, so the whole
/// table is one scalar update — see [`update_variate_table`]. Otherwise each row moves
/// independently, and because every observation belongs wholly to one row the weighted
/// least-squares step collapses to a scalar per row:
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
    means: &mut [f64],
    table_matches: &[u32],
    target: &[f64],
    weights: &[f64],
    offset: &[f64],
    row_exposure: &[f64],
    table: &RatingTable,
    loss_fn: &LossFunction,
    numer: &mut Vec<f64>,
    denom: &mut Vec<f64>,
) {
    if let TableSemantics::Variate { values, degree } = table.semantics() {
        update_variate_table(
            t, factors, eta, means, table_matches, target, weights, offset, values, *degree,
            loss_fn,
        );
        return;
    }
    let n_rows = factors[t].len();
    numer.clear();
    numer.resize(n_rows, 0.0);
    denom.clear();
    denom.resize(n_rows, 0.0);

    let power = loss_fn.log_link_variance_power();

    if let Some(chunk) = parallel_chunk(table_matches.len(), n_rows) {
        // Each worker accumulates into its own pair of row vectors and the partials are
        // summed at the end. `n_rows` is the width of one rating table, so replicating
        // it per worker is cheap - `parallel_chunk` declines when it would not be.
        let (par_numer, par_denom) = table_matches
            .par_chunks(chunk)
            .zip(target.par_chunks(chunk))
            .zip(weights.par_chunks(chunk))
            .zip(means.par_chunks(chunk))
            .fold(
                || (vec![0.0f64; n_rows], vec![0.0f64; n_rows]),
                |(mut nu, mut de), (((ms, ys), ws), mus)| {
                    accumulate_block(ms, ys, ws, mus, power, loss_fn, &mut nu, &mut de);
                    (nu, de)
                },
            )
            .reduce(
                || (vec![0.0f64; n_rows], vec![0.0f64; n_rows]),
                |(mut nu, mut de), (nu2, de2)| {
                    for r in 0..n_rows {
                        nu[r] += nu2[r];
                        de[r] += de2[r];
                    }
                    (nu, de)
                },
            );
        numer.copy_from_slice(&par_numer);
        denom.copy_from_slice(&par_denom);
    } else {
        accumulate_block(
            table_matches, target, weights, means, power, loss_fn, numer, denom,
        );
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
        let step_limit = loss_fn.step_limit();
        let eta_limit = loss_fn.eta_limit();
        let new = (old + step.clamp(-step_limit, step_limit)).clamp(-eta_limit, eta_limit);
        factors[t][r] = new;
        numer[r] = new - old; // reuse as the delta to apply to eta
    }

    // Fold the changes into the running linear predictor and mean. Reusing `numer` as
    // the per-row delta keeps this to a single pass with no extra allocation.
    for r in 0..n_rows {
        if row_exposure[r] <= 0.0 || table.is_row_offset(r) || !(denom[r] > 0.0) || !denom[r].is_finite() {
            numer[r] = 0.0;
        }
    }
    apply_row_deltas(loss_fn, table_matches, &numer[..n_rows], offset, eta, means);
}

/// Rows below which splitting the work across threads costs more than it saves.
const PARALLEL_ROWS: usize = 100_000;

/// Chunk size for a parallel pass over the observations, or `None` to stay serial.
///
/// Two things have to hold. The pass must be long enough to cover the cost of handing
/// work to a thread pool at all, which `PARALLEL_ROWS` sets. And where the pass
/// accumulates into per-worker copies of a table's rows — as the scatter-adds below do —
/// those copies have to be cheap relative to the scan itself. A rating table is
/// typically tens of rows against millions of observations, so they are; but a table
/// with a row per postcode against a small dataset would spend more time allocating and
/// reducing partials than reading the data, and that case stays serial.
///
/// `replicated` is the number of `f64`s each worker would need its own copy of, or 0 for
/// a pass that writes only to its own observation's slot.
fn parallel_chunk(n: usize, replicated: usize) -> Option<usize> {
    if n < PARALLEL_ROWS {
        return None;
    }
    let workers = rayon::current_num_threads().max(1);
    if workers < 2 {
        return None;
    }
    if replicated.saturating_mul(workers).saturating_mul(4) > n {
        return None;
    }
    Some((n / workers).max(1))
}

/// One worker's share of the scatter-add behind a step table's update.
///
/// Split out so the serial and parallel paths compute the same sums from the same code;
/// the only difference is what they accumulate into.
#[inline]
fn accumulate_block(
    table_matches: &[u32],
    target: &[f64],
    weights: &[f64],
    means: &[f64],
    power: Option<f64>,
    loss_fn: &LossFunction,
    numer: &mut [f64],
    denom: &mut [f64],
) {
    for (i, m) in table_matches.iter().enumerate() {
        if *m == NO_MATCH {
            continue;
        }
        let r = *m as usize;
        let a = weights[i];
        if a == 0.0 {
            continue;
        }
        let mu = means[i];
        match power {
            // A = sum(a * mu^(1-p) * y), E = sum(a * mu^(2-p))
            Some(p) => {
                let base = pow_special(mu, 1.0 - p);
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
}

/// Folds a table's per-row change into the running linear predictor and the running
/// mean, in one pass.
///
/// `mu` is carried alongside `eta` rather than re-derived from it, because the inverse
/// link is an `exp` for every log-link family and re-deriving costs one call per
/// observation *per table* — nine tables over the French motor data is six million
/// exponentials a sweep. A table's change is constant within a table row, so the whole
/// update needs only `n_rows` transcendental calls:
///
/// * **identity link** — `mu` moves by exactly what `eta` moves by;
/// * **log link** — `mu` scales by `exp(delta_r)`, one `exp` per row;
/// * **logit link** — no such shortcut, so the link is evaluated as before. Odds do
///   scale multiplicatively, but recovering `mu` from them gives up the strict
///   `(0, 1)` bound that [`LossFunction::eta_limit`] exists to guarantee.
///
/// The shortcuts are exact only where the eta clamp does not bind. That is the ordinary
/// case by an enormous margin — `|eta| < 500` admits a rating factor of 1e217 — but a
/// separating fit or a level with no positive response can reach it, so each
/// observation checks both its old and its new position and falls back to a full link
/// evaluation if either is clamped. Checking costs two adds and two comparisons against
/// the `exp` it replaces.
///
/// Where the clamp does not bind, `mu[i] == exp(eta[i] + offset[i])` exactly, by
/// induction: the fast path preserves it and the fallback restores it. Rounding cannot
/// accumulate either, because the caller re-derives `mu` from `eta` at the end of every
/// sweep — at most `n_tables` multiplications ever separate the two.
fn apply_row_deltas(
    loss_fn: &LossFunction,
    table_matches: &[u32],
    delta: &[f64],
    offset: &[f64],
    eta: &mut [f64],
    mu: &mut [f64],
) {
    // The whole point: `n_rows` exponentials, hoisted out of the pass over the data.
    let scale: Option<Vec<f64>> = loss_fn
        .log_link_variance_power()
        .map(|_| delta.iter().map(|d| d.exp()).collect());

    // Every observation writes only to its own slot, so the workers share nothing and
    // there is no reduction to pay for.
    match parallel_chunk(table_matches.len(), 0) {
        Some(chunk) => table_matches
            .par_chunks(chunk)
            .zip(offset.par_chunks(chunk))
            .zip(eta.par_chunks_mut(chunk))
            .zip(mu.par_chunks_mut(chunk))
            .for_each(|(((ms, offs), et), m)| {
                apply_deltas_block(loss_fn, ms, delta, scale.as_deref(), offs, et, m)
            }),
        None => apply_deltas_block(
            loss_fn,
            table_matches,
            delta,
            scale.as_deref(),
            offset,
            eta,
            mu,
        ),
    }
}

/// One worker's share of [`apply_row_deltas`]. `scale` holds `exp(delta_r)` for the
/// log-link families and is `None` otherwise.
#[inline]
fn apply_deltas_block(
    loss_fn: &LossFunction,
    table_matches: &[u32],
    delta: &[f64],
    scale: Option<&[f64]>,
    offset: &[f64],
    eta: &mut [f64],
    mu: &mut [f64],
) {
    match loss_fn {
        // mu is eta, and there is no clamp on the identity link, so this is not an
        // approximation of anything.
        LossFunction::Gaussian => {
            for (i, m) in table_matches.iter().enumerate() {
                if *m != NO_MATCH {
                    let d = delta[*m as usize];
                    eta[i] += d;
                    mu[i] += d;
                }
            }
        }

        LossFunction::Poisson | LossFunction::Gamma | LossFunction::Tweedie(_) => {
            let scale = scale.expect("log-link families always carry exp(delta)");
            let limit = loss_fn.eta_limit();
            for (i, m) in table_matches.iter().enumerate() {
                if *m == NO_MATCH {
                    continue;
                }
                let r = *m as usize;
                let before = eta[i] + offset[i];
                eta[i] += delta[r];
                let after = eta[i] + offset[i];
                if before.abs() < limit && after.abs() < limit {
                    mu[i] *= scale[r];
                } else {
                    mu[i] = loss_fn.inverse_link(after);
                }
            }
        }

        LossFunction::Binary => {
            for (i, m) in table_matches.iter().enumerate() {
                if *m != NO_MATCH {
                    eta[i] += delta[*m as usize];
                    mu[i] = loss_fn.inverse_link(eta[i] + offset[i]);
                }
            }
        }
    }
}

/// Updates a variate table: `degree` parameters for the whole table, however many rows
/// it has.
///
/// The row factors are tied together as `factor[r] = sum_m beta_m * values[r]^m + c`,
/// with `c` absorbed by the intercept, so there are exactly `degree` things to
/// estimate. The table's design is `degree` columns — the powers of the row's value —
/// and one IRLS step solves them jointly:
///
/// ```text
///   A[m][l] = sum a * w * phi_m * phi_l          (degree x degree, symmetric)
///   b[m]    = sum a * phi_m * (w * r)
///   A * d_beta = b
/// ```
///
/// then `factor[r] += sum_m d_beta_m * phi_m(r)` for every row. At degree 1 the matrix
/// is 1x1 and this is the scalar slope step; the two are the same formula.
///
/// Solving the powers *jointly* rather than one at a time matters: `v` and `v^2` are
/// strongly correlated over any range that does not straddle zero, and cycling between
/// them would converge at a crawl.
///
/// # Conditioning
///
/// Two things keep the small solve well behaved, and both are pure
/// reparameterisations — the span of the basis is unchanged, so the fit and its fixed
/// point are identical either way. Only the path there is shorter.
///
/// *Rescaling.* The powers are taken of `u = (v - centre) / half_range`, which lies in
/// `[-1, 1]`, rather than of `v` itself. Age to the fourth is around ten million while
/// age is around forty; without rescaling the normal matrix spans orders of magnitude.
///
/// *Centring.* Each column is then centred on its weight-weighted mean, which makes the
/// step orthogonal to the intercept under the current weights — the exact Newton step
/// for the shape *given that the level is free to adjust*. Without it, a slope column
/// that never crosses zero is nearly collinear with the intercept, and coordinate
/// descent between the two converges linearly and slowly: the deviance goes flat long
/// before the parameters have settled, so the fit reports convergence while the
/// coefficients are still drifting in the sixth decimal place.
///
/// Two further consequences. Rows with no exposure still move, because the curve is
/// estimated from the whole table and every row reads its factor off it — that is the
/// point of a variate. And the log-link closed form does not apply here: it relies on
/// the coefficient entering one level at a time, whereas the design varies across rows.
/// A Newton step on a convex objective converges perfectly well.
fn update_variate_table(
    t: usize,
    factors: &mut [Vec<f64>],
    eta: &mut [f64],
    means: &mut [f64],
    table_matches: &[u32],
    target: &[f64],
    weights: &[f64],
    offset: &[f64],
    values: &[f64],
    degree: usize,
    loss_fn: &LossFunction,
) {
    let n_rows = factors[t].len();
    let d = degree;
    let Some((centre, scale)) = variate_basis_params(values) else {
        return;
    };

    // phi[r][m] = u_r^(m+1), for m in 0..d
    let mut phi = vec![0.0f64; n_rows * d];
    for r in 0..n_rows {
        let u = (values[r] - centre) / scale;
        let mut p = 1.0;
        for m in 0..d {
            p *= u;
            phi[r * d + m] = p;
        }
    }

    // Accumulate the raw sums; centring follows algebraically, so this stays one pass.
    let mut s_w = 0.0f64; // sum of IRLS weights
    let mut s_wphi = vec![0.0f64; d]; // ... times phi_m
    let mut s_wphiphi = vec![0.0f64; d * d]; // ... times phi_m phi_l
    let mut s_r = 0.0f64; // sum of weighted link residuals
    let mut s_rphi = vec![0.0f64; d]; // ... times phi_m

    for (i, m) in table_matches.iter().enumerate() {
        if *m == NO_MATCH {
            continue;
        }
        let r = *m as usize;
        let a = weights[i];
        if a == 0.0 {
            continue;
        }
        let mu = means[i];
        let w = a * loss_fn.irls_weight(mu);
        let res = a * loss_fn.weighted_link_residual(target[i], mu);
        if !w.is_finite() || !res.is_finite() {
            continue;
        }
        let row = &phi[r * d..(r + 1) * d];
        s_w += w;
        s_r += res;
        for m in 0..d {
            s_wphi[m] += w * row[m];
            s_rphi[m] += res * row[m];
            for l in 0..d {
                s_wphiphi[m * d + l] += w * row[m] * row[l];
            }
        }
    }

    if !(s_w > 0.0) || !s_w.is_finite() {
        return;
    }

    // Centred normal equations:
    //   A[m][l] = sum W (phi_m - mean_m)(phi_l - mean_l) = S_mml - S_m S_l / S_w
    //   b[m]    = sum R (phi_m - mean_m)                 = S_rm  - S_r S_m / S_w
    let mut a_mat = vec![0.0f64; d * d];
    let mut b_vec = vec![0.0f64; d];
    for m in 0..d {
        b_vec[m] = s_rphi[m] - s_r * s_wphi[m] / s_w;
        for l in 0..d {
            a_mat[m * d + l] = s_wphiphi[m * d + l] - s_wphi[m] * s_wphi[l] / s_w;
        }
    }

    let Some(step) = solve_spd(&a_mat, &b_vec, d) else {
        return;
    };

    // Change to each row's factor, before clamping. `phi_means` are the basis columns'
    // weighted means, the centring described above — not the fitted means of the model.
    let phi_means: Vec<f64> = (0..d).map(|m| s_wphi[m] / s_w).collect();
    let mut delta = vec![0.0f64; n_rows];
    for r in 0..n_rows {
        let mut change = 0.0;
        for m in 0..d {
            change += step[m] * (phi[r * d + m] - phi_means[m]);
        }
        delta[r] = change;
    }

    // The step limit caps how far a factor may move in one sweep. Scale the whole step
    // down rather than clipping rows individually, which would bend the curve off its
    // polynomial. Infinite under the identity link, where the step is exact.
    let step_limit = loss_fn.step_limit();
    let max_change = delta.iter().fold(0.0f64, |acc, v| acc.max(v.abs()));
    if !max_change.is_finite() {
        return;
    }
    if max_change > step_limit {
        let shrink = step_limit / max_change;
        for v in delta.iter_mut() {
            *v *= shrink;
        }
    }

    let eta_limit = loss_fn.eta_limit();
    for r in 0..n_rows {
        let old = factors[t][r];
        let new = (old + delta[r]).clamp(-eta_limit, eta_limit);
        factors[t][r] = new;
        delta[r] = new - old;
    }

    apply_row_deltas(loss_fn, table_matches, &delta, offset, eta, means);
}

/// Shifts a constant out of each feature table and into the intercept table, leaving
/// every prediction unchanged. See [`Normalization`].
fn normalize(
    factors: &mut [Vec<f64>],
    row_exposure: &[Vec<f64>],
    tables: &[RatingTable],
    updatable: &[bool],
    mode: Normalization,
    eta_limit: f64,
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

    factors[0][0] = (factors[0][0] + shift_into_intercept).clamp(-eta_limit, eta_limit);
}

/// Scatters one observation's score contribution into every table it touches, and
/// returns its absolute value for the scaling denominator.
///
/// The contribution is the same quantity the IRLS step already uses, `a * w * r`, which
/// for the log-link families is `a * mu^(1-p) * (y - mu)` — the `A - E` of the exact
/// update, before the log.
#[inline]
#[allow(clippy::too_many_arguments)]
fn score_row(
    i: usize,
    loss_fn: &LossFunction,
    target: &[f64],
    weights: &[f64],
    means: &[f64],
    matches: &[Vec<u32>],
    updatable: &[bool],
    rows: &mut [Vec<f64>],
) -> f64 {
    let a = weights[i];
    if a == 0.0 {
        return 0.0;
    }
    let s = a * loss_fn.weighted_link_residual(target[i], means[i]);
    for (t, table_matches) in matches.iter().enumerate() {
        if !updatable[t] {
            continue;
        }
        let m = table_matches[i];
        if m != NO_MATCH {
            rows[t][m as usize] += s;
        }
    }
    s.abs()
}

/// Relative improvement in the score that counts as progress.
const STALL_IMPROVEMENT: f64 = 1e-3;

/// Sweeps without progress before giving up. Backfitting can converge genuinely
/// slowly, so this has to be loose enough not to cut off a fit that is still working.
const STALL_SWEEPS: usize = 12;

/// The largest absolute score component over every free parameter, scaled by the total
/// prior weight.
///
/// This is the convergence test, and it is a direct measure of what the caller
/// receives: at the optimum every free parameter's score is zero, so `max |g|` says how
/// far the *parameters* still have to move. The deviance change does not — deviance is
/// quadratic in the parameter error near the optimum, so a deviance tolerance of `1e-t`
/// buys only about `1e-(t/2)` in the fitted factors, and a fit can report convergence
/// while still visibly wrong. That is not a hypothetical: on the French motor data the
/// deviance test declared victory 1.1e-04 away from the answer.
///
/// The scaling by total weight matches what glum reports, so the tolerances mean
/// roughly the same thing in both. Without it the threshold would depend on the number
/// of observations.
///
/// `scratch` is per-table row storage, reused across sweeps.
#[allow(clippy::too_many_arguments)]
fn max_abs_score(
    loss_fn: &LossFunction,
    target: &[f64],
    weights: &[f64],
    means: &[f64],
    matches: &[Vec<u32>],
    tables: &[RatingTable],
    updatable: &[bool],
    row_exposure: &[Vec<f64>],
    variate_values: &[Option<(Vec<f64>, usize)>],
    scratch: &mut [Vec<f64>],
) -> f64 {
    for rows in scratch.iter_mut() {
        rows.iter_mut().for_each(|v| *v = 0.0);
    }

    // This pass costs what the whole update path costs — one scatter-add per table per
    // observation — so it gets the same treatment. Each worker fills its own copy of
    // every table's rows, which is why `parallel_chunk` is asked about the total width
    // of the model rather than one table's.
    let width: usize = scratch.iter().map(|s| s.len()).sum();
    let total_abs = match parallel_chunk(target.len(), width) {
        Some(chunk) => {
            let shape: Vec<usize> = scratch.iter().map(|s| s.len()).collect();
            let (partial, total_abs) = target
                .par_chunks(chunk)
                .enumerate()
                .fold(
                    || (shape.iter().map(|k| vec![0.0f64; *k]).collect::<Vec<_>>(), 0.0f64),
                    |(mut rows, mut abs), (c, block)| {
                        // `matches` is indexed by table first, so the pass needs the
                        // absolute observation index rather than a chunk-local one.
                        let start = c * chunk;
                        for k in 0..block.len() {
                            abs += score_row(
                                start + k,
                                loss_fn,
                                target,
                                weights,
                                means,
                                matches,
                                updatable,
                                &mut rows,
                            );
                        }
                        (rows, abs)
                    },
                )
                .reduce(
                    || (shape.iter().map(|k| vec![0.0f64; *k]).collect::<Vec<_>>(), 0.0f64),
                    |(mut rows, abs), (rows2, abs2)| {
                        for (a, b) in rows.iter_mut().zip(rows2.iter()) {
                            for (x, y) in a.iter_mut().zip(b.iter()) {
                                *x += y;
                            }
                        }
                        (rows, abs + abs2)
                    },
                );
            for (dst, src) in scratch.iter_mut().zip(partial.iter()) {
                dst.copy_from_slice(src);
            }
            total_abs
        }
        None => {
            let mut total_abs = 0.0f64;
            for i in 0..target.len() {
                total_abs +=
                    score_row(i, loss_fn, target, weights, means, matches, updatable, scratch);
            }
            total_abs
        }
    };

    let mut worst = 0.0f64;
    for t in 0..scratch.len() {
        if !updatable[t] {
            continue;
        }

        match &variate_values[t] {
            // A variate table's free parameters are its polynomial coefficients, not
            // its rows, so the row scores have to be projected onto the basis the fit
            // actually moves along.
            Some((values, degree)) => {
                let Some((centre, scale)) = variate_basis_params(values) else {
                    continue;
                };
                for m in 1..=*degree {
                    let mut g = 0.0;
                    for (r, v) in values.iter().enumerate() {
                        let u = (v - centre) / scale;
                        g += scratch[t][r] * u.powi(m as i32);
                    }
                    worst = worst.max(g.abs());
                }
            }
            None => {
                for r in 0..scratch[t].len() {
                    // A locked row or one with no exposure carries no free parameter,
                    // so its score is not ours to drive to zero.
                    if row_exposure[t][r] <= 0.0 || tables[t].is_row_offset(r) {
                        continue;
                    }
                    worst = worst.max(scratch[t][r].abs());
                }
            }
        }
    }

    // Scale by the total absolute residual, not by the weight. Both make the threshold
    // independent of the number of observations, but only this one makes it
    // independent of the units the response is measured in: the score carries the
    // response's scale, and dividing by a bare weight leaves it there. A Gaussian fit
    // on currency would otherwise need a different tolerance from one on log-odds.
    //
    // Read it as: the fraction of the residual signal still concentrated in the worst
    // single parameter. At the optimum the signed residuals cancel within every level,
    // so this goes to zero while the denominator stays put.
    if total_abs > 0.0 {
        worst / total_abs
    } else {
        worst
    }
}

/// Deviance of the best intercept-only fit, used as the reference for `pseudo_r2`.
fn null_deviance(loss_fn: &LossFunction, target: &[f64], weights: &[f64], offset: &[f64]) -> f64 {
    let mut beta = 0.0f64;
    let power = loss_fn.log_link_variance_power();
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
            match power {
                Some(p) => {
                    let base = pow_special(mu, 1.0 - p);
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
        let step = match power {
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
        let step_limit = loss_fn.step_limit();
        let eta_limit = loss_fn.eta_limit();
        let next = (beta + step.clamp(-step_limit, step_limit)).clamp(-eta_limit, eta_limit);
        if (next - beta).abs() < 1e-14 * beta.abs().max(1.0) {
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
    matches: &[Vec<u32>],
    n_rows: usize,
) -> Result<(), PolarsError> {
    for (t, table_matches) in matches.iter().enumerate() {
        let unmatched = table_matches.iter().filter(|m| **m == NO_MATCH).count();
        if unmatched == 0 {
            continue;
        }
        let first = table_matches.iter().position(|m| *m == NO_MATCH).unwrap();
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

    // A variate table's factors all come from one slope, so a pinned row is not
    // representable. (as_variate rejects this too; a row could be locked afterwards.)
    for (t, table) in model.tables.iter().enumerate() {
        if table.variate_values().is_none() {
            continue;
        }
        if let Some(r) = (0..table.data.height()).find(|r| table.is_row_offset(*r)) {
            return Err(PolarsError::ComputeError(
                format!(
                    "Table {} is a variate but row {} is locked. Every factor is derived from \
                     the one slope, so pinning a single row would break the line. Lock the \
                     whole table with as_offset() instead.",
                    t, r
                )
                .into(),
            ));
        }
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

#[cfg(test)]
mod tests {
    use super::*;

    /// `apply_row_deltas` carries `mu` forward as `mu *= exp(delta)` instead of
    /// re-deriving it from `eta`, which is only valid while the eta clamp does not bind.
    /// Whichever route it takes for a given observation, it must leave the invariant
    /// `mu[i] == inverse_link(eta[i] + offset[i])` intact.
    ///
    /// This is checked here rather than through a fit because the clamp is not reachable
    /// end-to-end without wrecking the rest of the model: an offset large enough to
    /// clamp needs a response near `exp(500)` to stay representable, and a response that
    /// size swamps the convergence test's residual scaling, so every other parameter is
    /// declared converged before it has moved. The invariant is the thing that matters,
    /// and it can be checked directly.
    #[test]
    fn apply_row_deltas_keeps_mu_consistent_with_eta() {
        for loss_fn in [
            LossFunction::Gaussian,
            LossFunction::Poisson,
            LossFunction::Gamma,
            LossFunction::Tweedie(1.5),
            LossFunction::Binary,
        ] {
            // Chosen to exercise every branch of the guard. Reading `eta + offset`
            // before and after the delta, row by row: 1 -> 9 (both inside the clamp,
            // fast path); 493 -> 485 (inside); -494.5 -> -502.5 (leaves the clamp);
            // 505 -> 497 (*enters* from outside, the case where `mu` was pinned at
            // exp(500) and scaling it would be wrong); -490 -> -482 (inside); and one
            // observation this table does not match at all.
            let offset = vec![0.0, 495.0, -495.0, 600.0, -600.0, 3.0];
            let table_matches = vec![0u32, 1, 1, 1, 0, NO_MATCH];
            let delta = vec![8.0, -8.0];
            let mut eta = vec![1.0, -2.0, 0.5, -95.0, 110.0, 2.0];

            let mut mu: Vec<f64> = eta
                .iter()
                .zip(offset.iter())
                .map(|(e, o)| loss_fn.inverse_link(e + o))
                .collect();

            apply_row_deltas(&loss_fn, &table_matches, &delta, &offset, &mut eta, &mut mu);

            for i in 0..eta.len() {
                let expected = loss_fn.inverse_link(eta[i] + offset[i]);
                let tol = 1e-12 * expected.abs().max(f64::MIN_POSITIVE);
                assert!(
                    (mu[i] - expected).abs() <= tol,
                    "{:?} row {}: carried mu = {:e}, but inverse_link(eta + offset) = {:e}",
                    loss_fn, i, mu[i], expected
                );
            }
        }
    }
}
