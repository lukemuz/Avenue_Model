use super::inference::{compute_inference, solve_spd, GLMInference};
use super::loss::{pow_special, LossFunction, MAX_STEP};
use super::matching::{precompute_all_matches, NO_MATCH};
use super::penalty::{soft_threshold, PenaltyPlan, TablePenalty, ANCHOR_ROW};
use super::redundancy::{collective_strength, table_correlations, TablePair};
use crate::rating_model::{variate_basis_params, RatingModel, RatingTable, TableSemantics};
use polars::prelude::*;
use rayon::prelude::*;

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

/// Which numerical strategy fits the GLM.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GLMSolver {
    /// Prefer the global solver and fall back to table descent when the model is not
    /// supported by the global path. This is the default.
    Auto,
    /// Avenue's low-memory block coordinate descent over rating tables.
    Table,
    /// Global IRLS over a compact treatment-coded Gram matrix.
    Global,
}

impl Default for GLMSolver {
    fn default() -> Self {
        Self::Auto
    }
}

/// Options for GLM fitting
#[derive(Debug, Clone)]
pub struct GLMOptions {
    pub solver: GLMSolver,
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
    /// Accelerate the sweep with SQUAREM extrapolation. See [`squarem_steplength`].
    ///
    /// Costs three parameter vectors of memory and pays for itself many times over on
    /// models with correlated tables. Safeguarded, so the worst case is a few wasted
    /// passes; turn it off only to reproduce the unaccelerated sequence exactly.
    pub accelerate: bool,
    /// Measure how much information each pair of tables shares, and use it.
    ///
    /// Two things are built from that one measurement: near-aliased pairs are updated as
    /// one block rather than one at a time, and the sweep visits the most strongly
    /// coupled table first (see [`sweep_order`]). Turning this off disables both and
    /// leaves the sweep in table order.
    ///
    /// Costs one pass over the data to measure how much information each pair of tables
    /// shares, and thereafter one extra scatter-add per observation for each pair that
    /// qualifies. Worth it whenever a plan carries two tables describing the same thing —
    /// a density band and an area code, an age band and a birth-year band — which
    /// otherwise converges at a crawl. See [`update_pair`].
    pub solve_aliased_pairs_jointly: bool,
    /// Penalty strength. Zero - the default - fits the unpenalised model, on exactly the
    /// code path it always did.
    ///
    /// Scaled to mean the same thing as glum's `alpha`: the objective is
    /// `D / (2 * sum(w)) + alpha * (l1_ratio * |b|_1 + (1 - l1_ratio)/2 * |b|^2)`, where
    /// `b` runs over each table's levels **measured against its base level**, and the
    /// intercept is excluded. See [`crate::glm::penalty`] for why the base level rather
    /// than zero, and what follows from it.
    ///
    /// Requires [`Normalization::BaseLevel`], which is the default.
    pub alpha: f64,
    /// The share of `alpha` spent on the L1 term: 0 is a pure ridge, 1 a pure lasso,
    /// anything between an elastic net.
    ///
    /// With the table solver, a soft threshold replaces a division in a step already
    /// taken per level, but L1 disables the near-aliased pair solve because that solve
    /// has no proximal form. The global solver instead performs coordinate descent on
    /// its Gram matrix.
    pub l1_ratio: f64,
}

impl Default for GLMOptions {
    fn default() -> Self {
        Self {
            solver: GLMSolver::default(),
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
            accelerate: true,
            solve_aliased_pairs_jointly: true,
            alpha: 0.0,
            l1_ratio: 0.0,
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
    /// How strongly the tables share a single common direction: 1.0 when they are
    /// orthogonal, rising to the number of tables when they all carry the same
    /// information.
    ///
    /// This is what sets the sweep count on a correlated plan, and no pairwise
    /// correlation substitutes for it - a hundred tables pairwise-correlated at only
    /// 0.28 score 28.5 here and need about a thousand sweeps. Above roughly 10 expect
    /// hundreds of sweeps; above 25, thousands. `None` when
    /// [`GLMOptions::solve_aliased_pairs_jointly`] is off, since the pairwise
    /// correlations it is built from are not computed.
    pub table_conditioning: Option<f64>,
    /// Extrapolation steps that were accepted, out of the cycles attempted.
    ///
    /// Zero on a fit that converges before the accelerator gets a chance, and zero when
    /// [`GLMOptions::accelerate`] is off. A large count next to a small
    /// [`iterations`](Self::iterations) is the accelerator earning its keep.
    pub accelerated_steps: usize,
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

/// Everything a sweep reads but never writes, gathered so the sweep and the accelerator
/// can both be expressed as ordinary calls rather than fifteen-argument functions.
struct FitContext<'a> {
    loss_fn: &'a LossFunction,
    target: &'a [f64],
    weights: &'a [f64],
    offset: &'a [f64],
    matches: &'a [Vec<u32>],
    tables: &'a [RatingTable],
    updatable: &'a [bool],
    row_exposure: &'a [Vec<f64>],
    variate_values: &'a [Option<(Vec<f64>, usize)>],
    normalization: Normalization,
    /// Near-aliased table pairs to update as one block, primary first. Disjoint: a table
    /// appears in at most one pair, so every table is still updated exactly once a sweep.
    joint_pairs: &'a [(usize, usize)],
    /// The order tables are visited in, from [`sweep_order`]. A permutation of every
    /// table index, so the sweep still visits each exactly once.
    order: &'a [usize],
    /// The penalty on the factors, or `None` for an unpenalised fit. `None` keeps every
    /// update on the exact code path it took before penalties existed.
    penalty: Option<&'a PenaltyPlan>,
}

impl<'a> FitContext<'a> {
    /// One pass over every updatable table, followed by the anchoring step.
    ///
    /// This is the fixed-point map the accelerator extrapolates on: it takes the whole
    /// parameter vector to its next iterate, and the fit is the sequence of its powers.
    fn sweep(
        &self,
        factors: &mut [Vec<f64>],
        eta: &mut [f64],
        means: &mut [f64],
        numer: &mut Vec<f64>,
        denom: &mut Vec<f64>,
    ) {
        let mut paired = vec![false; self.tables.len()];
        for (t, u) in self.joint_pairs {
            update_pair(
                *t,
                *u,
                factors,
                eta,
                means,
                self.matches,
                self.target,
                self.weights,
                self.offset,
                self.row_exposure,
                self.tables,
                self.loss_fn,
                self.penalty,
            );
            paired[*t] = true;
            paired[*u] = true;
        }

        for &t in self.order {
            if !self.updatable[t] || paired[t] {
                continue;
            }
            update_table(
                t,
                factors,
                eta,
                means,
                &self.matches[t],
                self.target,
                self.weights,
                self.offset,
                &self.row_exposure[t],
                &self.tables[t],
                self.loss_fn,
                self.penalty,
                numer,
                denom,
            );
        }

        if self.normalization != Normalization::None {
            normalize(
                factors,
                self.row_exposure,
                self.tables,
                self.updatable,
                self.normalization,
                self.loss_fn.eta_limit(),
            );
        }

        // Re-derive the means exactly. `apply_row_deltas` has been carrying them forward
        // multiplicatively through the sweep, which is worth one pass to reset: it costs
        // what a single table update used to, and it bounds the rounding those
        // increments can accumulate to a single sweep's worth.
        self.relink(eta, means);
    }

    /// Rebuilds `eta` from the factors and `means` from `eta`.
    ///
    /// The sweep maintains both incrementally, so this is only needed where the factors
    /// are written directly rather than reached by a sweep — which is exactly what the
    /// accelerator does.
    fn refresh(&self, factors: &[Vec<f64>], eta: &mut [f64], means: &mut [f64]) {
        eta.iter_mut().for_each(|v| *v = 0.0);
        for (t, table_matches) in self.matches.iter().enumerate() {
            for (i, m) in table_matches.iter().enumerate() {
                if *m != NO_MATCH {
                    eta[i] += factors[t][*m as usize];
                }
            }
        }
        self.relink(eta, means);
    }

    fn relink(&self, eta: &[f64], means: &mut [f64]) {
        for i in 0..eta.len() {
            means[i] = self.loss_fn.inverse_link(eta[i] + self.offset[i]);
        }
    }

    fn deviance(&self, means: &[f64]) -> f64 {
        self.loss_fn
            .total_deviance(self.target, means, self.weights)
    }

    /// What the fit actually minimises: the deviance plus the penalty, on the deviance
    /// scale.
    ///
    /// The stall rule and the SQUAREM acceptance test both ask whether the objective
    /// went down, and once a penalty is on, the objective is not the deviance. The
    /// deviance is still what gets reported, because it is a goodness-of-fit statistic
    /// measured against `null_deviance` and folding the penalty into it would quietly
    /// break `pseudo_r2`.
    fn objective(&self, factors: &[Vec<f64>], means: &[f64]) -> f64 {
        self.deviance(means) + self.penalty.map_or(0.0, |p| p.total(factors))
    }

    fn max_score(&self, factors: &[Vec<f64>], means: &[f64], scratch: &mut [Vec<f64>]) -> f64 {
        max_abs_score(
            self.loss_fn,
            self.target,
            self.weights,
            means,
            self.matches,
            self.tables,
            self.updatable,
            self.row_exposure,
            self.variate_values,
            factors,
            self.penalty,
            scratch,
        )
    }
}

/// How many times the extrapolated point may be pulled back toward plain iteration
/// before the cycle gives up on it.
///
/// Each attempt costs a rebuild of `eta` and a deviance evaluation — cheaper than a
/// sweep, but not by much, so a cycle that backtracks the whole way is most of the work
/// of the sweeps it was trying to save. That is affordable when jumps are landing and
/// pure waste when they are not, which is what [`SQUAREM_MAX_BACKOFF`] is for.
const SQUAREM_BACKTRACKS: usize = 6;

/// Slack on the deviance comparison that decides whether a jump is accepted.
///
/// Deviance goes flat to machine precision long before the parameters have settled —
/// that is the entire reason the convergence test looks at the score instead. Without
/// slack, every late cycle would reject its jump on rounding noise alone, which is
/// precisely when the accelerator is most needed.
const SQUAREM_DEVIANCE_SLACK: f64 = 1e-12;

/// Largest power-of-two backoff after consecutive failed cycles.
///
/// Some models simply do not have a single dominant mode for the extrapolation to catch,
/// and on those every cycle pays for backtracking it will never recoup. Doubling the gap
/// between attempts after each failure bounds that waste to a vanishing share of the fit
/// while still letting the accelerator back in if the problem's character changes — which
/// it does, since the modes that dominate early are rarely the ones that dominate late.
const SQUAREM_MAX_BACKOFF: u32 = 5;

/// The SQUAREM steplength from three successive iterates, or `None` where extrapolation
/// is not meaningful.
///
/// Backfitting converges linearly: the error decays as `rho^k` along a dominant mode set
/// by the canonical correlation between the tables. Two tables carrying nearly the same
/// information — a density band and an area code that rebands it — push `rho` toward 1,
/// and the sweep spends hundreds of passes walking down that one direction. Measured on
/// the French motor data, `rho = 0.943`: one decade of accuracy per 39 sweeps.
///
/// Three iterates are enough to identify that mode. Writing `r = t1 - t0` and
/// `v = t2 - 2*t1 + t0`, the steplength
///
/// ```text
///   alpha = -||r|| / ||v||
/// ```
///
/// applied as `t' = t0 - 2*alpha*r + alpha^2*v` lands *exactly* on the fixed point when
/// the error really is a single geometric mode. Substituting `t_k - t* = rho^k e`:
/// `r = (rho - 1) e` and `v = (rho - 1)^2 e`, so `alpha = -1/(1 - rho)` and the
/// correction collapses to `t0 - e`. At `rho = 0.943` that is a jump of about 17 sweeps'
/// worth in one step.
///
/// Real problems carry more than one mode, so the jump overshoots and the caller has to
/// safeguard it. The parameterisation makes that easy: `alpha = -1` reproduces `t2`
/// exactly, so pulling `alpha` toward `-1` interpolates smoothly between the full
/// extrapolation and doing nothing at all.
///
/// Reference: Varadhan & Roland (2008), *Simple and globally convergent methods for
/// accelerating the convergence of any EM algorithm*, Scandinavian Journal of
/// Statistics 35(2). This is their SqS3 steplength.
fn squarem_steplength(t0: &[f64], t1: &[f64], t2: &[f64]) -> Option<f64> {
    let mut r_sq = 0.0f64;
    let mut v_sq = 0.0f64;
    for i in 0..t0.len() {
        let r = t1[i] - t0[i];
        let v = t2[i] - 2.0 * t1[i] + t0[i];
        r_sq += r * r;
        v_sq += v * v;
    }
    if !(r_sq > 0.0) || !(v_sq > 0.0) || !r_sq.is_finite() || !v_sq.is_finite() {
        return None;
    }
    let alpha = -(r_sq / v_sq).sqrt();
    // `alpha` above -1 would be an *under*-relaxation of a sequence that is already
    // converging monotonically; there is nothing to gain and the safeguard would only
    // undo it. Curvature that small means the iterates are nearly collinear, which is
    // the converged case.
    if !alpha.is_finite() || alpha > -1.0 {
        return None;
    }
    Some(alpha)
}

/// Builds the extrapolated parameter vector `t0 - 2*alpha*r + alpha^2*v`, clamped so a
/// large jump cannot put a factor somewhere the link cannot evaluate.
fn squarem_extrapolate(
    t0: &[f64],
    t1: &[f64],
    t2: &[f64],
    alpha: f64,
    eta_limit: f64,
    out: &mut Vec<f64>,
) {
    out.clear();
    out.reserve(t0.len());
    for i in 0..t0.len() {
        let r = t1[i] - t0[i];
        let v = t2[i] - 2.0 * t1[i] + t0[i];
        let value = t0[i] - 2.0 * alpha * r + alpha * alpha * v;
        out.push(value.clamp(-eta_limit, eta_limit));
    }
}

fn flatten_factors(factors: &[Vec<f64>], out: &mut Vec<f64>) {
    out.clear();
    for f in factors {
        out.extend_from_slice(f);
    }
}

fn unflatten_factors(flat: &[f64], factors: &mut [Vec<f64>]) {
    let mut k = 0;
    for f in factors.iter_mut() {
        let len = f.len();
        f.copy_from_slice(&flat[k..k + len]);
        k += len;
    }
}

/// What the convergence bookkeeping decided after a sweep.
enum Status {
    Continue,
    Stop,
}

/// Convergence bookkeeping, kept together so a sweep can be recorded from any of the
/// three places a SQUAREM cycle performs one.
struct Progress {
    iterations: usize,
    converged: bool,
    max_gradient: f64,
    deviance: f64,
    sweeps_without_progress: usize,
    deviance_history: Vec<f64>,
    gradient_history: Vec<f64>,
    /// What the fit is actually minimising: the deviance plus the penalty. Equal to
    /// `deviance_history` for an unpenalised fit, and the series the stall rule reads,
    /// because a penalised fit can trade deviance for penalty and still be improving.
    objective_history: Vec<f64>,
}

impl Progress {
    fn record(
        &mut self,
        objective: f64,
        deviance: f64,
        gradient: f64,
        options: &GLMOptions,
    ) -> Status {
        self.iterations += 1;
        self.deviance = deviance;
        self.max_gradient = gradient;
        self.deviance_history.push(deviance);
        self.objective_history.push(objective);
        self.gradient_history.push(gradient);

        if options.verbose {
            println!(
                "Iteration {}: deviance = {:.10e}, max |score| = {:.3e}",
                self.iterations, deviance, gradient
            );
        }

        if gradient <= options.tolerance {
            self.converged = true;
            if options.verbose {
                println!("Converged after {} iterations", self.iterations);
            }
            return Status::Stop;
        }

        // A fit can run out of reachable precision above the tolerance - two near-aliased
        // tables trading a constant back and forth, or a threshold set below the noise
        // floor of the sums involved. Continuing cannot help, so stop.
        //
        // The test is on the deviance rather than on the score, because the score
        // *oscillates*: the error rotates as it decays, so the score can climb for many
        // sweeps while the fit improves throughout. On a hundred correlated tables it
        // bottoms at 7.2e-04 on sweep 34, is climbing again by sweep 46, and reaches the
        // tolerance on sweep 1119 - so any patience window shorter than that period
        // abandons a converging fit, and the period is a property of the data rather
        // than something that can be tuned for. Judging progress on the deviance has no
        // period to be shorter than.
        //
        // Convergence is still decided by the score, above; the deviance is far too weak
        // for that, being quadratic in the parameter error. This asks only whether there
        // is anything left to gain, and for that the two regimes are nine orders apart:
        // on the fit above the smallest real per-sweep improvement is 4.8e-06, while a
        // fit sitting on its rounding floor moves by about 5e-15.
        if let Some(&previous) = self.objective_history.iter().rev().nth(1) {
            let scale = previous.abs().max(f64::MIN_POSITIVE);
            if (previous - objective) / scale < DEVIANCE_STALL {
                self.sweeps_without_progress += 1;
                if self.sweeps_without_progress >= STALL_SWEEPS {
                    if options.verbose {
                        println!(
                            "Stopping after {} iterations: the deviance has not improved \
                             in {} sweeps and max |score| = {:.3e} is above the tolerance \
                             of {:.1e}",
                            self.iterations, STALL_SWEEPS, gradient, options.tolerance
                        );
                    }
                    return Status::Stop;
                }
            } else {
                self.sweeps_without_progress = 0;
            }
        }

        if self.iterations >= options.max_iterations {
            return Status::Stop;
        }
        Status::Continue
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
    fit_glm_with_diagnostics(model, df, target_col, weight_col, offset_col, options).map(|(m, _)| m)
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

    // A penalty is defined on each level's contrast against its table's base level,
    // which only means anything if something pins the base level down. `BaseLevel` is
    // what does that, and it is the default; under the other two modes the penalty and
    // the anchoring would be minimising different objectives every sweep. Refusing is
    // better than fitting something nobody could interpret - see [`crate::glm::penalty`].
    if options.alpha > 0.0 && options.normalization != Normalization::BaseLevel {
        return Err(PolarsError::ComputeError(
            format!(
                "A penalty (alpha = {}) requires Normalization::BaseLevel, but this fit                  asked for {:?}. The penalty shrinks every level toward its table's base                  level, and the other modes do not hold a base level still, so the two                  would pull against each other every sweep.",
                options.alpha, options.normalization
            )
            .into(),
        ));
    }
    if options.alpha < 0.0 || !options.alpha.is_finite() {
        return Err(PolarsError::ComputeError(
            format!(
                "alpha must be finite and not negative, got {}",
                options.alpha
            )
            .into(),
        ));
    }
    if !(0.0..=1.0).contains(&options.l1_ratio) {
        return Err(PolarsError::ComputeError(
            format!(
                "l1_ratio must be between 0 (ridge) and 1 (lasso), got {}",
                options.l1_ratio
            )
            .into(),
        ));
    }

    let n = df.height();
    let target = read_f64_column(df, target_col, "target")?;
    let weights = match weight_col {
        Some(col) => {
            let w = read_f64_column(df, col, "weight")?;
            if let Some(bad) = w.iter().position(|v| *v < 0.0) {
                return Err(PolarsError::ComputeError(
                    format!(
                        "Weight column '{}' has a negative value at row {}",
                        col, bad
                    )
                    .into(),
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

    // Reused by the convergence test; one slot per table row.
    let mut score_scratch: Vec<Vec<f64>> = factors.iter().map(|f| vec![0.0; f.len()]).collect();

    // Which tables carry so much of the same information that updating them one at a
    // time would crawl. Costs one pass over the data against the hundreds of sweeps it
    // saves, and the correlations are reported either way — a plan whose two geography
    // tables are the same geography is worth telling someone about.
    let table_shapes: Vec<usize> = factors.iter().map(|f| f.len()).collect();

    // The penalty, or `None` when `alpha` is zero. Built once: it is a pair of scalars
    // and a flag per table, and it costs nothing per sweep beyond the arithmetic in
    // [`next_factor`].
    let is_variate: Vec<bool> = variate_values.iter().map(|v| v.is_some()).collect();
    let penalty = PenaltyPlan::new(
        options.alpha,
        options.l1_ratio,
        weights.iter().sum::<f64>(),
        &table_shapes,
        &is_variate,
    );

    let global_supported = options.normalization == Normalization::BaseLevel
        && !factors.is_empty()
        && factors[0].len() == 1
        && updatable[0]
        && factors.iter().enumerate().all(|(t, rows)| {
            !updatable[t]
                || (variate_values[t].is_none()
                    && !(0..rows.len()).any(|r| working_model.tables[t].is_row_offset(r)))
        })
        && 1 + factors
            .iter()
            .enumerate()
            .skip(1)
            .map(|(t, rows)| {
                if updatable[t] {
                    (1..rows.len())
                        .filter(|&r| row_exposure[t][r] > 0.0)
                        .count()
                } else {
                    0
                }
            })
            .sum::<usize>()
            <= 6000;

    if options.solver == GLMSolver::Global
        || (options.solver == GLMSolver::Auto && global_supported)
    {
        return fit_global_irls(
            working_model,
            loss_fn,
            target,
            weights,
            offset,
            matches,
            factors,
            updatable,
            variate_values,
            row_exposure,
            penalty,
            null_deviance,
            options,
        );
    }

    let pairable: Vec<bool> = (0..n_tables)
        .map(|t| updatable[t] && variate_values[t].is_none())
        .collect();
    let correlations = if options.solve_aliased_pairs_jointly {
        table_correlations(&matches, &weights, &table_shapes, &pairable)
    } else {
        Vec::new()
    };
    // The joint solve for a near-aliased pair is a Newton step on a dense block, and a
    // Newton step has no proximal form - there is no way to soft-threshold two coupled
    // coordinates at once and land on the right pair of zeros. Under an L1 penalty the
    // pair falls back to being swept one table at a time, which is correct but gives up
    // the acceleration; a pure ridge keeps it, because ridge is exact on the diagonal.
    let joint_pairs = if penalty.as_ref().map_or(false, |p| p.selects()) {
        if options.verbose && !correlations.is_empty() {
            println!(
                "L1 penalty active: near-aliased pairs will be swept separately rather \
                 than solved as one block"
            );
        }
        Vec::new()
    } else {
        choose_joint_pairs(&correlations, &table_shapes)
    };
    let order = sweep_order(&correlations, n_tables);

    // Free, given the correlations: it is an eigenvalue of a matrix the size of the
    // table count. Reported whether or not any pair crossed the joint-solve threshold,
    // because the case it warns about is precisely the one where none of them does - a
    // plan whose tables are individually only mildly correlated but collectively cover
    // one shared direction converges at a crawl with nothing for the pair solve to find.
    let table_conditioning = if options.solve_aliased_pairs_jointly {
        Some(collective_strength(&correlations))
    } else {
        None
    };

    if options.verbose {
        if let Some(strength) = table_conditioning {
            println!(
                "Tables share a common direction at {:.2} of a possible {}{}",
                strength,
                table_shapes.iter().filter(|k| **k > 1).count(),
                if strength > 10.0 {
                    " - expect this fit to need hundreds of sweeps"
                } else {
                    ""
                }
            );
        }
        for pair in correlations.iter().filter(|p| p.is_near_aliased()) {
            println!(
                "Tables {} and {} share {:.1}% of their information (first canonical \
                 correlation {:.4}); they will be updated as one block",
                pair.first,
                pair.second,
                100.0 * pair.correlation,
                pair.correlation
            );
        }
    }

    let ctx = FitContext {
        loss_fn: &loss_fn,
        target: &target,
        weights: &weights,
        offset: &offset,
        matches: &matches,
        tables: &working_model.tables,
        updatable: &updatable,
        row_exposure: &row_exposure,
        variate_values: &variate_values,
        normalization: options.normalization,
        joint_pairs: &joint_pairs,
        order: &order,
        penalty: penalty.as_ref(),
    };

    let mut progress = Progress {
        iterations: 0,
        converged: false,
        max_gradient: f64::INFINITY,
        deviance: f64::INFINITY,
        sweeps_without_progress: 0,
        deviance_history: Vec::with_capacity(options.max_iterations),
        gradient_history: Vec::with_capacity(options.max_iterations),
        objective_history: Vec::with_capacity(options.max_iterations),
    };
    let mut accelerated_steps = 0usize;

    // SQUAREM works on the parameter vector as a whole, so the factors are flattened in
    // and out of these. Three of them, plus the candidate: `p` doubles each, against the
    // `n` the data already occupies.
    let mut theta0: Vec<f64> = Vec::new();
    let mut theta1: Vec<f64> = Vec::new();
    let mut theta2: Vec<f64> = Vec::new();
    let mut candidate: Vec<f64> = Vec::new();
    let eta_limit = loss_fn.eta_limit();
    let mut consecutive_failures: u32 = 0;
    let mut cycles_to_skip: usize = 0;

    'fitting: loop {
        // ---------------------------------------------------------- two plain sweeps
        if options.accelerate {
            flatten_factors(&factors, &mut theta0);
        }
        ctx.sweep(&mut factors, &mut eta, &mut means, &mut numer, &mut denom);
        if let Status::Stop = progress.record(
            ctx.objective(&factors, &means),
            ctx.deviance(&means),
            ctx.max_score(&factors, &means, &mut score_scratch),
            &options,
        ) {
            break 'fitting;
        }

        if !options.accelerate {
            continue;
        }

        flatten_factors(&factors, &mut theta1);
        ctx.sweep(&mut factors, &mut eta, &mut means, &mut numer, &mut denom);
        let deviance2 = ctx.deviance(&means);
        // Judged on the penalised objective, which is what the sweep is descending.
        // Equal to the deviance whenever no penalty is on.
        let objective2 = ctx.objective(&factors, &means);
        // What plain iteration achieved. An extrapolated jump has to beat this on the
        // same quantity the fit converges on, or it is not worth taking.
        let score2 = ctx.max_score(&factors, &means, &mut score_scratch);
        if let Status::Stop = progress.record(objective2, deviance2, score2, &options) {
            break 'fitting;
        }
        flatten_factors(&factors, &mut theta2);

        // ------------------------------------------------------------- extrapolation
        if cycles_to_skip > 0 {
            cycles_to_skip -= 1;
            continue;
        }

        let Some(mut alpha) = squarem_steplength(&theta0, &theta1, &theta2) else {
            continue;
        };
        let accept_at = objective2 + objective2.abs() * SQUAREM_DEVIANCE_SLACK;

        // Pull the jump back toward plain iteration until it is at least not worse than
        // the point it started from. Each attempt costs a rebuild and a deviance, not a
        // sweep. `alpha = -1` reproduces theta2 exactly, so this always terminates
        // somewhere useful.
        let mut landed = false;
        for _ in 0..SQUAREM_BACKTRACKS {
            squarem_extrapolate(&theta0, &theta1, &theta2, alpha, eta_limit, &mut candidate);
            unflatten_factors(&candidate, &mut factors);
            ctx.refresh(&factors, &mut eta, &mut means);
            // The extrapolated point is not normalised, which is exactly why the penalty
            // reads its anchor out of the factors rather than assuming it is zero.
            let d = ctx.objective(&factors, &means);
            if d.is_finite() && d <= accept_at {
                landed = true;
                break;
            }
            alpha = (alpha - 1.0) / 2.0;
            if alpha > -1.0 - 1e-3 {
                break;
            }
        }

        if !landed {
            // Nothing better than plain iteration was found. Put theta2 back, let the
            // next cycle sweep on from there, and wait longer before trying again.
            unflatten_factors(&theta2, &mut factors);
            ctx.refresh(&factors, &mut eta, &mut means);
            consecutive_failures = (consecutive_failures + 1).min(SQUAREM_MAX_BACKOFF);
            cycles_to_skip = (1usize << consecutive_failures) - 1;
            continue;
        }

        // A sweep from the extrapolated point, both to stabilise it and because an
        // accepted jump is only worth keeping if the map still improves on it.
        //
        // **Judged on the score, not the deviance.** Deviance is quadratic in the
        // parameter error near the optimum, so once the fit is close it cannot tell a
        // jump that halves the remaining error from one that doubles it - the same
        // reason the convergence test is on the score. Guarding here on deviance let
        // SQUAREM accept steps that raised the score, which both slowed the fit and
        // made the score sequence non-monotone, tripping the stall rule below. On 50
        // correlated tables that turned a 248-sweep fit into one that had not converged
        // after 5,000.
        //
        // `score3` is computed either way, to hand to `record` - so this costs nothing.
        ctx.sweep(&mut factors, &mut eta, &mut means, &mut numer, &mut denom);
        let deviance3 = ctx.deviance(&means);
        let objective3 = ctx.objective(&factors, &means);
        let score3 = ctx.max_score(&factors, &means, &mut score_scratch);
        if !(objective3.is_finite() && objective3 <= accept_at) || !(score3 <= score2) {
            unflatten_factors(&theta2, &mut factors);
            ctx.refresh(&factors, &mut eta, &mut means);
            consecutive_failures = (consecutive_failures + 1).min(SQUAREM_MAX_BACKOFF);
            cycles_to_skip = (1usize << consecutive_failures) - 1;
            continue;
        }

        consecutive_failures = 0;
        accelerated_steps += 1;
        if let Status::Stop = progress.record(objective3, deviance3, score3, &options) {
            break 'fitting;
        }
    }

    let Progress {
        iterations,
        converged,
        max_gradient,
        deviance_history,
        gradient_history,
        ..
    } = progress;

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
            penalty.as_ref(),
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
        table_conditioning,
        accelerated_steps,
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
    penalty: Option<&PenaltyPlan>,
    numer: &mut Vec<f64>,
    denom: &mut Vec<f64>,
) {
    if let TableSemantics::Variate { values, degree } = table.semantics() {
        update_variate_table(
            t,
            factors,
            eta,
            means,
            table_matches,
            target,
            weights,
            offset,
            values,
            *degree,
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
            table_matches,
            target,
            weights,
            means,
            power,
            loss_fn,
            numer,
            denom,
        );
    }

    let step_limit = loss_fn.step_limit();
    let eta_limit = loss_fn.eta_limit();

    // Every level of a penalised table is shrunk toward this one. Read from the factors
    // rather than assumed to be zero, so the penalty stays invariant to the anchoring
    // `normalize` performs - see [`crate::glm::penalty`].
    let penalty = penalty.filter(|p| p.covers(t));

    // Scratch for the base level's own step, which needs every other level's contrast
    // at once. One allocation the width of the table, on penalised fits only.
    let mut contrasts: Vec<f64> = match penalty {
        Some(_) => Vec::with_capacity(n_rows.saturating_sub(1)),
        None => Vec::new(),
    };

    for r in 0..n_rows {
        let old = factors[t][r];

        // Read after the base level has had its own turn, not before the loop, so every
        // other level is shrunk toward where the base actually is rather than toward
        // where it was a sweep ago. Reading it up front left the contrasts chasing a
        // reference that kept moving - a lasso strong enough to drop every level settled
        // at a contrast of 1.46 instead of zero. The cost is one array read per row.
        let anchor = match penalty {
            Some(_) => factors[t][ANCHOR_ROW],
            None => 0.0,
        };

        // Rows with no exposure, locked rows, and degenerate denominators keep whatever
        // factor they started with.
        let new = if row_exposure[r] <= 0.0
            || table.is_row_offset(r)
            || !(denom[r] > 0.0)
            || !denom[r].is_finite()
        {
            old
        } else if let (Some(pen), true) = (penalty, r == ANCHOR_ROW) {
            // The base level moves every contrast at once, so its step is not a scalar
            // solve - see [`TablePenalty::solve_anchor`]. Every other level is still
            // holding last sweep's value here, which is what makes these the contrasts
            // the step is defined against.
            contrasts.clear();
            contrasts.extend(
                (0..n_rows)
                    .filter(|s| *s != ANCHOR_ROW && pen.row(t, *s).is_some())
                    .map(|s| factors[t][s] - old),
            );
            anchor_factor(
                old,
                numer[r],
                denom[r],
                power,
                pen.row(t, 1).unwrap_or(TablePenalty { l1: 0.0, l2: 0.0 }),
                &mut contrasts,
                step_limit,
                eta_limit,
            )
            .unwrap_or(old)
        } else {
            next_factor(
                old,
                anchor,
                numer[r],
                denom[r],
                power,
                penalty.and_then(|p| p.row(t, r)),
                step_limit,
                eta_limit,
            )
            .unwrap_or(old)
        };

        factors[t][r] = new;
        // Reuse `numer` as the delta to fold into eta. Written on every path, including
        // the ones that decline to move, so no row can carry its accumulated numerator
        // through to `apply_row_deltas` as though it were a step.
        numer[r] = new - old;
    }

    // Fold the changes into the running linear predictor and mean. Reusing `numer` as
    // the per-row delta keeps this to a single pass with no extra allocation.
    apply_row_deltas(loss_fn, table_matches, &numer[..n_rows], offset, eta, means);
}

/// Where one level of a step table moves to, or `None` if the step is unusable.
///
/// This is phase two of a table update - the `O(n_rows)` arithmetic between the two
/// `O(n)` passes over the data - and it is the only phase a penalty touches. That is
/// the whole reason regularisation is close to free here: `numer` and `denom` are built
/// and applied by exactly the same code whether or not `penalty` is `Some`.
///
/// **Unpenalised**, the step is what it always was: `ln(A / E)` for a log link, where
/// `A = numer` and `E = denom`, and `numer / denom` otherwise.
///
/// **Penalised**, the same two numbers become the score and curvature of a local
/// quadratic model, and [`TablePenalty::solve`] returns the contrast that minimises it
/// plus the penalty. The two agree in the limit: at zero penalty the solve returns
/// `z + (A - E) / E`, so the raw step is `A / E - 1`, and the `ln(1 + .)` damping below
/// turns that back into `ln(A / E)` exactly. The penalised path is a generalisation of
/// the unpenalised one rather than a second algorithm - but it is still gated on
/// `penalty` being `Some`, so an unpenalised fit runs the arithmetic it always ran and
/// cannot drift by a rounding step.
///
/// The damping is the same trick [`update_pair`] uses, and for the same reason: a raw
/// Newton step is the wrong size on a log link while the fit is far out. `ln(1 + d)`
/// preserves sign, preserves zeros - so it moves neither the fixed point nor the local
/// rate - and damps the overshoot.
///
/// One case is deliberately not damped. When the threshold sends a coefficient to
/// exactly zero the level belongs exactly on the anchor, and landing anywhere near it
/// instead would leave a lasso fit with levels that are almost but not quite dropped.
/// So that step is taken whole, provided it is within the step limit; past the limit
/// the coefficient walks in over the next few sweeps and arrives at zero when it gets
/// there.
#[inline]
#[allow(clippy::too_many_arguments)]
fn next_factor(
    old: f64,
    anchor: f64,
    numer: f64,
    denom: f64,
    power: Option<f64>,
    penalty: Option<TablePenalty>,
    step_limit: f64,
    eta_limit: f64,
) -> Option<f64> {
    let step = match penalty {
        None => match power {
            Some(_) => {
                // ln(A / E); A <= 0 means the level has no positive response at all,
                // whose MLE is -inf. Fall back to the largest downward step allowed.
                if numer > 0.0 {
                    (numer / denom).ln()
                } else {
                    -MAX_STEP
                }
            }
            None => numer / denom,
        },
        Some(pen) => {
            // The score and curvature of the local model. For a log link the score is
            // `A - E` - the same quantity the convergence test scatters - and the
            // curvature is `E`.
            let (g, h) = match power {
                Some(_) => (numer - denom, denom),
                None => (numer, denom),
            };
            let z = old - anchor;
            let theta = pen.solve(g, h, z);
            if !theta.is_finite() {
                return None;
            }
            let raw = theta - z;
            if theta == 0.0 && raw.abs() <= step_limit {
                return Some(anchor.clamp(-eta_limit, eta_limit));
            }
            match power {
                Some(_) => {
                    if raw > -1.0 {
                        (1.0 + raw).ln()
                    } else {
                        -step_limit
                    }
                }
                None => raw,
            }
        }
    };

    if !step.is_finite() {
        return None;
    }
    Some((old + step.clamp(-step_limit, step_limit)).clamp(-eta_limit, eta_limit))
}

/// Where a penalised table's base level moves to, or `None` if the step is unusable.
///
/// [`TablePenalty::solve_anchor`] carries the argument for why this row is stepped at
/// all rather than pinned; the mechanics here are the same as [`next_factor`], including
/// the `ln(1 + .)` damping that a log link needs.
#[inline]
#[allow(clippy::too_many_arguments)]
fn anchor_factor(
    old: f64,
    numer: f64,
    denom: f64,
    power: Option<f64>,
    penalty: TablePenalty,
    contrasts: &mut [f64],
    step_limit: f64,
    eta_limit: f64,
) -> Option<f64> {
    let (g, h) = match power {
        Some(_) => (numer - denom, denom),
        None => (numer, denom),
    };
    let raw = penalty.solve_anchor(g, h, contrasts);
    if !raw.is_finite() {
        return None;
    }
    let step = match power {
        Some(_) => {
            if raw > -1.0 {
                (1.0 + raw).ln()
            } else {
                -step_limit
            }
        }
        None => raw,
    };
    if !step.is_finite() {
        return None;
    }
    Some((old + step.clamp(-step_limit, step_limit)).clamp(-eta_limit, eta_limit))
}

struct GlobalLayout {
    columns: Vec<Vec<Option<usize>>>,
    parameter_rows: Vec<Option<(usize, usize)>>,
    fitted_tables: Vec<bool>,
}

#[allow(clippy::too_many_arguments)]
fn fit_global_irls(
    mut model: RatingModel,
    loss_fn: LossFunction,
    target: Vec<f64>,
    weights: Vec<f64>,
    offset: Vec<f64>,
    matches: Vec<Vec<u32>>,
    mut factors: Vec<Vec<f64>>,
    updatable: Vec<bool>,
    variate_values: Vec<Option<(Vec<f64>, usize)>>,
    row_exposure: Vec<Vec<f64>>,
    penalty: Option<PenaltyPlan>,
    null_deviance: f64,
    options: GLMOptions,
) -> Result<(RatingModel, GLMDiagnostics), PolarsError> {
    if options.normalization != Normalization::BaseLevel {
        return Err(PolarsError::ComputeError(
            "The global solver currently requires normalization='base_level'.".into(),
        ));
    }
    if factors.is_empty() || factors[0].len() != 1 || !updatable[0] {
        return Err(PolarsError::ComputeError(
            "The global solver requires one updatable intercept row.".into(),
        ));
    }
    for t in 0..factors.len() {
        if updatable[t] && variate_values[t].is_some() {
            return Err(PolarsError::ComputeError(
                "The global solver does not yet support variate tables; use solver='table'.".into(),
            ));
        }
        if updatable[t] && (0..factors[t].len()).any(|r| model.tables[t].is_row_offset(r)) {
            return Err(PolarsError::ComputeError(
                "The global solver does not yet support locked rows inside an updatable table; use solver='table'."
                    .into(),
            ));
        }
    }

    let n_tables = factors.len();
    let mut columns: Vec<Vec<Option<usize>>> =
        factors.iter().map(|rows| vec![None; rows.len()]).collect();
    columns[0][0] = Some(0);
    let mut parameter_rows = vec![None];
    for t in 1..n_tables {
        if !updatable[t] {
            continue;
        }
        for r in 1..factors[t].len() {
            if row_exposure[t][r] <= 0.0 {
                continue;
            }
            let c = parameter_rows.len();
            columns[t][r] = Some(c);
            parameter_rows.push(Some((t, r)));
        }
    }
    let layout = GlobalLayout {
        columns,
        parameter_rows,
        fitted_tables: updatable.clone(),
    };
    let p = layout.parameter_rows.len();
    const MAX_GLOBAL_PARAMETERS: usize = 6000;
    if p > MAX_GLOBAL_PARAMETERS {
        return Err(PolarsError::ComputeError(
            format!(
                "The global solver needs a dense {p}x{p} Gram matrix, above its limit of \
                 {MAX_GLOBAL_PARAMETERS} parameters; use solver='table'."
            )
            .into(),
        ));
    }

    let mut beta = vec![0.0; p];
    beta[0] = factors[0][0];
    for t in 1..n_tables {
        if !updatable[t] {
            continue;
        }
        let base = factors[t][0];
        beta[0] += base;
        for r in 1..factors[t].len() {
            if let Some(c) = layout.columns[t][r] {
                beta[c] = factors[t][r] - base;
            }
        }
    }

    let mut fixed_eta = vec![0.0; target.len()];
    for t in 0..n_tables {
        if updatable[t] {
            continue;
        }
        for (i, m) in matches[t].iter().enumerate() {
            if *m != NO_MATCH {
                fixed_eta[i] += factors[t][*m as usize];
            }
        }
    }

    // IRLS is only locally quadratic. Starting a log-link fit at eta=0 means mu=1,
    // which is catastrophic for a response measured in house prices: the first Gamma
    // Newton step is hundreds of thousands and line search can spend hundreds of outer
    // iterations walking it back. The table solver already avoids this through its
    // exact intercept update. Give a fresh global fit the same closed-form null start.
    if beta.iter().skip(1).all(|v| *v == 0.0) {
        let effective_offset: Vec<f64> = offset
            .iter()
            .zip(fixed_eta.iter())
            .map(|(a, b)| a + b)
            .collect();
        beta[0] = null_intercept(&loss_fn, &target, &weights, &effective_offset);
    }

    let mut l1 = vec![0.0; p];
    let mut l2 = vec![0.0; p];
    if let Some(plan) = penalty.as_ref() {
        for c in 1..p {
            if let Some((t, r)) = layout.parameter_rows[c] {
                if let Some(term) = plan.row(t, r) {
                    l1[c] = term.l1;
                    l2[c] = term.l2;
                }
            }
        }
    }

    let pairable: Vec<bool> = (0..n_tables)
        .map(|t| updatable[t] && variate_values[t].is_none())
        .collect();
    let correlations = if options.solve_aliased_pairs_jointly {
        let shapes: Vec<usize> = factors.iter().map(Vec::len).collect();
        table_correlations(&matches, &weights, &shapes, &pairable)
    } else {
        Vec::new()
    };
    let table_conditioning = options
        .solve_aliased_pairs_jointly
        .then(|| collective_strength(&correlations));

    let mut eta = vec![0.0; target.len()];
    let mut means = vec![0.0; target.len()];
    global_refresh(
        &loss_fn, &beta, &layout, &matches, &fixed_eta, &offset, &mut eta, &mut means,
    );
    global_set_factors(&beta, &layout, &mut factors);

    let mut progress = Progress {
        iterations: 0,
        converged: false,
        max_gradient: f64::INFINITY,
        deviance: f64::INFINITY,
        sweeps_without_progress: 0,
        deviance_history: Vec::with_capacity(options.max_iterations),
        gradient_history: Vec::with_capacity(options.max_iterations),
        objective_history: Vec::with_capacity(options.max_iterations),
    };
    let mut score_scratch: Vec<Vec<f64>> = factors.iter().map(|f| vec![0.0; f.len()]).collect();
    loop {
        let (gram, score) = global_quadratic(
            &loss_fn, &target, &weights, &means, &matches, &updatable, &layout, p,
        );

        let mut q = score.clone();
        for j in 0..p {
            for k in 0..p {
                q[j] += gram[j * p + k] * beta[k];
            }
        }
        // An early IRLS quadratic is only a local approximation, so solving it to the
        // final KKT tolerance wastes coordinate cycles. Tighten with the outer residual
        // (an inexact-Newton forcing sequence), reaching full precision only near the
        // solution.
        let final_inner_tolerance = (options.tolerance * 0.01).max(1e-13);
        let inner_tolerance = final_inner_tolerance.max(
            (progress.max_gradient * 0.01)
                .min(1e-5)
                .max(final_inner_tolerance),
        );
        let proposed = if l1.iter().any(|v| *v > 0.0) {
            solve_gram_cd(&gram, &q, &beta, &l1, &l2, p, inner_tolerance)
        } else {
            let mut system = gram.clone();
            for j in 0..p {
                system[j * p + j] += l2[j];
            }
            solve_spd(&system, &q, p)
                .unwrap_or_else(|| solve_gram_cd(&gram, &q, &beta, &l1, &l2, p, inner_tolerance))
        };

        let current_objective = loss_fn.total_deviance(&target, &means, &weights)
            + penalty.as_ref().map_or(0.0, |plan| plan.total(&factors));
        let mut scale = 1.0;
        let mut accepted = false;
        let mut candidate = beta.clone();
        for _ in 0..24 {
            for j in 0..p {
                candidate[j] = beta[j] + scale * (proposed[j] - beta[j]);
            }
            global_set_factors(&candidate, &layout, &mut factors);
            global_refresh(
                &loss_fn, &candidate, &layout, &matches, &fixed_eta, &offset, &mut eta, &mut means,
            );
            let objective = loss_fn.total_deviance(&target, &means, &weights)
                + penalty.as_ref().map_or(0.0, |plan| plan.total(&factors));
            if objective.is_finite()
                && objective <= current_objective + current_objective.abs() * 1e-12
            {
                accepted = true;
                break;
            }
            scale *= 0.5;
        }
        if accepted {
            beta.copy_from_slice(&candidate);
        } else {
            global_set_factors(&beta, &layout, &mut factors);
            global_refresh(
                &loss_fn, &beta, &layout, &matches, &fixed_eta, &offset, &mut eta, &mut means,
            );
        }

        let deviance = loss_fn.total_deviance(&target, &means, &weights);
        let objective = deviance + penalty.as_ref().map_or(0.0, |plan| plan.total(&factors));
        let gradient = max_abs_score(
            &loss_fn,
            &target,
            &weights,
            &means,
            &matches,
            &model.tables,
            &updatable,
            &row_exposure,
            &variate_values,
            &factors,
            penalty.as_ref(),
            &mut score_scratch,
        );
        if let Status::Stop = progress.record(objective, deviance, gradient, &options) {
            break;
        }
    }

    let mut inference_error = None;
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
            penalty.as_ref(),
        ) {
            Ok(value) => Some(value),
            Err(error) => {
                inference_error = Some(error.to_string());
                None
            }
        }
    } else {
        None
    };
    for t in 0..n_tables {
        write_back_factors(&mut model.tables[t], &factors[t])?;
    }
    let mut unfitted_rows = Vec::new();
    for t in 0..n_tables {
        if !updatable[t] {
            continue;
        }
        for (r, exposure) in row_exposure[t].iter().enumerate() {
            if *exposure <= 0.0 {
                unfitted_rows.push((t, r));
            }
        }
    }
    Ok((
        model,
        GLMDiagnostics {
            iterations: progress.iterations,
            converged: progress.converged,
            max_gradient: progress.max_gradient,
            gradient_history: progress.gradient_history,
            deviance: progress.deviance,
            null_deviance,
            deviance_history: progress.deviance_history,
            unfitted_rows,
            table_conditioning,
            accelerated_steps: 0,
            inference,
            inference_error,
        },
    ))
}

fn global_set_factors(beta: &[f64], layout: &GlobalLayout, factors: &mut [Vec<f64>]) {
    factors[0][0] = beta[0];
    for t in 1..factors.len() {
        if !layout.fitted_tables[t] {
            continue;
        }
        factors[t][0] = 0.0;
        for r in 1..factors[t].len() {
            if let Some(c) = layout.columns[t][r] {
                factors[t][r] = beta[c];
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn global_refresh(
    loss_fn: &LossFunction,
    beta: &[f64],
    layout: &GlobalLayout,
    matches: &[Vec<u32>],
    fixed_eta: &[f64],
    offset: &[f64],
    eta: &mut [f64],
    means: &mut [f64],
) {
    for i in 0..eta.len() {
        let mut value = fixed_eta[i] + beta[0];
        for t in 1..matches.len() {
            let m = matches[t][i];
            if m != NO_MATCH {
                if let Some(c) = layout.columns[t][m as usize] {
                    value += beta[c];
                }
            }
        }
        eta[i] = value;
        means[i] = loss_fn.inverse_link(value + offset[i]);
    }
}

#[allow(clippy::too_many_arguments)]
fn global_quadratic(
    loss_fn: &LossFunction,
    target: &[f64],
    weights: &[f64],
    means: &[f64],
    matches: &[Vec<u32>],
    updatable: &[bool],
    layout: &GlobalLayout,
    p: usize,
) -> (Vec<f64>, Vec<f64>) {
    let accumulate = |start: usize, end: usize, gram: &mut [f64], score: &mut [f64]| {
        let mut active = Vec::with_capacity(matches.len());
        for i in start..end {
            let prior = weights[i];
            if !(prior > 0.0) {
                continue;
            }
            active.clear();
            active.push(0usize);
            for t in 1..matches.len() {
                if !updatable[t] {
                    continue;
                }
                let m = matches[t][i];
                if m != NO_MATCH {
                    if let Some(c) = layout.columns[t][m as usize] {
                        active.push(c);
                    }
                }
            }
            let w = prior * loss_fn.irls_weight(means[i]);
            let wr = prior * loss_fn.weighted_link_residual(target[i], means[i]);
            if !w.is_finite() || !(w > 0.0) || !wr.is_finite() {
                continue;
            }
            for (a_pos, &a) in active.iter().enumerate() {
                score[a] += wr;
                gram[a * p + a] += w;
                for &b in active.iter().skip(a_pos + 1) {
                    gram[a * p + b] += w;
                    gram[b * p + a] += w;
                }
            }
        }
    };

    // The global matrix is deliberately bounded, and for the ordinary rating-plan
    // widths this gives each worker a few hundred KB rather than sharing cache lines.
    // Below 100k rows the serial loop is faster than allocating and reducing copies.
    if target.len() >= PARALLEL_ROWS && p <= 1000 && rayon::current_num_threads() > 1 {
        let workers = rayon::current_num_threads();
        let chunk = (target.len() / workers).max(1);
        (0..target.len())
            .into_par_iter()
            .step_by(chunk)
            .map(|start| {
                let end = (start + chunk).min(target.len());
                let mut gram = vec![0.0; p * p];
                let mut score = vec![0.0; p];
                accumulate(start, end, &mut gram, &mut score);
                (gram, score)
            })
            .reduce(
                || (vec![0.0; p * p], vec![0.0; p]),
                |(mut ga, mut sa), (gb, sb)| {
                    for (a, b) in ga.iter_mut().zip(gb) {
                        *a += b;
                    }
                    for (a, b) in sa.iter_mut().zip(sb) {
                        *a += b;
                    }
                    (ga, sa)
                },
            )
    } else {
        let mut gram = vec![0.0; p * p];
        let mut score = vec![0.0; p];
        accumulate(0, target.len(), &mut gram, &mut score);
        (gram, score)
    }
}

fn solve_gram_cd(
    gram: &[f64],
    q: &[f64],
    start: &[f64],
    l1: &[f64],
    l2: &[f64],
    p: usize,
    tolerance: f64,
) -> Vec<f64> {
    let mut beta = start.to_vec();
    let mut residual = q.to_vec();
    for j in 0..p {
        for k in 0..p {
            residual[j] -= gram[j * p + k] * beta[k];
        }
        residual[j] -= l2[j] * beta[j];
    }
    for _ in 0..20_000 {
        let mut largest = 0.0f64;
        for j in 0..p {
            let curvature = gram[j * p + j] + l2[j];
            if !(curvature > 0.0) {
                continue;
            }
            let partial = residual[j] + curvature * beta[j];
            let next = if j == 0 {
                partial / curvature
            } else {
                soft_threshold(partial, l1[j]) / curvature
            };
            let change = next - beta[j];
            if change == 0.0 {
                continue;
            }
            beta[j] = next;
            largest = largest.max(change.abs());
            for k in 0..p {
                residual[k] -= gram[k * p + j] * change;
            }
            residual[j] -= l2[j] * change;
        }
        // Coefficients live on the link scale, so this must not be multiplied by `q`:
        // `q` grows with the row count and doing so made a large dataset accept a very
        // loose inner solve, throwing away the whole point of caching the Gram matrix.
        if largest <= tolerance {
            break;
        }
    }
    beta
}

/// The order tables are swept in: most strongly coupled first.
///
/// Backfitting is Gauss-Seidel, so a table updated early in a sweep is seen by every
/// table after it, while one updated late leaves the rest of the sweep working against a
/// stale value. Putting the table that shares the most information with the others first
/// is therefore the cheapest correction available: it is a permutation, decided once,
/// costing nothing at run time and changing no arithmetic - the fits it produces are
/// identical to the last digit of the deviance.
///
/// Measured against sweeping in table order, on the five real designs in the benchmark
/// suite: `house_sales` 50 sweeps to 44, `census_income` 36 to 33, freMTPL2 15 to 14,
/// `nyc_taxi` and the synthetic cases unchanged. Nothing got worse.
///
/// Two orderings that sound at least as plausible are worse, which is why this one is
/// not obvious. Sweeping *weakly* coupled tables first costs `census_income` 5 sweeps.
/// Chaining tables so that strongly coupled ones are adjacent - the arrangement that
/// most resembles solving them together - is the worst of all, costing `nyc_taxi` 10
/// sweeps and freMTPL2 2, presumably because being adjacent is worth much less than
/// being early.
///
/// With no correlations measured - [`GLMOptions::solve_aliased_pairs_jointly`] off -
/// every table scores zero and the tie breaks on the index, which is the original
/// table order.
fn sweep_order(correlations: &[TablePair], n_tables: usize) -> Vec<usize> {
    let mut coupling = vec![0.0f64; n_tables];
    for pair in correlations {
        coupling[pair.first] = coupling[pair.first].max(pair.correlation);
        coupling[pair.second] = coupling[pair.second].max(pair.correlation);
    }
    let mut order: Vec<usize> = (0..n_tables).collect();
    order.sort_by(|a, b| coupling[*b].total_cmp(&coupling[*a]).then(a.cmp(b)));
    order
}

/// Picks which near-aliased pairs to solve jointly, greedily and worst first.
///
/// The pairs have to be disjoint: a table updated inside two different blocks in the same
/// sweep would have the second block overwrite the first's view of it. Taking the worst
/// pair, then the worst remaining pair among untouched tables, and so on, keeps every
/// table updated exactly once per sweep. In practice one pair is the whole problem — on
/// the French motor data the worst pair measures 0.971 and the next 0.611.
///
/// Within a pair the table with **more levels** leads. Near-aliasing between rating
/// tables is almost always a coarsening — `Area` is `Density` rebanded into six bands —
/// and the finer table is the one that can express what the coarser one cannot. The
/// coarser table takes the ridge, so it keeps only what the finer one leaves behind.
fn choose_joint_pairs(correlations: &[TablePair], shapes: &[usize]) -> Vec<(usize, usize)> {
    let mut taken = vec![false; shapes.len()];
    let mut pairs = Vec::new();

    // `table_correlations` returns them worst first.
    for pair in correlations.iter().filter(|p| p.is_near_aliased()) {
        if taken[pair.first] || taken[pair.second] {
            continue;
        }
        taken[pair.first] = true;
        taken[pair.second] = true;
        pairs.push(if shapes[pair.first] >= shapes[pair.second] {
            (pair.first, pair.second)
        } else {
            (pair.second, pair.first)
        });
    }
    pairs
}

/// Relative ridge added to the secondary table's block when a near-aliased pair is
/// solved jointly.
///
/// The combined design of two step tables is rank deficient no matter how well separated
/// they are — each one's levels sum to the same total, so a constant can be moved from
/// one to the other without changing a prediction — and a near-aliased pair adds
/// directions that are nearly null on top of that. Something has to pick among the
/// solutions, and this picks the one with the smallest secondary factors.
///
/// It is scaled to the block's own diagonal, so it means "one part in a billion of the
/// exposure behind this table" rather than a fixed number of claims. Large enough to make
/// the factorisation succeed, far too small to shrink a factor anyone would notice: this
/// resolves the tie, it does not express a view about it. For deliberate shrinkage —
/// keeping two tables in a plan while making one of them carry the signal — see the
/// per-table credibility discussion in the module README.
const PAIR_RIDGE: f64 = 1e-9;

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

/// Updates two near-aliased tables together, as one block, instead of one after the
/// other.
///
/// Backfitting's rate between two blocks is their first canonical correlation squared.
/// On the French motor data `Area` is a six-band rebanding of `Density`, which measures
/// 0.971 — so the shared direction survives 94% of each sweep and the fit spends
/// hundreds of passes handing a constant back and forth. Solving the pair jointly removes
/// that direction outright: whatever the two tables share is settled inside one step
/// rather than negotiated across many.
///
/// The block is assembled from quantities the ordinary update already produces. Writing
/// `w` for the IRLS weight `a * mu^(2-p)` and `g` for the score `A - E`:
///
/// ```text
///   [ D_t  C  ] [d_t]   [g_t]        D_t = diag(sum of w per level of t)  = `denom`
///   [ C'   D_u] [d_u] = [g_u]        C[r][s] = sum of w over cells (r, s)
///                                    g_t     = A_t - E_t = `numer` - `denom`
/// ```
///
/// The diagonal blocks are diagonal because each observation belongs to exactly one level
/// of each table, so only the cross block `C` is new — one extra scatter-add per
/// observation, against the `T(T+1)/2` a full normal matrix would need. The system is
/// `(k_t + k_u)` square, which for a pair of rating tables is a hundred-odd entries.
///
/// Note this is a Newton step on the quadratic approximation, not the exact `ln(A/E)`
/// coordinate solve the tables would get singly. That is the price of coupling them, and
/// it is the same step glum takes for every table on every iteration.
#[allow(clippy::too_many_arguments)]
fn update_pair(
    t: usize,
    u: usize,
    factors: &mut [Vec<f64>],
    eta: &mut [f64],
    means: &mut [f64],
    matches: &[Vec<u32>],
    target: &[f64],
    weights: &[f64],
    offset: &[f64],
    row_exposure: &[Vec<f64>],
    tables: &[RatingTable],
    loss_fn: &LossFunction,
    penalty: Option<&PenaltyPlan>,
) {
    let (k_t, k_u) = (factors[t].len(), factors[u].len());
    let n = k_t + k_u;

    let power = loss_fn.log_link_variance_power();
    let mut d_t = vec![0.0f64; k_t];
    let mut d_u = vec![0.0f64; k_u];
    let mut g_t = vec![0.0f64; k_t];
    let mut g_u = vec![0.0f64; k_u];
    let mut cross = vec![0.0f64; k_t * k_u];

    for i in 0..target.len() {
        let a = weights[i];
        if a == 0.0 {
            continue;
        }
        let (mt, mu_idx) = (matches[t][i], matches[u][i]);
        if mt == NO_MATCH || mu_idx == NO_MATCH {
            continue;
        }
        let (r, s) = (mt as usize, mu_idx as usize);
        let mean = means[i];

        let (w, g) = match power {
            Some(p) => {
                let base = pow_special(mean, 1.0 - p);
                (a * base * mean, a * base * (target[i] - mean))
            }
            None => (
                a * loss_fn.irls_weight(mean),
                a * loss_fn.weighted_link_residual(target[i], mean),
            ),
        };
        if !w.is_finite() || !g.is_finite() {
            continue;
        }

        d_t[r] += w;
        d_u[s] += w;
        g_t[r] += g;
        g_u[s] += g;
        cross[r * k_u + s] += w;
    }

    // A level that is locked, or that no observation reached, carries no free parameter.
    // Zeroing its row and column and pinning its diagonal leaves the rest of the system
    // untouched and its own step at zero.
    let frozen_t: Vec<bool> = (0..k_t)
        .map(|r| row_exposure[t][r] <= 0.0 || tables[t].is_row_offset(r) || !(d_t[r] > 0.0))
        .collect();
    let frozen_u: Vec<bool> = (0..k_u)
        .map(|s| row_exposure[u][s] <= 0.0 || tables[u].is_row_offset(s) || !(d_u[s] > 0.0))
        .collect();

    // The secondary table takes the ridge, so the shared component lands on the primary.
    let scale: f64 = d_u.iter().sum::<f64>() / (k_u as f64).max(1.0);
    let ridge = PAIR_RIDGE * scale.max(f64::MIN_POSITIVE);

    // An L2 penalty is exact here: it adds a constant to each level's curvature, a
    // linear term to its score, and - because every contrast is measured against the
    // base level - a `-l2` coupling between each level and the base, which this system
    // is already dense enough to carry. An L1 penalty has no equivalent, since there is
    // no way to soft-threshold two coupled coordinates at once and land on the right
    // pair of zeros, so the caller empties `joint_pairs` when one is active and the
    // lambdas below are then all zero.
    let pen_t = penalty.filter(|p| p.covers(t));
    let pen_u = penalty.filter(|p| p.covers(u));
    let anchor_t = factors[t].get(ANCHOR_ROW).copied().unwrap_or(0.0);
    let anchor_u = factors[u].get(ANCHOR_ROW).copied().unwrap_or(0.0);
    let l2_of = |p: Option<&PenaltyPlan>, table: usize, row: usize| -> f64 {
        p.and_then(|p| p.row(table, row)).map_or(0.0, |q| q.l2)
    };

    let mut a_mat = vec![0.0f64; n * n];
    let mut b_vec = vec![0.0f64; n];
    for r in 0..k_t {
        if frozen_t[r] {
            a_mat[r * n + r] = 1.0;
            continue;
        }
        if r == ANCHOR_ROW {
            // The base level carries every contrast's penalty with the opposite sign.
            let mut curvature = 0.0;
            let mut score = 0.0;
            for other in 0..k_t {
                if other == ANCHOR_ROW || frozen_t[other] {
                    continue;
                }
                let l2 = l2_of(pen_t, t, other);
                curvature += l2;
                score += l2 * (factors[t][other] - anchor_t);
                a_mat[r * n + other] -= l2;
                a_mat[other * n + r] -= l2;
            }
            a_mat[r * n + r] = d_t[r] + curvature;
            b_vec[r] = g_t[r] + score;
        } else {
            let l2 = l2_of(pen_t, t, r);
            a_mat[r * n + r] = d_t[r] + l2;
            b_vec[r] = g_t[r] - l2 * (factors[t][r] - anchor_t);
        }
        for s in 0..k_u {
            if frozen_u[s] {
                continue;
            }
            let c = cross[r * k_u + s];
            a_mat[r * n + (k_t + s)] = c;
            a_mat[(k_t + s) * n + r] = c;
        }
    }
    for s in 0..k_u {
        let j = k_t + s;
        if frozen_u[s] {
            a_mat[j * n + j] = 1.0;
            continue;
        }
        if s == ANCHOR_ROW {
            let mut curvature = 0.0;
            let mut score = 0.0;
            for other in 0..k_u {
                if other == ANCHOR_ROW || frozen_u[other] {
                    continue;
                }
                let l2 = l2_of(pen_u, u, other);
                curvature += l2;
                score += l2 * (factors[u][other] - anchor_u);
                a_mat[j * n + (k_t + other)] -= l2;
                a_mat[(k_t + other) * n + j] -= l2;
            }
            a_mat[j * n + j] = d_u[s] + ridge + curvature;
            b_vec[j] = g_u[s] + score;
        } else {
            let l2 = l2_of(pen_u, u, s);
            a_mat[j * n + j] = d_u[s] + ridge + l2;
            b_vec[j] = g_u[s] - l2 * (factors[u][s] - anchor_u);
        }
    }

    let Some(step) = solve_spd(&a_mat, &b_vec, n) else {
        // A block that will not factorise is one the pair cannot be separated on at all.
        // Fall through to the ordinary one-at-a-time updates rather than doing nothing.
        return;
    };

    let step_limit = loss_fn.step_limit();
    let eta_limit = loss_fn.eta_limit();

    // A raw Newton step is the wrong size on a log link when the fit is still far out,
    // and it oscillates rather than converging - which is precisely why the single-table
    // update uses the exact solve instead. The two are related exactly: for one level in
    // isolation the Newton step is `A/E - 1` and the exact solve is `ln(A/E)`, so
    // `ln(1 + step)` recovers the exact update whenever the coupling vanishes, damps the
    // overshoot identically when it does not, and leaves both the fixed point and the
    // local convergence rate untouched (`ln(1+d) -> d` as `d -> 0`).
    //
    // A step at or below -1 is asking for a mean of zero or less, which has no finite
    // answer; the largest downward step allowed stands in, as it does elsewhere.
    let damp = |raw: f64| -> f64 {
        match power {
            Some(_) => {
                if raw > -1.0 {
                    (1.0 + raw).ln()
                } else {
                    -step_limit
                }
            }
            None => raw,
        }
    };

    let mut delta_t = vec![0.0f64; k_t];
    for r in 0..k_t {
        if frozen_t[r] || !step[r].is_finite() {
            continue;
        }
        let old = factors[t][r];
        let new = (old + damp(step[r]).clamp(-step_limit, step_limit)).clamp(-eta_limit, eta_limit);
        factors[t][r] = new;
        delta_t[r] = new - old;
    }

    let mut delta_u = vec![0.0f64; k_u];
    for s in 0..k_u {
        if frozen_u[s] || !step[k_t + s].is_finite() {
            continue;
        }
        let old = factors[u][s];
        let new =
            (old + damp(step[k_t + s]).clamp(-step_limit, step_limit)).clamp(-eta_limit, eta_limit);
        factors[u][s] = new;
        delta_u[s] = new - old;
    }

    apply_row_deltas(loss_fn, &matches[t], &delta_t, offset, eta, means);
    apply_row_deltas(loss_fn, &matches[u], &delta_u, offset, eta, means);
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

        // Anchoring is a pure change of gauge: it moves a constant out of a table and
        // into the intercept, leaving every prediction — and therefore the running `eta`
        // — untouched. That is what makes it safe to do mid-fit without recomputing
        // anything.
        //
        // It stops being true the moment a factor it produces is not representable. A
        // level with no positive response has an MLE of -infinity, so it walks down by
        // `MAX_STEP` every sweep until it sits on the clamp; anchoring a table against
        // such a base level then asks the intercept to go past the clamp as well. Truncating
        // it there would leave the factors no longer summing to `eta`, and every later
        // sweep would work from a linear predictor that describes a different model —
        // which shows up as the fit locking solid at a gradient it can never improve.
        //
        // So the shift is skipped instead. The fit is unaffected; only its presentation
        // is, and a table anchored on a level the data cannot identify was never going to
        // read as a sensible set of relativities anyway.
        let intercept_after = factors[0][0] + shift_into_intercept + anchor;
        if intercept_after.abs() > eta_limit {
            continue;
        }
        if factors[t].iter().any(|f| (f - anchor).abs() > eta_limit) {
            continue;
        }

        for f in factors[t].iter_mut() {
            *f -= anchor;
        }
        shift_into_intercept += anchor;
    }

    factors[0][0] += shift_into_intercept;
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

/// Relative improvement in the deviance, per sweep, below which a sweep counts as having
/// made no progress.
///
/// Set at the deviance's own rounding floor, which is where it stops carrying any
/// information - a fit sitting on that floor moves by about 5e-15 relative, in either
/// direction. It cannot be set any higher: the deviance is *quadratic* in the parameter
/// error, so on the final approach to a tight score tolerance its improvements are
/// legitimately tiny, and a threshold of 1e-12 stops fits that are still converging
/// perfectly well (it cut one off at `max|score| = 4.4e-09` against a 1e-9 tolerance).
///
/// The consequence is deliberate: past a score of roughly 1e-8 the deviance can no
/// longer certify that anything is happening, so a fit that truly stalls beyond that
/// point runs to `max_iterations` instead of stopping early. Wasting sweeps is a much
/// smaller failure than abandoning a converging fit and returning the iterate.
const DEVIANCE_STALL: f64 = 1e-15;

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
    factors: &[Vec<f64>],
    penalty: Option<&PenaltyPlan>,
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
                    || {
                        (
                            shape.iter().map(|k| vec![0.0f64; *k]).collect::<Vec<_>>(),
                            0.0f64,
                        )
                    },
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
                    || {
                        (
                            shape.iter().map(|k| vec![0.0f64; *k]).collect::<Vec<_>>(),
                            0.0f64,
                        )
                    },
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
                total_abs += score_row(
                    i, loss_fn, target, weights, means, matches, updatable, scratch,
                );
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
                // Under a penalty the score at the optimum is not zero - it is equal
                // and opposite to the penalty gradient, which is the entire point of
                // penalising. Testing the raw score would mean a penalised fit never
                // reported convergence and always ran to `max_iterations`. See
                // [`TablePenalty::subgradient`], and note it also handles the L1 kink,
                // where optimality is the inclusion `|g| <= l1` rather than an equation.
                let table_penalty = penalty.filter(|p| p.covers(t));
                let anchor = match table_penalty {
                    Some(_) => factors[t][ANCHOR_ROW],
                    None => 0.0,
                };
                // The base level's own condition, which is the sum of every other
                // level's with the sign flipped. Built once per table.
                let base_contrasts: Vec<f64> = match table_penalty {
                    Some(plan) => (0..scratch[t].len())
                        .filter(|s| *s != ANCHOR_ROW && plan.row(t, *s).is_some())
                        .map(|s| factors[t][s] - anchor)
                        .collect(),
                    None => Vec::new(),
                };

                for r in 0..scratch[t].len() {
                    // A locked row or one with no exposure carries no free parameter,
                    // so its score is not ours to drive to zero.
                    if row_exposure[t][r] <= 0.0 || tables[t].is_row_offset(r) {
                        continue;
                    }
                    let g = match (table_penalty, r == ANCHOR_ROW) {
                        (Some(plan), true) => plan.row(t, 1).map_or(scratch[t][r], |pen| {
                            pen.anchor_subgradient(scratch[t][r], &base_contrasts)
                        }),
                        (Some(plan), false) => match plan.row(t, r) {
                            Some(pen) => pen.subgradient(scratch[t][r], factors[t][r] - anchor),
                            None => scratch[t][r],
                        },
                        (None, _) => scratch[t][r],
                    };
                    worst = worst.max(g.abs());
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
    let beta = null_intercept(loss_fn, target, weights, offset);
    let means: Vec<f64> = offset
        .iter()
        .map(|o| loss_fn.inverse_link(beta + o))
        .collect();
    loss_fn.total_deviance(target, &means, weights)
}

/// Iteration cap for the links that have no closed form. Reached only by a fit that is
/// not converging at all; the ordinary exit is the noise-floor test below.
const MAX_NULL_ITERATIONS: usize = 50;

/// The intercept-only fit the null deviance is measured at: the fitter's own coordinate
/// update applied to a single global level.
///
/// **Under a log link this is one pass, not an iteration.** Write `A` for
/// `sum a·e^((1-p)·o)·y` and `B` for `sum a·e^((2-p)·o)`. With `mu = e^(beta + o)` the
/// update from any starting `beta` is
///
/// ```text
/// step(beta) = ln( e^((1-p)·beta)·A / e^((2-p)·beta)·B ) = ln(A / B) - beta
/// ```
///
/// so `beta + step(beta) = ln(A / B)` regardless of where it started. The first pass
/// lands on the answer and every later one only re-derives it.
///
/// This used to iterate to a fixed point instead, and the cost was not small: the exit
/// test compared a step that jitters at the rounding noise of an `n`-term sequential sum
/// (~1e-13 over 678k rows) against a threshold of `1e-14`, so it usually ran its full
/// 200 passes. On freMTPL2 that was ~490 ms of a ~750 ms fit — twice what the sweeps
/// themselves cost — to compute a statistic that is only reported.
fn null_intercept(loss_fn: &LossFunction, target: &[f64], weights: &[f64], offset: &[f64]) -> f64 {
    let eta_limit = loss_fn.eta_limit();

    if let Some(p) = loss_fn.log_link_variance_power() {
        let mut numer = 0.0;
        let mut denom = 0.0;
        for i in 0..target.len() {
            let a = weights[i];
            if a == 0.0 {
                continue;
            }
            // `beta = 0`, so this is exactly the first pass of the old loop.
            let mu = loss_fn.inverse_link(offset[i]);
            let base = pow_special(mu, 1.0 - p);
            numer += a * base * target[i];
            denom += a * base * mu;
        }
        if !(denom > 0.0) || !denom.is_finite() {
            return 0.0;
        }
        if !(numer > 0.0) {
            // Nothing positive to explain. The likelihood drives the mean to zero and
            // the link's bound is as far down as the linear predictor goes.
            return -eta_limit;
        }
        let beta = (numer / denom).ln();
        if !beta.is_finite() {
            // An overflowed or underflowed ratio says nothing about where the intercept
            // belongs. The old loop stopped on the same condition and kept its start.
            return 0.0;
        }
        return beta.clamp(-eta_limit, eta_limit);
    }

    // Identity and logit links have no such collapse, so iterate. The identity link is
    // exact in one step; the logit takes a handful.
    let mut beta = 0.0f64;
    let mut last_delta = f64::INFINITY;
    for _ in 0..MAX_NULL_ITERATIONS {
        let mut numer = 0.0;
        let mut denom = 0.0;
        for i in 0..target.len() {
            let a = weights[i];
            if a == 0.0 {
                continue;
            }
            let mu = loss_fn.inverse_link(beta + offset[i]);
            numer += a * loss_fn.weighted_link_residual(target[i], mu);
            denom += a * loss_fn.irls_weight(mu);
        }
        if !(denom > 0.0) || !denom.is_finite() {
            break;
        }
        let step = numer / denom;
        if !step.is_finite() {
            break;
        }
        let next = (beta + step.clamp(-loss_fn.step_limit(), loss_fn.step_limit()))
            .clamp(-eta_limit, eta_limit);
        let delta = (next - beta).abs();
        beta = next;
        // Converged, or as close as this many summands can resolve: once the step stops
        // shrinking it is rounding noise, and further passes only re-measure it.
        if delta < 1e-14 * beta.abs().max(1.0) || delta >= last_delta {
            break;
        }
        last_delta = delta;
    }

    beta
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
            format!(
                "{} column '{}' must be Float64, found {:?}",
                role,
                name,
                df.column(name).unwrap().dtype()
            )
            .into(),
        )
    })?;

    let mut out = Vec::with_capacity(ca.len());
    for i in 0..ca.len() {
        match ca.get(i) {
            Some(v) if v.is_finite() => out.push(v),
            Some(v) => {
                return Err(PolarsError::ComputeError(
                    format!(
                        "{} column '{}' has a non-finite value ({}) at row {}",
                        role, name, v, i
                    )
                    .into(),
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
        let features: Vec<String> = model.tables[t].get_feature_info().keys().cloned().collect();
        return Err(PolarsError::ComputeError(
            format!(
                "Table {} matched no row for {} of {} observations (first at row {}). \
                 Table features: [{}]. Every observation must fall in some row: check that \
                 those columns are present with the expected dtype, that numeric tables have \
                 a final unbounded (inf) row, and that categorical tables cover every level \
                 or carry a -999 wildcard.",
                t,
                unmatched,
                n_rows,
                first,
                features.join(", ")
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
        return Err(PolarsError::ComputeError(
            "Training data has no rows".into(),
        ));
    }

    // Check target column exists
    if df.column(target_col).is_err() {
        return Err(PolarsError::ColumnNotFound(
            format!("Target column '{}' not found", target_col).into(),
        ));
    }

    // Check weight column exists if specified
    if let Some(wcol) = weight_col {
        if df.column(wcol).is_err() {
            return Err(PolarsError::ColumnNotFound(
                format!("Weight column '{}' not found", wcol).into(),
            ));
        }
    }

    // Check offset column exists if specified
    if let Some(ocol) = offset_col {
        if df.column(ocol).is_err() {
            return Err(PolarsError::ColumnNotFound(
                format!("Offset column '{}' not found", ocol).into(),
            ));
        }
    }

    // Check that model has at least 2 tables (mean + at least one feature table)
    if model.tables.len() < 2 {
        return Err(PolarsError::ComputeError(
            "Model must have at least 2 tables (mean + feature tables)".into(),
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

    fn fresh_progress(capacity: usize) -> Progress {
        Progress {
            iterations: 0,
            converged: false,
            max_gradient: f64::INFINITY,
            deviance: f64::INFINITY,
            sweeps_without_progress: 0,
            deviance_history: Vec::with_capacity(capacity),
            gradient_history: Vec::with_capacity(capacity),
            objective_history: Vec::with_capacity(capacity),
        }
    }

    /// The defect the give-up rule was rewritten for.
    ///
    /// On a badly conditioned fit the error rotates as it decays, so `max|score|` rises
    /// and falls with a period of tens of sweeps while the fit improves throughout. A
    /// rule that asks the score whether progress is being made abandons that fit - on a
    /// hundred correlated tables it quit at sweep 42 and returned a model 95% wrong,
    /// where the same fit reaches the tolerance at sweep 1119. The deviance cannot
    /// behave that way, because every coordinate update is an exact minimiser along its
    /// own coordinate.
    #[test]
    fn an_oscillating_score_does_not_abandon_a_converging_fit() {
        let options = GLMOptions {
            tolerance: 1e-12,
            max_iterations: 10_000,
            ..Default::default()
        };
        let mut progress = fresh_progress(300);

        let mut deviance = 1_000.0f64;
        for sweep in 0..300 {
            // Falling every sweep, but slowly - this is what a badly conditioned fit
            // that is nonetheless converging looks like.
            deviance -= deviance * 1e-7;
            // Trending down over a period far longer than STALL_SWEEPS, exactly as
            // measured on the correlated case.
            let phase = (sweep as f64) / 5.0;
            let score = 1e-3 * (1.0 - sweep as f64 / 600.0) * (1.5 + phase.sin());
            assert!(
                matches!(
                    progress.record(deviance, deviance, score, &options),
                    Status::Continue
                ),
                "gave up at sweep {} with the deviance still falling",
                sweep
            );
        }
    }

    /// The case the rule does exist for: nothing further is reachable, so stop.
    #[test]
    fn a_deviance_at_its_rounding_floor_ends_the_fit() {
        let options = GLMOptions {
            tolerance: 1e-12,
            max_iterations: 10_000,
            ..Default::default()
        };
        let mut progress = fresh_progress(64);

        let mut deviance = 1_000.0f64;
        let mut stopped_at = None;
        for sweep in 0..(4 * STALL_SWEEPS) {
            // Moving only by rounding, which carries no information either way.
            deviance -= deviance * 1e-16;
            if let Status::Stop = progress.record(deviance, deviance, 1e-6, &options) {
                stopped_at = Some(sweep);
                break;
            }
        }
        let stopped_at = stopped_at.expect("a fit on its floor has to stop");
        assert!(
            stopped_at <= STALL_SWEEPS + 2,
            "took {} sweeps to notice a fit that cannot move",
            stopped_at
        );
        assert!(!progress.converged, "stopping short is not convergence");
    }

    /// The log-link null intercept is the one that reproduces the weighted total, and
    /// it is reached in a single pass whatever the offsets are.
    #[test]
    fn poisson_null_intercept_matches_the_closed_form() {
        let target: [f64; 6] = [0.0, 3.0, 1.0, 0.0, 7.0, 2.0];
        let weights: [f64; 6] = [1.0, 2.0, 1.0, 0.5, 1.0, 3.0];
        let offset: [f64; 6] = [-0.4, 0.9, 0.0, 1.7, -2.1, 0.3];

        // Poisson has variance power 1, so the weights on both sums are the prior
        // weights alone: beta = ln( sum a·y / sum a·e^o ).
        let a_y: f64 = target.iter().zip(weights).map(|(y, a)| a * y).sum();
        let a_e: f64 = offset.iter().zip(weights).map(|(o, a)| a * o.exp()).sum();

        let beta = null_intercept(&LossFunction::Poisson, &target, &weights, &offset);
        assert!(
            (beta - (a_y / a_e).ln()).abs() < 1e-12,
            "beta {beta} is not ln(A/E) = {}",
            (a_y / a_e).ln()
        );
    }

    /// The claim the one-pass form rests on: the update's fixed point does not depend on
    /// where it starts, so re-applying it cannot move the answer. Checked on Gamma,
    /// whose variance power puts a non-trivial weight on both sums.
    #[test]
    fn log_link_null_intercept_is_a_fixed_point() {
        let target = [2.0, 11.0, 0.5, 4.0, 30.0];
        let weights = [1.0, 1.0, 2.0, 1.0, 0.5];
        let offset = [0.2, -1.1, 0.0, 3.0, 0.7];
        let loss = LossFunction::Gamma;
        let p = loss
            .log_link_variance_power()
            .expect("gamma has a variance power");

        let beta = null_intercept(&loss, &target, &weights, &offset);

        // One more turn of the same coordinate update, by hand.
        let mut numer = 0.0;
        let mut denom = 0.0;
        for i in 0..target.len() {
            let mu = loss.inverse_link(beta + offset[i]);
            let base = pow_special(mu, 1.0 - p);
            numer += weights[i] * base * target[i];
            denom += weights[i] * base * mu;
        }
        let step = (numer / denom).ln();
        assert!(
            step.abs() < 1e-12,
            "step {step} should be zero at the fixed point"
        );
    }

    /// A response with no positive part has no finite log-link mean to fit; the answer
    /// is the bound rather than a walk towards it.
    #[test]
    fn log_link_null_intercept_bottoms_out_on_an_all_zero_response() {
        let target = [0.0, 0.0, 0.0];
        let weights = [1.0, 1.0, 1.0];
        let offset = [0.0, 0.5, -0.5];
        let beta = null_intercept(&LossFunction::Poisson, &target, &weights, &offset);
        assert_eq!(beta, -LossFunction::Poisson.eta_limit());
    }

    /// The identity link is exact in one step, and the loop must not spin past it.
    #[test]
    fn gaussian_null_intercept_is_the_weighted_mean() {
        let target = [1.0, 2.0, 3.0, 10.0];
        let weights = [1.0, 1.0, 2.0, 1.0];
        let offset = [0.0, 0.0, 0.0, 0.0];
        let mean: f64 = target.iter().zip(weights).map(|(y, a)| a * y).sum::<f64>()
            / weights.iter().sum::<f64>();
        let beta = null_intercept(&LossFunction::Gaussian, &target, &weights, &offset);
        assert!(
            (beta - mean).abs() < 1e-12,
            "beta {beta} is not the weighted mean {mean}"
        );
    }

    /// The property the accelerator is built on: when the error really is a single
    /// geometric mode, three iterates identify the fixed point exactly.
    ///
    /// Backfitting's error decays as `rho^k` along the direction the correlated tables
    /// share, so this is the shape of the real problem, not a convenient abstraction —
    /// the French motor fit tracks `rho = 0.943` for two hundred sweeps in a row.
    #[test]
    fn squarem_lands_on_the_fixed_point_of_a_single_mode() {
        let rho = 0.943_f64;
        let star = [2.0, -1.0, 0.5, 7.25];
        let err = [0.3, 0.7, -0.4, 1.1];
        let iterate = |k: i32| -> Vec<f64> {
            star.iter()
                .zip(err.iter())
                .map(|(s, e)| s + e * rho.powi(k))
                .collect()
        };

        let (t0, t1, t2) = (iterate(0), iterate(1), iterate(2));
        let alpha =
            squarem_steplength(&t0, &t1, &t2).expect("a geometric sequence has a steplength");

        // alpha = -1/(1 - rho) for a pure mode.
        assert!(
            (alpha - -1.0 / (1.0 - rho)).abs() < 1e-9,
            "steplength was {}, expected {}",
            alpha,
            -1.0 / (1.0 - rho)
        );

        let mut out = Vec::new();
        squarem_extrapolate(&t0, &t1, &t2, alpha, f64::INFINITY, &mut out);
        for (got, want) in out.iter().zip(star.iter()) {
            assert!(
                (got - want).abs() < 1e-9,
                "extrapolated to {:?}, fixed point is {:?}",
                out,
                star
            );
        }
    }

    /// Iterates that are already at the fixed point, or that carry no curvature to read
    /// a mode from, must not produce a steplength — there is nothing to extrapolate and
    /// the ratio that defines `alpha` is 0/0.
    #[test]
    fn squarem_declines_a_degenerate_sequence() {
        let fixed = vec![1.0, 2.0, 3.0];
        assert!(squarem_steplength(&fixed, &fixed, &fixed).is_none());

        // A straight line: r is non-zero but v vanishes.
        let (t0, t1, t2) = (vec![0.0, 0.0], vec![1.0, 1.0], vec![2.0, 2.0]);
        assert!(squarem_steplength(&t0, &t1, &t2).is_none());
    }

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
                    loss_fn,
                    i,
                    mu[i],
                    expected
                );
            }
        }
    }
}
