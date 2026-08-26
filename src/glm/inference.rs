//! Standard errors and fit statistics for a fitted GLM.
//!
//! # Why this needs care
//!
//! A rating model carries an intercept table *and* a free factor for every level of
//! every table. That design matrix is rank deficient: adding a constant to one table
//! and subtracting it from the intercept leaves every prediction untouched. `X'WX` is
//! therefore singular and the individual factors have no standard error at all.
//!
//! What *is* estimable is any contrast that is invariant to that shift — level
//! differences within a table, and the fit's overall level. So inference is done in a
//! reduced, full-rank basis (an intercept plus every level except each table's first,
//! which is exactly treatment coding), and the standard error reported against each
//! table row is the standard error of the contrast that row actually represents under
//! the model's anchoring.
//!
//! Under [`Normalization::BaseLevel`](super::Normalization::BaseLevel) that contrast is
//! simply the reduced parameter itself, so the numbers line up directly with what a
//! treatment-coded GLM reports.

use polars::prelude::*;

use super::fitting::Normalization;
use super::loss::LossFunction;

/// Largest reduced-basis design permitted. `X'WX` is dense at `p^2` and inverted in
/// `p^3`, so this bounds the work at roughly 200 MB and a few seconds.
const MAX_PARAMETERS: usize = 5000;

/// Standard errors and fit statistics for a fitted model.
#[derive(Debug, Clone)]
pub struct GLMInference {
    /// Standard error of each table's rows, matching the model's table layout.
    ///
    /// A row that is the anchoring reference has a standard error of exactly 0 — it
    /// is fixed by construction, not estimated. A row with no exposure, or one held
    /// as an offset, is `NaN`.
    pub standard_errors: Vec<Vec<f64>>,
    /// Estimated dispersion. Fixed at 1 for Poisson and Binomial; Pearson chi-squared
    /// over residual degrees of freedom for Gaussian, Gamma and Tweedie.
    pub dispersion: f64,
    /// Free parameters in the reduced basis, i.e. the model's rank.
    pub n_parameters: usize,
    /// Table rows whose parameter is a linear combination of others, as
    /// `(table_index, row_index)`. Their factors are still valid predictions, but the
    /// data cannot separate their effect from another parameter's, so they have no
    /// standard error. The usual causes are two tables keyed on the same feature and
    /// a completely separated level.
    pub aliased_rows: Vec<(usize, usize)>,
    /// Observations minus free parameters.
    pub df_residual: f64,
    /// Pearson chi-squared statistic.
    pub pearson_chi2: f64,
    /// Log-likelihood, where the family has a tractable one.
    ///
    /// `None` for Tweedie: its density is an infinite series with no closed form, so
    /// any number here would be a quietly substituted approximation.
    pub log_likelihood: Option<f64>,
    /// Akaike information criterion, counting the mean parameters only. `None` when
    /// the log-likelihood is unavailable.
    pub aic: Option<f64>,
    /// Bayesian information criterion, counting the mean parameters only. `None` when
    /// the log-likelihood is unavailable.
    pub bic: Option<f64>,
}

impl GLMInference {
    /// Wald z statistic for a table row: factor divided by its standard error.
    ///
    /// `None` where the standard error is zero (a reference level) or unavailable.
    pub fn z_value(&self, table_idx: usize, row_idx: usize, factor: f64) -> Option<f64> {
        let se = *self.standard_errors.get(table_idx)?.get(row_idx)?;
        if se > 0.0 && se.is_finite() {
            Some(factor / se)
        } else {
            None
        }
    }
}

/// Where a table row sits in the reduced, full-rank basis.
enum ReducedColumn {
    /// This row is the anchoring reference and carries no free parameter.
    Reference,
    /// This row maps to the given column of the reduced design.
    Column(usize),
    /// No exposure, or held fixed — excluded from inference entirely.
    Excluded,
}

/// Computes standard errors and fit statistics from a converged fit.
///
/// `eta` is the linear predictor including any offset, `means` the corresponding fitted
/// means. `matches[t][i]` is the row of table `t` that observation `i` fell in.
#[allow(clippy::too_many_arguments)]
pub fn compute_inference(
    loss_fn: &LossFunction,
    target: &[f64],
    weights: &[f64],
    means: &[f64],
    matches: &[Vec<Option<usize>>],
    factors: &[Vec<f64>],
    row_exposure: &[Vec<f64>],
    updatable: &[bool],
    normalization: Normalization,
) -> Result<GLMInference, PolarsError> {
    let n_obs = target.len();
    let n_tables = factors.len();

    // ---- 1. Lay out the reduced basis -----------------------------------------
    //
    // Column 0 is the intercept. Each updatable feature table contributes one column
    // per row except its reference row, whose effect the intercept absorbs.
    let mut layout: Vec<Vec<ReducedColumn>> = Vec::with_capacity(n_tables);
    let mut n_params = 1usize;

    for t in 0..n_tables {
        let n_rows = factors[t].len();
        let mut table_layout = Vec::with_capacity(n_rows);

        if t == 0 {
            // The intercept table itself is column 0.
            for r in 0..n_rows {
                table_layout.push(if r == 0 && updatable[0] {
                    ReducedColumn::Column(0)
                } else {
                    ReducedColumn::Excluded
                });
            }
        } else if !updatable[t] {
            for _ in 0..n_rows {
                table_layout.push(ReducedColumn::Excluded);
            }
        } else {
            let reference = reference_row(&row_exposure[t]);
            for r in 0..n_rows {
                if row_exposure[t][r] <= 0.0 {
                    table_layout.push(ReducedColumn::Excluded);
                } else if Some(r) == reference {
                    table_layout.push(ReducedColumn::Reference);
                } else {
                    table_layout.push(ReducedColumn::Column(n_params));
                    n_params += 1;
                }
            }
        }
        layout.push(table_layout);
    }

    if n_params > MAX_PARAMETERS {
        return Err(PolarsError::ComputeError(
            format!(
                "Standard errors need a {p}x{p} matrix ({p} free parameters), above the \
                 limit of {max}. Fit without inference, or reduce the number of levels.",
                p = n_params,
                max = MAX_PARAMETERS
            )
            .into(),
        ));
    }

    // ---- 2. Accumulate X'WX ----------------------------------------------------
    //
    // Every row of X is an indicator pattern: a 1 in the intercept column and a 1 in
    // at most one column per table. So the outer product is just a handful of
    // increments, and the whole accumulation is O(n * tables^2).
    let mut xtwx = vec![0.0f64; n_params * n_params];
    let mut pearson_chi2 = 0.0f64;
    let mut active_obs = 0usize;
    let mut cols: Vec<usize> = Vec::with_capacity(n_tables + 1);

    for i in 0..n_obs {
        let a = weights[i];
        if a <= 0.0 {
            continue;
        }
        active_obs += 1;

        let mu = means[i];
        let w = a * loss_fn.irls_weight(mu);

        // Pearson residual, for the dispersion estimate.
        let v = loss_fn.variance(mu);
        if v > 0.0 {
            pearson_chi2 += a * (target[i] - mu).powi(2) / v;
        }

        if !(w > 0.0) || !w.is_finite() {
            continue;
        }

        cols.clear();
        for t in 0..n_tables {
            if let Some(r) = matches[t][i] {
                if let ReducedColumn::Column(c) = layout[t][r] {
                    cols.push(c);
                }
            }
        }

        for (a_idx, &u) in cols.iter().enumerate() {
            xtwx[u * n_params + u] += w;
            for &v_col in cols.iter().skip(a_idx + 1) {
                xtwx[u * n_params + v_col] += w;
                xtwx[v_col * n_params + u] += w;
            }
        }
    }

    // ---- 3. Rank ---------------------------------------------------------------
    //
    // Rank deficiency here is information, not failure. Two tables keyed on the same
    // feature are collinear; a completely separated level has zero IRLS weight and
    // confounds with the intercept. Identify those parameters, set them aside, and
    // invert what remains so one aliased level does not cost every other level its
    // standard error.
    let aliased = find_aliased(&xtwx, n_params);
    let active: Vec<usize> = (0..n_params).filter(|j| !aliased[*j]).collect();
    let rank = active.len();

    // ---- 4. Dispersion ---------------------------------------------------------
    //
    // Degrees of freedom spend the model's rank, not its parameter count: an aliased
    // parameter costs nothing because it estimates nothing.
    let df_residual = active_obs as f64 - rank as f64;
    let dispersion = if loss_fn.has_fixed_dispersion() {
        1.0
    } else if df_residual > 0.0 {
        pearson_chi2 / df_residual
    } else {
        f64::NAN
    };

    // ---- 5. Invert -------------------------------------------------------------
    let mut compact_of = vec![None; n_params];
    for (c, &j) in active.iter().enumerate() {
        compact_of[j] = Some(c);
    }

    let k = rank;
    let mut compact = vec![0.0f64; k * k];
    for (ci, &i) in active.iter().enumerate() {
        for (cj, &j) in active.iter().enumerate() {
            compact[ci * k + cj] = xtwx[i * n_params + j];
        }
    }
    let compact_cov = invert_spd(&compact, k)?;

    // Scatter back into full-size coordinates; dropped columns stay zero and are
    // caught by the `compact_of` check when contrasts are formed.
    let mut cov = vec![0.0f64; n_params * n_params];
    for (ci, &i) in active.iter().enumerate() {
        for (cj, &j) in active.iter().enumerate() {
            cov[i * n_params + j] = compact_cov[ci * k + cj];
        }
    }

    // ---- 5. Standard error of the contrast each row actually represents ---------
    let mut standard_errors: Vec<Vec<f64>> = Vec::with_capacity(n_tables);
    let mut aliased_rows: Vec<(usize, usize)> = Vec::new();
    for t in 0..n_tables {
        let n_rows = factors[t].len();
        let mut ses = vec![f64::NAN; n_rows];

        // Under WeightedMean anchoring a reported factor is the level's parameter
        // minus the table's exposure-weighted average, so the contrast touches every
        // level of the table rather than just one.
        let shares: Option<Vec<f64>> = if t > 0 && normalization == Normalization::WeightedMean {
            let total: f64 = row_exposure[t].iter().sum();
            if total > 0.0 {
                Some(row_exposure[t].iter().map(|e| e / total).collect())
            } else {
                None
            }
        } else {
            None
        };

        for r in 0..n_rows {
            match layout[t][r] {
                ReducedColumn::Excluded => ses[r] = f64::NAN,
                ReducedColumn::Reference | ReducedColumn::Column(_) => {
                    // Build the contrast vector for this row, sparsely.
                    let mut contrast: Vec<(usize, f64)> = Vec::new();
                    if let ReducedColumn::Column(c) = layout[t][r] {
                        contrast.push((c, 1.0));
                    }
                    if let Some(p) = &shares {
                        for (s, share) in p.iter().enumerate() {
                            if *share == 0.0 {
                                continue;
                            }
                            if let ReducedColumn::Column(c) = layout[t][s] {
                                match contrast.iter_mut().find(|(idx, _)| *idx == c) {
                                    Some(entry) => entry.1 -= share,
                                    None => contrast.push((c, -share)),
                                }
                            }
                        }
                    }

                    if contrast.is_empty() {
                        // A pure reference level: fixed by construction, not estimated.
                        ses[r] = 0.0;
                        continue;
                    }

                    // A contrast touching an aliased parameter is not estimable.
                    if contrast.iter().any(|(c, w)| *w != 0.0 && compact_of[*c].is_none()) {
                        ses[r] = f64::NAN;
                        aliased_rows.push((t, r));
                        continue;
                    }

                    let mut quad = 0.0;
                    for (iu, &(u, cu)) in contrast.iter().enumerate() {
                        quad += cu * cu * cov[u * n_params + u];
                        for &(v, cv) in contrast.iter().skip(iu + 1) {
                            quad += 2.0 * cu * cv * cov[u * n_params + v];
                        }
                    }
                    ses[r] = if quad >= 0.0 {
                        (dispersion * quad).sqrt()
                    } else {
                        f64::NAN
                    };
                }
            }
        }
        standard_errors.push(ses);
    }

    // Without an anchor the factors are not identified, so a per-row standard error
    // would be describing a quantity the fit does not pin down.
    if normalization == Normalization::None {
        for ses in standard_errors.iter_mut() {
            for se in ses.iter_mut() {
                *se = f64::NAN;
            }
        }
    }

    // ---- 6. Likelihood-based statistics ---------------------------------------
    let log_likelihood = loss_fn.log_likelihood(target, means, weights, dispersion);
    let (aic, bic) = match log_likelihood {
        Some(llf) => {
            // Parameter count follows statsmodels: the mean parameters only, and the
            // rank rather than the nominal count. An estimated dispersion is not
            // counted, so these line up with what statsmodels and most GLM software
            // report.
            let k = rank as f64;
            (
                Some(-2.0 * llf + 2.0 * k),
                Some(-2.0 * llf + k * (active_obs as f64).ln()),
            )
        }
        None => (None, None),
    };

    Ok(GLMInference {
        standard_errors,
        aliased_rows,
        dispersion,
        n_parameters: rank,
        df_residual,
        pearson_chi2,
        log_likelihood,
        aic,
        bic,
    })
}

/// The row a table is anchored on: its first row carrying exposure.
fn reference_row(row_exposure: &[f64]) -> Option<usize> {
    row_exposure.iter().position(|e| *e > 0.0)
}

/// Finds parameters that are linear combinations of earlier ones.
///
/// A rank-deficient `X'WX` is not an error condition here — it is a statement about
/// the data. Two tables keyed on the same feature are collinear; a completely
/// separated level carries no information and confounds with the intercept. Either
/// way the estimable parameters still have perfectly good standard errors, and only
/// the aliased ones do not.
///
/// So rather than refusing to produce anything, run the Cholesky and record which
/// pivots collapse. Those parameters get no standard error and are reported by name;
/// the rest are inverted normally. This is what R does when it marks coefficients NA.
fn find_aliased(a: &[f64], n: usize) -> Vec<bool> {
    let mut aliased = vec![false; n];
    let mut l = vec![0.0f64; n * n];

    for i in 0..n {
        // Relative to this parameter's own information, so the threshold means the
        // same thing whether weights are exposure-years or fractions.
        let scale = a[i * n + i].abs();
        let tol = 1e-10 * scale.max(f64::MIN_POSITIVE);

        for j in 0..=i {
            if aliased[j] {
                continue; // this column contributes nothing
            }
            let mut sum = a[i * n + j];
            for k in 0..j {
                sum -= l[i * n + k] * l[j * n + k];
            }
            if i == j {
                if !(sum > tol) {
                    aliased[i] = true;
                    for k in 0..=i {
                        l[i * n + k] = 0.0;
                    }
                    break;
                }
                l[i * n + j] = sum.sqrt();
            } else {
                l[i * n + j] = sum / l[j * n + j];
            }
        }
    }

    aliased
}

/// Inverts a symmetric positive-definite matrix by Cholesky decomposition.
///
/// Callers must strip rank-deficient rows and columns first — see [`find_aliased`].
fn invert_spd(a: &[f64], n: usize) -> Result<Vec<f64>, PolarsError> {
    if n == 0 {
        return Ok(Vec::new());
    }

    // Cholesky: A = L L'
    let mut l = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut sum = a[i * n + j];
            for k in 0..j {
                sum -= l[i * n + k] * l[j * n + k];
            }
            if i == j {
                // Scale the tolerance to the matrix so it means the same thing whether
                // weights are counts or fractions of exposure.
                let tol = 1e-12 * a[i * n + i].abs().max(1.0);
                if sum <= tol {
                    return Err(PolarsError::ComputeError(
                        format!(
                            "Cannot compute standard errors: the design is singular at free \
                             parameter {} (pivot {:.3e}) even after removing aliased \
                             parameters. This should not happen; please report it.",
                            i, sum
                        )
                        .into(),
                    ));
                }
                l[i * n + j] = sum.sqrt();
            } else {
                l[i * n + j] = sum / l[j * n + j];
            }
        }
    }

    // Invert L (lower triangular), then A^-1 = L^-1' L^-1.
    let mut linv = vec![0.0f64; n * n];
    for i in 0..n {
        linv[i * n + i] = 1.0 / l[i * n + i];
        for j in 0..i {
            let mut sum = 0.0;
            for k in j..i {
                sum += l[i * n + k] * linv[k * n + j];
            }
            linv[i * n + j] = -sum / l[i * n + i];
        }
    }

    let mut inv = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut sum = 0.0;
            for k in i.max(j)..n {
                sum += linv[k * n + i] * linv[k * n + j];
            }
            inv[i * n + j] = sum;
            inv[j * n + i] = sum;
        }
    }

    Ok(inv)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inverts_a_known_matrix() {
        // [[4, 2], [2, 3]] has inverse [[0.375, -0.25], [-0.25, 0.5]]
        let a = vec![4.0, 2.0, 2.0, 3.0];
        let inv = invert_spd(&a, 2).unwrap();
        let expected = [0.375, -0.25, -0.25, 0.5];
        for (got, want) in inv.iter().zip(expected.iter()) {
            assert!((got - want).abs() < 1e-12, "got {:?}, want {:?}", inv, expected);
        }
    }

    #[test]
    fn inverse_times_original_is_identity() {
        let n = 4;
        // A well-conditioned SPD matrix: diagonally dominant.
        let mut a = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                a[i * n + j] = if i == j { 5.0 + i as f64 } else { 1.0 / (1 + i + j) as f64 };
            }
        }
        let inv = invert_spd(&a, n).unwrap();
        for i in 0..n {
            for j in 0..n {
                let mut s = 0.0;
                for k in 0..n {
                    s += a[i * n + k] * inv[k * n + j];
                }
                let want = if i == j { 1.0 } else { 0.0 };
                assert!((s - want).abs() < 1e-10, "({},{}) = {}", i, j, s);
            }
        }
    }

    #[test]
    fn collinear_column_is_detected_as_aliased() {
        // Second column is a multiple of the first.
        let a = vec![1.0, 2.0, 2.0, 4.0];
        assert_eq!(find_aliased(&a, 2), vec![false, true]);
    }

    #[test]
    fn full_rank_matrix_has_nothing_aliased() {
        let a = vec![4.0, 2.0, 2.0, 3.0];
        assert_eq!(find_aliased(&a, 2), vec![false, false]);
    }

    #[test]
    fn zero_information_column_is_aliased() {
        // A parameter with no weight behind it at all.
        let a = vec![4.0, 0.0, 0.0, 0.0];
        assert_eq!(find_aliased(&a, 2), vec![false, true]);
    }
}
