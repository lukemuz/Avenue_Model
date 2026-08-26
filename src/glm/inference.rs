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

use crate::rating_model::variate_basis_params;

use super::fitting::Normalization;
use super::loss::LossFunction;
use super::matching::NO_MATCH;

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
    /// The fitted polynomial behind each variate table, one entry per variate table.
    pub variate_terms: Vec<VariateTerms>,
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

/// How a table row enters the reduced, full-rank design.
///
/// A step row is either the reference or its own column with coefficient 1. A variate
/// row loads on its table's `degree` shared columns, with the coefficients being the
/// powers of its value measured from the reference row's — so every row of a variate
/// table draws on the same handful of parameters, which is exactly what makes the table
/// cost `degree` degrees of freedom instead of one per row.
#[derive(Clone)]
enum ReducedColumn {
    /// This row is the anchoring reference and carries no free parameter.
    Reference,
    /// This row loads on the listed columns with the listed coefficients.
    Loadings(Vec<(usize, f64)>),
    /// No exposure, or held fixed — excluded from inference entirely.
    Excluded,
}

/// The fitted polynomial behind a variate table.
#[derive(Debug, Clone)]
pub struct VariateTerms {
    /// Which table this describes.
    pub table_index: usize,
    /// Polynomial degree.
    pub degree: usize,
    /// Coefficients on the raw scale, `[beta_1, ..., beta_degree]`, so that
    /// `factor[r] = constant + sum of beta_m * values[r]^m`. This is the form to write
    /// the fitted curve down in.
    pub coefficients: Vec<f64>,
    /// The same curve expressed on the rescaled basis the fit actually uses, where the
    /// driver is mapped onto `[-1, 1]`. Pairs with `standard_errors`.
    pub scaled_coefficients: Vec<f64>,
    /// Standard error of each degree's coefficient, in the rescaled basis the fit uses.
    ///
    /// The basis is triangular — the `m`th column involves no power above `m` — so the
    /// **top** degree's z statistic is the same whatever scale the lower terms are
    /// expressed on. That is the one that answers the question worth asking: does this
    /// curve need to bend, or would one fewer degree do?
    pub standard_errors: Vec<f64>,
}

impl VariateTerms {
    /// Wald z statistic for the top degree: is the highest power earning its place?
    ///
    /// Compare against a normal quantile as usual. The lower degrees' z statistics
    /// depend on how the basis is centred and are not reported for that reason; to
    /// judge them, refit at a lower degree and compare deviance.
    pub fn top_degree_z(&self) -> Option<f64> {
        let se = *self.standard_errors.last()?;
        let coef = *self.scaled_coefficients.last()?;
        (se > 0.0 && se.is_finite()).then(|| coef / se)
    }
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
    matches: &[Vec<u32>],
    factors: &[Vec<f64>],
    row_exposure: &[Vec<f64>],
    updatable: &[bool],
    variate_values: &[Option<(Vec<f64>, usize)>],
    normalization: Normalization,
) -> Result<GLMInference, PolarsError> {
    let n_obs = target.len();
    let n_tables = factors.len();

    // ---- 1. Lay out the reduced basis -----------------------------------------
    //
    // Column 0 is the intercept. A step table then contributes one column per row
    // except its reference row, whose effect the intercept absorbs. A variate table
    // contributes exactly one column however many rows it has.
    let mut layout: Vec<Vec<ReducedColumn>> = Vec::with_capacity(n_tables);
    let mut n_params = 1usize;

    for t in 0..n_tables {
        let n_rows = factors[t].len();
        let mut table_layout = Vec::with_capacity(n_rows);

        if t == 0 {
            // The intercept table itself is column 0.
            for r in 0..n_rows {
                table_layout.push(if r == 0 && updatable[0] {
                    ReducedColumn::Loadings(vec![(0, 1.0)])
                } else {
                    ReducedColumn::Excluded
                });
            }
        } else if !updatable[t] {
            for _ in 0..n_rows {
                table_layout.push(ReducedColumn::Excluded);
            }
        } else if let Some((values, degree)) = &variate_values[t] {
            // `degree` shared columns for the whole table. Loadings are the powers of
            // the row's value measured from the reference row's, so the reference row
            // reads as exactly zero — matching how the factors themselves are anchored.
            let reference = reference_row(&row_exposure[t]).unwrap_or(0);
            let Some((centre, scale)) = variate_basis_params(values) else {
                for _ in 0..n_rows {
                    table_layout.push(ReducedColumn::Excluded);
                }
                layout.push(table_layout);
                continue;
            };
            let first_col = n_params;
            n_params += degree;

            let powers = |v: f64| -> Vec<f64> {
                let u = (v - centre) / scale;
                let mut out = Vec::with_capacity(*degree);
                let mut p = 1.0;
                for _ in 0..*degree {
                    p *= u;
                    out.push(p);
                }
                out
            };
            let ref_powers = powers(values[reference]);

            for r in 0..n_rows {
                let row_powers = powers(values[r]);
                let loadings: Vec<(usize, f64)> = (0..*degree)
                    .map(|m| (first_col + m, row_powers[m] - ref_powers[m]))
                    .filter(|(_, coef)| *coef != 0.0)
                    .collect();
                table_layout.push(if loadings.is_empty() {
                    ReducedColumn::Reference
                } else {
                    ReducedColumn::Loadings(loadings)
                });
            }
        } else {
            let reference = reference_row(&row_exposure[t]);
            for r in 0..n_rows {
                if row_exposure[t][r] <= 0.0 {
                    table_layout.push(ReducedColumn::Excluded);
                } else if Some(r) == reference {
                    table_layout.push(ReducedColumn::Reference);
                } else {
                    table_layout.push(ReducedColumn::Loadings(vec![(n_params, 1.0)]));
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
    let mut cols: Vec<(usize, f64)> = Vec::with_capacity(n_tables + 1);

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
            let m = matches[t][i];
            if m != NO_MATCH {
                if let ReducedColumn::Loadings(loadings) = &layout[t][m as usize] {
                    cols.extend_from_slice(loadings);
                }
            }
        }

        for (a_idx, &(u, cu)) in cols.iter().enumerate() {
            xtwx[u * n_params + u] += w * cu * cu;
            for &(v_col, cv) in cols.iter().skip(a_idx + 1) {
                let contribution = w * cu * cv;
                xtwx[u * n_params + v_col] += contribution;
                xtwx[v_col * n_params + u] += contribution;
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
                ReducedColumn::Reference | ReducedColumn::Loadings(_) => {
                    // Build the contrast vector for this row, sparsely.
                    let mut contrast: Vec<(usize, f64)> = Vec::new();
                    if let ReducedColumn::Loadings(loadings) = &layout[t][r] {
                        contrast.extend_from_slice(loadings);
                    }
                    if let Some(p) = &shares {
                        for (s, share) in p.iter().enumerate() {
                            if *share == 0.0 {
                                continue;
                            }
                            if let ReducedColumn::Loadings(loadings) = &layout[t][s] {
                                for (c, coef) in loadings {
                                    let adjustment = share * coef;
                                    match contrast.iter_mut().find(|(idx, _)| idx == c) {
                                        Some(entry) => entry.1 -= adjustment,
                                        None => contrast.push((*c, -adjustment)),
                                    }
                                }
                            }
                        }
                    }
                    contrast.retain(|(_, w)| *w != 0.0);

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

    // ---- 7. Per-degree statistics for variate tables ---------------------------
    //
    // The reduced design gives these directly: a variate table owns `degree` adjacent
    // columns, so its coefficients are the corresponding entries of the fitted
    // parameter vector and their standard errors the diagonal of the covariance there.
    let mut variate_terms = Vec::new();
    for t in 0..n_tables {
        let Some((values, degree)) = &variate_values[t] else {
            continue;
        };
        let Some((centre, scale)) = variate_basis_params(values) else {
            continue;
        };

        // Find the table's first column by looking at any row that carries loadings.
        let Some(first_col) = layout[t].iter().find_map(|c| match c {
            ReducedColumn::Loadings(l) => l.first().map(|(col, _)| *col),
            _ => None,
        }) else {
            continue;
        };
        // Loadings are filtered of zero coefficients, so the lowest column seen is the
        // table's base column only if every degree appears somewhere. Take the minimum.
        let first_col = layout[t]
            .iter()
            .filter_map(|c| match c {
                ReducedColumn::Loadings(l) => l.iter().map(|(col, _)| *col).min(),
                _ => None,
            })
            .min()
            .unwrap_or(first_col);

        // Recover the fitted coefficients on the rescaled basis from the anchored
        // factors, which lie exactly on the polynomial by construction.
        let scaled_coefficients = fit_polynomial_to_factors(&factors[t], values, *degree, centre, scale);
        let Some(scaled_coefficients) = scaled_coefficients else {
            continue;
        };

        let standard_errors: Vec<f64> = (0..*degree)
            .map(|m| {
                let c = first_col + m;
                if compact_of.get(c).copied().flatten().is_none() {
                    return f64::NAN;
                }
                let var = dispersion * cov[c * n_params + c];
                if var >= 0.0 { var.sqrt() } else { f64::NAN }
            })
            .collect();

        variate_terms.push(VariateTerms {
            table_index: t,
            degree: *degree,
            coefficients: expand_to_raw_scale(&scaled_coefficients, centre, scale),
            scaled_coefficients,
            standard_errors,
        });
    }

    Ok(GLMInference {
        standard_errors,
        aliased_rows,
        dispersion,
        variate_terms,
        n_parameters: rank,
        df_residual,
        pearson_chi2,
        log_likelihood,
        aic,
        bic,
    })
}

/// Recovers a variate's coefficients on the rescaled basis from its fitted factors.
///
/// The factors lie exactly on the polynomial by construction, so this is a consistent
/// system and the least-squares solution is the exact one. Returns `[a_1, ..., a_d]`,
/// dropping the constant, which anchoring has already moved into the intercept.
fn fit_polynomial_to_factors(
    factors: &[f64],
    values: &[f64],
    degree: usize,
    centre: f64,
    scale: f64,
) -> Option<Vec<f64>> {
    let k = degree + 1;
    let mut ata = vec![0.0f64; k * k];
    let mut atb = vec![0.0f64; k];
    let mut basis = vec![0.0f64; k];

    for r in 0..values.len() {
        let u = (values[r] - centre) / scale;
        let mut p = 1.0;
        for b in basis.iter_mut() {
            *b = p;
            p *= u;
        }
        let f = factors[r];
        if !f.is_finite() {
            return None;
        }
        for a in 0..k {
            atb[a] += basis[a] * f;
            for b in 0..k {
                ata[a * k + b] += basis[a] * basis[b];
            }
        }
    }

    solve_spd(&ata, &atb, k).map(|mut c| {
        c.remove(0);
        c
    })
}

/// Expands `sum a_m ((v - centre)/scale)^m` into coefficients on powers of `v`.
fn expand_to_raw_scale(scaled: &[f64], centre: f64, scale: f64) -> Vec<f64> {
    let degree = scaled.len();
    let mut raw = vec![0.0f64; degree];
    for m in 1..=degree {
        let a_m = scaled[m - 1] / scale.powi(m as i32);
        for j in 1..=m {
            let mut binom = 1.0f64;
            for i in 0..j {
                binom = binom * (m - i) as f64 / (i + 1) as f64;
            }
            raw[j - 1] += a_m * binom * (-centre).powi((m - j) as i32);
        }
    }
    raw
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

/// Solves `A x = b` for a small symmetric positive-definite `A`, by Cholesky.
///
/// Returns `None` rather than an error when `A` is singular or nearly so: callers here
/// are in a position to skip the step and try again on better-conditioned weights.
pub fn solve_spd(a: &[f64], b: &[f64], n: usize) -> Option<Vec<f64>> {
    if n == 0 {
        return Some(Vec::new());
    }

    let mut l = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut sum = a[i * n + j];
            for k in 0..j {
                sum -= l[i * n + k] * l[j * n + k];
            }
            if i == j {
                let tol = 1e-13 * a[i * n + i].abs().max(f64::MIN_POSITIVE);
                if !(sum > tol) || !sum.is_finite() {
                    return None;
                }
                l[i * n + i] = sum.sqrt();
            } else {
                l[i * n + j] = sum / l[j * n + j];
            }
        }
    }

    // Forward substitution: L y = b
    let mut y = vec![0.0f64; n];
    for i in 0..n {
        let mut sum = b[i];
        for k in 0..i {
            sum -= l[i * n + k] * y[k];
        }
        y[i] = sum / l[i * n + i];
    }

    // Back substitution: L' x = y
    let mut x = vec![0.0f64; n];
    for i in (0..n).rev() {
        let mut sum = y[i];
        for k in (i + 1)..n {
            sum -= l[k * n + i] * x[k];
        }
        x[i] = sum / l[i * n + i];
    }

    x.iter().all(|v| v.is_finite()).then_some(x)
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
