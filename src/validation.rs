//! Holdout validation for a fitted rating model.
//!
//! [`validate`] answers "is this model any good, and is anything quietly wrong with
//! it" in **one call**, returning a structured verdict rather than something printed.
//!
//! The design constraint is that the caller may be an agent with no eyes. It cannot
//! look at a plot and notice the top decile is inverted, so every judgement this
//! module is capable of making, it makes — and reports as a [`Warning`] carrying both
//! a machine-readable `code` to branch on and a `message` written to be relayed to a
//! person verbatim. A caller that ignores everything except `warnings` should still
//! never tell a user a broken model is fine.
//!
//! Everything is computed in a single pass over the observation-to-row matches, which
//! is also what makes unmatched observations visible: scoring alone turns them into
//! `NaN` predictions that average into a metric without complaint.

use crate::glm::loss::LossFunction;
use crate::glm::matching::{precompute_all_matches, NO_MATCH};
use crate::glm::GLMDiagnostics;
use crate::rating_model::RatingModel;
use polars::prelude::*;

/// How much a finding should worry the caller.
///
/// `High` means the model should not be used as it stands. `Medium` means it is
/// usable but the caveat belongs in whatever is reported to a person. `Low` is
/// informational.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Severity {
    Low,
    Medium,
    High,
}

impl Severity {
    pub fn as_str(&self) -> &'static str {
        match self {
            Severity::Low => "low",
            Severity::Medium => "medium",
            Severity::High => "high",
        }
    }
}

/// One finding about a fitted model.
///
/// `code` is stable and meant to be matched on. `message` is meant to be shown to a
/// person unchanged — it states what is wrong and, where there is one, what to do
/// about it. `rows` locates the finding in the model as `(table_index, row_index)`
/// pairs, aligned with [`RatingModel::model_tables`].
#[derive(Debug, Clone)]
pub struct Warning {
    pub severity: Severity,
    pub code: String,
    pub message: String,
    pub rows: Vec<(usize, usize)>,
}

impl Warning {
    fn new(severity: Severity, code: &str, message: String) -> Self {
        Self {
            severity,
            code: code.to_string(),
            message,
            rows: Vec::new(),
        }
    }

    fn with_rows(mut self, rows: Vec<(usize, usize)>) -> Self {
        self.rows = rows;
        self
    }
}

/// Thresholds that decide when [`validate`] raises a warning.
///
/// The defaults are deliberately conversational rather than statistical: they are set
/// where a pricing actuary would start asking questions, not where a hypothesis test
/// would reject. Tighten them for a model heading to a filing.
#[derive(Debug, Clone)]
pub struct ValidationOptions {
    /// Equal-exposure buckets for the calibration and lift table.
    pub bins: usize,
    /// Overall actual-over-expected outside `1 +/- this` is a medium warning.
    pub calibration_tolerance: f64,
    /// Overall actual-over-expected outside `1 +/- this` is a high warning.
    pub calibration_tolerance_high: f64,
    /// A single bucket's actual-over-expected outside `1 +/- this` is a warning.
    pub bucket_tolerance: f64,
    /// A table row holding less than this share of total exposure is thinly
    /// estimated. Expressed as a fraction, so it scales with the dataset.
    pub thin_exposure_share: f64,
    /// A Gini below this means the model barely orders risk at all.
    pub min_gini: f64,
}

impl Default for ValidationOptions {
    fn default() -> Self {
        Self {
            bins: 10,
            calibration_tolerance: 0.02,
            calibration_tolerance_high: 0.10,
            bucket_tolerance: 0.10,
            thin_exposure_share: 0.001,
            min_gini: 0.05,
        }
    }
}

/// The verdict on a fitted model, measured against a dataset it did not fit.
///
/// Every field is data. Nothing here prints itself, and nothing is only available by
/// reading a formatted summary.
#[derive(Debug, Clone)]
pub struct Validation {
    /// Rows in the validation frame.
    pub n_rows: usize,
    /// Rows that failed to match some table and were therefore excluded. Non-zero
    /// means the remaining figures describe a subset of the data.
    pub unmatched_rows: usize,
    /// Rows actually scored: `n_rows - unmatched_rows`.
    pub n_scored: usize,

    /// Weighted deviance on this data.
    pub deviance: f64,
    /// Weighted deviance of an intercept-only model on this data.
    pub null_deviance: f64,
    /// Share of the null deviance the model explains here. Out-of-sample this can be
    /// negative, which means the model does worse than the overall mean.
    pub pseudo_r2: f64,

    /// Exposure-weighted totals. `total_actual / total_expected` is the headline
    /// calibration number.
    pub total_weight: f64,
    pub total_actual: f64,
    pub total_expected: f64,
    /// `total_actual / total_expected`. 1.0 is perfectly calibrated in aggregate.
    pub ae_ratio: f64,

    /// Equal-exposure buckets ordered by predicted value, with actual and expected in
    /// each. This is the calibration and lift exhibit in one table: columns are
    /// `bin`, `n`, `weight`, `mean_predicted`, `actual`, `expected`, `actual_rate`,
    /// `expected_rate`, `ae_ratio`.
    pub calibration: DataFrame,
    /// Actual rate in the top bucket over the bottom one. How much risk the model
    /// separates end to end.
    pub lift: f64,
    /// Weighted Gini on the predicted ordering. 0 is no discrimination.
    pub gini: f64,

    /// One frame per table, aligned with [`RatingModel::model_tables`]: the table's
    /// own columns plus `Exposure`, `Actual`, `Expected`, `AE_Ratio` and `N`.
    /// This is the actual-versus-expected exhibit, per rating factor.
    pub actual_vs_expected: Vec<DataFrame>,

    /// Everything the module found worth saying, most severe first.
    pub warnings: Vec<Warning>,
}

impl Validation {
    /// Warnings at or above a severity, most severe first.
    pub fn warnings_at_least(&self, severity: Severity) -> Vec<&Warning> {
        self.warnings
            .iter()
            .filter(|w| w.severity >= severity)
            .collect()
    }

    /// True when nothing was found that should stop the model being used.
    pub fn is_usable(&self) -> bool {
        !self.warnings.iter().any(|w| w.severity == Severity::High)
    }
}

/// Read a required `Float64` column into a plain vector, or default it.
fn column_or(df: &DataFrame, name: Option<&str>, default: f64) -> Result<Vec<f64>, PolarsError> {
    match name {
        None => Ok(vec![default; df.height()]),
        Some(col) => {
            let series = df.column(col).map_err(|_| {
                PolarsError::ColumnNotFound(
                    format!(
                        "Column '{}' not found in the validation data. Columns present: {}",
                        col,
                        df.get_column_names()
                            .iter()
                            .map(|c| c.as_str())
                            .collect::<Vec<_>>()
                            .join(", ")
                    )
                    .into(),
                )
            })?;
            let ca = series.f64().map_err(|_| {
                PolarsError::ComputeError(
                    format!(
                        "Column '{}' has dtype {:?}, but validation needs Float64. \
                         Cast it with df.with_columns(pl.col('{}').cast(pl.Float64)).",
                        col,
                        series.dtype(),
                        col
                    )
                    .into(),
                )
            })?;
            Ok(ca.into_iter().map(|v| v.unwrap_or(f64::NAN)).collect())
        }
    }
}

/// Validate a fitted model against data, in one pass.
///
/// `weight_col` and `offset_col` must describe the data the same way the fit did:
/// actual is `sum(weight * target)` and expected is `sum(weight * fitted mean)`, which
/// is the right definition whether exposure entered as a weight or as an offset.
///
/// `diagnostics` is optional. Supplying the ones from the fit lets the verdict include
/// convergence and aliasing, which cannot be recovered from the tables alone.
pub fn validate(
    model: &RatingModel,
    df: &DataFrame,
    target_col: &str,
    weight_col: Option<&str>,
    offset_col: Option<&str>,
    family: &str,
    tweedie_power: f64,
    diagnostics: Option<&GLMDiagnostics>,
    options: &ValidationOptions,
) -> Result<Validation, PolarsError> {
    if df.height() == 0 {
        return Err(PolarsError::ComputeError(
            "Validation data is empty. Supply at least one row.".into(),
        ));
    }
    if options.bins == 0 {
        return Err(PolarsError::ComputeError(
            "ValidationOptions.bins must be at least 1.".into(),
        ));
    }

    let mut loss = LossFunction::from_objective(family);
    if let LossFunction::Tweedie(_) = loss {
        loss = LossFunction::Tweedie(tweedie_power);
    }

    let target = column_or(df, Some(target_col), 0.0)?;
    let weight = column_or(df, weight_col, 1.0)?;
    let offset = column_or(df, offset_col, 0.0)?;

    if let Some(i) = weight.iter().position(|w| *w < 0.0) {
        return Err(PolarsError::ComputeError(
            format!(
                "Weight column '{}' has a negative value ({}) at row {}. Weights are \
                 exposures and cannot be negative.",
                weight_col.unwrap_or("<none>"),
                weight[i],
                i
            )
            .into(),
        ));
    }

    // One pass of matching gives both the linear predictor and the per-row grouping
    // the actual-versus-expected exhibits need.
    let matches = precompute_all_matches(model, df)?;
    let n_rows = df.height();
    let n_tables = model.tables.len();

    let factors: Vec<Vec<f64>> = model
        .tables
        .iter()
        .map(|t| {
            let ca = t.data.column("Rating_Factor")?.f64()?;
            Ok((0..ca.len()).map(|i| ca.get(i).unwrap_or(f64::NAN)).collect())
        })
        .collect::<Result<_, PolarsError>>()?;

    let mut eta = vec![0.0f64; n_rows];
    let mut scored = vec![true; n_rows];
    // Which table first failed to match, for a message that names it.
    let mut unmatched_by_table = vec![0usize; n_tables];

    for t in 0..n_tables {
        for i in 0..n_rows {
            let m = matches[t][i];
            if m == NO_MATCH {
                if scored[i] {
                    unmatched_by_table[t] += 1;
                }
                scored[i] = false;
            } else {
                eta[i] += factors[t][m as usize];
            }
        }
    }

    let mut warnings: Vec<Warning> = Vec::new();

    let unmatched_rows = scored.iter().filter(|s| !**s).count();
    let n_scored = n_rows - unmatched_rows;
    if n_scored == 0 {
        return Err(PolarsError::ComputeError(
            format!(
                "None of the {} validation rows matched every table, so nothing could \
                 be scored. Check that the validation data covers the same levels and \
                 bands as the training data.",
                n_rows
            )
            .into(),
        ));
    }
    if unmatched_rows > 0 {
        let worst = unmatched_by_table
            .iter()
            .enumerate()
            .max_by_key(|(_, c)| **c)
            .map(|(t, c)| format!(" The largest contributor is table {} with {} rows.", t, c))
            .unwrap_or_default();
        warnings.push(Warning::new(
            Severity::High,
            "unmatched_observations",
            format!(
                "{} of {} validation rows ({:.2}%) did not match every rating table and \
                 were excluded. Every figure below describes the remaining {} rows.{} \
                 Unmatched rows normally mean a level or band present in validation is \
                 absent from the model.",
                unmatched_rows,
                n_rows,
                100.0 * unmatched_rows as f64 / n_rows as f64,
                n_scored,
                worst
            ),
        ));
    }

    // Fitted means on the response scale, for scored rows only.
    let mut mu = vec![f64::NAN; n_rows];
    for i in 0..n_rows {
        if scored[i] {
            mu[i] = loss.inverse_link(eta[i] + offset[i]);
        }
    }

    let bad_target = (0..n_rows)
        .filter(|&i| scored[i] && !target[i].is_finite())
        .count();
    if bad_target > 0 {
        return Err(PolarsError::ComputeError(
            format!(
                "Target column '{}' has {} null or non-finite values among the scored \
                 rows. Drop or impute them before validating.",
                target_col, bad_target
            )
            .into(),
        ));
    }

    let mut total_weight = 0.0;
    let mut total_actual = 0.0;
    let mut total_expected = 0.0;
    for i in 0..n_rows {
        if scored[i] {
            total_weight += weight[i];
            total_actual += weight[i] * target[i];
            total_expected += weight[i] * mu[i];
        }
    }

    let nan_predictions = (0..n_rows)
        .filter(|&i| scored[i] && !mu[i].is_finite())
        .count();
    if nan_predictions > 0 {
        warnings.push(Warning::new(
            Severity::High,
            "non_finite_predictions",
            format!(
                "{} scored rows produced a non-finite prediction. This usually means a \
                 rating factor is NaN or infinite — check the fitted tables for rows \
                 flagged as having no exposure.",
                nan_predictions
            ),
        ));
    }

    // Deviance here, and against an intercept-only model on the same data.
    let scored_idx: Vec<usize> = (0..n_rows).filter(|&i| scored[i]).collect();
    let y_s: Vec<f64> = scored_idx.iter().map(|&i| target[i]).collect();
    let w_s: Vec<f64> = scored_idx.iter().map(|&i| weight[i]).collect();
    let mu_s: Vec<f64> = scored_idx.iter().map(|&i| mu[i]).collect();

    let deviance = loss.total_deviance(&y_s, &mu_s, &w_s);
    let null_mean = if total_weight > 0.0 {
        total_actual / total_weight
    } else {
        0.0
    };
    let null_means = vec![null_mean; scored_idx.len()];
    let null_deviance = loss.total_deviance(&y_s, &null_means, &w_s);
    let pseudo_r2 = if null_deviance > 0.0 {
        1.0 - deviance / null_deviance
    } else {
        0.0
    };

    let ae_ratio = if total_expected != 0.0 {
        total_actual / total_expected
    } else {
        f64::NAN
    };

    // ---- calibration and lift, on equal-exposure buckets ordered by prediction
    let (calibration, lift, gini) = calibration_table(&mu_s, &y_s, &w_s, options.bins)?;

    // ---- actual versus expected, per table
    let mut actual_vs_expected = Vec::with_capacity(n_tables);
    let mut thin_rows: Vec<(usize, usize)> = Vec::new();
    let mut unseen_rows: Vec<(usize, usize)> = Vec::new();

    for t in 0..n_tables {
        let n_table_rows = model.tables[t].data.height();
        let mut row_weight = vec![0.0; n_table_rows];
        let mut row_actual = vec![0.0; n_table_rows];
        let mut row_expected = vec![0.0; n_table_rows];
        let mut row_count = vec![0i64; n_table_rows];

        for &i in &scored_idx {
            let r = matches[t][i] as usize;
            row_weight[r] += weight[i];
            row_actual[r] += weight[i] * target[i];
            row_expected[r] += weight[i] * mu[i];
            row_count[r] += 1;
        }

        for r in 0..n_table_rows {
            if row_count[r] == 0 {
                unseen_rows.push((t, r));
            } else if total_weight > 0.0
                && row_weight[r] / total_weight < options.thin_exposure_share
            {
                thin_rows.push((t, r));
            }
        }

        let ratio: Vec<f64> = (0..n_table_rows)
            .map(|r| {
                if row_expected[r] != 0.0 {
                    row_actual[r] / row_expected[r]
                } else {
                    f64::NAN
                }
            })
            .collect();

        let mut data = model.tables[t].data.clone();
        data.with_column(Series::new("N".into(), row_count))?;
        data.with_column(Series::new("Exposure".into(), row_weight))?;
        data.with_column(Series::new("Actual".into(), row_actual))?;
        data.with_column(Series::new("Expected".into(), row_expected))?;
        data.with_column(Series::new("AE_Ratio".into(), ratio))?;
        actual_vs_expected.push(data);
    }

    // ---- warnings drawn from the fit itself
    if let Some(diag) = diagnostics {
        if !diag.converged {
            warnings.push(Warning::new(
                Severity::High,
                "not_converged",
                format!(
                    "The fit did not converge: it stopped after {} sweeps with a score of \
                     {:.2e} against a tolerance of the same scale. The factors had not \
                     settled, so these relativities are not the maximum-likelihood ones. \
                     Near-aliased tables are the usual cause — check table_conditioning \
                     and the reported correlations.",
                    diag.iterations, diag.max_gradient
                ),
            ));
        }
        if let Some(inference) = diag.inference.as_ref() {
            if !inference.aliased_rows.is_empty() {
                warnings.push(
                    Warning::new(
                        Severity::High,
                        "aliased_levels",
                        format!(
                            "{} table rows are aliased: their effect cannot be separated \
                             from another parameter's, so their factors are arbitrary and \
                             they carry no standard error. This is usually two tables \
                             keyed on the same driver, or a completely separated level. \
                             Drop one of the tables or combine the levels.",
                            inference.aliased_rows.len()
                        ),
                    )
                    .with_rows(inference.aliased_rows.clone()),
                );
            }
        }
        if !diag.unfitted_rows.is_empty() {
            warnings.push(
                Warning::new(
                    Severity::Medium,
                    "unfitted_levels",
                    format!(
                        "{} table rows saw no exposure during fitting and kept their \
                         starting factor. They were not estimated from data, so any \
                         business written into them will be priced off a placeholder.",
                        diag.unfitted_rows.len()
                    ),
                )
                .with_rows(diag.unfitted_rows.clone()),
            );
        }
    }

    // ---- warnings drawn from the validation data
    let drift = (ae_ratio - 1.0).abs();
    if ae_ratio.is_finite() && drift > options.calibration_tolerance {
        let severity = if drift > options.calibration_tolerance_high {
            Severity::High
        } else {
            Severity::Medium
        };
        warnings.push(Warning::new(
            severity,
            "calibration_drift",
            format!(
                "Overall actual over expected is {:.4}: the model predicts {:.2}% {} than \
                 this data shows in aggregate ({:.4} actual against {:.4} expected). \
                 Rebase the intercept before using these relativities to set a rate level.",
                ae_ratio,
                100.0 * drift,
                if ae_ratio > 1.0 { "less" } else { "more" },
                total_actual,
                total_expected
            ),
        ));
    }

    if let Ok(bad) = miscalibrated_bins(&calibration, options.bucket_tolerance) {
        if !bad.is_empty() {
            warnings.push(Warning::new(
                Severity::Medium,
                "bucket_miscalibration",
                format!(
                    "{} of {} equal-exposure buckets have actual over expected outside \
                     {:.0}%: buckets {}. The model is calibrated in aggregate but not \
                     across the risk range, which is the pattern a missing interaction or \
                     a mis-specified band produces.",
                    bad.len(),
                    options.bins,
                    100.0 * options.bucket_tolerance,
                    bad.iter()
                        .map(|b| b.to_string())
                        .collect::<Vec<_>>()
                        .join(", ")
                ),
            ));
        }
    }

    if gini.is_finite() && gini < options.min_gini {
        warnings.push(Warning::new(
            Severity::High,
            "no_discrimination",
            format!(
                "The Gini is {:.4} and the top bucket's actual rate is {:.2}x the bottom \
                 one's. The model barely separates risk on this data, so it will not \
                 support a rating plan even if it is well calibrated on average.",
                gini, lift
            ),
        ));
    }

    if !unseen_rows.is_empty() {
        warnings.push(
            Warning::new(
                Severity::Medium,
                "unseen_levels",
                format!(
                    "{} table rows received no exposure in this validation data, so \
                     nothing here tests them. Their factors are being extrapolated \
                     whenever business lands in them.",
                    unseen_rows.len()
                ),
            )
            .with_rows(unseen_rows),
        );
    }

    if !thin_rows.is_empty() {
        warnings.push(
            Warning::new(
                Severity::Low,
                "thin_levels",
                format!(
                    "{} table rows hold less than {:.2}% of total exposure each. Their \
                     actual-versus-expected figures are noisy and should not be read as \
                     evidence on their own.",
                    thin_rows.len(),
                    100.0 * options.thin_exposure_share
                ),
            )
            .with_rows(thin_rows),
        );
    }

    if pseudo_r2 < 0.0 {
        warnings.push(Warning::new(
            Severity::High,
            "worse_than_intercept",
            format!(
                "Out-of-sample deviance is worse than an intercept-only model on this \
                 data (pseudo R-squared {:.4}). The model is not generalising.",
                pseudo_r2
            ),
        ));
    }

    warnings.sort_by(|a, b| b.severity.cmp(&a.severity));

    Ok(Validation {
        n_rows,
        unmatched_rows,
        n_scored,
        deviance,
        null_deviance,
        pseudo_r2,
        total_weight,
        total_actual,
        total_expected,
        ae_ratio,
        calibration,
        lift,
        gini,
        actual_vs_expected,
        warnings,
    })
}

/// Bucket indices whose actual-over-expected sits outside `1 +/- tolerance`.
fn miscalibrated_bins(calibration: &DataFrame, tolerance: f64) -> Result<Vec<i64>, PolarsError> {
    let bins = calibration.column("bin")?.i64()?;
    let ratios = calibration.column("ae_ratio")?.f64()?;
    let mut out = Vec::new();
    for i in 0..ratios.len() {
        if let (Some(bin), Some(r)) = (bins.get(i), ratios.get(i)) {
            if r.is_finite() && (r - 1.0).abs() > tolerance {
                out.push(bin);
            }
        }
    }
    Ok(out)
}

/// Equal-exposure buckets ordered by prediction, plus lift and Gini.
///
/// Buckets are cut on cumulative *weight* rather than on count, which is what makes
/// the exhibit read the way a pricing actuary expects: every bucket carries the same
/// exposure, so the actual rates are directly comparable.
fn calibration_table(
    mu: &[f64],
    y: &[f64],
    w: &[f64],
    bins: usize,
) -> Result<(DataFrame, f64, f64), PolarsError> {
    let n = mu.len();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| mu[a].partial_cmp(&mu[b]).unwrap_or(std::cmp::Ordering::Equal));

    let total_weight: f64 = w.iter().sum();
    if total_weight <= 0.0 {
        return Err(PolarsError::ComputeError(
            "Total weight across the validation data is zero, so no exposure-weighted \
             calibration is possible."
                .into(),
        ));
    }

    let mut bin_of = vec![0usize; n];
    let mut cumulative = 0.0;
    for (rank, &i) in order.iter().enumerate() {
        // Assign on the exposure *entering* this observation so a single heavy row
        // cannot push the whole bucket boundary past itself.
        let bucket = ((cumulative / total_weight) * bins as f64).floor() as usize;
        bin_of[i] = bucket.min(bins - 1);
        cumulative += w[i];
        let _ = rank;
    }

    let mut b_n = vec![0i64; bins];
    let mut b_w = vec![0.0; bins];
    let mut b_actual = vec![0.0; bins];
    let mut b_expected = vec![0.0; bins];
    let mut b_mu_weight = vec![0.0; bins];
    for i in 0..n {
        let b = bin_of[i];
        b_n[b] += 1;
        b_w[b] += w[i];
        b_actual[b] += w[i] * y[i];
        b_expected[b] += w[i] * mu[i];
        b_mu_weight[b] += w[i] * mu[i];
    }

    let mean_predicted: Vec<f64> = (0..bins)
        .map(|b| {
            if b_w[b] > 0.0 {
                b_mu_weight[b] / b_w[b]
            } else {
                f64::NAN
            }
        })
        .collect();
    let actual_rate: Vec<f64> = (0..bins)
        .map(|b| {
            if b_w[b] > 0.0 {
                b_actual[b] / b_w[b]
            } else {
                f64::NAN
            }
        })
        .collect();
    let expected_rate: Vec<f64> = (0..bins)
        .map(|b| {
            if b_w[b] > 0.0 {
                b_expected[b] / b_w[b]
            } else {
                f64::NAN
            }
        })
        .collect();
    let ae: Vec<f64> = (0..bins)
        .map(|b| {
            if b_expected[b] != 0.0 {
                b_actual[b] / b_expected[b]
            } else {
                f64::NAN
            }
        })
        .collect();

    // Lift across the populated range, so an empty tail bucket cannot fabricate one.
    let first = (0..bins).find(|&b| b_w[b] > 0.0);
    let last = (0..bins).rev().find(|&b| b_w[b] > 0.0);
    let lift = match (first, last) {
        (Some(f), Some(l)) if f != l && actual_rate[f] != 0.0 => actual_rate[l] / actual_rate[f],
        _ => f64::NAN,
    };

    let df = DataFrame::new(vec![
        Series::new("bin".into(), (0..bins as i64).collect::<Vec<i64>>()).into(),
        Series::new("n".into(), b_n).into(),
        Series::new("weight".into(), b_w).into(),
        Series::new("mean_predicted".into(), mean_predicted).into(),
        Series::new("actual".into(), b_actual).into(),
        Series::new("expected".into(), b_expected).into(),
        Series::new("actual_rate".into(), actual_rate).into(),
        Series::new("expected_rate".into(), expected_rate).into(),
        Series::new("ae_ratio".into(), ae).into(),
    ])?;

    Ok((df, lift, weighted_gini(&order, mu, y, w)))
}

/// Weighted Gini of the predicted ordering.
///
/// The Lorenz curve plots cumulative share of exposure against cumulative share of
/// actual response, walking observations from lowest predicted value to highest. A
/// model that orders risk correctly holds the curve below the diagonal, so the area
/// falls short of a half and the Gini is positive. Zero is no ordering ability;
/// negative means the ordering is inverted.
///
/// **Ties are advanced as one block.** Observations the model scores identically carry
/// no information about which is riskier, so the order the sort happened to put them
/// in must not become discrimination the model does not have. Stepping through them
/// one at a time lets whatever order the input arrived in bleed into the statistic —
/// a model predicting a single constant would score a small non-zero Gini purely from
/// how the rows were laid out. Advancing the whole tie group at once draws the
/// straight chord that a tie actually implies.
fn weighted_gini(order: &[usize], mu: &[f64], y: &[f64], w: &[f64]) -> f64 {
    let total_weight: f64 = w.iter().sum();
    let total_actual: f64 = order.iter().map(|&i| w[i] * y[i]).sum();
    if total_weight <= 0.0 || total_actual == 0.0 {
        return f64::NAN;
    }

    let mut cum_w = 0.0;
    let mut cum_a = 0.0;
    let mut prev_x = 0.0;
    let mut prev_y = 0.0;
    let mut area = 0.0;

    let mut start = 0usize;
    while start < order.len() {
        // Extend over every observation sharing this predicted value.
        let value = mu[order[start]];
        let mut end = start + 1;
        while end < order.len() && mu[order[end]] == value {
            end += 1;
        }
        for &i in &order[start..end] {
            cum_w += w[i];
            cum_a += w[i] * y[i];
        }
        let x = cum_w / total_weight;
        let y_cum = cum_a / total_actual;
        area += (x - prev_x) * (y_cum + prev_y) / 2.0;
        prev_x = x;
        prev_y = y_cum;
        start = end;
    }
    1.0 - 2.0 * area
}
