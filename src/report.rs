//! One artifact carrying what the model is, whether it works, and why to believe it.
//!
//! A fitted model, its diagnostics, its validation and its rating tables are the same
//! document viewed from four angles, and assembling them by hand is where a caller's
//! caveats get lost. A [`ModelReport`] holds them together, adds the plan that
//! produced them, and reduces the whole thing to one [`Verdict`] to branch on and one
//! `headline` to relay.
//!
//! It exists because "the user feels confident" is the wrong target. The right one is
//! that their confidence is *calibrated* — high when the model is good, appropriately
//! low when it is not. A report that omitted the caveats would score better on the
//! wrong target, so every finding travels with the numbers rather than beside them.
//!
//! Same artifact, three jobs: what an agent shows a person, what a filing appendix
//! needs, and what an implementation team reads off. Because the tables *are* the
//! model, those were always one document.

use crate::plan::{FittedModel, PlanCheck, ResolvedTerm};
use crate::validation::{Severity, Validation, ValidationOptions, Warning};
use polars::prelude::*;

/// Whether the model can be used, in one value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verdict {
    /// Nothing found that should stop it being used.
    Usable,
    /// Usable, but findings belong in whatever is reported to a person.
    UsableWithCaveats,
    /// Something is wrong enough that the model should not be used as it stands.
    NotUsable,
}

impl Verdict {
    pub fn as_str(&self) -> &'static str {
        match self {
            Verdict::Usable => "usable",
            Verdict::UsableWithCaveats => "usable_with_caveats",
            Verdict::NotUsable => "not_usable",
        }
    }
}

/// A finding, wherever it came from.
#[derive(Debug, Clone)]
pub struct Finding {
    pub severity: Severity,
    pub code: String,
    pub message: String,
    /// `plan` for something found before fitting, `fit` or `validation` after.
    pub stage: String,
}

/// How the fit itself went.
#[derive(Debug, Clone)]
pub struct FitSummary {
    pub converged: bool,
    pub iterations: usize,
    pub max_gradient: f64,
    pub deviance: f64,
    pub null_deviance: f64,
    pub pseudo_r2: f64,
    pub n_parameters: Option<usize>,
    pub dispersion: Option<f64>,
    pub aic: Option<f64>,
    pub bic: Option<f64>,
    pub table_conditioning: Option<f64>,
}

/// Everything about one fitted model, in one place.
pub struct ModelReport {
    pub family: String,
    pub target: String,
    pub table_names: Vec<String>,
    /// What the plan decided, one entry per table.
    pub resolved: Vec<ResolvedTerm>,
    /// The plan as JSON: the model's source code, so the report is reproducible.
    pub plan_json: String,
    /// A short digest of the plan, to tell two reports apart at a glance.
    pub fingerprint: String,
    /// How the fit went. `None` for a model that was loaded or converted rather than
    /// fitted here, so a report never implies a fit that did not happen.
    ///
    /// Named for the noun it is: `fit` is the verb on [`crate::plan::Plan`], and one
    /// word meaning both a thing and an action is a name a reader has to disambiguate.
    pub fit_summary: Option<FitSummary>,
    /// Present when the report was built against data.
    pub validation: Option<Validation>,
    /// Rating tables with coefficients, standard errors, status and relativities.
    pub rating_tables: Vec<DataFrame>,
    /// Everything worth saying, most severe first.
    pub findings: Vec<Finding>,
    pub verdict: Verdict,
    /// One sentence, written to be relayed to a person unchanged.
    pub headline: String,
}

impl FittedModel {
    /// Assemble the full report, validating against `data` when it is given.
    ///
    /// Pass the [`PlanCheck`] from before fitting to carry its findings through;
    /// several of them — a thin level, a near-aliased pair — are about the plan rather
    /// than the fit, and are not recoverable from the fitted model.
    pub fn report(
        &self,
        data: Option<&DataFrame>,
        options: &ValidationOptions,
    ) -> Result<ModelReport, PolarsError> {
        let validation = match data {
            Some(df) => Some(self.validate(df, options)?),
            None => None,
        };

        let mut findings: Vec<Finding> = Vec::new();
        // The check comes from the fit rather than from the caller. Several of its
        // findings are about the plan and cannot be recovered from the fitted model,
        // so asking for it back meant forgetting produced a cleaner report.
        if let Some(check) = self.check.as_ref() {
            for issue in &check.issues {
                findings.push(Finding {
                    severity: issue.severity,
                    code: issue.code.clone(),
                    message: issue.message.clone(),
                    stage: "plan".to_string(),
                });
            }
        }
        if let Some(validation) = validation.as_ref() {
            for warning in &validation.warnings {
                findings.push(Finding {
                    severity: warning.severity,
                    code: warning.code.clone(),
                    message: warning.message.clone(),
                    stage: stage_of(warning).to_string(),
                });
            }
        } else {
            findings.push(Finding {
                severity: Severity::Medium,
                code: "not_validated".to_string(),
                message: "The model has not been measured against held-out data. Validate it before treating it as ready for use.".to_string(),
                stage: "validation".to_string(),
            });
            if let Some(diagnostics) = self.diagnostics.as_ref().filter(|d| !d.converged) {
                // Without validation the convergence flag would otherwise go unreported.
                findings.push(Finding {
                    severity: Severity::High,
                    code: "not_converged".to_string(),
                    message: format!(
                        "The fit did not converge: it stopped after {} sweeps with a score \
                         of {:.2e}. The factors had not settled, so these are not the \
                         maximum-likelihood relativities.",
                        diagnostics.iterations, diagnostics.max_gradient
                    ),
                    stage: "fit".to_string(),
                });
            }
        }

        // A finding reported before the fit and again after it is one finding.
        findings.sort_by(|a, b| b.severity.cmp(&a.severity).then(a.code.cmp(&b.code)));
        findings.dedup_by(|a, b| a.code == b.code && a.message == b.message);

        let verdict = if findings.iter().any(|f| f.severity == Severity::High) {
            Verdict::NotUsable
        } else if findings.iter().any(|f| f.severity == Severity::Medium) {
            Verdict::UsableWithCaveats
        } else {
            Verdict::Usable
        };

        let fit = self.diagnostics.as_ref().map(|diagnostics| {
            let inference = diagnostics.inference.as_ref();
            FitSummary {
                converged: diagnostics.converged,
                iterations: diagnostics.iterations,
                max_gradient: diagnostics.max_gradient,
                deviance: diagnostics.deviance,
                null_deviance: diagnostics.null_deviance,
                pseudo_r2: diagnostics.pseudo_r2(),
                n_parameters: inference.map(|i| i.n_parameters),
                dispersion: inference.map(|i| i.dispersion),
                aic: inference.and_then(|i| i.aic),
                bic: inference.and_then(|i| i.bic),
                table_conditioning: diagnostics.table_conditioning,
            }
        });

        // A model with no plan has nothing to fingerprint but its tables, so the
        // fingerprint follows whatever describes it.
        let plan_json = match self.plan.as_ref() {
            Some(plan) => plan.to_json()?,
            None => String::new(),
        };
        let fingerprint_source = if plan_json.is_empty() {
            // A loaded or converted model has no plan, so fingerprint the artifact
            // itself. Remove volatile provenance first: two saves of the same model
            // must identify the same model.
            let mut workbook = self.to_workbook(Some(crate::workbook::Scale::Factor))?;
            workbook.manifest.created = None;
            for table in &mut workbook.manifest.tables {
                table.file = None;
            }
            workbook.to_json()?
        } else {
            plan_json.clone()
        };
        let fingerprint = fingerprint(&fingerprint_source);
        let headline = headline(verdict, fit.as_ref(), validation.as_ref(), &findings);

        Ok(ModelReport {
            family: self.family.clone(),
            target: self.target.clone().unwrap_or_else(|| "unknown".to_string()),
            table_names: self.table_names.clone(),
            resolved: self.resolved.clone(),
            plan_json,
            fingerprint,
            fit_summary: fit,
            validation,
            rating_tables: self.rating_tables()?,
            findings,
            verdict,
            headline,
        })
    }
}

/// Which stage a validation warning really belongs to, so the report can say where to
/// go and fix it.
fn stage_of(warning: &Warning) -> &'static str {
    match warning.code.as_str() {
        "not_converged" | "aliased_levels" | "unfitted_levels" => "fit",
        _ => "validation",
    }
}

/// A short, stable digest of the plan.
fn fingerprint(plan_json: &str) -> String {
    let digest = ring::digest::digest(&ring::digest::SHA256, plan_json.as_bytes());
    digest
        .as_ref()
        .iter()
        .take(6)
        .map(|b| format!("{:02x}", b))
        .collect()
}

fn headline(
    verdict: Verdict,
    fit: Option<&FitSummary>,
    validation: Option<&Validation>,
    findings: &[Finding],
) -> String {
    let worst = findings.first();
    // What the numbers rest on. This is stated whatever the verdict: a headline that
    // omitted it would let a caller relay an unvalidated model as though it had been
    // measured, which is exactly the false confidence the report exists to prevent.
    let quality = match validation {
        Some(v) => format!(
            "It explains {:.1}% of the deviance on the data it was measured against, \
             with actual over expected at {:.4} and a Gini of {:.3}",
            100.0 * v.pseudo_r2,
            v.ae_ratio,
            v.gini
        ),
        None => match fit {
            Some(fit) => format!(
                "It explains {:.1}% of the deviance in training, and has not been measured \
                 against held-out data",
                100.0 * fit.pseudo_r2
            ),
            // Loaded or converted: there is no training deviance to quote either.
            None => "It has not been measured against any data".to_string(),
        },
    };

    match verdict {
        Verdict::Usable => format!(
            "{}. Nothing was found that should stop it being used.",
            quality
        ),
        Verdict::UsableWithCaveats => format!(
            "{}. It is usable, with {} caveat{} to carry forward — the first is: {}",
            quality,
            findings.len(),
            if findings.len() == 1 { "" } else { "s" },
            worst.map(|f| f.message.as_str()).unwrap_or(""),
        ),
        // Lead with the problem, but still say what the numbers rest on.
        Verdict::NotUsable => format!(
            "This model should not be used as it stands. {} {}.",
            worst.map(|f| f.message.as_str()).unwrap_or(""),
            quality
        ),
    }
}

// ---------------------------------------------------------------- rendering

impl ModelReport {
    /// The report as Markdown, for showing to a person.
    ///
    /// Findings come first. A reader who stops after the first screen should already
    /// know whether to trust the numbers below it.
    pub fn to_markdown(&self) -> String {
        let mut out = String::new();

        out.push_str(&format!(
            "# {} model of `{}`\n\n",
            title_case(&self.family),
            self.target
        ));
        out.push_str(&format!(
            "**{}** · plan `{}`\n\n{}\n\n",
            match self.verdict {
                Verdict::Usable => "Usable",
                Verdict::UsableWithCaveats => "Usable with caveats",
                Verdict::NotUsable => "Not usable as it stands",
            },
            self.fingerprint,
            self.headline
        ));

        if !self.findings.is_empty() {
            out.push_str("## Findings\n\n");
            out.push_str("| Severity | Stage | Finding |\n|---|---|---|\n");
            for finding in &self.findings {
                out.push_str(&format!(
                    "| {} | {} | {} |\n",
                    finding.severity.as_str(),
                    finding.stage,
                    escape_pipes(&finding.message)
                ));
            }
            out.push('\n');
        }

        out.push_str("## The model\n\n");
        out.push_str("| Term | Kind | Rows | Parameters | Base |\n|---|---|---:|---:|---|\n");
        for term in &self.resolved {
            out.push_str(&format!(
                "| {} | {} | {} | {} | {} |\n",
                term.name,
                term.kind,
                term.rows,
                term.parameters,
                term.base_level.as_deref().unwrap_or("—")
            ));
        }
        out.push('\n');

        if let Some(fit) = self.fit_summary.as_ref() {
            out.push_str("## Fit\n\n");
            out.push_str(&format!(
                "| | |\n|---|---|\n\
                 | Converged | {} after {} sweeps (score {:.2e}) |\n\
                 | Deviance | {:.6} against a null of {:.6} |\n\
                 | Pseudo R-squared | {:.4} |\n",
                if fit.converged { "yes" } else { "**no**" },
                fit.iterations,
                fit.max_gradient,
                fit.deviance,
                fit.null_deviance,
                fit.pseudo_r2,
            ));
            if let Some(p) = fit.n_parameters {
                out.push_str(&format!("| Parameters | {} |\n", p));
            }
            if let Some(d) = fit.dispersion {
                out.push_str(&format!("| Dispersion | {:.6} |\n", d));
            }
            if let Some(a) = fit.aic {
                out.push_str(&format!("| AIC | {:.4} |\n", a));
            }
            if let Some(b) = fit.bic {
                out.push_str(&format!("| BIC | {:.4} |\n", b));
            }
            if let Some(c) = fit.table_conditioning {
                out.push_str(&format!(
                    "| Table conditioning | {:.2}{} |\n",
                    c,
                    if c > 10.0 {
                        " — above 10, expect a slow fit"
                    } else {
                        ""
                    }
                ));
            }
            out.push('\n');
        } else {
            out.push_str(
                "## Fit\n\nThis model was loaded or converted rather than fitted here, so \
                 there are no fit statistics to report.\n\n",
            );
        }

        if let Some(v) = self.validation.as_ref() {
            out.push_str("## Validation\n\n");
            out.push_str(&format!(
                "| | |\n|---|---|\n\
                 | Rows | {} scored of {} |\n\
                 | Actual / expected | {:.4} |\n\
                 | Gini | {:.4} |\n\
                 | Lift, top over bottom | {:.2}x |\n\
                 | Out-of-sample pseudo R-squared | {:.4} |\n\n",
                v.n_scored, v.n_rows, v.ae_ratio, v.gini, v.lift, v.pseudo_r2
            ));

            out.push_str("### Calibration, by equal-exposure bucket\n\n");
            out.push_str(&frame_to_markdown(&v.calibration, 20));
            out.push('\n');

            out.push_str("### Actual versus expected\n\n");
            for (name, frame) in self.table_names.iter().zip(v.actual_vs_expected.iter()) {
                if frame.height() <= 1 {
                    continue; // the intercept says nothing here
                }
                out.push_str(&format!("**{}**\n\n", name));
                out.push_str(&frame_to_markdown(frame, 30));
                out.push('\n');
            }
        }

        out.push_str("## Rating tables\n\n");
        for (name, frame) in self.table_names.iter().zip(self.rating_tables.iter()) {
            out.push_str(&format!("**{}**\n\n", name));
            out.push_str(&frame_to_markdown(frame, 30));
            out.push('\n');
        }

        if !self.plan_json.is_empty() {
            out.push_str("## Plan\n\n");
            out.push_str("The model's source code. Save it, edit it, re-run it.\n\n```json\n");
            out.push_str(&self.plan_json);
            out.push_str("\n```\n");
        }

        out
    }
}

fn title_case(text: &str) -> String {
    let mut chars = text.chars();
    match chars.next() {
        Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
        None => String::new(),
    }
}

fn escape_pipes(text: &str) -> String {
    text.replace('|', "\\|").replace('\n', " ")
}

/// Render a frame as a Markdown table, truncating long ones.
fn frame_to_markdown(df: &DataFrame, max_rows: usize) -> String {
    let names: Vec<String> = df
        .get_column_names()
        .iter()
        .map(|c| c.to_string())
        .collect();
    if names.is_empty() {
        return String::new();
    }

    let mut out = String::new();
    out.push_str(&format!("| {} |\n", names.join(" | ")));
    out.push_str(&format!(
        "|{}|\n",
        names.iter().map(|_| "---").collect::<Vec<_>>().join("|")
    ));

    let shown = df.height().min(max_rows);
    for row in 0..shown {
        let cells: Vec<String> = names
            .iter()
            .map(|name| match df.column(name) {
                Ok(column) => format_cell(column, row),
                Err(_) => String::from("—"),
            })
            .collect();
        out.push_str(&format!("| {} |\n", cells.join(" | ")));
    }
    if df.height() > shown {
        out.push_str(&format!(
            "\n_{} further rows not shown._\n",
            df.height() - shown
        ));
    }
    out
}

fn format_cell(column: &Column, row: usize) -> String {
    match column.dtype() {
        DataType::Float64 => match column.f64().ok().and_then(|c| c.get(row)) {
            Some(v) if v.is_infinite() => {
                if v > 0.0 {
                    "inf".into()
                } else {
                    "-inf".into()
                }
            }
            Some(v) if v.is_nan() => "—".into(),
            // Enough digits to be useful, few enough to read.
            Some(v) if v != 0.0 && v.abs() < 1e-4 => format!("{:.2e}", v),
            Some(v) => format!("{:.4}", v),
            None => "—".into(),
        },
        DataType::Int32 => column
            .i32()
            .ok()
            .and_then(|c| c.get(row))
            .map(|v| v.to_string())
            .unwrap_or_else(|| "—".into()),
        DataType::Int64 => column
            .i64()
            .ok()
            .and_then(|c| c.get(row))
            .map(|v| v.to_string())
            .unwrap_or_else(|| "—".into()),
        DataType::String => column
            .str()
            .ok()
            .and_then(|c| c.get(row))
            .map(escape_pipes)
            .unwrap_or_else(|| "—".into()),
        _ => match column.get(row) {
            Ok(value) => escape_pipes(&value.to_string()),
            Err(_) => "—".into(),
        },
    }
}
