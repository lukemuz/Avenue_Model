//! Tests for the assembled model report.
//!
//! The report's job is to make confidence *calibrated* rather than high, so what is
//! pinned here is mostly that caveats survive: a model with something wrong must not
//! be able to produce a report that reads as fine, and the numbers must never appear
//! without the findings that qualify them.

#[cfg(test)]
mod report_tests {
    use crate::glm::GLMOptions;
    use crate::plan::{Breaks, Plan, Term};
    use crate::report::Verdict;
    use crate::validation::{Severity, ValidationOptions};
    use polars::prelude::*;

    /// Motor-shaped data whose true structure is *multiplicative*, matching the log
    /// link the model fits with.
    ///
    /// This matters more than it looks. An additive truth fitted multiplicatively is
    /// genuinely miscalibrated across the risk range — the two diverge most at the low
    /// end — and `validate` correctly says so. A fixture built that way cannot be used
    /// to test that a *sound* model reports as sound, because the model is not sound.
    fn motor(n: usize) -> DataFrame {
        let regions = ["north", "south", "east", "west"];
        let region: Vec<&str> = (0..n).map(|i| regions[i % 4]).collect();
        let age: Vec<i64> = (0..n).map(|i| 18 + ((i * 7) % 55) as i64).collect();
        let exposure: Vec<f64> = (0..n).map(|i| 0.5 + ((i % 4) as f64) / 4.0).collect();
        let claims: Vec<f64> = (0..n)
            .map(|i| {
                // log rate = log(0.8) + region effect + a step at age 45, which the
                // model's region factor and age bands can represent exactly.
                let region_multiplier = [1.0, 1.25, 1.5, 1.75][i % 4];
                let age_multiplier = if age[i] > 45 { 1.6 } else { 1.0 };
                let rate = 0.8 * region_multiplier * age_multiplier;
                // Within-cell variation with a mean of exactly 1, so the cell means
                // stay exact while the residual deviance does not collapse.
                let variation = [0.75, 1.0, 1.25][(i / 4) % 3];
                rate * exposure[i] * variation
            })
            .collect();
        DataFrame::new(vec![
            Series::new("region".into(), region).into(),
            Series::new("driver_age".into(), age).into(),
            Series::new("exposure".into(), exposure).into(),
            Series::new("claims".into(), claims).into(),
        ])
        .unwrap()
    }

    fn plan() -> Plan {
        Plan::frequency("exposure")
            .with(Term::banded(
                "driver_age",
                Breaks::explicit(vec![30.0, 45.0, 60.0]),
            ))
            .with(Term::categorical("region"))
    }

    fn options() -> GLMOptions {
        GLMOptions {
            max_iterations: 500,
            tolerance: 1e-10,
            ..Default::default()
        }
    }

    #[test]
    fn a_sound_model_reports_as_usable_and_carries_its_evidence() {
        let df = motor(480);
        let plan = plan();
        let check = plan.check(&df, "claims").unwrap();
        let fitted = plan.fit(&df, "claims", options()).unwrap();

        let report = fitted
            .report(Some(&df), Some(&check), &ValidationOptions::default())
            .unwrap();

        assert_eq!(report.verdict, Verdict::Usable, "findings: {:?}", report.findings);
        assert!(report.findings.is_empty(), "{:?}", report.findings);
        assert!(report.fit.converged);
        assert!(report.validation.is_some());

        // The evidence a reader needs is attached, not left to be fetched.
        assert_eq!(report.rating_tables.len(), 3);
        assert_eq!(report.resolved.len(), 3);
        assert!(report.fit.aic.is_some(), "model comparison needs AIC");
        assert!(!report.plan_json.is_empty());

        // The headline is one sentence meant to be relayed as written.
        assert!(
            report.headline.contains("actual over expected"),
            "headline was: {}",
            report.headline
        );
    }

    #[test]
    fn a_broken_model_cannot_produce_a_report_that_reads_as_fine() {
        let df = motor(480);
        let plan = plan();
        let fitted = plan.fit(&df, "claims", options()).unwrap();

        // Holdout containing a region the model has never seen.
        let mut regions: Vec<&str> = (0..480)
            .map(|i| ["north", "south", "east", "west"][i % 4])
            .collect();
        for i in 0..48 {
            regions[i * 10] = "atlantis";
        }
        let holdout = DataFrame::new(vec![
            Series::new("region".into(), regions).into(),
            Series::new(
                "driver_age".into(),
                (0..480).map(|i| 18 + ((i * 7) % 55) as i64).collect::<Vec<i64>>(),
            )
            .into(),
            Series::new("exposure".into(), vec![1.0f64; 480]).into(),
            Series::new(
                "claims".into(),
                (0..480).map(|i| (i % 3) as f64).collect::<Vec<f64>>(),
            )
            .into(),
        ])
        .unwrap();

        let report = fitted
            .report(Some(&holdout), None, &ValidationOptions::default())
            .unwrap();

        assert_eq!(report.verdict, Verdict::NotUsable);
        assert!(report
            .findings
            .iter()
            .any(|f| f.code == "unmatched_observations" && f.severity == Severity::High));

        // The headline leads with the problem rather than the metrics.
        assert!(
            report.headline.starts_with("This model should not be used"),
            "headline was: {}",
            report.headline
        );

        // And the rendered document says so before it shows a single number.
        let markdown = report.to_markdown();
        let verdict_at = markdown.find("Not usable as it stands").expect("verdict missing");
        let findings_at = markdown.find("## Findings").expect("findings missing");
        let fit_at = markdown.find("## Fit").expect("fit section missing");
        assert!(
            verdict_at < findings_at && findings_at < fit_at,
            "a reader who stops early must already know not to trust the numbers"
        );
    }

    #[test]
    fn findings_are_ordered_by_severity_and_never_duplicated_across_stages() {
        let df = motor(480);
        // `band` coarsens `driver_age`, so the plan check and the fit both have
        // something to say about the same redundancy.
        let band: Vec<i32> = (0..480)
            .map(|i| if 18 + ((i * 7) % 55) < 45 { 0 } else { 1 })
            .collect();
        let df = df.hstack(&[Series::new("age_band".into(), band).into()]).unwrap();

        let plan = Plan::frequency("exposure")
            .with(Term::banded("driver_age", Breaks::explicit(vec![44.0])))
            .with(Term::categorical("age_band"));
        let check = plan.check(&df, "claims").unwrap();
        let fitted = plan.fit(&df, "claims", options()).unwrap();
        let report = fitted
            .report(Some(&df), Some(&check), &ValidationOptions::default())
            .unwrap();

        for pair in report.findings.windows(2) {
            assert!(
                pair[0].severity >= pair[1].severity,
                "findings must be ordered by severity: {:?} then {:?}",
                pair[0].code,
                pair[1].code
            );
        }
        // The same finding reported before and after the fit is one finding.
        let mut seen = std::collections::HashSet::new();
        for finding in &report.findings {
            assert!(
                seen.insert((finding.code.clone(), finding.message.clone())),
                "duplicated finding: {}",
                finding.code
            );
        }
        // Findings say where to go and fix them.
        assert!(report
            .findings
            .iter()
            .all(|f| ["plan", "fit", "validation"].contains(&f.stage.as_str())));
    }

    #[test]
    fn a_report_without_data_still_reports_a_failed_fit() {
        let df = motor(240);
        let plan = plan();
        // One sweep is nowhere near enough to converge.
        let fitted = plan
            .fit(
                &df,
                "claims",
                GLMOptions {
                    max_iterations: 1,
                    tolerance: 1e-12,
                    ..Default::default()
                },
            )
            .unwrap();

        let report = fitted.report(None, None, &ValidationOptions::default()).unwrap();
        assert!(report.validation.is_none());
        assert_eq!(report.verdict, Verdict::NotUsable);
        assert!(
            report.findings.iter().any(|f| f.code == "not_converged"),
            "without validation the convergence flag must still be reported: {:?}",
            report.findings
        );
        assert!(
            report.headline.contains("has not been measured against held-out data"),
            "the report must not imply a validation it did not do: {}",
            report.headline
        );
    }

    #[test]
    fn the_fingerprint_tracks_the_plan_and_nothing_else() {
        let df = motor(240);

        let a = plan().fit(&df, "claims", options()).unwrap();
        let b = plan().fit(&df, "claims", options()).unwrap();
        let ra = a.report(None, None, &ValidationOptions::default()).unwrap();
        let rb = b.report(None, None, &ValidationOptions::default()).unwrap();
        assert_eq!(ra.fingerprint, rb.fingerprint, "the same plan must fingerprint alike");
        assert_eq!(ra.fingerprint.len(), 12);

        let different = Plan::frequency("exposure")
            .with(Term::banded("driver_age", Breaks::explicit(vec![40.0])))
            .with(Term::categorical("region"));
        let rc = different
            .fit(&df, "claims", options())
            .unwrap()
            .report(None, None, &ValidationOptions::default())
            .unwrap();
        assert_ne!(ra.fingerprint, rc.fingerprint, "a changed plan must fingerprint differently");
    }

    #[test]
    fn the_markdown_carries_the_plan_back_out_so_the_report_is_reproducible() {
        let df = motor(240);
        let plan = plan();
        let fitted = plan.fit(&df, "claims", options()).unwrap();
        let report = fitted.report(Some(&df), None, &ValidationOptions::default()).unwrap();
        let markdown = report.to_markdown();

        assert!(markdown.contains("## Plan"));
        assert!(markdown.contains("```json"));
        // The embedded plan must be the real thing, not a description of it.
        let start = markdown.find("```json").unwrap() + "```json\n".len();
        let end = markdown[start..].find("```").unwrap() + start;
        let embedded = markdown[start..end].trim();
        let recovered = Plan::from_json(embedded).expect("the embedded plan must parse");
        assert_eq!(recovered, plan, "the report must round-trip the plan it describes");
    }

    #[test]
    fn the_markdown_renders_every_section_and_reads_as_a_table() {
        let df = motor(480);
        let plan = plan();
        let check = plan.check(&df, "claims").unwrap();
        let fitted = plan.fit(&df, "claims", options()).unwrap();
        let report = fitted
            .report(Some(&df), Some(&check), &ValidationOptions::default())
            .unwrap();
        let markdown = report.to_markdown();

        for section in [
            "# Poisson model of `claims`",
            "## The model",
            "## Fit",
            "## Validation",
            "### Calibration, by equal-exposure bucket",
            "### Actual versus expected",
            "## Rating tables",
            "## Plan",
        ] {
            assert!(markdown.contains(section), "missing section: {}", section);
        }

        // Level text, not just codes: a report a person reads has to name the levels.
        assert!(markdown.contains("region_Level"), "level labels must be rendered");
        assert!(markdown.contains("north"), "the actual level names must appear");
        assert!(markdown.contains("Relativity"), "a log-link report quotes relativities");

        // Every table row must have the same cell count as its header, or the
        // Markdown collapses into unreadable soup.
        let mut header: Option<usize> = None;
        for line in markdown.lines() {
            if !line.starts_with('|') {
                header = None;
                continue;
            }
            let cells = line.matches('|').count() - line.matches("\\|").count();
            match header {
                None => header = Some(cells),
                Some(expected) => assert_eq!(
                    cells, expected,
                    "ragged markdown table row: {}",
                    line
                ),
            }
        }
    }

    #[test]
    fn infinite_band_edges_and_missing_standard_errors_render_readably() {
        let df = motor(240);
        let fitted = plan().fit(&df, "claims", options()).unwrap();
        let report = fitted.report(None, None, &ValidationOptions::default()).unwrap();
        let markdown = report.to_markdown();

        // The top band's bound is infinite, and the base level's standard error is
        // exactly zero while an unestimated one is NaN. None of those may print as
        // "NaN" or "inf" noise in a document a person reads.
        assert!(markdown.contains("inf"), "the unbounded top band should say so");
        assert!(!markdown.contains("NaN"), "NaN must render as an em dash");
    }
}
