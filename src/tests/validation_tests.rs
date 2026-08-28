//! Tests for holdout validation.
//!
//! The point of `validate` is that a caller who reads nothing but `warnings` is still
//! told when a model is broken. So most of these assert that a *specific* defect
//! produces a *specific* code at a *specific* severity — a warning set that quietly
//! stops firing is the failure this module exists to prevent.

#[cfg(test)]
mod validation_tests {
    use crate::glm::{fit_glm_with_diagnostics, GLMOptions};
    use crate::rating_model::RatingModel;
    use crate::validation::{validate, Severity, ValidationOptions};
    use polars::prelude::*;

    fn intercept(value: f64) -> DataFrame {
        DataFrame::new(vec![Series::new("Rating_Factor".into(), vec![value]).into()]).unwrap()
    }

    /// A categorical table over codes `0..k`, all factors zero unless given.
    fn cat_table(col: &str, k: i32, factors: Vec<f64>) -> DataFrame {
        DataFrame::new(vec![
            Series::new(col.into(), (0..k).collect::<Vec<i32>>()).into(),
            Series::new("Rating_Factor".into(), factors).into(),
        ])
        .unwrap()
    }

    /// Deterministic data: `region` in 0..4 with mean claim rates 1, 2, 3, 4 and unit
    /// exposure. `n` should be a multiple of 12.
    ///
    /// Claims vary *within* each region — the counts cycle `r, r+1, r+2` around a mean
    /// of `r+1` — so the group means stay exact while the residual deviance does not
    /// collapse to zero. Data with no within-group variation is not merely unrealistic
    /// here, it is degenerate: the fitter scales its convergence score by the total
    /// absolute score, and on an exactly saturated fit both that scale and the score
    /// itself fall to floating-point noise, so their ratio is meaningless. See
    /// `an_exactly_saturated_fit_is_a_degenerate_case` below.
    fn data(n: usize) -> DataFrame {
        let region: Vec<i32> = (0..n).map(|i| (i % 4) as i32).collect();
        let claims: Vec<f64> = (0..n)
            .map(|i| {
                let r = (i % 4) as f64;
                let variant = ((i / 4) % 3) as f64 - 1.0; // -1, 0, +1
                (r + 1.0) + variant
            })
            .collect();
        let exposure: Vec<f64> = vec![1.0; n];
        DataFrame::new(vec![
            Series::new("region".into(), region).into(),
            Series::new("claims".into(), claims).into(),
            Series::new("exposure".into(), exposure).into(),
        ])
        .unwrap()
    }

    fn opts() -> GLMOptions {
        GLMOptions {
            objective: "poisson".to_string(),
            max_iterations: 500,
            tolerance: 1e-12,
            ..Default::default()
        }
    }

    /// Fit a saturated region model, which reproduces each region's mean exactly.
    fn fitted(df: &DataFrame) -> (RatingModel, crate::glm::GLMDiagnostics) {
        let model = RatingModel::from_dataframes(
            vec![intercept(0.0), cat_table("region", 4, vec![0.0; 4])],
            "poisson",
            None,
            None,
        )
        .unwrap();
        fit_glm_with_diagnostics(&model, df, "claims", Some("exposure"), None, opts()).unwrap()
    }

    #[test]
    fn a_perfectly_calibrated_model_raises_nothing_serious() {
        let df = data(480);
        let (model, diag) = fitted(&df);
        let v = validate(
            &model,
            &df,
            "claims",
            Some("exposure"),
            None,
            "poisson",
            1.5,
            Some(&diag),
            &ValidationOptions::default(),
        )
        .unwrap();

        assert_eq!(v.n_rows, 480);
        assert_eq!(v.unmatched_rows, 0);
        assert_eq!(v.n_scored, 480);
        assert!(
            (v.ae_ratio - 1.0).abs() < 1e-9,
            "a saturated fit should be exactly calibrated, got {}",
            v.ae_ratio
        );
        assert!(
            (v.total_actual - v.total_expected).abs() < 1e-8,
            "actual {} expected {}",
            v.total_actual,
            v.total_expected
        );
        assert!(
            v.is_usable(),
            "unexpected high-severity warnings: {:?}",
            v.warnings_at_least(Severity::High)
                .iter()
                .map(|w| &w.code)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn totals_are_conserved_across_every_exhibit() {
        let df = data(480);
        let (model, diag) = fitted(&df);
        let v = validate(
            &model,
            &df,
            "claims",
            Some("exposure"),
            None,
            "poisson",
            1.5,
            Some(&diag),
            &ValidationOptions::default(),
        )
        .unwrap();

        // The calibration buckets partition the same exposure and the same actuals.
        let bucket_weight: f64 = v
            .calibration
            .column("weight")
            .unwrap()
            .f64()
            .unwrap()
            .sum()
            .unwrap();
        let bucket_actual: f64 = v
            .calibration
            .column("actual")
            .unwrap()
            .f64()
            .unwrap()
            .sum()
            .unwrap();
        assert!((bucket_weight - v.total_weight).abs() < 1e-9);
        assert!((bucket_actual - v.total_actual).abs() < 1e-9);

        // So does every table's actual-versus-expected exhibit.
        for (t, ave) in v.actual_vs_expected.iter().enumerate() {
            let a: f64 = ave.column("Actual").unwrap().f64().unwrap().sum().unwrap();
            let e: f64 = ave
                .column("Expected")
                .unwrap()
                .f64()
                .unwrap()
                .sum()
                .unwrap();
            let n: i64 = ave.column("N").unwrap().i64().unwrap().sum().unwrap();
            assert!(
                (a - v.total_actual).abs() < 1e-8,
                "table {} actual {} != total {}",
                t,
                a,
                v.total_actual
            );
            assert!((e - v.total_expected).abs() < 1e-8, "table {} expected", t);
            assert_eq!(n as usize, v.n_scored, "table {} row count", t);
        }
    }

    #[test]
    fn calibration_drift_is_detected_and_graded() {
        let df = data(480);
        let (model, _) = fitted(&df);

        // Shift the intercept on the log scale: predictions become 1.30x actual, so
        // actual over expected lands near 0.77 — well past the high threshold.
        let mut inflated = model.clone();
        let bumped = intercept(
            inflated.tables[0]
                .data
                .column("Rating_Factor")
                .unwrap()
                .f64()
                .unwrap()
                .get(0)
                .unwrap()
                + 0.3f64,
        );
        inflated.tables[0] = crate::rating_model::RatingTable::new(bumped, None);

        let v = validate(
            &inflated,
            &df,
            "claims",
            Some("exposure"),
            None,
            "poisson",
            1.5,
            None,
            &ValidationOptions::default(),
        )
        .unwrap();

        let drift = v
            .warnings
            .iter()
            .find(|w| w.code == "calibration_drift")
            .expect("a 30% rate-level error must be reported");
        assert_eq!(drift.severity, Severity::High);
        assert!(!v.is_usable());
        assert!(
            (v.ae_ratio - (-0.3f64).exp()).abs() < 1e-6,
            "ae_ratio {} should be exp(-0.3)",
            v.ae_ratio
        );

        // A small drift is reported, but only as a caveat.
        let mut nudged = model.clone();
        nudged.tables[0] = crate::rating_model::RatingTable::new(
            intercept(
                model.tables[0]
                    .data
                    .column("Rating_Factor")
                    .unwrap()
                    .f64()
                    .unwrap()
                    .get(0)
                    .unwrap()
                    + 0.05f64,
            ),
            None,
        );
        let v2 = validate(
            &nudged,
            &df,
            "claims",
            Some("exposure"),
            None,
            "poisson",
            1.5,
            None,
            &ValidationOptions::default(),
        )
        .unwrap();
        let w = v2
            .warnings
            .iter()
            .find(|w| w.code == "calibration_drift")
            .expect("a 5% drift is still worth saying");
        assert_eq!(w.severity, Severity::Medium);
    }

    #[test]
    fn unmatched_observations_are_reported_not_silently_dropped() {
        let df = data(480);
        let (model, diag) = fitted(&df);

        // A validation frame containing a region the model has never seen. Scoring
        // this through predict() alone yields NaN and no complaint.
        let mut region: Vec<i32> = (0..480).map(|i| (i % 4) as i32).collect();
        for i in 0..48 {
            region[i * 10] = 99;
        }
        let claims: Vec<f64> = (0..480)
            .map(|i| {
                let r = (i % 4) as f64;
                let variant = ((i / 4) % 3) as f64 - 1.0;
                (r + 1.0) + variant
            })
            .collect();
        let holdout = DataFrame::new(vec![
            Series::new("region".into(), region).into(),
            Series::new("claims".into(), claims).into(),
            Series::new("exposure".into(), vec![1.0; 480]).into(),
        ])
        .unwrap();

        let v = validate(
            &model,
            &holdout,
            "claims",
            Some("exposure"),
            None,
            "poisson",
            1.5,
            Some(&diag),
            &ValidationOptions::default(),
        )
        .unwrap();

        assert_eq!(v.n_rows, 480);
        assert_eq!(v.unmatched_rows, 48);
        assert_eq!(v.n_scored, 432);
        let w = v
            .warnings
            .iter()
            .find(|w| w.code == "unmatched_observations")
            .expect("unmatched rows must be reported");
        assert_eq!(w.severity, Severity::High);
        assert!(!v.is_usable());
        // The message has to carry the numbers, since it is what gets shown to a person.
        assert!(
            w.message.contains("48 of 480"),
            "message was: {}",
            w.message
        );
    }

    #[test]
    fn a_model_that_orders_risk_scores_a_positive_gini_and_a_flat_one_does_not() {
        let df = data(480);
        let (model, diag) = fitted(&df);
        let v = validate(
            &model,
            &df,
            "claims",
            Some("exposure"),
            None,
            "poisson",
            1.5,
            Some(&diag),
            &ValidationOptions::default(),
        )
        .unwrap();
        assert!(
            v.gini > 0.15,
            "a model separating rates 1..4 should order risk, gini was {}",
            v.gini
        );
        assert!(
            v.lift > 3.0,
            "top over bottom rate should be near 4, got {}",
            v.lift
        );

        // An intercept-only model predicts one number for everyone: no ordering at all.
        let flat = RatingModel::from_dataframes(
            vec![
                intercept((2.5f64).ln()),
                cat_table("region", 4, vec![0.0; 4]),
            ],
            "poisson",
            None,
            None,
        )
        .unwrap();
        let vf = validate(
            &flat,
            &df,
            "claims",
            Some("exposure"),
            None,
            "poisson",
            1.5,
            None,
            &ValidationOptions::default(),
        )
        .unwrap();
        assert!(
            vf.gini.abs() < 1e-9,
            "a constant prediction cannot order risk, gini was {}",
            vf.gini
        );
        assert!(
            vf.warnings.iter().any(|w| w.code == "no_discrimination"),
            "a model with no lift must say so"
        );
    }

    #[test]
    fn unseen_levels_are_flagged_as_extrapolation() {
        let df = data(480);
        // A model carrying a fifth region that the data never exercises.
        let model = RatingModel::from_dataframes(
            vec![intercept(0.0), cat_table("region", 5, vec![0.0; 5])],
            "poisson",
            None,
            None,
        )
        .unwrap();
        let (fit, diag) =
            fit_glm_with_diagnostics(&model, &df, "claims", Some("exposure"), None, opts())
                .unwrap();

        let v = validate(
            &fit,
            &df,
            "claims",
            Some("exposure"),
            None,
            "poisson",
            1.5,
            Some(&diag),
            &ValidationOptions::default(),
        )
        .unwrap();

        let w = v
            .warnings
            .iter()
            .find(|w| w.code == "unseen_levels")
            .expect("a level with no exposure must be reported");
        assert!(
            w.rows.contains(&(1, 4)),
            "row (1,4) is the unseen one, got {:?}",
            w.rows
        );
    }

    #[test]
    fn pseudo_r2_is_zero_for_an_intercept_only_model_and_positive_for_a_real_one() {
        let df = data(480);

        let flat = RatingModel::from_dataframes(
            vec![
                intercept((2.5f64).ln()),
                cat_table("region", 4, vec![0.0; 4]),
            ],
            "poisson",
            None,
            None,
        )
        .unwrap();
        let vf = validate(
            &flat,
            &df,
            "claims",
            Some("exposure"),
            None,
            "poisson",
            1.5,
            None,
            &ValidationOptions::default(),
        )
        .unwrap();
        assert!(
            vf.pseudo_r2.abs() < 1e-9,
            "predicting the overall mean explains none of the null deviance, got {}",
            vf.pseudo_r2
        );

        let (model, diag) = fitted(&df);
        let v = validate(
            &model,
            &df,
            "claims",
            Some("exposure"),
            None,
            "poisson",
            1.5,
            Some(&diag),
            &ValidationOptions::default(),
        )
        .unwrap();
        assert!(v.pseudo_r2 > 0.0);
        assert!(
            v.deviance < v.null_deviance,
            "the fitted model must beat the mean: {} vs {}",
            v.deviance,
            v.null_deviance
        );
    }

    #[test]
    fn a_model_worse_than_the_mean_is_called_out() {
        let df = data(480);
        // Relativities pointing the wrong way: region 0 loaded, region 3 discounted.
        let inverted = RatingModel::from_dataframes(
            vec![
                intercept((2.5f64).ln()),
                cat_table("region", 4, vec![0.9, 0.3, -0.3, -0.9]),
            ],
            "poisson",
            None,
            None,
        )
        .unwrap();
        let v = validate(
            &inverted,
            &df,
            "claims",
            Some("exposure"),
            None,
            "poisson",
            1.5,
            None,
            &ValidationOptions::default(),
        )
        .unwrap();
        assert!(v.pseudo_r2 < 0.0, "pseudo_r2 was {}", v.pseudo_r2);
        assert!(v
            .warnings
            .iter()
            .any(|w| w.code == "worse_than_intercept" && w.severity == Severity::High));
        assert!(
            v.gini < 0.0,
            "an inverted ordering should give a negative gini"
        );
    }

    #[test]
    fn warnings_come_back_most_severe_first() {
        let df = data(480);
        let inverted = RatingModel::from_dataframes(
            vec![
                intercept(1.5),
                cat_table("region", 4, vec![0.9, 0.3, -0.3, -0.9]),
            ],
            "poisson",
            None,
            None,
        )
        .unwrap();
        let v = validate(
            &inverted,
            &df,
            "claims",
            Some("exposure"),
            None,
            "poisson",
            1.5,
            None,
            &ValidationOptions::default(),
        )
        .unwrap();
        assert!(v.warnings.len() >= 2, "expected several findings here");
        for pair in v.warnings.windows(2) {
            assert!(
                pair[0].severity >= pair[1].severity,
                "warnings must be ordered by severity: {:?} then {:?}",
                pair[0].code,
                pair[1].code
            );
        }
    }

    #[test]
    fn calibration_buckets_carry_comparable_exposure() {
        let df = data(1200);
        let (model, diag) = fitted(&df);
        let v = validate(
            &model,
            &df,
            "claims",
            Some("exposure"),
            None,
            "poisson",
            1.5,
            Some(&diag),
            &ValidationOptions {
                bins: 4,
                ..Default::default()
            },
        )
        .unwrap();

        assert_eq!(v.calibration.height(), 4);
        let w = v.calibration.column("weight").unwrap().f64().unwrap();
        let target = v.total_weight / 4.0;
        for i in 0..4 {
            let got = w.get(i).unwrap();
            assert!(
                (got - target).abs() / target < 0.35,
                "bucket {} holds {} against a target of {}",
                i,
                got,
                target
            );
        }
        // Ordered by prediction, so predicted rates ascend across buckets.
        let mp = v
            .calibration
            .column("mean_predicted")
            .unwrap()
            .f64()
            .unwrap();
        for i in 1..4 {
            assert!(
                mp.get(i).unwrap() >= mp.get(i - 1).unwrap(),
                "buckets must be ordered by prediction"
            );
        }
    }

    #[test]
    fn a_non_convergent_or_aliased_fit_is_reported_from_the_diagnostics() {
        let df = data(480);
        // Two tables keyed on the same feature: exactly aliased.
        let model = RatingModel::from_dataframes(
            vec![
                intercept(0.0),
                cat_table("region", 4, vec![0.0; 4]),
                cat_table("region", 4, vec![0.0; 4]),
            ],
            "poisson",
            None,
            None,
        )
        .unwrap();
        let (fit, diag) =
            fit_glm_with_diagnostics(&model, &df, "claims", Some("exposure"), None, opts())
                .unwrap();

        let v = validate(
            &fit,
            &df,
            "claims",
            Some("exposure"),
            None,
            "poisson",
            1.5,
            Some(&diag),
            &ValidationOptions::default(),
        )
        .unwrap();

        assert!(
            v.warnings
                .iter()
                .any(|w| w.code == "aliased_levels" || w.code == "not_converged"),
            "two tables on the same feature must produce a finding, got {:?}",
            v.warnings.iter().map(|w| &w.code).collect::<Vec<_>>()
        );
    }

    #[test]
    fn bad_inputs_explain_the_repair() {
        let df = data(120);
        let (model, _) = fitted(&df);

        let missing = validate(
            &model,
            &df,
            "claims",
            Some("nope"),
            None,
            "poisson",
            1.5,
            None,
            &ValidationOptions::default(),
        );
        let msg = format!("{}", missing.unwrap_err());
        assert!(
            msg.contains("nope"),
            "message must name the column: {}",
            msg
        );
        assert!(
            msg.contains("Columns present"),
            "message must list what is available: {}",
            msg
        );

        let empty = DataFrame::new(vec![
            Series::new("region".into(), Vec::<i32>::new()).into(),
            Series::new("claims".into(), Vec::<f64>::new()).into(),
            Series::new("exposure".into(), Vec::<f64>::new()).into(),
        ])
        .unwrap();
        assert!(validate(
            &model,
            &empty,
            "claims",
            Some("exposure"),
            None,
            "poisson",
            1.5,
            None,
            &ValidationOptions::default(),
        )
        .is_err());
    }

    #[test]
    fn offsets_are_honoured_the_same_way_the_fit_used_them() {
        // Exposure as an offset rather than a weight: target is a count, and the
        // model has to reproduce counts rather than rates.
        let n = 400;
        let region: Vec<i32> = (0..n).map(|i| (i % 4) as i32).collect();
        let exposure: Vec<f64> = (0..n).map(|i| 1.0 + (i % 3) as f64).collect();
        // Vary claims within each region so the fit is not exactly saturated; see the
        // note on `data` above.
        let claims: Vec<f64> = region
            .iter()
            .zip(exposure.iter())
            .enumerate()
            .map(|(i, (&r, &e))| {
                let variant = ((i / 4) % 3) as f64 - 1.0;
                ((r + 1) as f64 + variant) * e
            })
            .collect();
        let log_exposure: Vec<f64> = exposure.iter().map(|e| e.ln()).collect();
        let df = DataFrame::new(vec![
            Series::new("region".into(), region).into(),
            Series::new("claims".into(), claims).into(),
            Series::new("log_exposure".into(), log_exposure).into(),
        ])
        .unwrap();

        let model = RatingModel::from_dataframes(
            vec![intercept(0.0), cat_table("region", 4, vec![0.0; 4])],
            "poisson",
            None,
            None,
        )
        .unwrap();
        let (fit, diag) =
            fit_glm_with_diagnostics(&model, &df, "claims", None, Some("log_exposure"), opts())
                .unwrap();

        let v = validate(
            &fit,
            &df,
            "claims",
            None,
            Some("log_exposure"),
            "poisson",
            1.5,
            Some(&diag),
            &ValidationOptions::default(),
        )
        .unwrap();

        assert!(
            (v.ae_ratio - 1.0).abs() < 1e-6,
            "an offset fit should still balance, got {}",
            v.ae_ratio
        );
        assert!(
            v.is_usable(),
            "{:?}",
            v.warnings.iter().map(|w| &w.code).collect::<Vec<_>>()
        );
    }

    /// Data with no within-group variation makes a saturated fit *exactly* right, and
    /// that is a degenerate input rather than an ideal one.
    ///
    /// The fitter reports convergence as the largest absolute score divided by the
    /// total absolute score. When the fit is exact both fall to floating-point noise,
    /// so the ratio is whatever the noise happens to be — here it lands near 2/3
    /// against a tolerance of 1e-12, and `converged` comes back false for a model that
    /// reproduces the data to 1e-14. Nothing downstream is wrong, but a caller reading
    /// `converged` alone would report a false alarm, so this is pinned to keep the
    /// behaviour visible. The guard in `max_abs_score` tests `total_abs > 0.0`, which
    /// tiny-but-nonzero noise passes.
    #[test]
    fn an_exactly_saturated_fit_is_a_degenerate_case() {
        let n = 400;
        let region: Vec<i32> = (0..n).map(|i| (i % 4) as i32).collect();
        // No within-region variation at all: every row is its group mean.
        let claims: Vec<f64> = region.iter().map(|&r| (r + 1) as f64).collect();
        let df = DataFrame::new(vec![
            Series::new("region".into(), region).into(),
            Series::new("claims".into(), claims).into(),
            Series::new("exposure".into(), vec![1.0; n]).into(),
        ])
        .unwrap();

        let (model, diag) = {
            let m = RatingModel::from_dataframes(
                vec![intercept(0.0), cat_table("region", 4, vec![0.0; 4])],
                "poisson",
                None,
                None,
            )
            .unwrap();
            fit_glm_with_diagnostics(&m, &df, "claims", Some("exposure"), None, opts()).unwrap()
        };

        // The fit is numerically exact ...
        assert!(
            diag.deviance.abs() < 1e-10,
            "deviance should be zero, got {}",
            diag.deviance
        );
        // ... and yet the score is nowhere near the tolerance.
        assert!(
            !diag.converged && diag.max_gradient > 1e-3,
            "expected the degenerate score, got converged={} score={:e}",
            diag.converged,
            diag.max_gradient
        );

        // Validation still measures the model correctly: it is perfectly calibrated,
        // and the only high-severity finding is the convergence flag it was handed.
        let v = validate(
            &model,
            &df,
            "claims",
            Some("exposure"),
            None,
            "poisson",
            1.5,
            Some(&diag),
            &ValidationOptions::default(),
        )
        .unwrap();
        assert!((v.ae_ratio - 1.0).abs() < 1e-9);
        let high: Vec<&str> = v
            .warnings_at_least(Severity::High)
            .iter()
            .map(|w| w.code.as_str())
            .collect();
        assert_eq!(high, vec!["not_converged"], "got {:?}", high);
    }

    /// A model predicting one number for everyone orders nothing, whatever order the
    /// rows happened to arrive in. Ties must advance as a block or the row layout
    /// leaks into the statistic.
    #[test]
    fn ties_contribute_no_discrimination() {
        let df = data(480);
        let flat = RatingModel::from_dataframes(
            vec![
                intercept((2.5f64).ln()),
                cat_table("region", 4, vec![0.0; 4]),
            ],
            "poisson",
            None,
            None,
        )
        .unwrap();
        let v = validate(
            &flat,
            &df,
            "claims",
            Some("exposure"),
            None,
            "poisson",
            1.5,
            None,
            &ValidationOptions::default(),
        )
        .unwrap();
        assert!(
            v.gini.abs() < 1e-12,
            "a constant prediction must score exactly zero, got {}",
            v.gini
        );
    }
}
