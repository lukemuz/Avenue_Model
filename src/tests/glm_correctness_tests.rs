//! Correctness tests for GLM fitting on rating tables.
//!
//! These assert on *numbers*, not on "did it return two tables". Two kinds of checks:
//!
//!   1. **Closed-form.** A saturated model - an intercept plus one factor with every
//!      level free - has an exact maximum-likelihood solution for every exponential
//!      family: the fitted mean of each level equals that level's weighted mean of `y`.
//!      No iteration or reference implementation needed to know the right answer.
//!
//!   2. **External reference.** Fits pinned from statsmodels in
//!      `glm_reference_data.rs`. Avenue's table parameterisation carries an intercept
//!      *and* every level, so raw coefficients are not comparable to statsmodels'
//!      treatment coding. We compare the two quantities that are invariant to the
//!      parameterisation: fitted means per row, and level contrasts within a table.

#[cfg(test)]
mod glm_correctness_tests {
    use crate::glm::{fit_glm, GLMOptions};
    use crate::rating_model::RatingModel;
    use crate::tests::glm_reference_data as refdata;
    use polars::prelude::*;

    // Backfitting converges linearly, so give it room and a tight stopping rule;
    // the assertions below are what actually define "converged".
    const FIT_TOL: f64 = 1e-13;
    const MAX_ITER: usize = 2000;

    /// Tolerance for agreement with statsmodels. Both implementations solve the same
    /// optimisation to near machine precision, so this is generous.
    const REF_TOL: f64 = 1e-7;

    // ---------------------------------------------------------------- helpers

    fn intercept_table() -> DataFrame {
        DataFrame::new(vec![
            Series::new("Rating_Factor".into(), vec![0.0]).into()
        ]).unwrap()
    }

    /// A step table over `col`: `bounds` are inclusive upper bounds, ascending,
    /// last one normally `f64::INFINITY`.
    fn factor_table(col: &str, bounds: &[f64]) -> DataFrame {
        DataFrame::new(vec![
            Series::new(col.into(), bounds.to_vec()).into(),
            Series::new("Rating_Factor".into(), vec![0.0; bounds.len()]).into(),
        ]).unwrap()
    }

    fn options(objective: &str, tweedie_power: f64) -> GLMOptions {
        GLMOptions {
            objective: objective.to_string(),
            max_iterations: MAX_ITER,
            tolerance: FIT_TOL,
            verbose: false,
            tweedie_power,
            ..Default::default()
        }
    }

    fn rating_factors(model: &RatingModel, table_idx: usize) -> Vec<f64> {
        let ca = model.tables[table_idx].data.column("Rating_Factor").unwrap().f64().unwrap();
        (0..ca.len()).map(|i| ca.get(i).unwrap()).collect()
    }

    /// Level contrasts relative to level 0 - invariant to how the intercept is split.
    fn contrasts(model: &RatingModel, table_idx: usize) -> Vec<f64> {
        let f = rating_factors(model, table_idx);
        let base = f[0];
        f.iter().map(|v| v - base).collect()
    }

    fn predictions(model: &RatingModel, df: &DataFrame) -> Vec<f64> {
        let s = model.predict(df).unwrap();
        let ca = s.f64().unwrap();
        (0..ca.len()).map(|i| ca.get(i).unwrap()).collect()
    }

    fn assert_all_close(actual: &[f64], expected: &[f64], tol: f64, what: &str) {
        assert_eq!(actual.len(), expected.len(), "{}: length mismatch", what);
        let mut worst = (0usize, 0.0f64);
        for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
            // Relative where the magnitude warrants it, absolute near zero.
            let denom = e.abs().max(1.0);
            let err = (a - e).abs() / denom;
            if err > worst.1 {
                worst = (i, err);
            }
        }
        assert!(
            worst.1 <= tol,
            "{}: worst mismatch at index {} - got {:.12e}, expected {:.12e} (rel err {:.3e} > {:.3e})",
            what, worst.0, actual[worst.0], expected[worst.0], worst.1, tol
        );
    }

    /// Weighted mean of `y` within each level of `level_of`.
    fn weighted_group_means(level_of: &[usize], y: &[f64], w: &[f64], n_levels: usize) -> Vec<f64> {
        let mut num = vec![0.0; n_levels];
        let mut den = vec![0.0; n_levels];
        for i in 0..y.len() {
            num[level_of[i]] += w[i] * y[i];
            den[level_of[i]] += w[i];
        }
        (0..n_levels).map(|j| num[j] / den[j]).collect()
    }

    // ------------------------------------------------- 1. closed-form checks

    /// For ANY exponential-family GLM, a saturated one-factor model must reproduce
    /// each level's weighted mean of `y`. This holds for every link, which makes it
    /// the sharpest possible check that the link scale is being handled correctly.
    fn saturated_one_factor(objective: &str, tweedie_power: f64, y: Vec<f64>) {
        let x: Vec<f64> = vec![1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0];
        let w: Vec<f64> = vec![1.0, 2.0, 0.5, 3.0, 1.0, 1.5, 0.25, 2.0, 1.0];
        assert_eq!(y.len(), 9, "{}: fixture must have 9 rows", objective);

        let level_of: Vec<usize> = x.iter().map(|v| (*v as usize) - 1).collect();
        let expected_levels = weighted_group_means(&level_of, &y, &w, 3);
        let expected: Vec<f64> = level_of.iter().map(|&j| expected_levels[j]).collect();

        let df = DataFrame::new(vec![
            Series::new("x".into(), x).into(),
            Series::new("y".into(), y).into(),
            Series::new("w".into(), w).into(),
        ]).unwrap();

        let model = RatingModel::from_dataframes(
            vec![intercept_table(), factor_table("x", &[1.0, 2.0, f64::INFINITY])],
            objective, None, None,
        ).unwrap();

        let fitted = fit_glm(&model, &df, "y", Some("w"), None,
                             options(objective, tweedie_power)).unwrap();

        assert_all_close(
            &predictions(&fitted, &df), &expected, 1e-9,
            &format!("saturated {} - fitted mean must equal weighted group mean", objective),
        );
    }

    #[test]
    fn saturated_gaussian_reproduces_group_means() {
        saturated_one_factor("gaussian", 1.5,
            vec![3.0, 5.0, 4.0, 30.0, 50.0, 40.0, -2.0, -6.0, -1.0]);
    }

    #[test]
    fn saturated_poisson_reproduces_group_means() {
        saturated_one_factor("poisson", 1.5,
            vec![2.0, 4.0, 3.0, 10.0, 14.0, 12.0, 30.0, 40.0, 35.0]);
    }

    #[test]
    fn saturated_gamma_reproduces_group_means() {
        saturated_one_factor("gamma", 1.5,
            vec![7.5, 9.8, 8.1, 22.8, 30.2, 25.0, 70.0, 92.6, 81.0]);
    }

    #[test]
    fn saturated_tweedie_reproduces_group_means() {
        saturated_one_factor("tweedie", 1.5,
            vec![0.0, 4.0, 3.0, 0.0, 14.0, 12.0, 30.0, 0.0, 35.0]);
    }

    #[test]
    fn saturated_binary_reproduces_group_rates() {
        // Deliberately no all-0 or all-1 level: separation is covered separately below.
        saturated_one_factor("binary", 1.5,
            vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0]);
    }

    /// A level where every observation is a 1 has no finite MLE - the coefficient
    /// diverges. The fit must stay finite and drive the probability to the boundary
    /// rather than producing NaN or overflowing.
    #[test]
    fn binary_separation_stays_finite() {
        let df = DataFrame::new(vec![
            Series::new("x".into(), vec![1.0, 1.0, 1.0, 2.0, 2.0, 2.0]).into(),
            Series::new("y".into(), vec![1.0, 1.0, 1.0, 0.0, 1.0, 0.0]).into(),
        ]).unwrap();

        let model = RatingModel::from_dataframes(
            vec![intercept_table(), factor_table("x", &[1.0, f64::INFINITY])],
            "binary", None, None,
        ).unwrap();
        let fitted = fit_glm(&model, &df, "y", None, None, options("binary", 1.5)).unwrap();

        let preds = predictions(&fitted, &df);
        assert!(preds.iter().all(|p| p.is_finite()), "separation produced non-finite predictions: {:?}", preds);
        assert!(preds.iter().all(|p| (0.0..=1.0).contains(p)), "predictions outside [0,1]: {:?}", preds);
        assert!(preds[0] > 0.999, "separated level should push to the boundary, got {}", preds[0]);
        // The unseparated level still has a well-defined rate of 1/3.
        assert!((preds[3] - 1.0 / 3.0).abs() < 1e-6, "expected 1/3, got {}", preds[3]);
        for f in rating_factors(&fitted, 1) {
            assert!(f.is_finite(), "rating factor diverged to {}", f);
        }
    }

    /// A table whose only row matches every observation is fully confounded with the
    /// intercept. Whatever the split between them, the fitted mean must be the overall
    /// weighted mean of `y`. This fails outright if the intercept is never updated.
    #[test]
    fn intercept_is_fitted_not_frozen() {
        let y = vec![2.0, 4.0, 3.0, 10.0, 14.0, 12.0, 30.0, 40.0, 35.0];
        let w = vec![1.0, 2.0, 0.5, 3.0, 1.0, 1.5, 0.25, 2.0, 1.0];
        let x = vec![1.0; 9];

        let grand: f64 = y.iter().zip(&w).map(|(a, b)| a * b).sum::<f64>()
            / w.iter().sum::<f64>();

        let df = DataFrame::new(vec![
            Series::new("x".into(), x).into(),
            Series::new("y".into(), y).into(),
            Series::new("w".into(), w).into(),
        ]).unwrap();

        for objective in ["gaussian", "poisson", "gamma", "tweedie"] {
            let model = RatingModel::from_dataframes(
                vec![intercept_table(), factor_table("x", &[f64::INFINITY])],
                objective, None, None,
            ).unwrap();
            let fitted = fit_glm(&model, &df, "y", Some("w"), None,
                                 options(objective, 1.5)).unwrap();
            let preds = predictions(&fitted, &df);
            assert_all_close(&preds, &vec![grand; 9], 1e-9,
                &format!("{} - single-level model must fit the grand weighted mean", objective));
        }

        // Same check for logit, which needs a target in [0, 1].
        let y_bin = vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0];
        let w_bin = vec![1.0, 2.0, 0.5, 3.0, 1.0, 1.5, 0.25, 2.0, 1.0];
        let grand_bin: f64 = y_bin.iter().zip(&w_bin).map(|(a, b)| a * b).sum::<f64>()
            / w_bin.iter().sum::<f64>();
        let df_bin = DataFrame::new(vec![
            Series::new("x".into(), vec![1.0; 9]).into(),
            Series::new("y".into(), y_bin).into(),
            Series::new("w".into(), w_bin).into(),
        ]).unwrap();
        let model = RatingModel::from_dataframes(
            vec![intercept_table(), factor_table("x", &[f64::INFINITY])],
            "binary", None, None,
        ).unwrap();
        let fitted = fit_glm(&model, &df_bin, "y", Some("w"), None,
                             options("binary", 1.5)).unwrap();
        assert_all_close(&predictions(&fitted, &df_bin), &vec![grand_bin; 9], 1e-9,
            "binary - single-level model must fit the grand weighted rate");
    }

    /// Two observations of weight 1 must be indistinguishable from one of weight 2.
    #[test]
    fn weights_behave_like_replication() {
        let x_rep = vec![1.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 3.0];
        let y_rep = vec![3.0, 3.0, 7.0, 11.0, 11.0, 4.0, 9.0, 9.0];

        let x_w = vec![1.0, 1.0, 2.0, 3.0, 3.0];
        let y_w = vec![3.0, 7.0, 11.0, 4.0, 9.0];
        let w_w = vec![2.0, 1.0, 2.0, 1.0, 2.0];

        let probe = DataFrame::new(vec![
            Series::new("x".into(), vec![1.0, 2.0, 3.0]).into(),
        ]).unwrap();

        for objective in ["gaussian", "poisson", "gamma"] {
            let tables = || vec![intercept_table(), factor_table("x", &[1.0, 2.0, f64::INFINITY])];

            let df_rep = DataFrame::new(vec![
                Series::new("x".into(), x_rep.clone()).into(),
                Series::new("y".into(), y_rep.clone()).into(),
            ]).unwrap();
            let m_rep = RatingModel::from_dataframes(tables(), objective, None, None).unwrap();
            let f_rep = fit_glm(&m_rep, &df_rep, "y", None, None,
                                options(objective, 1.5)).unwrap();

            let df_w = DataFrame::new(vec![
                Series::new("x".into(), x_w.clone()).into(),
                Series::new("y".into(), y_w.clone()).into(),
                Series::new("w".into(), w_w.clone()).into(),
            ]).unwrap();
            let m_w = RatingModel::from_dataframes(tables(), objective, None, None).unwrap();
            let f_w = fit_glm(&m_w, &df_w, "y", Some("w"), None,
                              options(objective, 1.5)).unwrap();

            assert_all_close(
                &predictions(&f_w, &probe), &predictions(&f_rep, &probe), 1e-9,
                &format!("{} - weight 2 must equal two replicated rows", objective),
            );
        }
    }

    // ------------------------------------------------ 1b. identifiability

    /// The tables are the deliverable, so the same data must always produce the same
    /// tables. An intercept plus every level of every factor is over-parameterised, so
    /// without an anchor the fit can settle anywhere along that flat direction -
    /// different starting values would give different, equally valid, tables.
    #[test]
    fn fitted_tables_are_independent_of_starting_values() {
        use refdata::PoissonTwoFactor as C;

        let df = DataFrame::new(vec![
            Series::new("x1".into(), C::X1.to_vec()).into(),
            Series::new("x2".into(), C::X2.to_vec()).into(),
            Series::new("y".into(), C::Y.to_vec()).into(),
            Series::new("w".into(), C::WEIGHT.to_vec()).into(),
        ]).unwrap();

        let build = |seed: f64| {
            let bump = |df: DataFrame, k: f64| {
                let h = df.height();
                let mut d = df;
                let f: Vec<f64> = (0..h).map(|i| k * (i as f64 + 1.0)).collect();
                d.with_column(Series::new("Rating_Factor".into(), f)).unwrap();
                d
            };
            RatingModel::from_dataframes(
                vec![
                    bump(intercept_table(), seed),
                    bump(factor_table("x1", &refdata::X1_BOUNDS), -seed),
                    bump(factor_table("x2", &refdata::X2_BOUNDS), seed * 0.5),
                ],
                "poisson", None, None,
            ).unwrap()
        };

        let a = fit_glm(&build(0.0), &df, "y", Some("w"), None, options("poisson", 1.5)).unwrap();
        let b = fit_glm(&build(0.7), &df, "y", Some("w"), None, options("poisson", 1.5)).unwrap();

        for t in 0..3 {
            assert_all_close(&rating_factors(&a, t), &rating_factors(&b, t), 1e-9,
                &format!("table {} must not depend on starting values", t));
        }

        // The default anchor puts every feature table's base level at zero, so the
        // remaining factors read directly as relativities.
        for t in 1..3 {
            let f = rating_factors(&a, t);
            assert!(f[0].abs() < 1e-12, "table {} base level should be 0, got {}", t, f[0]);
        }
    }

    /// Anchoring changes how the fit is split between the intercept and the tables,
    /// never what it predicts.
    #[test]
    fn normalization_does_not_change_predictions() {
        use crate::glm::Normalization;
        use refdata::GammaTwoFactor as C;

        let df = DataFrame::new(vec![
            Series::new("x1".into(), C::X1.to_vec()).into(),
            Series::new("x2".into(), C::X2.to_vec()).into(),
            Series::new("y".into(), C::Y.to_vec()).into(),
            Series::new("w".into(), C::WEIGHT.to_vec()).into(),
        ]).unwrap();

        let tables = || vec![
            intercept_table(),
            factor_table("x1", &refdata::X1_BOUNDS),
            factor_table("x2", &refdata::X2_BOUNDS),
        ];

        let fit_with = |norm: Normalization| {
            let model = RatingModel::from_dataframes(tables(), "gamma", None, None).unwrap();
            let mut opts = options("gamma", 1.5);
            opts.normalization = norm;
            fit_glm(&model, &df, "y", Some("w"), None, opts).unwrap()
        };

        let base = fit_with(Normalization::BaseLevel);
        let wmean = fit_with(Normalization::WeightedMean);
        let none = fit_with(Normalization::None);

        assert_all_close(&predictions(&wmean, &df), &predictions(&base, &df), 1e-9,
            "WeightedMean anchoring must not move predictions");
        assert_all_close(&predictions(&none, &df), &predictions(&base, &df), 1e-9,
            "unanchored fit must not move predictions");

        assert_all_close(&contrasts(&wmean, 1), &contrasts(&base, 1), 1e-9,
            "contrasts are invariant to the anchor");

        // Under WeightedMean the exposure-weighted average factor of each table is zero.
        for (t, x) in [(1usize, &C::X1[..]), (2usize, &C::X2[..])] {
            let f = rating_factors(&wmean, t);
            let mut num = 0.0;
            let mut den = 0.0;
            for i in 0..x.len() {
                let level = x[i] as usize - 1;
                num += C::WEIGHT[i] * f[level];
                den += C::WEIGHT[i];
            }
            assert!((num / den).abs() < 1e-9,
                "table {} exposure-weighted mean factor should be 0, got {:.3e} from {:?}",
                t, num / den, f);
        }
    }

    // -------------------------------------------- 2. statsmodels reference fits

    struct RefCase<'a> {
        name: &'a str,
        objective: &'a str,
        tweedie_power: f64,
        x1: &'a [f64],
        x2: &'a [f64],
        y: &'a [f64],
        weight: &'a [f64],
        offset: Option<&'a [f64]>,
        mu: &'a [f64],
        x1_contrasts: &'a [f64],
        x2_contrasts: &'a [f64],
        deviance: f64,
    }

    fn check_reference(case: RefCase) {
        let mut cols: Vec<Column> = vec![
            Series::new("x1".into(), case.x1.to_vec()).into(),
            Series::new("x2".into(), case.x2.to_vec()).into(),
            Series::new("y".into(), case.y.to_vec()).into(),
            Series::new("w".into(), case.weight.to_vec()).into(),
        ];
        if let Some(off) = case.offset {
            cols.push(Series::new("off".into(), off.to_vec()).into());
        }
        let df = DataFrame::new(cols).unwrap();

        let model = RatingModel::from_dataframes(
            vec![
                intercept_table(),
                factor_table("x1", &refdata::X1_BOUNDS),
                factor_table("x2", &refdata::X2_BOUNDS),
            ],
            case.objective, None, None,
        ).unwrap();

        let (fitted, diag) = crate::glm::fit_glm_with_diagnostics(
            &model, &df, "y", Some("w"),
            case.offset.map(|_| "off"),
            options(case.objective, case.tweedie_power),
        ).unwrap();

        assert!(diag.converged, "{} - fit did not converge in {} sweeps", case.name, diag.iterations);
        assert!(
            diag.deviance <= diag.null_deviance + 1e-9,
            "{} - fit deviance {} exceeds null deviance {}",
            case.name, diag.deviance, diag.null_deviance
        );
        // Deviance must decrease monotonically; the log-link coordinate solve is exact,
        // so any increase would mean the update rule is wrong.
        for w in diag.deviance_history.windows(2) {
            assert!(
                w[1] <= w[0] + 1e-9 * w[0].abs().max(1.0),
                "{} - deviance rose from {} to {}", case.name, w[0], w[1]
            );
        }

        // Predictions from the model itself do not know about the offset column, so
        // for offset cases we rebuild the linear predictor and add it back.
        let mu: Vec<f64> = match case.offset {
            None => predictions(&fitted, &df),
            Some(off) => fitted
                .predict_linear(&df)
                .unwrap()
                .iter()
                .zip(off.iter())
                .map(|(eta, o)| (eta + o).exp())
                .collect(),
        };

        assert_all_close(&mu, case.mu, REF_TOL,
            &format!("{} - fitted means vs statsmodels", case.name));
        assert_all_close(&contrasts(&fitted, 1), case.x1_contrasts, REF_TOL,
            &format!("{} - x1 level contrasts vs statsmodels", case.name));
        assert_all_close(&contrasts(&fitted, 2), case.x2_contrasts, REF_TOL,
            &format!("{} - x2 level contrasts vs statsmodels", case.name));
        assert_all_close(&[diag.deviance], &[case.deviance], REF_TOL,
            &format!("{} - deviance vs statsmodels", case.name));
    }

    #[test]
    fn matches_statsmodels_gaussian() {
        use refdata::GaussianTwoFactor as C;
        check_reference(RefCase {
            name: "gaussian/identity", objective: "gaussian", tweedie_power: 1.5,
            x1: &C::X1, x2: &C::X2, y: &C::Y, weight: &C::WEIGHT, offset: None,
            mu: &C::MU, x1_contrasts: &C::X1_CONTRASTS, x2_contrasts: &C::X2_CONTRASTS, deviance: C::DEVIANCE,
        });
    }

    #[test]
    fn matches_statsmodels_poisson() {
        use refdata::PoissonTwoFactor as C;
        check_reference(RefCase {
            name: "poisson/log", objective: "poisson", tweedie_power: 1.5,
            x1: &C::X1, x2: &C::X2, y: &C::Y, weight: &C::WEIGHT, offset: None,
            mu: &C::MU, x1_contrasts: &C::X1_CONTRASTS, x2_contrasts: &C::X2_CONTRASTS, deviance: C::DEVIANCE,
        });
    }

    #[test]
    fn matches_statsmodels_poisson_with_offset() {
        use refdata::PoissonOffset as C;
        check_reference(RefCase {
            name: "poisson/log + offset", objective: "poisson", tweedie_power: 1.5,
            x1: &C::X1, x2: &C::X2, y: &C::Y, weight: &C::WEIGHT, offset: Some(&C::OFFSET),
            mu: &C::MU, x1_contrasts: &C::X1_CONTRASTS, x2_contrasts: &C::X2_CONTRASTS, deviance: C::DEVIANCE,
        });
    }

    #[test]
    fn matches_statsmodels_gamma() {
        use refdata::GammaTwoFactor as C;
        check_reference(RefCase {
            name: "gamma/log", objective: "gamma", tweedie_power: 1.5,
            x1: &C::X1, x2: &C::X2, y: &C::Y, weight: &C::WEIGHT, offset: None,
            mu: &C::MU, x1_contrasts: &C::X1_CONTRASTS, x2_contrasts: &C::X2_CONTRASTS, deviance: C::DEVIANCE,
        });
    }

    #[test]
    fn matches_statsmodels_binary() {
        use refdata::BinaryTwoFactor as C;
        check_reference(RefCase {
            name: "binomial/logit", objective: "binary", tweedie_power: 1.5,
            x1: &C::X1, x2: &C::X2, y: &C::Y, weight: &C::WEIGHT, offset: None,
            mu: &C::MU, x1_contrasts: &C::X1_CONTRASTS, x2_contrasts: &C::X2_CONTRASTS, deviance: C::DEVIANCE,
        });
    }

    #[test]
    fn matches_statsmodels_tweedie() {
        use refdata::TweedieTwoFactor as C;
        check_reference(RefCase {
            name: "tweedie(1.5)/log", objective: "tweedie", tweedie_power: 1.5,
            x1: &C::X1, x2: &C::X2, y: &C::Y, weight: &C::WEIGHT, offset: None,
            mu: &C::MU, x1_contrasts: &C::X1_CONTRASTS, x2_contrasts: &C::X2_CONTRASTS, deviance: C::DEVIANCE,
        });
    }
}
