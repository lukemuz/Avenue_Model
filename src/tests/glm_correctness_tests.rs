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

    /// Standard errors go through a matrix inversion on both sides, so they carry a
    /// little more numerical noise than the coefficients themselves.
    const SE_TOL: f64 = 1e-6;

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
        x1_se: &'a [f64],
        x2_se: &'a [f64],
        intercept_se: f64,
        scale: f64,
        df_resid: f64,
        /// statsmodels values. Not compared for Tweedie, whose density has no closed
        /// form: statsmodels substitutes an approximation and Avenue reports None.
        llf: Option<f64>,
        aic: Option<f64>,
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

        // Standard errors, dispersion and residual degrees of freedom. Under the
        // default base-level anchoring the reported factors ARE the treatment-coded
        // contrasts, so these line up directly with statsmodels' bse.
        let inf = diag.inference.as_ref().expect("inference should be computed by default");
        assert_all_close(&[inf.dispersion], &[case.scale], REF_TOL,
            &format!("{} - dispersion vs statsmodels", case.name));
        assert_all_close(&[inf.df_residual], &[case.df_resid], REF_TOL,
            &format!("{} - residual df vs statsmodels", case.name));
        assert_all_close(&[inf.standard_errors[0][0]], &[case.intercept_se], SE_TOL,
            &format!("{} - intercept standard error vs statsmodels", case.name));
        assert_all_close(&inf.standard_errors[1], case.x1_se, SE_TOL,
            &format!("{} - x1 standard errors vs statsmodels", case.name));
        assert_all_close(&inf.standard_errors[2], case.x2_se, SE_TOL,
            &format!("{} - x2 standard errors vs statsmodels", case.name));

        match case.llf {
            Some(llf) => {
                assert_all_close(
                    &[inf.log_likelihood.expect("log-likelihood should be available")],
                    &[llf], SE_TOL,
                    &format!("{} - log-likelihood vs statsmodels", case.name));
                assert_all_close(
                    &[inf.aic.expect("AIC should be available")],
                    &[case.aic.unwrap()], SE_TOL,
                    &format!("{} - AIC vs statsmodels", case.name));
            }
            None => {
                assert!(inf.log_likelihood.is_none(),
                    "{} - log-likelihood should be None, got {:?}", case.name, inf.log_likelihood);
                assert!(inf.aic.is_none(),
                    "{} - AIC should be None, got {:?}", case.name, inf.aic);
            }
        }
    }

    // ------------------------------------------------------- 3. linear variates

    /// A variate table's factors must lie exactly on a straight line through the
    /// per-row values. This is the defining property: whatever the fit does, the five
    /// factors are one slope, not five free numbers.
    fn assert_on_a_line(factors: &[f64], values: &[f64], what: &str) {
        let slope = (factors[1] - factors[0]) / (values[1] - values[0]);
        for r in 0..factors.len() {
            let expected = factors[0] + slope * (values[r] - values[0]);
            assert!(
                (factors[r] - expected).abs() < 1e-9,
                "{}: row {} is {:.12} but the line through rows 0 and 1 gives {:.12} \
                 (slope {:.12}); factors {:?}",
                what, r, factors[r], expected, slope, factors
            );
        }
    }

    /// The headline behaviour: five rows, one parameter, factors on a line.
    #[test]
    fn variate_factors_lie_on_a_line() {
        use refdata::LinearVariate as C;

        let df = DataFrame::new(vec![
            Series::new("x1".into(), C::X1.to_vec()).into(),
            Series::new("age".into(), C::AGE.to_vec()).into(),
            Series::new("y".into(), C::Y.to_vec()).into(),
            Series::new("w".into(), C::WEIGHT.to_vec()).into(),
        ]).unwrap();

        let fitted = fit_variate_model(&df);
        let f = rating_factors(&fitted, 2);
        assert_on_a_line(&f, &C::AGE_VALUES, "age variate");

        // Under base-level anchoring the first row is the reference, so it is zero and
        // every other factor reads as a relativity against it.
        assert!(f[0].abs() < 1e-12, "base row should be 0, got {}", f[0]);
    }

    fn fit_variate_model(df: &DataFrame) -> RatingModel {
        crate::glm::fit_glm(&variate_model(), df, "y", Some("w"), None,
                            options("poisson", 1.5)).unwrap()
    }

    fn variate_model() -> RatingModel {
        use crate::rating_model::RatingTable;
        use refdata::LinearVariate as C;

        let age_table = RatingTable::new(factor_table("age", &C::AGE_BOUNDS), None)
            .as_variate(C::AGE_VALUES.to_vec())
            .expect("age table should be a valid variate");

        RatingModel::new(
            vec![
                RatingTable::new(intercept_table(), None),
                RatingTable::new(factor_table("x1", &refdata::X1_BOUNDS), None),
                age_table,
            ],
            crate::rating_model::LinkFunction::from_objective("poisson"),
        )
    }

    /// The slope, its standard error, the fitted means and the companion step table
    /// must all agree with the equivalent GLM fitted by statsmodels, where the variate
    /// is an ordinary continuous covariate taking each record's band value.
    #[test]
    fn variate_matches_statsmodels() {
        use refdata::LinearVariate as C;

        let df = DataFrame::new(vec![
            Series::new("x1".into(), C::X1.to_vec()).into(),
            Series::new("age".into(), C::AGE.to_vec()).into(),
            Series::new("y".into(), C::Y.to_vec()).into(),
            Series::new("w".into(), C::WEIGHT.to_vec()).into(),
        ]).unwrap();

        let (fitted, diag) = crate::glm::fit_glm_with_diagnostics(
            &variate_model(), &df, "y", Some("w"), None, options("poisson", 1.5)).unwrap();

        assert!(diag.converged, "variate fit did not converge in {} sweeps", diag.iterations);
        assert_all_close(&predictions(&fitted, &df), &C::MU, REF_TOL,
            "variate - fitted means vs statsmodels");
        assert_all_close(&[diag.deviance], &[C::DEVIANCE], REF_TOL,
            "variate - deviance vs statsmodels");
        assert_all_close(&contrasts(&fitted, 1), &C::X1_CONTRASTS, REF_TOL,
            "variate - companion step table contrasts vs statsmodels");

        // The slope itself.
        let slope = fitted.tables[2].variate_slope().expect("table 2 is a variate");
        assert_all_close(&[slope], &[C::SLOPE], REF_TOL,
            "variate - slope vs statsmodels");

        // Each row's factor is the slope times its distance from the base value.
        let f = rating_factors(&fitted, 2);
        let expected: Vec<f64> = C::AGE_VALUES.iter()
            .map(|v| C::SLOPE * (v - C::AGE_VALUES[0]))
            .collect();
        assert_all_close(&f, &expected, REF_TOL, "variate - row factors vs statsmodels");

        // And each row's standard error is the slope's, scaled the same way.
        let inf = diag.inference.expect("inference should be computed");
        let expected_se: Vec<f64> = C::AGE_VALUES.iter()
            .map(|v| C::SLOPE_SE * (v - C::AGE_VALUES[0]).abs())
            .collect();
        assert_all_close(&inf.standard_errors[2], &expected_se, SE_TOL,
            "variate - row standard errors vs statsmodels");
        assert_all_close(&inf.standard_errors[1], &C::X1_SE, SE_TOL,
            "variate - companion step table standard errors vs statsmodels");
    }

    /// Centring the variate column is what keeps the slope from crawling toward its
    /// answer alongside the intercept. Without it the fit reports convergence while the
    /// slope is still drifting, so pin the iteration count: a well-conditioned fit
    /// settles in a handful of sweeps, not hundreds.
    #[test]
    fn variate_converges_quickly() {
        use refdata::LinearVariate as C;

        let df = DataFrame::new(vec![
            Series::new("x1".into(), C::X1.to_vec()).into(),
            Series::new("age".into(), C::AGE.to_vec()).into(),
            Series::new("y".into(), C::Y.to_vec()).into(),
            Series::new("w".into(), C::WEIGHT.to_vec()).into(),
        ]).unwrap();

        let (_, diag) = crate::glm::fit_glm_with_diagnostics(
            &variate_model(), &df, "y", Some("w"), None, options("poisson", 1.5)).unwrap();

        assert!(diag.converged, "did not converge");
        assert!(diag.iterations < 40,
            "took {} sweeps to converge; an uncentred slope column is the usual cause",
            diag.iterations);
    }

    /// The whole point of a variate: a five-row table costs one parameter, not four.
    #[test]
    fn variate_costs_one_parameter() {
        use refdata::LinearVariate as C;

        let df = DataFrame::new(vec![
            Series::new("x1".into(), C::X1.to_vec()).into(),
            Series::new("age".into(), C::AGE.to_vec()).into(),
            Series::new("y".into(), C::Y.to_vec()).into(),
            Series::new("w".into(), C::WEIGHT.to_vec()).into(),
        ]).unwrap();

        let (_, diag) = crate::glm::fit_glm_with_diagnostics(
            &variate_model(), &df, "y", Some("w"), None, options("poisson", 1.5)).unwrap();

        // intercept (1) + x1 with 3 levels (2) + age variate (1) = 4
        let inf = diag.inference.unwrap();
        assert_eq!(inf.n_parameters, 4,
            "expected 4 parameters, got {}; a 5-row variate must not spend 4 on its own",
            inf.n_parameters);
        assert_all_close(&[inf.df_residual], &[C::DF_RESID], REF_TOL,
            "variate - residual df vs statsmodels");

        // The same tables as free step factors would spend three more.
        let free = RatingModel::from_dataframes(
            vec![
                intercept_table(),
                factor_table("x1", &refdata::X1_BOUNDS),
                factor_table("age", &C::AGE_BOUNDS),
            ],
            "poisson", None, None,
        ).unwrap();
        let (_, free_diag) = crate::glm::fit_glm_with_diagnostics(
            &free, &df, "y", Some("w"), None, options("poisson", 1.5)).unwrap();
        assert_eq!(free_diag.inference.unwrap().n_parameters, 7);
    }

    /// A band with no exposure still gets a factor, read off the line. A free step
    /// table would leave it stranded at its starting value.
    #[test]
    fn variate_fills_in_empty_bands() {
        use crate::rating_model::RatingTable;

        // Nobody in the 30-40 band.
        let ages = vec![22.0, 25.0, 28.0, 44.0, 47.0, 49.0, 55.0, 61.0, 70.0];
        let y = vec![1.0, 2.0, 1.0, 4.0, 5.0, 4.0, 7.0, 8.0, 9.0];
        let df = DataFrame::new(vec![
            Series::new("age".into(), ages).into(),
            Series::new("y".into(), y).into(),
        ]).unwrap();

        let bounds = [20.0, 30.0, 40.0, 50.0, f64::INFINITY];
        let values = vec![20.0, 30.0, 40.0, 50.0, 65.0];

        let model = RatingModel::new(
            vec![
                RatingTable::new(intercept_table(), None),
                RatingTable::new(factor_table("age", &bounds), None)
                    .as_variate(values.clone()).unwrap(),
            ],
            crate::rating_model::LinkFunction::from_objective("poisson"),
        );

        let (fitted, diag) = crate::glm::fit_glm_with_diagnostics(
            &model, &df, "y", None, None, options("poisson", 1.5)).unwrap();

        let f = rating_factors(&fitted, 1);
        assert_on_a_line(&f, &values, "variate with an empty band");
        // Row 2 (30-40) saw no data but is not stranded at its starting value.
        assert!(f[2].abs() > 1e-6, "empty band should be filled from the line, got {}", f[2]);
        assert!(diag.unfitted_rows.is_empty(),
            "a variate row without exposure is still fitted, got {:?}", diag.unfitted_rows);
        // It has a standard error too, since it borrows the slope's.
        let inf = diag.inference.unwrap();
        assert!(inf.standard_errors[1][2].is_finite() && inf.standard_errors[1][2] > 0.0,
            "empty band should carry the slope's standard error, got {}",
            inf.standard_errors[1][2]);
    }

    /// Anchoring changes where the line sits, never its slope or the predictions.
    #[test]
    fn variate_slope_is_invariant_to_anchoring() {
        use crate::glm::Normalization;
        use refdata::LinearVariate as C;

        let df = DataFrame::new(vec![
            Series::new("x1".into(), C::X1.to_vec()).into(),
            Series::new("age".into(), C::AGE.to_vec()).into(),
            Series::new("y".into(), C::Y.to_vec()).into(),
            Series::new("w".into(), C::WEIGHT.to_vec()).into(),
        ]).unwrap();

        let fit_with = |norm: Normalization| {
            let mut opts = options("poisson", 1.5);
            opts.normalization = norm;
            crate::glm::fit_glm(&variate_model(), &df, "y", Some("w"), None, opts).unwrap()
        };

        let base = fit_with(Normalization::BaseLevel);
        let wmean = fit_with(Normalization::WeightedMean);
        let none = fit_with(Normalization::None);

        for (name, m) in [("WeightedMean", &wmean), ("None", &none)] {
            assert_all_close(&predictions(m, &df), &predictions(&base, &df), 1e-9,
                &format!("{} anchoring must not move predictions", name));
            assert_all_close(
                &[m.tables[2].variate_slope().unwrap()],
                &[base.tables[2].variate_slope().unwrap()], 1e-9,
                &format!("{} anchoring must not change the slope", name));
            assert_on_a_line(&rating_factors(m, 2), &C::AGE_VALUES,
                &format!("{} anchoring", name));
        }
    }

    /// A variate table is still an ordinary step table to anything reading it, so a
    /// deployed lookup reproduces the fit exactly - no interpolation, no approximation.
    #[test]
    fn variate_predicts_by_step_lookup() {
        use refdata::LinearVariate as C;

        let df = DataFrame::new(vec![
            Series::new("x1".into(), C::X1.to_vec()).into(),
            Series::new("age".into(), C::AGE.to_vec()).into(),
            Series::new("y".into(), C::Y.to_vec()).into(),
            Series::new("w".into(), C::WEIGHT.to_vec()).into(),
        ]).unwrap();
        let fitted = fit_variate_model(&df);

        // Two ages in the same band must get identical predictions.
        let probe = DataFrame::new(vec![
            Series::new("x1".into(), vec![1.0, 1.0, 1.0]).into(),
            Series::new("age".into(), vec![31.0, 39.0, 41.0]).into(),
        ]).unwrap();
        let p = predictions(&fitted, &probe);
        assert!((p[0] - p[1]).abs() < 1e-12,
            "ages 31 and 39 are in the same band and must predict alike: {} vs {}", p[0], p[1]);
        assert!((p[1] - p[2]).abs() > 1e-9,
            "ages 39 and 41 are in different bands and must differ");
    }

    /// Values that cannot describe a line are rejected at construction, with the
    /// reason, rather than producing a fit nobody can interpret.
    #[test]
    fn invalid_variate_values_are_rejected() {
        use crate::rating_model::RatingTable;
        let bounds = [20.0, 30.0, f64::INFINITY];

        let cases: Vec<(Vec<f64>, &str)> = vec![
            (vec![20.0, 30.0], "one value per row"),
            (vec![20.0, 30.0, 40.0, 50.0], "one value per row"),
            (vec![20.0, 30.0, f64::INFINITY], "not finite"),
            (vec![25.0, 25.0, 25.0], "no slope to estimate"),
        ];

        for (values, expected) in cases {
            let err = RatingTable::new(factor_table("age", &bounds), None)
                .as_variate(values.clone())
                .expect_err(&format!("{:?} should be rejected", values))
                .to_string();
            assert!(err.contains(expected),
                "for {:?} expected a message mentioning {:?}, got: {}", values, expected, err);
        }

        // A locked row cannot coexist with a slope-derived factor.
        let mut table = RatingTable::new(factor_table("age", &bounds), None);
        table.set_row_offset(1, true);
        let err = table.as_variate(vec![20.0, 30.0, 40.0]).unwrap_err().to_string();
        assert!(err.contains("locked rows"), "unhelpful message: {}", err);
    }

    // --------------------------------------------------- 3b. polynomial variates

    /// A variate table's factors must lie exactly on a polynomial of the declared
    /// degree in the per-row values. Checked by fitting the polynomial back through
    /// the factors and requiring a residual of zero.
    fn assert_on_a_polynomial(factors: &[f64], values: &[f64], degree: usize, what: &str) {
        // Normal equations on a basis rescaled to [-1, 1], the same way the library
        // does it, so this checks the shape and not the conditioning.
        let (centre, scale) = crate::rating_model::variate_basis_params(values)
            .expect("values must vary");
        let k = degree + 1;
        let mut ata = vec![0.0f64; k * k];
        let mut atb = vec![0.0f64; k];
        for r in 0..values.len() {
            let u = (values[r] - centre) / scale;
            let basis: Vec<f64> = (0..k).map(|m| u.powi(m as i32)).collect();
            for a in 0..k {
                atb[a] += basis[a] * factors[r];
                for b in 0..k {
                    ata[a * k + b] += basis[a] * basis[b];
                }
            }
        }
        let coefs = crate::glm::solve_spd(&ata, &atb, k).expect("basis should be solvable");

        for r in 0..values.len() {
            let u = (values[r] - centre) / scale;
            let predicted: f64 = (0..k).map(|m| coefs[m] * u.powi(m as i32)).sum();
            assert!(
                (factors[r] - predicted).abs() < 1e-9,
                "{}: row {} is {:.12} but the degree-{} polynomial through the table gives \
                 {:.12}; factors {:?}",
                what, r, factors[r], degree, predicted, factors
            );
        }
    }

    fn quadratic_model(degree: usize) -> RatingModel {
        use crate::rating_model::RatingTable;
        use refdata::QuadraticVariate as C;

        let age_table = RatingTable::new(factor_table("age", &C::AGE_BOUNDS), None)
            .as_polynomial_variate(C::AGE_VALUES.to_vec(), degree)
            .expect("age table should be a valid variate");

        RatingModel::new(
            vec![
                RatingTable::new(intercept_table(), None),
                RatingTable::new(factor_table("x1", &refdata::X1_BOUNDS), None),
                age_table,
            ],
            crate::rating_model::LinkFunction::from_objective("poisson"),
        )
    }

    fn quadratic_df() -> DataFrame {
        use refdata::QuadraticVariate as C;
        DataFrame::new(vec![
            Series::new("x1".into(), C::X1.to_vec()).into(),
            Series::new("age".into(), C::AGE.to_vec()).into(),
            Series::new("y".into(), C::Y.to_vec()).into(),
            Series::new("w".into(), C::WEIGHT.to_vec()).into(),
        ]).unwrap()
    }

    /// A degree-2 variate must reproduce the equivalent GLM carrying both z and z^2,
    /// including the raw-scale coefficients on each power.
    #[test]
    fn quadratic_variate_matches_statsmodels() {
        use refdata::QuadraticVariate as C;
        let df = quadratic_df();

        let (fitted, diag) = crate::glm::fit_glm_with_diagnostics(
            &quadratic_model(2), &df, "y", Some("w"), None, options("poisson", 1.5)).unwrap();

        assert!(diag.converged, "quadratic fit did not converge in {} sweeps", diag.iterations);
        assert_all_close(&predictions(&fitted, &df), &C::MU, REF_TOL,
            "quadratic variate - fitted means vs statsmodels");
        assert_all_close(&[diag.deviance], &[C::DEVIANCE], REF_TOL,
            "quadratic variate - deviance vs statsmodels");
        assert_all_close(&contrasts(&fitted, 1), &C::X1_CONTRASTS, REF_TOL,
            "quadratic variate - step table contrasts vs statsmodels");

        // The two raw-scale coefficients, recovered from the fitted table.
        let coefs = fitted.tables[2].variate_coefficients().expect("table 2 is a variate");
        assert_eq!(coefs.len(), 2);
        assert_all_close(&coefs, &C::COEFFICIENTS, SE_TOL,
            "quadratic variate - raw-scale coefficients vs statsmodels");

        // And the same numbers via the diagnostics.
        let inf = diag.inference.expect("inference should be computed");
        assert_eq!(inf.variate_terms.len(), 1);
        let terms = &inf.variate_terms[0];
        assert_eq!(terms.table_index, 2);
        assert_eq!(terms.degree, 2);
        assert_all_close(&terms.coefficients, &C::COEFFICIENTS, SE_TOL,
            "quadratic variate - reported coefficients vs statsmodels");
    }

    /// The factors must sit exactly on the quadratic - that is the constraint.
    #[test]
    fn quadratic_variate_factors_lie_on_a_curve() {
        use refdata::QuadraticVariate as C;
        let fitted = crate::glm::fit_glm(
            &quadratic_model(2), &quadratic_df(), "y", Some("w"), None,
            options("poisson", 1.5)).unwrap();

        let f = rating_factors(&fitted, 2);
        assert_on_a_polynomial(&f, &C::AGE_VALUES, 2, "quadratic age variate");
        assert!(f[0].abs() < 1e-12, "base row should be 0, got {}", f[0]);

        // A genuine bend: the curve is not a straight line through the same points.
        let slope_lo = (f[1] - f[0]) / (C::AGE_VALUES[1] - C::AGE_VALUES[0]);
        let slope_hi = (f[4] - f[3]) / (C::AGE_VALUES[4] - C::AGE_VALUES[3]);
        assert!((slope_hi - slope_lo).abs() > 1e-3,
            "expected curvature, but the ends have slopes {:.6} and {:.6}", slope_lo, slope_hi);
    }

    /// A degree-d variate costs exactly d parameters, whatever the row count.
    #[test]
    fn polynomial_degree_sets_the_parameter_count() {
        use refdata::QuadraticVariate as C;
        let df = quadratic_df();

        // intercept (1) + x1 with 3 levels (2) + age variate (degree)
        for degree in 1..=4 {
            let (_, diag) = crate::glm::fit_glm_with_diagnostics(
                &quadratic_model(degree), &df, "y", Some("w"), None,
                options("poisson", 1.5)).unwrap();
            let inf = diag.inference.unwrap();
            assert_eq!(inf.n_parameters, 3 + degree,
                "degree {} should cost {} parameters, got {}", degree, 3 + degree, inf.n_parameters);
            assert_eq!(inf.variate_terms[0].degree, degree);
            assert_eq!(inf.variate_terms[0].coefficients.len(), degree);
        }

        // Degree 4 through 5 distinct values passes through every row exactly, so it
        // must reproduce the free-level fit.
        let free = RatingModel::from_dataframes(
            vec![
                intercept_table(),
                factor_table("x1", &refdata::X1_BOUNDS),
                factor_table("age", &C::AGE_BOUNDS),
            ],
            "poisson", None, None,
        ).unwrap();
        let free_fit = crate::glm::fit_glm(&free, &df, "y", Some("w"), None,
                                           options("poisson", 1.5)).unwrap();
        let sat_fit = crate::glm::fit_glm(&quadratic_model(4), &df, "y", Some("w"), None,
                                          options("poisson", 1.5)).unwrap();
        assert_all_close(&predictions(&sat_fit, &df), &predictions(&free_fit, &df), 1e-7,
            "a degree-4 variate over 5 values is saturated and must equal free levels");
    }

    /// Raising the degree can only improve the fit, and each degree is nested in the
    /// next - a basic sanity property that catches a mis-specified basis.
    #[test]
    fn higher_degree_never_fits_worse() {
        let df = quadratic_df();
        let mut previous = f64::INFINITY;
        for degree in 1..=4 {
            let (_, diag) = crate::glm::fit_glm_with_diagnostics(
                &quadratic_model(degree), &df, "y", Some("w"), None,
                options("poisson", 1.5)).unwrap();
            assert!(diag.deviance <= previous + 1e-6,
                "degree {} has deviance {} against {} at degree {}",
                degree, diag.deviance, previous, degree - 1);
            previous = diag.deviance;
        }
    }

    /// The top degree's z statistic is what tells you whether the curve needs to bend.
    /// On data generated with real curvature the quadratic term must be significant;
    /// on data generated from a straight line it must not be.
    #[test]
    fn top_degree_z_detects_curvature() {
        use refdata::LinearVariate as L;

        let (_, curved) = crate::glm::fit_glm_with_diagnostics(
            &quadratic_model(2), &quadratic_df(), "y", Some("w"), None,
            options("poisson", 1.5)).unwrap();
        let z_curved = curved.inference.unwrap().variate_terms[0]
            .top_degree_z().expect("quadratic term should have a z");
        assert!(z_curved.abs() > 3.0,
            "curved data should show a significant quadratic term, got z = {:.3}", z_curved);

        // The linear dataset, fitted with a spare degree it does not need.
        use crate::rating_model::RatingTable;
        let linear_df = DataFrame::new(vec![
            Series::new("x1".into(), L::X1.to_vec()).into(),
            Series::new("age".into(), L::AGE.to_vec()).into(),
            Series::new("y".into(), L::Y.to_vec()).into(),
            Series::new("w".into(), L::WEIGHT.to_vec()).into(),
        ]).unwrap();
        let model = RatingModel::new(
            vec![
                RatingTable::new(intercept_table(), None),
                RatingTable::new(factor_table("x1", &refdata::X1_BOUNDS), None),
                RatingTable::new(factor_table("age", &L::AGE_BOUNDS), None)
                    .as_polynomial_variate(L::AGE_VALUES.to_vec(), 2).unwrap(),
            ],
            crate::rating_model::LinkFunction::from_objective("poisson"),
        );
        let (_, straight) = crate::glm::fit_glm_with_diagnostics(
            &model, &linear_df, "y", Some("w"), None, options("poisson", 1.5)).unwrap();
        let z_straight = straight.inference.unwrap().variate_terms[0]
            .top_degree_z().expect("quadratic term should have a z");
        assert!(z_straight.abs() < 2.0,
            "data generated from a line should not show a significant quadratic term, \
             got z = {:.3}", z_straight);
    }

    /// Anchoring moves the curve up and down, never its shape.
    #[test]
    fn quadratic_coefficients_are_invariant_to_anchoring() {
        use crate::glm::Normalization;
        use refdata::QuadraticVariate as C;
        let df = quadratic_df();

        let fit_with = |norm: Normalization| {
            let mut opts = options("poisson", 1.5);
            opts.normalization = norm;
            crate::glm::fit_glm(&quadratic_model(2), &df, "y", Some("w"), None, opts).unwrap()
        };
        let base = fit_with(Normalization::BaseLevel);

        for (name, norm) in [
            ("WeightedMean", Normalization::WeightedMean),
            ("None", Normalization::None),
        ] {
            let m = fit_with(norm);
            assert_all_close(&predictions(&m, &df), &predictions(&base, &df), 1e-9,
                &format!("{} anchoring must not move predictions", name));
            assert_all_close(
                &m.tables[2].variate_coefficients().unwrap(),
                &base.tables[2].variate_coefficients().unwrap(), 1e-7,
                &format!("{} anchoring must not change the curve", name));
            assert_on_a_polynomial(&rating_factors(&m, 2), &C::AGE_VALUES, 2,
                &format!("{} anchoring", name));
        }
    }

    /// variate_slope is a linear-only convenience; a curve has no single slope.
    #[test]
    fn variate_slope_is_none_above_degree_one() {
        let fitted = crate::glm::fit_glm(
            &quadratic_model(2), &quadratic_df(), "y", Some("w"), None,
            options("poisson", 1.5)).unwrap();
        assert!(fitted.tables[2].variate_slope().is_none(),
            "a degree-2 variate should not report a single slope");
        assert_eq!(fitted.tables[2].variate_degree(), Some(2));
        // Table 1 is an ordinary step table.
        assert!(fitted.tables[1].variate_slope().is_none());
        assert_eq!(fitted.tables[1].variate_degree(), None);
        // Whereas a degree-1 variate does report one, equal to its only coefficient.
        let linear = fit_variate_model(&{
            use refdata::LinearVariate as L;
            DataFrame::new(vec![
                Series::new("x1".into(), L::X1.to_vec()).into(),
                Series::new("age".into(), L::AGE.to_vec()).into(),
                Series::new("y".into(), L::Y.to_vec()).into(),
                Series::new("w".into(), L::WEIGHT.to_vec()).into(),
            ]).unwrap()
        });
        let slope = linear.tables[2].variate_slope().unwrap();
        let coefs = linear.tables[2].variate_coefficients().unwrap();
        assert_eq!(coefs.len(), 1);
        assert!((slope - coefs[0]).abs() < 1e-9,
            "slope {} and coefficient {} should agree", slope, coefs[0]);
    }

    /// Degrees that cannot be identified are rejected at construction.
    #[test]
    fn invalid_degrees_are_rejected() {
        use crate::rating_model::{RatingTable, MAX_VARIATE_DEGREE};
        let bounds = [20.0, 30.0, 40.0, f64::INFINITY];
        let values = vec![20.0, 30.0, 40.0, 55.0];

        let table = || RatingTable::new(factor_table("age", &bounds), None);

        let err = table().as_polynomial_variate(values.clone(), 0).unwrap_err().to_string();
        assert!(err.contains("degree 0"), "unhelpful message: {}", err);

        // 4 distinct values support at most degree 3.
        let err = table().as_polynomial_variate(values.clone(), 4).unwrap_err().to_string();
        assert!(err.contains("distinct"), "unhelpful message: {}", err);
        assert!(table().as_polynomial_variate(values.clone(), 3).is_ok(),
            "degree 3 through 4 distinct values should be allowed");

        let err = table()
            .as_polynomial_variate(values, MAX_VARIATE_DEGREE + 1)
            .unwrap_err().to_string();
        assert!(err.contains("limit is"), "unhelpful message: {}", err);
    }

    // ------------------------------------------------- 4. inference edge cases

    /// A completely separated level has zero IRLS weight, so it confounds with the
    /// intercept and its coefficient is not estimable. It must be reported as aliased
    /// rather than given a confident-looking standard error, and it must not cost the
    /// rest of the model its standard errors.
    #[test]
    fn separated_level_is_reported_as_aliased() {
        let df = DataFrame::new(vec![
            Series::new("x".into(), vec![1.0, 1.0, 1.0, 2.0, 2.0, 2.0]).into(),
            Series::new("y".into(), vec![1.0, 1.0, 1.0, 0.0, 1.0, 0.0]).into(),
        ]).unwrap();
        let model = RatingModel::from_dataframes(
            vec![intercept_table(), factor_table("x", &[1.0, f64::INFINITY])],
            "binary", None, None,
        ).unwrap();

        let (_, diag) = crate::glm::fit_glm_with_diagnostics(
            &model, &df, "y", None, None, options("binary", 1.5)).unwrap();

        let inf = diag.inference.expect("separation must not suppress inference entirely");
        assert!(diag.inference_error.is_none(),
            "fit should not have recorded an inference failure: {:?}", diag.inference_error);
        // Level 0 is the base level, pinned at zero by construction, not estimated.
        assert_eq!(inf.standard_errors[1][0], 0.0);
        // Level 1 cannot be separated from the intercept once level 0 is saturated.
        assert!(inf.standard_errors[1][1].is_nan(),
            "aliased level should have no standard error, got {}", inf.standard_errors[1][1]);
        assert!(inf.aliased_rows.contains(&(1, 1)),
            "aliased level should be listed, got {:?}", inf.aliased_rows);
        assert_eq!(inf.n_parameters, 1, "only the intercept is estimable here");
    }

    /// Two tables keyed on the same column are perfectly collinear: their effects
    /// cannot be separated. The second table's levels must come back aliased while
    /// the first table keeps usable standard errors, and the fit itself is unharmed.
    #[test]
    fn collinear_tables_are_reported_as_aliased() {
        use refdata::PoissonTwoFactor as C;
        let df = DataFrame::new(vec![
            Series::new("x1".into(), C::X1.to_vec()).into(),
            Series::new("y".into(), C::Y.to_vec()).into(),
            Series::new("w".into(), C::WEIGHT.to_vec()).into(),
        ]).unwrap();

        let model = RatingModel::from_dataframes(
            vec![
                intercept_table(),
                factor_table("x1", &refdata::X1_BOUNDS),
                factor_table("x1", &refdata::X1_BOUNDS), // same feature, same cuts
            ],
            "poisson", None, None,
        ).unwrap();

        let (fitted, diag) = crate::glm::fit_glm_with_diagnostics(
            &model, &df, "y", Some("w"), None, options("poisson", 1.5)).unwrap();

        let inf = diag.inference.expect("a collinear design still has estimable parameters");
        assert!(diag.inference_error.is_none(), "should not be an error: {:?}", diag.inference_error);

        // The duplicate table's non-base levels carry no separable information.
        assert!(inf.aliased_rows.contains(&(2, 1)) && inf.aliased_rows.contains(&(2, 2)),
            "duplicate table should be aliased, got {:?}", inf.aliased_rows);
        // The first table is still fully estimable.
        assert!(inf.standard_errors[1][1].is_finite() && inf.standard_errors[1][2].is_finite(),
            "first table should keep its standard errors, got {:?}", inf.standard_errors[1]);
        assert_eq!(inf.n_parameters, 3, "intercept plus two estimable levels");

        // The fit itself is unharmed.
        assert!(predictions(&fitted, &df).iter().all(|p| p.is_finite()));
    }

    /// Doubling every prior weight doubles the information, so standard errors shrink
    /// by sqrt(2) while the fitted factors do not move.
    #[test]
    fn standard_errors_scale_with_weight() {
        use refdata::PoissonTwoFactor as C;

        let build = |scale: f64| {
            DataFrame::new(vec![
                Series::new("x1".into(), C::X1.to_vec()).into(),
                Series::new("x2".into(), C::X2.to_vec()).into(),
                Series::new("y".into(), C::Y.to_vec()).into(),
                Series::new("w".into(), C::WEIGHT.iter().map(|w| w * scale).collect::<Vec<_>>()).into(),
            ]).unwrap()
        };
        let tables = || vec![
            intercept_table(),
            factor_table("x1", &refdata::X1_BOUNDS),
            factor_table("x2", &refdata::X2_BOUNDS),
        ];
        let run = |scale: f64| {
            let model = RatingModel::from_dataframes(tables(), "poisson", None, None).unwrap();
            crate::glm::fit_glm_with_diagnostics(
                &model, &build(scale), "y", Some("w"), None, options("poisson", 1.5)).unwrap()
        };

        let (m1, d1) = run(1.0);
        let (m2, d2) = run(4.0);

        assert_all_close(&rating_factors(&m2, 1), &rating_factors(&m1, 1), 1e-9,
            "scaling all weights must not move the factors");

        let se1 = &d1.inference.unwrap().standard_errors[1];
        let se2 = &d2.inference.unwrap().standard_errors[1];
        let halved: Vec<f64> = se1.iter().map(|s| s / 2.0).collect();
        assert_all_close(se2, &halved, 1e-9,
            "quadrupling weights should halve the standard errors");
    }

    /// A level with no observations cannot be estimated, and must be flagged rather
    /// than reported with a confident-looking standard error.
    #[test]
    fn empty_level_is_flagged_and_has_no_standard_error() {
        let df = DataFrame::new(vec![
            // Nothing lands in the middle bin.
            Series::new("x".into(), vec![1.0, 1.0, 1.0, 3.0, 3.0, 3.0]).into(),
            Series::new("y".into(), vec![2.0, 4.0, 3.0, 30.0, 40.0, 35.0]).into(),
        ]).unwrap();
        let model = RatingModel::from_dataframes(
            vec![intercept_table(), factor_table("x", &[1.0, 2.0, f64::INFINITY])],
            "poisson", None, None,
        ).unwrap();

        let (_, diag) = crate::glm::fit_glm_with_diagnostics(
            &model, &df, "y", None, None, options("poisson", 1.5)).unwrap();

        assert!(diag.unfitted_rows.contains(&(1, 1)),
            "empty level should be listed as unfitted, got {:?}", diag.unfitted_rows);
        let inf = diag.inference.unwrap();
        assert!(inf.standard_errors[1][1].is_nan(),
            "empty level should have no standard error, got {}", inf.standard_errors[1][1]);
        assert!(inf.standard_errors[1][2].is_finite(),
            "populated levels should still be estimable");
    }

    #[test]
    fn matches_statsmodels_gaussian() {
        use refdata::GaussianTwoFactor as C;
        check_reference(RefCase {
            name: "gaussian/identity", objective: "gaussian", tweedie_power: 1.5,
            x1: &C::X1, x2: &C::X2, y: &C::Y, weight: &C::WEIGHT, offset: None,
            mu: &C::MU, x1_contrasts: &C::X1_CONTRASTS, x2_contrasts: &C::X2_CONTRASTS, deviance: C::DEVIANCE,
            x1_se: &C::X1_SE, x2_se: &C::X2_SE, intercept_se: C::INTERCEPT_SE,
            scale: C::SCALE, df_resid: C::DF_RESID,
            llf: Some(C::LLF), aic: Some(C::AIC),
        });
    }

    #[test]
    fn matches_statsmodels_poisson() {
        use refdata::PoissonTwoFactor as C;
        check_reference(RefCase {
            name: "poisson/log", objective: "poisson", tweedie_power: 1.5,
            x1: &C::X1, x2: &C::X2, y: &C::Y, weight: &C::WEIGHT, offset: None,
            mu: &C::MU, x1_contrasts: &C::X1_CONTRASTS, x2_contrasts: &C::X2_CONTRASTS, deviance: C::DEVIANCE,
            x1_se: &C::X1_SE, x2_se: &C::X2_SE, intercept_se: C::INTERCEPT_SE,
            scale: C::SCALE, df_resid: C::DF_RESID,
            llf: Some(C::LLF), aic: Some(C::AIC),
        });
    }

    #[test]
    fn matches_statsmodels_poisson_with_offset() {
        use refdata::PoissonOffset as C;
        check_reference(RefCase {
            name: "poisson/log + offset", objective: "poisson", tweedie_power: 1.5,
            x1: &C::X1, x2: &C::X2, y: &C::Y, weight: &C::WEIGHT, offset: Some(&C::OFFSET),
            mu: &C::MU, x1_contrasts: &C::X1_CONTRASTS, x2_contrasts: &C::X2_CONTRASTS, deviance: C::DEVIANCE,
            x1_se: &C::X1_SE, x2_se: &C::X2_SE, intercept_se: C::INTERCEPT_SE,
            scale: C::SCALE, df_resid: C::DF_RESID,
            llf: Some(C::LLF), aic: Some(C::AIC),
        });
    }

    #[test]
    fn matches_statsmodels_gamma() {
        use refdata::GammaTwoFactor as C;
        check_reference(RefCase {
            name: "gamma/log", objective: "gamma", tweedie_power: 1.5,
            x1: &C::X1, x2: &C::X2, y: &C::Y, weight: &C::WEIGHT, offset: None,
            mu: &C::MU, x1_contrasts: &C::X1_CONTRASTS, x2_contrasts: &C::X2_CONTRASTS, deviance: C::DEVIANCE,
            x1_se: &C::X1_SE, x2_se: &C::X2_SE, intercept_se: C::INTERCEPT_SE,
            scale: C::SCALE, df_resid: C::DF_RESID,
            llf: Some(C::LLF), aic: Some(C::AIC),
        });
    }

    #[test]
    fn matches_statsmodels_binary() {
        use refdata::BinaryTwoFactor as C;
        check_reference(RefCase {
            name: "binomial/logit", objective: "binary", tweedie_power: 1.5,
            x1: &C::X1, x2: &C::X2, y: &C::Y, weight: &C::WEIGHT, offset: None,
            mu: &C::MU, x1_contrasts: &C::X1_CONTRASTS, x2_contrasts: &C::X2_CONTRASTS, deviance: C::DEVIANCE,
            x1_se: &C::X1_SE, x2_se: &C::X2_SE, intercept_se: C::INTERCEPT_SE,
            scale: C::SCALE, df_resid: C::DF_RESID,
            llf: Some(C::LLF), aic: Some(C::AIC),
        });
    }

    #[test]
    fn matches_statsmodels_tweedie() {
        use refdata::TweedieTwoFactor as C;
        check_reference(RefCase {
            name: "tweedie(1.5)/log", objective: "tweedie", tweedie_power: 1.5,
            x1: &C::X1, x2: &C::X2, y: &C::Y, weight: &C::WEIGHT, offset: None,
            mu: &C::MU, x1_contrasts: &C::X1_CONTRASTS, x2_contrasts: &C::X2_CONTRASTS, deviance: C::DEVIANCE,
            x1_se: &C::X1_SE, x2_se: &C::X2_SE, intercept_se: C::INTERCEPT_SE,
            scale: C::SCALE, df_resid: C::DF_RESID,
            llf: None, aic: None,
        });
    }
}
