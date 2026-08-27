//! Correctness tests for L1 and L2 penalties.
//!
//! Same philosophy as `glm_correctness_tests`: assert on numbers that can be derived
//! without running the thing under test.
//!
//! A Gaussian model with an intercept and one saturated categorical table has a closed
//! form under a ridge penalty, which is what most of this file checks. Writing `n_j` for
//! the weight behind level `j`, `ybar_j` for its weighted mean of `y`, and taking level 0
//! as the base level that the penalty shrinks toward:
//!
//! ```text
//! theta_j = n_j * (ybar_j - b0) / (n_j + l2)          for j >= 1, theta_0 = 0
//! b0      = sum_j c_j * ybar_j / sum_j c_j
//!           where c_0 = n_0 and c_j = n_j * l2 / (n_j + l2)
//! ```
//!
//! Both fall straight out of the two stationarity conditions and neither involves an
//! iteration, so agreement is real evidence rather than two implementations of the same
//! mistake. The lasso cases use the KKT conditions the same way.

#[cfg(test)]
mod glm_penalty_tests {
    use crate::glm::penalty::soft_threshold;
    use crate::glm::{fit_glm, fit_glm_with_diagnostics, GLMOptions, Normalization};
    use crate::rating_model::RatingModel;
    use polars::prelude::*;

    const FIT_TOL: f64 = 1e-13;
    const MAX_ITER: usize = 5000;

    // ---------------------------------------------------------------- fixture

    /// Six levels with deliberately uneven weight, so that shrinkage has something to
    /// bite on: a thin level is pulled much harder toward the base than a heavy one, and
    /// a test that gave every level the same weight could not tell the two apart.
    fn levels() -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let mut x = Vec::new();
        let mut y = Vec::new();
        let mut w = Vec::new();
        // (level, count, y values scaled around a level effect)
        let spec: [(f64, &[f64], f64); 6] = [
            (0.0, &[10.0, 12.0, 11.0, 9.0, 13.0, 10.5, 11.5, 9.5], 1.0),
            (1.0, &[18.0, 22.0, 20.0, 19.0], 1.0),
            (2.0, &[10.4, 9.8], 2.0),
            (3.0, &[31.0, 29.0, 30.0, 33.0, 27.0, 30.5], 0.5),
            (4.0, &[11.2], 3.0),
            (5.0, &[4.0, 6.0, 5.0], 1.5),
        ];
        for (level, ys, weight) in spec {
            for v in ys {
                x.push(level);
                y.push(*v);
                w.push(weight);
            }
        }
        (x, y, w)
    }

    fn fixture() -> (DataFrame, RatingModel, Vec<usize>, Vec<f64>, Vec<f64>) {
        let (x, y, w) = levels();
        let level_of: Vec<usize> = x.iter().map(|v| *v as usize).collect();
        let df = DataFrame::new(vec![
            Series::new("x".into(), x).into(),
            Series::new("y".into(), y.clone()).into(),
            Series::new("w".into(), w.clone()).into(),
        ])
        .unwrap();
        let bounds: Vec<f64> = vec![0.0, 1.0, 2.0, 3.0, 4.0, f64::INFINITY];
        let model = RatingModel::from_dataframes(
            vec![
                DataFrame::new(vec![Series::new("Rating_Factor".into(), vec![0.0]).into()])
                    .unwrap(),
                DataFrame::new(vec![
                    Series::new("x".into(), bounds.clone()).into(),
                    Series::new("Rating_Factor".into(), vec![0.0; bounds.len()]).into(),
                ])
                .unwrap(),
            ],
            "gaussian",
            None,
            None,
        )
        .unwrap();
        (df, model, level_of, y, w)
    }

    /// `n_j` and `ybar_j` per level, plus the total weight the penalty is scaled by.
    fn level_stats(level_of: &[usize], y: &[f64], w: &[f64]) -> (Vec<f64>, Vec<f64>, f64) {
        let k = level_of.iter().max().unwrap() + 1;
        let mut n = vec![0.0; k];
        let mut s = vec![0.0; k];
        for i in 0..y.len() {
            n[level_of[i]] += w[i];
            s[level_of[i]] += w[i] * y[i];
        }
        let total = n.iter().sum::<f64>();
        let ybar = (0..k).map(|j| s[j] / n[j]).collect();
        (n, ybar, total)
    }

    fn options(alpha: f64, l1_ratio: f64) -> GLMOptions {
        GLMOptions {
            objective: "gaussian".to_string(),
            max_iterations: MAX_ITER,
            tolerance: FIT_TOL,
            alpha,
            l1_ratio,
            ..Default::default()
        }
    }

    fn factors(model: &RatingModel, t: usize) -> Vec<f64> {
        let ca = model.tables[t]
            .data
            .column("Rating_Factor")
            .unwrap()
            .f64()
            .unwrap();
        (0..ca.len()).map(|i| ca.get(i).unwrap()).collect()
    }

    /// Level factors measured against the base level - the quantity the penalty acts on,
    /// and the only one that does not depend on how the intercept is split out.
    fn contrasts(model: &RatingModel) -> Vec<f64> {
        let f = factors(model, 1);
        f.iter().map(|v| v - f[0]).collect()
    }

    fn assert_close(actual: &[f64], expected: &[f64], tol: f64, what: &str) {
        assert_eq!(actual.len(), expected.len(), "{}: length", what);
        for (j, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (a - e).abs() <= tol * e.abs().max(1.0),
                "{}: level {} got {:.12e}, expected {:.12e}",
                what,
                j,
                a,
                e
            );
        }
    }

    // ---------------------------------------------------------------- ridge

    /// The closed form above, at three strengths spanning almost no shrinkage to almost
    /// total shrinkage.
    #[test]
    fn ridge_matches_the_closed_form() {
        let (df, model, level_of, y, w) = fixture();
        let (n, ybar, total) = level_stats(&level_of, &y, &w);

        for alpha in [1e-4, 0.05, 2.0] {
            let l2 = total * alpha;

            // b0 = sum c_j ybar_j / sum c_j, with c_0 = n_0 and c_j = n_j l2 / (n_j + l2).
            let c: Vec<f64> = (0..n.len())
                .map(|j| {
                    if j == 0 {
                        n[j]
                    } else {
                        n[j] * l2 / (n[j] + l2)
                    }
                })
                .collect();
            let b0: f64 = c.iter().zip(ybar.iter()).map(|(c, m)| c * m).sum::<f64>()
                / c.iter().sum::<f64>();
            let expected: Vec<f64> = (0..n.len())
                .map(|j| {
                    if j == 0 {
                        0.0
                    } else {
                        n[j] * (ybar[j] - b0) / (n[j] + l2)
                    }
                })
                .collect();

            let (fitted, diag) = fit_glm_with_diagnostics(
                &model,
                &df,
                "y",
                Some("w"),
                None,
                options(alpha, 0.0),
            )
            .unwrap();
            assert!(diag.converged, "alpha = {alpha}: did not converge");
            assert_close(
                &contrasts(&fitted),
                &expected,
                1e-9,
                &format!("ridge contrasts at alpha = {alpha}"),
            );
            // The intercept carries the base level, so it is b0 plus level 0's factor -
            // which the closed form puts at exactly b0, since theta_0 is the gauge.
            let intercept = factors(&fitted, 0)[0] + factors(&fitted, 1)[0];
            assert!(
                (intercept - b0).abs() < 1e-9 * b0.abs().max(1.0),
                "ridge intercept at alpha = {alpha}: got {intercept:.12e}, expected {b0:.12e}"
            );
        }
    }

    /// The ridge path has to shrink as `alpha` grows. On the *norm* of the contrast
    /// vector, not level by level: the intercept is unpenalised and moves along the path
    /// too, so an individual level whose mean sits on the far side of the moving
    /// intercept can grow while the table as a whole is shrinking. The closed-form test
    /// above is what pins the individual levels.
    #[test]
    fn ridge_shrinks_more_as_alpha_grows() {
        let (df, model, _, _, _) = fixture();
        let norm = |c: &[f64]| c.iter().map(|v| v * v).sum::<f64>().sqrt();
        let mut previous: Option<f64> = None;
        for alpha in [0.0, 1e-3, 1e-2, 1e-1, 1.0, 10.0] {
            let fitted =
                fit_glm(&model, &df, "y", Some("w"), None, options(alpha, 0.0)).unwrap();
            let size = norm(&contrasts(&fitted));
            if let Some(prev) = previous {
                assert!(
                    size <= prev + 1e-9,
                    "alpha = {alpha}: the table grew from {prev:.6} to {size:.6}"
                );
            }
            previous = Some(size);
        }
        // And at a large enough alpha everything has collapsed onto the base level.
        let fitted = fit_glm(&model, &df, "y", Some("w"), None, options(1e6, 0.0)).unwrap();
        for (j, v) in contrasts(&fitted).iter().enumerate() {
            assert!(v.abs() < 1e-3, "level {j} survived a huge ridge at {v:.3e}");
        }
    }

    // ---------------------------------------------------------------- lasso

    /// The KKT conditions for the lasso on this design, checked against the fit's own
    /// intercept: a level is at zero exactly when the data pulling it away from the base
    /// level is worth less than the threshold.
    #[test]
    fn lasso_zeroes_exactly_the_levels_the_data_cannot_pay_for() {
        let (df, model, level_of, y, w) = fixture();
        let (n, ybar, total) = level_stats(&level_of, &y, &w);

        let mut ever_zero = false;
        let mut ever_nonzero = false;
        for alpha in [1e-3, 0.02, 0.1, 0.5, 2.0] {
            let l1 = total * alpha;
            let (fitted, diag) =
                fit_glm_with_diagnostics(&model, &df, "y", Some("w"), None, options(alpha, 1.0))
                    .unwrap();
            assert!(diag.converged, "alpha = {alpha}: did not converge");

            let c = contrasts(&fitted);
            let b0 = factors(&fitted, 0)[0] + factors(&fitted, 1)[0];

            for j in 1..n.len() {
                // theta_j = S(n_j (ybar_j - b0), l1) / n_j, the one-dimensional lasso
                // solution given the intercept the fit settled on.
                let expected = soft_threshold(n[j] * (ybar[j] - b0), l1) / n[j];
                assert!(
                    (c[j] - expected).abs() < 1e-8 * expected.abs().max(1.0),
                    "alpha = {alpha}, level {j}: got {:.12e}, KKT says {:.12e}",
                    c[j],
                    expected
                );
                if c[j] == 0.0 {
                    ever_zero = true;
                } else {
                    ever_nonzero = true;
                }
            }
        }
        assert!(ever_zero, "no level was ever zeroed - the sweep is not testing L1");
        assert!(ever_nonzero, "every level was zeroed at every alpha");
    }

    /// Zeros have to be *exact*, not merely small. A level that is dropped and a level
    /// worth 1e-12 read the same on a report but are different models, and the whole
    /// reason coordinate descent suits the lasso is that it can land on the kink.
    #[test]
    fn a_dropped_level_is_exactly_the_base_level() {
        let (df, model, _, _, _) = fixture();
        let fitted = fit_glm(&model, &df, "y", Some("w"), None, options(5.0, 1.0)).unwrap();
        let f = factors(&fitted, 1);
        for (j, v) in f.iter().enumerate() {
            assert_eq!(
                *v, f[0],
                "level {j} sits at {v} rather than exactly on the base level {}",
                f[0]
            );
        }
    }

    /// With every level collapsed onto the base, the model is the null model, so the
    /// intercept must be the overall weighted mean. This pins the intercept's own
    /// exemption from the penalty: if it were being shrunk too, it would come back short.
    #[test]
    fn a_collapsed_lasso_leaves_the_weighted_mean_in_the_intercept() {
        let (df, model, _, y, w) = fixture();
        let mean = y.iter().zip(w.iter()).map(|(y, w)| y * w).sum::<f64>()
            / w.iter().sum::<f64>();
        let fitted = fit_glm(&model, &df, "y", Some("w"), None, options(5.0, 1.0)).unwrap();
        let intercept = factors(&fitted, 0)[0] + factors(&fitted, 1)[0];
        assert!(
            (intercept - mean).abs() < 1e-9 * mean.abs(),
            "got {intercept:.12e}, weighted mean is {mean:.12e}"
        );
    }

    /// An elastic net has to sit between its two ends rather than reducing to either.
    #[test]
    fn an_elastic_net_lies_between_ridge_and_lasso() {
        let (df, model, _, _, _) = fixture();
        let fit = |ratio: f64| {
            let m = fit_glm(&model, &df, "y", Some("w"), None, options(0.05, ratio)).unwrap();
            contrasts(&m)
        };
        let (ridge, mixed, lasso) = (fit(0.0), fit(0.5), fit(1.0));
        let mut differs_from_both = 0;
        for j in 1..ridge.len() {
            if (mixed[j] - ridge[j]).abs() > 1e-9 && (mixed[j] - lasso[j]).abs() > 1e-9 {
                differs_from_both += 1;
            }
        }
        assert!(
            differs_from_both > 0,
            "the elastic net reproduced one of its endpoints exactly"
        );
    }

    // ---------------------------------------------------------------- invariants

    /// `alpha = 0` must be the unpenalised fit to the last bit, not merely close to it.
    /// The penalised step reduces to the unpenalised one algebraically, but it is gated
    /// rather than relied on precisely so that turning the feature off cannot move an
    /// existing fit by a rounding step.
    #[test]
    fn alpha_zero_is_bit_for_bit_the_unpenalised_fit() {
        let (df, model, _, _, _) = fixture();
        let plain = GLMOptions {
            objective: "gaussian".to_string(),
            max_iterations: MAX_ITER,
            tolerance: FIT_TOL,
            ..Default::default()
        };
        let a = fit_glm(&model, &df, "y", Some("w"), None, plain).unwrap();
        let b = fit_glm(&model, &df, "y", Some("w"), None, options(0.0, 0.7)).unwrap();
        assert_eq!(factors(&a, 0), factors(&b, 0));
        assert_eq!(factors(&a, 1), factors(&b, 1));
    }

    /// The penalty is defined on contrasts against the base level, so where the fit
    /// starts - which is a pure change of gauge - must not change where it lands. This
    /// is the property that would break first if the penalty were applied to levels
    /// instead, because `normalize` shifts every table after every sweep.
    #[test]
    fn a_penalised_fit_does_not_depend_on_the_starting_gauge() {
        let (df, model, _, _, _) = fixture();
        // Same model, started from a different point on the flat direction: every level
        // of the table lifted by a constant, which changes no prediction and no contrast.
        let shifted = {
            let mut m = model.clone();
            let h = m.tables[1].data.height();
            m.tables[1]
                .data
                .with_column(Series::new("Rating_Factor".into(), vec![3.75; h]))
                .unwrap();
            m
        };
        for (alpha, ratio) in [(0.05, 0.0), (0.05, 1.0), (0.05, 0.5)] {
            let a = fit_glm(&model, &df, "y", Some("w"), None, options(alpha, ratio)).unwrap();
            let b = fit_glm(&shifted, &df, "y", Some("w"), None, options(alpha, ratio)).unwrap();
            assert_close(
                &contrasts(&b),
                &contrasts(&a),
                1e-8,
                &format!("contrasts at alpha = {alpha}, l1_ratio = {ratio}"),
            );
        }
    }

    /// A penalty against a base level nobody is holding still is not a well-posed
    /// problem, and quietly fitting one would be worse than refusing.
    #[test]
    fn a_penalty_needs_the_base_level_gauge() {
        let (df, model, _, _, _) = fixture();
        for mode in [Normalization::None, Normalization::WeightedMean] {
            let opts = GLMOptions {
                normalization: mode,
                ..options(0.05, 0.5)
            };
            let Err(err) = fit_glm(&model, &df, "y", Some("w"), None, opts) else {
                panic!("{mode:?} should have been refused");
            };
            assert!(
                err.to_string().contains("BaseLevel"),
                "unhelpful message: {err}"
            );
        }
        // Unpenalised, every mode is still fine.
        for mode in [Normalization::None, Normalization::WeightedMean] {
            let opts = GLMOptions {
                normalization: mode,
                ..options(0.0, 0.0)
            };
            assert!(fit_glm(&model, &df, "y", Some("w"), None, opts).is_ok());
        }
    }

    #[test]
    fn a_nonsense_l1_ratio_is_refused() {
        let (df, model, _, _, _) = fixture();
        for ratio in [-0.1, 1.5, f64::NAN] {
            assert!(
                fit_glm(&model, &df, "y", Some("w"), None, options(0.05, ratio)).is_err(),
                "l1_ratio = {ratio} was accepted"
            );
        }
        assert!(fit_glm(&model, &df, "y", Some("w"), None, options(-1.0, 0.0)).is_err());
    }

    // ---------------------------------------------------------------- inference

    /// A general square inverse, written out here so the check does not lean on the
    /// solver it is checking. Gauss-Jordan with partial pivoting; the matrices are 6x6.
    fn invert(a: &[f64], n: usize) -> Vec<f64> {
        let mut m = vec![0.0; n * 2 * n];
        for i in 0..n {
            for j in 0..n {
                m[i * 2 * n + j] = a[i * n + j];
            }
            m[i * 2 * n + n + i] = 1.0;
        }
        for c in 0..n {
            let mut best = c;
            for r in c + 1..n {
                if m[r * 2 * n + c].abs() > m[best * 2 * n + c].abs() {
                    best = r;
                }
            }
            for j in 0..2 * n {
                m.swap(c * 2 * n + j, best * 2 * n + j);
            }
            let pivot = m[c * 2 * n + c];
            assert!(pivot.abs() > 1e-14, "singular at column {c}");
            for j in 0..2 * n {
                m[c * 2 * n + j] /= pivot;
            }
            for r in 0..n {
                if r == c {
                    continue;
                }
                let f = m[r * 2 * n + c];
                if f == 0.0 {
                    continue;
                }
                for j in 0..2 * n {
                    m[r * 2 * n + j] -= f * m[c * 2 * n + j];
                }
            }
        }
        let mut out = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                out[i * n + j] = m[i * 2 * n + n + j];
            }
        }
        out
    }

    fn matmul(a: &[f64], b: &[f64], n: usize) -> Vec<f64> {
        let mut out = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                out[i * n + j] = (0..n).map(|l| a[i * n + l] * b[l * n + j]).sum();
            }
        }
        out
    }

    /// The one that matters. A penalised estimator solves `score(b) = P b`, so its
    /// variance is the sandwich `(H + P)^-1 H (H + P)^-1`, **not** `(H + P)^-1` and not
    /// `H^-1`. All three agree at zero penalty and diverge quickly, and reaching for the
    /// wrong one is the natural mistake, so this checks the reported standard errors
    /// against the sandwich and confirms they are not either alternative.
    ///
    /// For this design `H` is an arrow matrix that can be written down by hand: the
    /// intercept column carries every observation, each level column carries its own.
    #[test]
    fn ridge_standard_errors_are_the_sandwich() {
        let (df, model, level_of, y, w) = fixture();
        let (n, _, total) = level_stats(&level_of, &y, &w);
        let k = n.len(); // intercept plus one column per non-base level

        for alpha in [0.02, 0.5] {
            let l2 = total * alpha;

            // H = X'WX for [intercept, level 1 .. level 5], Gaussian so the IRLS weight
            // is the prior weight.
            let mut h = vec![0.0; k * k];
            h[0] = n.iter().sum::<f64>();
            for j in 1..k {
                h[j] = n[j];
                h[j * k] = n[j];
                h[j * k + j] = n[j];
            }
            let mut hp = h.clone();
            for j in 1..k {
                hp[j * k + j] += l2;
            }

            let hp_inv = invert(&hp, k);
            let sandwich = matmul(&matmul(&hp_inv, &h, k), &hp_inv, k);
            let h_inv = invert(&h, k);

            let (fitted, diag) =
                fit_glm_with_diagnostics(&model, &df, "y", Some("w"), None, options(alpha, 0.0))
                    .unwrap();
            let inf = diag.inference.expect("inference");
            let phi = inf.dispersion;

            for j in 1..k {
                let got = inf.standard_errors[1][j];
                let expected = (phi * sandwich[j * k + j]).sqrt();
                assert!(
                    (got - expected).abs() < 1e-8 * expected,
                    "alpha = {alpha}, level {j}: got {got:.10e}, sandwich says {expected:.10e}"
                );
                // And it is not one of the two things it could have been by mistake.
                let naive = (phi * hp_inv[j * k + j]).sqrt();
                let unpenalised = (phi * h_inv[j * k + j]).sqrt();
                assert!(
                    (got - naive).abs() > 1e-10 * expected,
                    "alpha = {alpha}, level {j}: reported (H+P)^-1 instead of the sandwich"
                );
                assert!(
                    (got - unpenalised).abs() > 1e-10 * expected,
                    "alpha = {alpha}, level {j}: reported H^-1, ignoring the penalty"
                );
            }
            // Level 0 is the reference: fixed by construction, so exactly zero.
            assert_eq!(inf.standard_errors[1][0], 0.0);
            // The trace of the hat matrix, from the same two matrices.
            let expected_edf: f64 = (0..k).map(|i| matmul(&hp_inv, &h, k)[i * k + i]).sum();
            assert!(
                (inf.effective_parameters - expected_edf).abs() < 1e-9,
                "alpha = {alpha}: edf {} vs {expected_edf}",
                inf.effective_parameters
            );
            let _ = &fitted;
        }
    }

    /// A penalty buys bias for variance, so it has to spend less than a free fit and the
    /// spend has to fall as the penalty grows - toward one, the unpenalised intercept.
    #[test]
    fn a_ridge_spends_less_than_its_parameter_count() {
        let (df, model, _, _, _) = fixture();
        let spend = |alpha: f64| {
            let (_, diag) =
                fit_glm_with_diagnostics(&model, &df, "y", Some("w"), None, options(alpha, 0.0))
                    .unwrap();
            let inf = diag.inference.expect("inference");
            (inf.effective_parameters, inf.n_parameters as f64)
        };

        let (free, rank) = spend(0.0);
        assert_eq!(free, rank, "an unpenalised fit spends exactly its rank");

        let mut previous = free;
        for alpha in [1e-3, 1e-2, 1e-1, 1.0, 100.0] {
            let (used, _) = spend(alpha);
            assert!(used < previous, "alpha = {alpha}: spend rose to {used}");
            assert!(used > 0.99, "alpha = {alpha}: spend fell below the intercept");
            previous = used;
        }
        assert!(
            previous < 1.2,
            "a huge ridge should leave little more than the intercept, got {previous}"
        );
    }

    /// A lasso picks its levels from the data it then estimates on. A Wald interval that
    /// ignores that is the wrong quantity rather than a wide one, so none is reported -
    /// and the spend is the count of levels kept.
    #[test]
    fn a_lasso_reports_no_standard_errors_and_spends_what_it_kept() {
        let (df, model, _, _, _) = fixture();
        for alpha in [0.02, 0.2] {
            let (fitted, diag) =
                fit_glm_with_diagnostics(&model, &df, "y", Some("w"), None, options(alpha, 1.0))
                    .unwrap();
            let inf = diag.inference.expect("inference");
            for (t, ses) in inf.standard_errors.iter().enumerate() {
                for (r, se) in ses.iter().enumerate() {
                    assert!(se.is_nan(), "alpha = {alpha}: table {t} row {r} reported {se}");
                }
            }
            let kept = contrasts(&fitted).iter().filter(|v| **v != 0.0).count() as f64;
            assert_eq!(
                inf.effective_parameters,
                kept + 1.0,
                "alpha = {alpha}: spend should be the intercept plus the {kept} levels kept"
            );
        }
    }

    // ---------------------------------------------------------------- other families

    /// The log-link update is an exact multiplicative solve rather than a Newton step,
    /// so it takes a different branch to the Gaussian one above. No closed form here,
    /// but the fit still has to converge, shrink monotonically, and reach exact zeros.
    #[test]
    fn a_poisson_table_shrinks_toward_its_base_level() {
        let x: Vec<f64> = (0..300).map(|i| (i % 6) as f64).collect();
        // Counts around a level effect, with within-level scatter that is not a function
        // of the level. Without the scatter the model is saturated, the deviance falls to
        // its rounding floor, and the fit stops on the stall rule rather than on the
        // score - true of the unpenalised fitter too, and nothing to do with penalties.
        let y: Vec<f64> = x
            .iter()
            .enumerate()
            .map(|(i, v)| {
                let noise = [0.0, 2.0, -1.0, 1.0, 3.0, -2.0, 0.0][(i * 7 + i / 6) % 7];
                (((v * 0.4).exp() * 3.0) + noise).round().max(0.0)
            })
            .collect();
        let df = DataFrame::new(vec![
            Series::new("x".into(), x).into(),
            Series::new("y".into(), y).into(),
        ])
        .unwrap();
        let bounds: Vec<f64> = vec![0.0, 1.0, 2.0, 3.0, 4.0, f64::INFINITY];
        let model = RatingModel::from_dataframes(
            vec![
                DataFrame::new(vec![Series::new("Rating_Factor".into(), vec![0.0]).into()])
                    .unwrap(),
                DataFrame::new(vec![
                    Series::new("x".into(), bounds.clone()).into(),
                    Series::new("Rating_Factor".into(), vec![0.0; bounds.len()]).into(),
                ])
                .unwrap(),
            ],
            "poisson",
            None,
            None,
        )
        .unwrap();

        let poisson = |alpha: f64, ratio: f64| GLMOptions {
            objective: "poisson".to_string(),
            max_iterations: MAX_ITER,
            tolerance: 1e-11,
            alpha,
            l1_ratio: ratio,
            ..Default::default()
        };

        let norm = |c: &[f64]| c.iter().map(|v| v * v).sum::<f64>().sqrt();
        let mut previous: Option<f64> = None;
        for alpha in [0.0, 1e-4, 1e-3, 1e-2] {
            let (fitted, diag) =
                fit_glm_with_diagnostics(&model, &df, "y", None, None, poisson(alpha, 0.0))
                    .unwrap();
            assert!(diag.converged, "poisson ridge at alpha = {alpha} did not converge");
            let size = norm(&contrasts(&fitted));
            if let Some(prev) = previous {
                assert!(
                    size <= prev + 1e-8,
                    "alpha = {alpha}: the table grew from {prev:.6} to {size:.6}"
                );
            }
            previous = Some(size);
        }

        // A lasso strong enough to drop every level must land exactly on the base, on
        // the log-link path too - that is the case the `ln(1 + .)` damping is bypassed
        // for, and getting it wrong leaves levels near but not at zero.
        let (fitted, diag) =
            fit_glm_with_diagnostics(&model, &df, "y", None, None, poisson(20.0, 1.0)).unwrap();
        assert!(diag.converged, "poisson lasso did not converge");
        let f = factors(&fitted, 1);
        for (j, v) in f.iter().enumerate() {
            assert_eq!(*v, f[0], "poisson level {j} is at {v}, base is {}", f[0]);
        }
    }
}
