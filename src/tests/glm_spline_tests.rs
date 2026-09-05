#![cfg(test)]
use crate::glm::{fit_glm_with_diagnostics, GLMOptions, GLMSolver, Normalization};
use crate::plan::Encoding;
use crate::rating_model::RatingModel;
use crate::workbook::{Scale, Workbook};
use polars::prelude::*;
use std::collections::BTreeMap;

fn model(family: &str) -> RatingModel {
    let mut model = RatingModel::from_dataframes(
        vec![
            df!("Rating_Factor" => [0.]).unwrap(),
            df!("category" => [0i32, 1], "Rating_Factor" => [0., 0.]).unwrap(),
            df!("x" => [0., 0.4, 1.7, 3.], "Rating_Factor" => [0.;4]).unwrap(),
        ],
        family,
        None,
        None,
    )
    .unwrap();
    model.tables[2] = model.tables[2].clone().as_natural_cubic().unwrap();
    model
}
fn close(a: f64, b: f64) {
    assert!((a - b).abs() < 2e-7 * (1. + b.abs()), "{a} != {b}");
}

#[test]
fn continuous_glms_match_independent_dense_scipy_reference_in_five_families() {
    let f: serde_json::Value =
        serde_json::from_str(include_str!("../../tests/fixtures/spline_fit.json")).unwrap();
    let numbers = |v: &serde_json::Value| {
        v.as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap())
            .collect::<Vec<_>>()
    };
    let weights = numbers(&f["weights"]);
    let offset = numbers(&f["offset"]);
    for case in f["cases"].as_array().unwrap() {
        let family = case["family"].as_str().unwrap();
        let family = if family == "binomial" {
            "binary"
        } else {
            family
        };
        let data = df!("x" => numbers(&f["x"]), "category" => f["category"].as_array().unwrap().iter().map(|x| x.as_i64().unwrap() as i32).collect::<Vec<_>>(),
            "y" => numbers(&case["target"]), "w" => weights.clone(), "offset" => offset.clone()).unwrap();
        let reference = numbers(&case["means"]);
        for normalization in [
            Normalization::BaseLevel,
            Normalization::WeightedMean,
            Normalization::None,
        ] {
            let options = GLMOptions {
                objective: family.into(),
                tweedie_power: 1.5,
                normalization,
                tolerance: 1e-10,
                max_iterations: 1000,
                ..Default::default()
            };
            let (fitted, diag) = fit_glm_with_diagnostics(
                &model(family),
                &data,
                "y",
                Some("w"),
                Some("offset"),
                options,
            )
            .unwrap();
            assert!(
                diag.converged,
                "{family} {normalization:?}: {}",
                diag.max_gradient
            );
            assert_eq!(diag.solver_used, GLMSolver::Table);
            assert_eq!(diag.accelerated_steps, 0);
            assert!(diag.inference.is_none());
            assert!(diag.inference_error.unwrap().contains("spline"));
            assert!(diag.table_conditioning.is_none());
            assert!(diag.unfitted_rows.is_empty());
            let eta: Vec<f64> = fitted
                .predict_linear(&data)
                .unwrap()
                .iter()
                .zip(&offset)
                .map(|(e, o)| e + o)
                .collect();
            let means = fitted.apply_link_function(eta);
            for (a, b) in means.iter().zip(&reference) {
                close(*a, *b);
            }
            let curve = fitted.tables[2].predict_batch(&data);
            if normalization == Normalization::WeightedMean {
                let avg: f64 = curve.iter().zip(&weights).map(|(v, w)| v * w).sum::<f64>()
                    / weights.iter().sum::<f64>();
                close(avg, 0.);
            }
            if normalization == Normalization::BaseLevel {
                close(fitted.tables[2].get_rating_factor(0), 0.);
                let beta = numbers(&case["coefficients"]);
                close(fitted.tables[0].get_rating_factor(0), beta[0]);
                close(fitted.tables[1].get_rating_factor(1), beta[1]);
                for r in 1..4 {
                    close(fitted.tables[2].get_rating_factor(r), beta[r + 1]);
                }
            }
            let book = Workbook::from_model(
                &fitted,
                family,
                &["intercept".into(), "category".into(), "curve".into()],
                &Encoding {
                    maps: BTreeMap::new(),
                },
                1.5,
                Some(Scale::Factor),
                None,
            )
            .unwrap();
            let loaded = Workbook::from_json(&book.to_json().unwrap())
                .unwrap()
                .to_model()
                .unwrap();
            for (a, b) in loaded
                .model
                .predict_linear(&data)
                .unwrap()
                .iter()
                .zip(fitted.predict_linear(&data).unwrap())
            {
                close(*a, b);
            }
            assert!(diag
                .deviance_history
                .windows(2)
                .all(|v| v[1] <= v[0] + 1e-10 * (1. + v[0].abs())));
        }
    }
}

#[test]
fn spline_fitting_rejects_unidentified_geometry_and_unsupported_options() {
    let data = df!("x" => [1.;12], "category" => [0i32;12], "y" => [1.;12]).unwrap();
    let base = model("poisson");
    match fit_glm_with_diagnostics(&base, &data, "y", None, None, GLMOptions::default()) {
        Err(e) => assert!(e.to_string().contains("singular")),
        Ok(_) => panic!("unidentified knot values must not be fitted as step levels"),
    }
    for options in [
        GLMOptions {
            alpha: 1.,
            ..Default::default()
        },
        GLMOptions {
            solver: GLMSolver::Global,
            ..Default::default()
        },
        GLMOptions {
            robust_standard_errors: true,
            ..Default::default()
        },
    ] {
        assert!(fit_glm_with_diagnostics(&base, &data, "y", None, None, options).is_err());
    }
    let mut locked = base.clone();
    locked.tables[2].set_row_offset(0, true);
    assert!(
        fit_glm_with_diagnostics(&locked, &data, "y", None, None, GLMOptions::default()).is_err()
    );
    // A fully fixed curve needs no identifiable knot information: it contributes
    // its known continuous offset while the remaining tables are estimated.
    let mut fixed = base.clone();
    fixed.tables[2] = fixed.tables[2].clone().as_offset();
    assert!(
        fit_glm_with_diagnostics(&fixed, &data, "y", None, None, GLMOptions::default())
            .unwrap()
            .1
            .converged
    );
}

#[test]
fn two_continuous_effects_recover_a_known_curve_without_observations_on_knots() {
    let x: Vec<f64> = (0..240).map(|i| -0.3 + (i % 20) as f64 * 0.19).collect();
    let z: Vec<f64> = (0..240)
        .map(|i| -0.25 + ((i * 7) % 31) as f64 * 0.12)
        .collect();
    let data = df!("x" => x, "z" => z).unwrap();
    let mut truth = model("poisson");
    truth.tables.remove(1);
    truth.tables[0]
        .data
        .with_column(Series::new("Rating_Factor".into(), [0.35]))
        .unwrap();
    truth.tables[1]
        .data
        .with_column(Series::new("Rating_Factor".into(), [0., 0.2, -0.25, 0.1]))
        .unwrap();
    let second = crate::rating_model::RatingTable::new(
        df!("z" => [0., 0.5, 2., 3.], "Rating_Factor" => [0., -0.15, 0.25, -0.1]).unwrap(),
        None,
    )
    .as_natural_cubic()
    .unwrap();
    truth.tables.push(second);
    let means = truth.predict(&data).unwrap();
    let mut data = data;
    let mut y = means.clone();
    y.rename("y".into());
    data.with_column(y).unwrap();
    let mut start = truth.clone();
    for table in &mut start.tables {
        table
            .data
            .with_column(Series::new(
                "Rating_Factor".into(),
                vec![0.; table.data.height()],
            ))
            .unwrap();
    }
    let options = GLMOptions {
        objective: "poisson".into(),
        tolerance: 1e-10,
        max_iterations: 1000,
        ..Default::default()
    };
    let (fitted, diag) = fit_glm_with_diagnostics(&start, &data, "y", None, None, options).unwrap();
    assert!(diag.converged, "{}", diag.max_gradient);
    // Recovery includes new quotes and both tails, not only fitted observations.
    let quotes =
        df!("x" => [-1., 0., 0.4, 1., 1.7, 3., 4.], "z" => [4., 3., 2., 1.5, 0.5, 0., -1.])
            .unwrap();
    for (a, b) in fitted
        .predict_linear(&quotes)
        .unwrap()
        .iter()
        .zip(truth.predict_linear(&quotes).unwrap())
    {
        close(*a, b);
    }
}
