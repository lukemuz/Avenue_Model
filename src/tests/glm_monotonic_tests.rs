#![cfg(test)]
use crate::glm::{fit_glm_with_diagnostics, GLMOptions, GLMSolver, Normalization};
use crate::plan::Encoding;
use crate::rating_model::{Monotonicity, RatingModel};
use crate::workbook::{Scale, Workbook};
use polars::prelude::*;

fn model(direction: Monotonicity, family: &str) -> RatingModel {
    let intercept = df!("Rating_Factor" => [0.0]).unwrap();
    let bands = df!("x" => [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, f64::INFINITY],
        "Rating_Factor" => [0.0; 7])
    .unwrap();
    let mut model =
        RatingModel::from_dataframes(vec![intercept, bands], family, None, None).unwrap();
    model.tables[1] = model.tables[1].clone().as_monotone(direction).unwrap();
    model
}

#[test]
fn ordered_fits_match_exact_weighted_offset_optima_and_roundtrip() {
    let q = [0.5_f64, 2.0, 1.5];
    let y = [8.0_f64, 2.0, 20.0];
    let w = [1.0_f64, 3.0, 2.0];
    let data = df!("x" => [1.0, 3.0, 5.0], "y" => y, "w" => w,
        "offset" => q.map(f64::ln))
    .unwrap();
    for (family, p) in [("poisson", 1.0), ("tweedie", 1.5), ("gamma", 2.0)] {
        for direction in [Monotonicity::Increasing, Monotonicity::Decreasing] {
            for normalization in [Normalization::BaseLevel, Normalization::WeightedMean] {
                let options = GLMOptions {
                    objective: family.into(),
                    tweedie_power: p,
                    tolerance: 1e-10,
                    normalization,
                    ..Default::default()
                };
                let (fitted, diag) = fit_glm_with_diagnostics(
                    &model(direction, family),
                    &data,
                    "y",
                    Some("w"),
                    Some("offset"),
                    options,
                )
                .unwrap();
                assert!(diag.converged, "{family}: {}", diag.max_gradient);
                assert_eq!(diag.solver_used, GLMSolver::Table);
                assert_eq!(diag.accelerated_steps, 0);
                assert!(diag.inference.is_none());
                assert!(diag.inference_error.as_ref().unwrap().contains("Monotonic"));
                let pooled = if direction.is_increasing() {
                    0..2
                } else {
                    1..3
                };
                let a: f64 = pooled
                    .clone()
                    .map(|i| w[i] * y[i] * q[i].powf(1.0 - p))
                    .sum();
                let b: f64 = pooled.clone().map(|i| w[i] * q[i].powf(2.0 - p)).sum();
                let expected: Vec<f64> = (0..3)
                    .map(|i| {
                        if pooled.contains(&i) {
                            a / b
                        } else {
                            y[i] / q[i]
                        }
                    })
                    .collect();
                let predictions = fitted.predict(&data).unwrap();
                for (actual, expected) in
                    predictions.f64().unwrap().into_no_null_iter().zip(expected)
                {
                    assert!(
                        (actual - expected).abs() < 1e-8,
                        "{family}: {actual} vs {expected}"
                    );
                }
                let factors: Vec<f64> = fitted.tables[1]
                    .data
                    .column("Rating_Factor")
                    .unwrap()
                    .f64()
                    .unwrap()
                    .into_no_null_iter()
                    .collect();
                assert!(factors.windows(2).all(|v| if direction.is_increasing() {
                    v[0] <= v[1]
                } else {
                    v[0] >= v[1]
                }));
                assert_eq!(factors[0], factors[1]);
                assert_eq!(factors[1], factors[2]);
                assert_eq!(factors[3], factors[4]);
                assert_eq!(factors[5], factors[6]);
                let workbook = Workbook::from_model(
                    &fitted,
                    family,
                    &["intercept".into(), "x".into()],
                    &Encoding::default(),
                    p,
                    Some(Scale::Factor),
                    None,
                )
                .unwrap();
                assert_eq!(workbook.manifest.format_version, 3);
                let mut restored = Workbook::from_json(&workbook.to_json().unwrap()).unwrap();
                let loaded = restored.to_model().unwrap();
                assert_eq!(
                    loaded.model.tables[1].metadata.monotonicity,
                    Some(direction)
                );
                assert_eq!(loaded.predict(&data).unwrap(), predictions);
                let mut edited = factors;
                edited[0] = if direction.is_increasing() {
                    edited[1] + 1.0
                } else {
                    edited[1] - 1.0
                };
                restored.tables[1]
                    .with_column(Series::new("Rating_Factor".into(), edited))
                    .unwrap();
                let edited = restored.to_model().unwrap();
                assert!(edited
                    .notes
                    .iter()
                    .any(|note| note.code == "monotonicity_violated"));
                assert!(edited
                    .report(None, &Default::default())
                    .unwrap()
                    .findings
                    .iter()
                    .any(|finding| finding.code == "monotonicity_violated"));
            }
        }
    }
}

#[test]
fn incompatible_solver_and_constraint_combinations_fail_before_fitting() {
    let data = df!("x" => [1.0, 3.0, 5.0], "y" => [1.0, 2.0, 3.0]).unwrap();
    for options in [
        GLMOptions {
            objective: "poisson".into(),
            solver: GLMSolver::Global,
            ..Default::default()
        },
        GLMOptions {
            objective: "poisson".into(),
            alpha: 0.1,
            ..Default::default()
        },
        GLMOptions {
            objective: "gaussian".into(),
            ..Default::default()
        },
        GLMOptions {
            objective: "poisson".into(),
            robust_standard_errors: true,
            ..Default::default()
        },
    ] {
        assert!(fit_glm_with_diagnostics(
            &model(Monotonicity::Increasing, "poisson"),
            &data,
            "y",
            None,
            None,
            options
        )
        .is_err());
    }
    let mut locked = model(Monotonicity::Increasing, "poisson");
    locked.tables[1].set_row_offset(1, true);
    assert!(fit_glm_with_diagnostics(
        &locked,
        &data,
        "y",
        None,
        None,
        GLMOptions {
            objective: "poisson".into(),
            ..Default::default()
        }
    )
    .is_err());
}
