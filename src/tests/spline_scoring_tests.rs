use crate::glm::{fit_glm_with_diagnostics, GLMOptions};
use crate::plan::Encoding;
use crate::rating_model::{FeatureValue, LinkFunction, RatingModel, RatingTable};
use crate::validation::ValidationOptions;
use crate::workbook::{Scale, Workbook};
use polars::prelude::*;
use std::collections::{BTreeMap, HashMap};

fn table(knots: Vec<f64>, values: Vec<f64>) -> RatingTable {
    RatingTable::new(df!("x" => knots, "Rating_Factor" => values).unwrap(), None)
        .as_natural_cubic()
        .unwrap()
}
fn model(spline: RatingTable) -> RatingModel {
    RatingModel::new(
        vec![
            RatingTable::new(df!("Rating_Factor" => [0.2]).unwrap(), None),
            spline,
        ],
        LinkFunction::Log,
    )
}
fn book(model: &RatingModel, scale: Scale) -> Workbook {
    Workbook::from_model(
        model,
        "poisson",
        &["intercept".into(), "curve".into()],
        &Encoding {
            maps: BTreeMap::new(),
        },
        1.5,
        Some(scale),
        Some(("y", None, None)),
    )
    .unwrap()
}
fn close(a: f64, b: f64) {
    assert!((a - b).abs() <= 1e-9 + 1e-10 * b.abs(), "{a} != {b}");
}

#[test]
fn spline_scoring_and_both_workbook_formats_match_scipy_at_every_probe() {
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("../../tests/fixtures/natural_spline.json")).unwrap();
    for (case_index, case) in fixture["cases"].as_array().unwrap().iter().enumerate() {
        let numbers = |key: &str| {
            case[key]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_f64().unwrap())
                .collect()
        };
        let spline = table(numbers("knots"), numbers("values"));
        let probes = case["probes"].as_array().unwrap();
        let xs: Vec<f64> = probes.iter().map(|p| p["x"].as_f64().unwrap()).collect();
        let df = df!("x" => xs.clone()).unwrap();
        let batch = spline.predict_batch(&df);
        for (i, p) in probes.iter().enumerate() {
            let expected = p["derivatives"][0].as_f64().unwrap();
            close(batch[i], expected);
            close(
                spline.predict(&HashMap::from([("x".into(), FeatureValue::Numeric(xs[i]))])),
                expected,
            );
        }
        let model = model(spline);
        for (actual, p) in model.predict_linear(&df).unwrap().iter().zip(probes) {
            close(*actual, 0.2 + p["derivatives"][0].as_f64().unwrap());
        }
        // Factor scale preserves exact coefficients even for extreme fixture curves.
        let book = book(&model, Scale::Factor);
        assert_eq!(book.manifest.format_version, 4);
        let json = book.to_json().unwrap();
        assert!(json.contains("natural_cubic_linear_tails"));
        let from_json = Workbook::from_json(&json).unwrap().to_model().unwrap();
        let dir = std::env::temp_dir().join(format!(
            "avenue_spline_{}_{}",
            std::process::id(),
            case_index
        ));
        book.save_csv_dir(&dir).unwrap();
        let from_csv = Workbook::load_csv_dir(&dir).unwrap().to_model().unwrap();
        for loaded in [&from_json, &from_csv] {
            let actual = loaded.model.predict_linear(&df).unwrap();
            for (value, p) in actual.iter().zip(probes) {
                close(*value, 0.2 + p["derivatives"][0].as_f64().unwrap());
            }
        }
        std::fs::remove_dir_all(dir).unwrap();
    }
}

#[test]
fn spline_explanations_validation_edits_and_composition_share_continuous_scores() {
    let model = model(table(vec![0., 1., 2.], vec![0., 0.4, 0.]));
    // Natural cubic at the half knots is .275; endpoint slopes are +/-.6.
    let xs = vec![-1., 0., 0.5, 1., 1.5, 2., 3.];
    let expected = vec![-0.6, 0., 0.275, 0.4, 0.275, 0., -0.6];
    let ys: Vec<f64> = expected.iter().map(|v| (0.2_f64 + v).exp()).collect();
    let df = df!("x" => xs, "y" => ys.clone()).unwrap();
    for scale in [Scale::Factor, Scale::Relativity] {
        let book = book(&model, scale);
        let loaded = book.to_model().unwrap();
        let predicted = loaded.predict(&df).unwrap();
        for (a, b) in predicted.f64().unwrap().into_no_null_iter().zip(&ys) {
            close(a, *b);
        }
        let (_, contributions) = loaded.explain(&df).unwrap();
        for i in 0..7 {
            assert_eq!(
                contributions
                    .column("kind")
                    .unwrap()
                    .str()
                    .unwrap()
                    .get(7 + i),
                Some("spline")
            );
            assert_eq!(
                contributions
                    .column("table_row")
                    .unwrap()
                    .u64()
                    .unwrap()
                    .get(7 + i),
                None
            );
            close(
                contributions
                    .column("coefficient")
                    .unwrap()
                    .f64()
                    .unwrap()
                    .get(7 + i)
                    .unwrap(),
                expected[i],
            );
        }
        let validation = loaded.validate(&df, &ValidationOptions::default()).unwrap();
        close(validation.ae_ratio, 1.);
        let exhibit = &validation.actual_vs_expected[1];
        assert!(exhibit.column("Rating_Factor").is_err());
        assert_eq!(
            exhibit
                .column("N")
                .unwrap()
                .i64()
                .unwrap()
                .into_no_null_iter()
                .collect::<Vec<_>>(),
            vec![2, 2, 3]
        );
        assert_eq!(
            exhibit
                .column("Support_Upper")
                .unwrap()
                .f64()
                .unwrap()
                .get(2),
            Some(f64::INFINITY)
        );
        let combined = loaded.combine(&loaded).unwrap();
        assert_eq!(combined.model.tables.len(), 3);
        for (a, b) in combined
            .predict(&df)
            .unwrap()
            .f64()
            .unwrap()
            .into_no_null_iter()
            .zip(&ys)
        {
            close(a, b * b);
        }
    }
    let mut edited = book(&model, Scale::Factor);
    edited.tables[1]
        .with_column(Series::new("Rating_Factor".into(), vec![0., 0.8, 0.]))
        .unwrap();
    let changed = edited.to_model().unwrap();
    for (a, b) in changed
        .model
        .predict_linear(&df)
        .unwrap()
        .iter()
        .zip(&expected)
    {
        close(*a, 0.2 + 2. * b);
    }
    assert!(fit_glm_with_diagnostics(&model, &df, "y", None, None, GLMOptions::default()).is_err());
}

#[test]
fn spline_invalid_quotes_and_malformed_edits_fail_explicitly() {
    let model = model(table(vec![0., 1., 2.], vec![0., 0.4, 0.]));
    let loaded = book(&model, Scale::Factor).to_model().unwrap();
    let df =
        df!("x" => [Some(0.5), None, Some(f64::NAN), Some(f64::INFINITY), Some(f64::NEG_INFINITY)])
            .unwrap();
    let diag = loaded.predict_diagnostics(&df).unwrap();
    assert!(loaded.predict(&df).is_err());
    assert_eq!(diag.column("predictions").unwrap().null_count(), 4);
    for knots in [
        vec![0., 0., 2.],
        vec![0., 2., 1.],
        vec![0., 1., f64::INFINITY],
    ] {
        let mut bad = book(&model, Scale::Factor);
        bad.tables[1]
            .with_column(Series::new("x".into(), knots))
            .unwrap();
        assert!(bad.to_model().is_err());
    }
    let mut bad = book(&model, Scale::Factor);
    bad.tables[1]
        .with_column(Series::new("Rating_Factor".into(), vec![0., f64::NAN, 0.]))
        .unwrap();
    assert!(bad.to_model().is_err());
    bad = book(&model, Scale::Factor);
    bad.manifest.format_version = 3;
    assert!(bad.to_model().is_err());
    let invalid_kind = book(&model, Scale::Factor)
        .to_json()
        .unwrap()
        .replace("natural_cubic_linear_tails", "cubic_extrapolation");
    assert!(Workbook::from_json(&invalid_kind).is_err());
}
