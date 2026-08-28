//! Tests for building models out of tables that already exist.
//!
//! This is the workflow the artifact exists for: a plan that is already in force is
//! loaded from a file, held fixed, and a new factor is fitted on top of it. Or its
//! shape is kept while its numbers are refreshed. Both are ordinary pricing work, and
//! both were previously impossible to express.

#[cfg(test)]
mod composition_tests {
    use crate::glm::GLMOptions;
    use crate::plan::{Breaks, GivenRole, Plan, Term};
    use crate::rating_model::RatingModel;
    use crate::validation::ValidationOptions;
    use crate::workbook::{Scale, Workbook};
    use polars::prelude::*;

    /// Motor data whose rate is driven by region and, more weakly, by a new factor the
    /// existing plan does not contain.
    fn motor(n: usize) -> DataFrame {
        let regions = ["north", "south", "east", "west"];
        let region: Vec<&str> = (0..n).map(|i| regions[i % 4]).collect();
        let telematics: Vec<i32> = (0..n).map(|i| ((i / 4) % 3) as i32).collect();
        let exposure: Vec<f64> = (0..n).map(|i| 0.5 + ((i % 4) as f64) / 4.0).collect();
        let claims: Vec<f64> = (0..n)
            .map(|i| {
                let region_multiplier = [1.0, 1.25, 1.5, 1.75][i % 4];
                let telematics_multiplier = [0.8, 1.0, 1.2][(i / 4) % 3];
                let variation = [0.9, 1.0, 1.1][(i / 12) % 3];
                0.8 * region_multiplier * telematics_multiplier * exposure[i] * variation
            })
            .collect();
        DataFrame::new(vec![
            Series::new("region".into(), region).into(),
            Series::new("telematics".into(), telematics).into(),
            Series::new("exposure".into(), exposure).into(),
            Series::new("claims".into(), claims).into(),
        ])
        .unwrap()
    }

    fn options() -> GLMOptions {
        GLMOptions {
            max_iterations: 500,
            tolerance: 1e-10,
            ..Default::default()
        }
    }

    fn factors(model: &RatingModel, table: usize) -> Vec<f64> {
        model.tables[table]
            .data
            .column("Rating_Factor")
            .unwrap()
            .f64()
            .unwrap()
            .into_no_null_iter()
            .collect()
    }

    /// Last year's plan: region only.
    fn existing_plan(df: &DataFrame) -> crate::plan::FittedModel {
        Plan::frequency("exposure")
            .with(Term::categorical("region"))
            .fit(df, "claims", options())
            .unwrap()
    }

    #[test]
    fn an_existing_plan_can_be_held_fixed_while_a_new_factor_is_fitted_on_top() {
        let df = motor(720);
        let existing = existing_plan(&df);
        let before = factors(&existing.model, 1);

        // Carry every table of the filed plan as an offset, and rate one new factor.
        let plan = Plan::frequency("exposure")
            .with_encoding(existing.encoding.clone())
            .with_offset_model(&existing.model, &existing.table_names, "prior")
            .unwrap()
            .with(Term::categorical("telematics"));

        let fitted = plan.fit(&df, "claims", options()).unwrap();
        assert_eq!(fitted.converged(), Some(true));

        // Tables: the new intercept, the two carried offsets, then the new factor.
        assert_eq!(fitted.table_names.len(), 4);
        assert_eq!(fitted.table_names[3], "telematics");

        // The carried plan is untouched.
        assert_eq!(
            factors(&fitted.model, 2),
            before,
            "an offset table must come out exactly as it went in"
        );

        // And the new factor recovered the multipliers it was supposed to. Read them
        // by level code rather than by row: the base defaults to the most exposed
        // level, so which row comes first is decided by the data, not by the codes.
        let telematics = factors(&fitted.model, 3);
        let codes: Vec<i32> = fitted.model.tables[3]
            .data
            .column("telematics")
            .unwrap()
            .i32()
            .unwrap()
            .into_no_null_iter()
            .collect();
        let factor_of = |code: i32| -> f64 {
            telematics[codes.iter().position(|c| *c == code).unwrap()]
        };
        // True multipliers are 0.8, 1.0, 1.2, so against code 0 they are 1.0, 1.25, 1.5.
        for (code, want) in [(1i32, 1.25f64), (2, 1.5)] {
            let got = (factor_of(code) - factor_of(0)).exp();
            assert!(
                (got - want).abs() < 0.02,
                "telematics level {} came back at {:.4} against an expected {:.4}",
                code,
                got,
                want
            );
        }

        // It prices, and it balances.
        let v = fitted.validate(&df, &ValidationOptions::default()).unwrap();
        assert_eq!(v.unmatched_rows, 0);
        assert!((v.ae_ratio - 1.0).abs() < 1e-6, "ae_ratio {}", v.ae_ratio);
    }

    #[test]
    fn an_offset_spends_no_parameters_and_says_so() {
        let df = motor(720);
        let existing = existing_plan(&df);

        let plan = Plan::frequency("exposure")
            .with_encoding(existing.encoding.clone())
            .with_offset_model(&existing.model, &existing.table_names, "prior")
            .unwrap()
            .with(Term::categorical("telematics"));

        let check = plan.check(&df, "claims").unwrap();
        let by_name: std::collections::HashMap<&str, &crate::plan::ResolvedTerm> =
            check.resolved.iter().map(|r| (r.name.as_str(), r)).collect();

        assert_eq!(by_name["prior.region"].kind, "offset");
        assert_eq!(
            by_name["prior.region"].parameters, 0,
            "a carried table is not estimated, so it costs nothing"
        );
        assert_eq!(by_name["telematics"].kind, "categorical");
        assert_eq!(by_name["telematics"].parameters, 2);
        assert!(check.is_fittable(), "{:?}", check.issues);
    }

    #[test]
    fn a_supplied_table_can_instead_define_the_shape_and_have_its_numbers_refreshed() {
        let df = motor(720);
        let existing = existing_plan(&df);
        let last_year = existing.model.tables[1].data.clone();

        // Same levels, refitted factors.
        let plan = Plan::frequency("exposure")
            .with_encoding(existing.encoding.clone())
            .with(Term::given("region", &last_year).unwrap());
        let fitted = plan.fit(&df, "claims", options()).unwrap();

        assert_eq!(fitted.resolved[1].kind, "given");
        assert_eq!(fitted.resolved[1].parameters, 3, "a given structure is estimated");
        assert_eq!(
            fitted.model.tables[1].data.height(),
            last_year.height(),
            "the shape is kept"
        );

        // Fitting the same data through the same shape reproduces the same relativities.
        let refreshed = factors(&fitted.model, 1);
        let original = factors(&existing.model, 1);
        for (a, b) in refreshed.iter().zip(original.iter()) {
            assert!(
                (a - b).abs() < 1e-6,
                "refreshed {:?} should match {:?}",
                refreshed,
                original
            );
        }
    }

    #[test]
    fn a_supplied_table_is_checked_as_strictly_as_one_that_is_loaded() {
        let df = motor(240);
        // An out-of-order band: the edit that silently mis-bins.
        let scrambled = DataFrame::new(vec![
            Series::new("exposure".into(), vec![2.0f64, 1.0, f64::INFINITY]).into(),
            Series::new("Rating_Factor".into(), vec![0.0f64, 0.1, 0.2]).into(),
        ])
        .unwrap();
        let plan = Plan::frequency("exposure")
            .with(Term::offset("banded", &scrambled).unwrap());

        let check = plan.check(&df, "claims").unwrap();
        assert!(!check.is_fittable(), "a bad supplied table must block the plan");
        assert!(
            check
                .issues
                .iter()
                .any(|i| i.message.contains("bounds_not_ascending")),
            "{:?}",
            check.issues.iter().map(|i| &i.code).collect::<Vec<_>>()
        );
    }

    // ------------------------------------------------------------ full circle

    #[test]
    fn fit_export_hand_edit_reload_refit() {
        let dir = std::env::temp_dir().join(format!("avenue_circle_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let df = motor(720);

        // 1. Fit a plan and write it out as a spreadsheet someone can open.
        let existing = existing_plan(&df);
        existing.to_workbook(None).unwrap().save_csv_dir(&dir).unwrap();

        // 2. Someone opens 01_region.csv and applies a 10% loading to every region.
        let path = dir.join("01_region.csv");
        let text = std::fs::read_to_string(&path).unwrap();
        let mut lines = text.lines();
        let header = lines.next().unwrap().to_string();
        assert!(header.ends_with(",Relativity"), "header was: {}", header);
        let mut edited = vec![header];
        for line in lines {
            let mut cells: Vec<String> = line.split(',').map(|c| c.to_string()).collect();
            let relativity: f64 = cells.last().unwrap().parse().unwrap();
            *cells.last_mut().unwrap() = format!("{}", relativity * 1.1);
            edited.push(cells.join(","));
        }
        std::fs::write(&path, edited.join("\n")).unwrap();

        // 3. Read it back. It is a model again, and it carries the loading.
        let loaded = Workbook::load_csv_dir(&dir).unwrap().to_model().unwrap();
        assert!(loaded.notes.is_empty(), "{:?}", loaded.notes);
        for (after, before) in factors(&loaded.model, 1).iter().zip(factors(&existing.model, 1).iter())
        {
            assert!(
                (after - before - 1.1f64.ln()).abs() < 1e-9,
                "a 10% loading on the relativity is ln(1.1) on the factor: {} vs {}",
                after,
                before
            );
        }

        // 4. Carry the edited plan as an offset and fit a new factor on top of it.
        let plan = Plan::frequency("exposure")
            .with_encoding(loaded.encoding.clone())
            .with_offset_model(&loaded.model, &loaded.table_names, "prior")
            .unwrap()
            .with(Term::categorical("telematics"));
        let refitted = plan.fit(&df, "claims", options()).unwrap();
        assert_eq!(refitted.converged(), Some(true));

        // The loading survived the whole trip untouched.
        assert_eq!(factors(&refitted.model, 2), factors(&loaded.model, 1));

        // And the new model still balances: the fitted intercept absorbed the 10%.
        let v = refitted.validate(&df, &ValidationOptions::default()).unwrap();
        assert!((v.ae_ratio - 1.0).abs() < 1e-6, "ae_ratio {}", v.ae_ratio);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_plan_carrying_a_supplied_table_still_round_trips_as_json() {
        let df = motor(240);
        let existing = existing_plan(&df);

        let plan = Plan::frequency("exposure")
            .with_encoding(existing.encoding.clone())
            .with_offset_model(&existing.model, &existing.table_names, "prior")
            .unwrap()
            .with(Term::banded("exposure", Breaks::quantile(3)));

        let json = plan.to_json().unwrap();
        let back = Plan::from_json(&json).unwrap();
        assert_eq!(
            plan, back,
            "a plan carrying tables must still be a complete description of its model"
        );

        // The table travels inside the plan, not by reference to a file.
        assert!(json.contains("\"role\": \"offset\""), "{}", &json[..400.min(json.len())]);
        assert!(
            matches!(&back.terms[0], Term::Given { role, .. } if *role == GivenRole::Offset)
        );

        // And the encoding travels with it, so the codes cannot drift.
        assert_eq!(back.encoding.as_ref().unwrap().label_for("region", 0), Some("east"));
    }

    // ------------------------------------------------------------ one type

    /// A model loaded from a workbook is not a lesser object: it scores, it validates,
    /// it reports, and it saves again. Before this, `validate` and `report` existed
    /// only on a freshly fitted plan, so saving a model and loading it back cost you
    /// the ability to measure it.
    #[test]
    fn a_loaded_model_can_do_everything_a_fitted_one_can() {
        let dir = std::env::temp_dir().join(format!("avenue_one_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let df = motor(720);

        let fitted = existing_plan(&df);
        fitted.to_workbook(None).unwrap().save_csv_dir(&dir).unwrap();

        let loaded = Workbook::load_csv_dir(&dir).unwrap().to_model().unwrap();
        assert!(!loaded.was_fitted(), "it was loaded, not fitted here");
        assert_eq!(loaded.converged(), None, "and it must not claim a fit it never did");

        // The manifest carried the response, so it knows what it is explaining.
        assert_eq!(loaded.target.as_deref(), Some("claims"));

        // Validate: the same numbers the fitted model gives.
        let a = fitted.validate(&df, &ValidationOptions::default()).unwrap();
        let b = loaded.validate(&df, &ValidationOptions::default()).unwrap();
        assert!((a.ae_ratio - b.ae_ratio).abs() < 1e-9);
        assert!((a.gini - b.gini).abs() < 1e-9);

        // Report: no fit section, and a headline that says so rather than implying one.
        let report = loaded.report(Some(&df), &ValidationOptions::default()).unwrap();
        assert!(report.fit.is_none());
        assert!(report.to_markdown().contains("loaded or converted rather than fitted"));
        assert!(report.headline.contains("actual over expected"));

        // Save again, and round-trip once more.
        loaded.to_workbook(None).unwrap().save_json(dir.join("again.json")).unwrap();
        let again = Workbook::load_json(dir.join("again.json")).unwrap().to_model().unwrap();
        assert_eq!(again.target.as_deref(), Some("claims"));

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A model with no plan cannot be measured until it is told what it explains, and
    /// the refusal says how to fix that rather than guessing at a column.
    #[test]
    fn a_model_that_does_not_know_its_response_says_so() {
        let df = motor(240);
        let fitted = existing_plan(&df);
        let bare = crate::plan::FittedModel::from_model(
            fitted.model.clone(),
            "poisson",
            fitted.table_names.clone(),
            fitted.encoding.clone(),
        );

        match bare.validate(&df, &ValidationOptions::default()) {
            Ok(_) => panic!("it cannot know what to measure"),
            Err(error) => {
                let text = format!("{}", error);
                assert!(text.contains("with_response"), "the repair must travel: {}", text);
            }
        }

        // Told what it explains, it measures like anything else — and it prepares the
        // raw frame itself, encoding `region` and deriving log(exposure) from what its
        // own tables require.
        let told = bare.with_response(
            "claims",
            Some("exposure"),
            Some(crate::plan::ExposureRole::Offset),
        );
        let v = told.validate(&df, &ValidationOptions::default()).unwrap();
        assert_eq!(v.unmatched_rows, 0, "a plan-less model must still prepare its own data");
        assert!((v.ae_ratio - 1.0).abs() < 1e-6, "ae_ratio {}", v.ae_ratio);
    }

    /// Frequency plus severity is pure premium: under a log link the factors add, so
    /// the fitted means multiply.
    #[test]
    fn a_frequency_and_a_severity_model_combine_into_pure_premium() {
        let df = motor(720);

        let frequency = existing_plan(&df);
        let severity = Plan::severity("claims")
            .with(Term::categorical("telematics"))
            .fit(&df, "claims", options())
            .unwrap();

        let pure_premium = frequency.combine(&severity).unwrap();

        // Every risk's pure premium is its frequency times its severity.
        let expected: Vec<f64> = frequency
            .predict(&df)
            .unwrap()
            .f64()
            .unwrap()
            .into_no_null_iter()
            .zip(
                severity
                    .predict(&df)
                    .unwrap()
                    .f64()
                    .unwrap()
                    .into_no_null_iter(),
            )
            .map(|(f, s)| f * s)
            .collect();
        let got: Vec<f64> = pure_premium
            .predict(&df)
            .unwrap()
            .f64()
            .unwrap()
            .into_no_null_iter()
            .collect();
        for (a, b) in got.iter().zip(expected.iter()) {
            assert!(
                (a - b).abs() < 1e-9 * b.abs().max(1.0),
                "{} should be the product {}",
                a,
                b
            );
        }

        // It is a model like any other: it saves, and it carries no fit statistics of
        // its own, because it was composed rather than fitted.
        assert!(!pure_premium.was_fitted());
        assert!(pure_premium.to_workbook(None).is_ok());
    }

    /// A LightGBM conversion is a model too — the same scoring, measuring, reporting
    /// and saving as anything else, rather than a separate object with none of it.
    #[test]
    fn a_converted_lightgbm_model_is_a_model_like_any_other() {
        let json = crate::tests::testing_utils::simple_lgbm_json();
        let converted = crate::plan::FittedModel::from_lgbm_json(&json, "max").unwrap();

        assert!(!converted.was_fitted());
        assert!(!converted.table_names.is_empty());
        // It saves as a workbook, which is what makes a converted ensemble deployable
        // as a rating plan rather than only inspectable in memory.
        assert!(converted.to_workbook(None).is_ok());
    }
}
