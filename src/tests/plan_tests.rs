//! Tests for declarative plans.
//!
//! Two things are being pinned. First, that a plan builds the tables a careful person
//! would have built by hand — the conventions it exists to absorb are exactly the ones
//! that fail silently, so "it produced a model" is not enough and the structure is
//! asserted directly. Second, that `check` reports what it decided and what is wrong
//! *before* a fit, since the whole point is to replace a sequence of failed attempts
//! with one call.

#[cfg(test)]
mod plan_tests {
    use crate::glm::GLMOptions;
    use crate::plan::{Base, Breaks, ExposureRole, Plan, Term};
    use crate::validation::{Severity, ValidationOptions};
    use polars::prelude::*;

    /// Motor-shaped data: an age band and a region drive the rate, exposure varies.
    ///
    /// Region arrives as a *string*, and age as `Int64` — numpy's default integer and
    /// the dtype `pandas.Categorical(...).codes` widens to. Both are shapes the
    /// matcher cannot read, so a plan that does not normalise them cannot fit this.
    fn motor(n: usize) -> DataFrame {
        let regions = ["north", "south", "east", "west"];
        let region: Vec<&str> = (0..n).map(|i| regions[i % 4]).collect();
        let age: Vec<i64> = (0..n).map(|i| 18 + ((i * 7) % 55) as i64).collect();
        let exposure: Vec<f64> = (0..n).map(|i| 0.5 + ((i % 4) as f64) / 4.0).collect();
        // Rate rises with region index and with age band; realised with variation.
        let claims: Vec<f64> = (0..n)
            .map(|i| {
                let base = 1.0 + (i % 4) as f64 * 0.5 + if age[i] > 45 { 1.0f64 } else { 0.0 };
                let variant = ((i / 4) % 3) as f64 - 1.0;
                ((base + variant * 0.25).max(0.0)) * exposure[i]
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

    /// `unwrap_err` needs `T: Debug`, and a `BuiltPlan` carries a `RatingModel`,
    /// which does not implement it.
    fn expect_err<T>(result: Result<T, PolarsError>) -> String {
        match result {
            Ok(_) => panic!("expected an error, got a value"),
            Err(e) => format!("{}", e),
        }
    }

    fn options() -> GLMOptions {
        GLMOptions {
            max_iterations: 500,
            tolerance: 1e-10,
            ..Default::default()
        }
    }

    // ------------------------------------------------------------ structure

    #[test]
    fn banded_terms_end_in_an_unbounded_row() {
        let df = motor(240);
        let plan = Plan::frequency("exposure").with(Term::banded(
            "driver_age",
            Breaks::explicit(vec![25.0, 40.0, 60.0]),
        ));
        let prepared = plan.prepare(&df, None).unwrap();
        let built = plan.build(&prepared).unwrap();

        // Intercept first, then one table per term.
        assert_eq!(built.model.tables.len(), 2);
        assert_eq!(built.table_names, vec!["intercept", "driver_age"]);
        assert_eq!(built.model.tables[0].data.height(), 1);

        let bounds: Vec<f64> = built.model.tables[1]
            .data
            .column("driver_age")
            .unwrap()
            .f64()
            .unwrap()
            .into_no_null_iter()
            .collect();
        // Three cut points become four bands, the last one unbounded — an observation
        // matching no row would otherwise be dropped from the term.
        assert_eq!(bounds, vec![25.0, 40.0, 60.0, f64::INFINITY]);
    }

    #[test]
    fn a_string_column_is_encoded_and_the_mapping_is_kept() {
        let df = motor(240);
        let plan = Plan::frequency("exposure").with(Term::categorical("region"));
        let prepared = plan.prepare(&df, None).unwrap();

        assert_eq!(prepared.df.column("region").unwrap().dtype(), &DataType::Int32);
        let built = plan.build(&prepared).unwrap();

        // Codes follow sorted level text, so encoding is independent of row order.
        assert_eq!(built.encoding.label_for("region", 0), Some("east"));
        assert_eq!(built.encoding.label_for("region", 1), Some("north"));
        assert_eq!(built.encoding.label_for("region", 3), Some("west"));
        assert_eq!(built.model.tables[1].data.height(), 4);
    }

    #[test]
    fn an_int64_column_is_accepted_where_the_matcher_would_have_refused_it() {
        let df = motor(240);
        assert_eq!(
            df.column("driver_age").unwrap().dtype(),
            &DataType::Int64,
            "the fixture must actually carry the awkward dtype"
        );
        let plan = Plan::frequency("exposure")
            .with(Term::banded("driver_age", Breaks::quantile(4)));
        let fitted = plan.fit(&df, "claims", options()).unwrap();
        assert_eq!(fitted.converged(), Some(true));
    }

    #[test]
    fn the_base_level_is_chosen_stated_and_placed_first() {
        let mut regions = vec!["rare"; 10];
        regions.extend(vec!["common"; 200]);
        let n = regions.len();
        let df = DataFrame::new(vec![
            Series::new("region".into(), regions).into(),
            Series::new("claims".into(), (0..n).map(|i| (i % 3) as f64).collect::<Vec<f64>>())
                .into(),
            Series::new("exposure".into(), vec![1.0; n]).into(),
        ])
        .unwrap();

        // The default anchors on the most exposed level, not the alphabetically first.
        let plan = Plan::frequency("exposure").with(Term::categorical("region"));
        let built = plan.build(&plan.prepare(&df, None).unwrap()).unwrap();
        assert_eq!(built.resolved[1].base_level.as_deref(), Some("common"));
        let codes: Vec<i32> = built.model.tables[1]
            .data
            .column("region")
            .unwrap()
            .i32()
            .unwrap()
            .into_no_null_iter()
            .collect();
        assert_eq!(
            built.encoding.label_for("region", codes[0]),
            Some("common"),
            "the base must be row 0, which is what the anchoring fixes at zero"
        );

        // And it can be named outright.
        let pinned = Plan::frequency("exposure").with(Term::categorical_based_on(
            "region",
            Base::Level { value: "rare".to_string() },
        ));
        let built = pinned.build(&pinned.prepare(&df, None).unwrap()).unwrap();
        assert_eq!(built.resolved[1].base_level.as_deref(), Some("rare"));

        // Naming one that does not exist says so, and lists what does.
        let wrong = Plan::frequency("exposure").with(Term::categorical_based_on(
            "region",
            Base::Level { value: "nowhere".to_string() },
        ));
        let err = expect_err(wrong.build(&wrong.prepare(&df, None).unwrap()));
        assert!(err.contains("nowhere"), "{}", err);
        assert!(err.contains("common"), "the message must list the real levels: {}", err);
    }

    #[test]
    fn every_rate_preset_uses_its_denominator_as_a_weight() {
        let df = motor(240);

        let frequency = Plan::frequency("exposure");
        assert_eq!(frequency.resolved_exposure_role(), ExposureRole::Weight);
        let prepared = frequency.prepare(&df, None).unwrap();
        assert!(prepared.offset_col.is_none());
        assert_eq!(prepared.weight_col.as_deref(), Some("exposure"));

        let severity = Plan::severity("claim_count");
        assert_eq!(severity.resolved_exposure_role(), ExposureRole::Weight);
        assert_eq!(severity.family, "gamma");
        let pure_premium = Plan::pure_premium("exposure");
        assert_eq!(pure_premium.family, "tweedie");
        assert_eq!(pure_premium.resolved_exposure_role(), ExposureRole::Weight);
    }

    #[test]
    fn a_variate_spends_its_degree_and_not_its_rows() {
        let df = motor(240);
        let plan = Plan::frequency("exposure")
            .with(Term::variate("driver_age", Breaks::quantile(8), 2));
        let prepared = plan.prepare(&df, None).unwrap();
        let built = plan.build(&prepared).unwrap();

        let term = &built.resolved[1];
        assert_eq!(term.kind, "variate");
        assert!(term.rows >= 4, "the table keeps its bands: {}", term.rows);
        assert_eq!(term.parameters, 2, "but spends only its degree");
        assert_eq!(
            built.model.tables[1].variate_degree(),
            Some(2),
            "the table must actually carry the variate semantics"
        );
        // Values default to band midpoints, one per row, and are reported.
        assert_eq!(term.variate_values.as_ref().unwrap().len(), term.rows);
    }

    #[test]
    fn an_interaction_crosses_its_axes_in_first_match_order() {
        let df = motor(240);
        let plan = Plan::frequency("exposure").with(Term::interaction(
            vec!["driver_age", "region"],
            vec![Some(Breaks::explicit(vec![30.0, 50.0])), None],
        ));
        let prepared = plan.prepare(&df, None).unwrap();
        let built = plan.build(&prepared).unwrap();

        // Three age bands crossed with four regions.
        assert_eq!(built.resolved[1].rows, 12);
        assert_eq!(built.table_names[1], "driver_age x region");

        // The numeric axis must be non-decreasing down the table: first-match lookup
        // over a conjunction of upper bounds is only correct in lexicographic order.
        let ages: Vec<f64> = built.model.tables[1]
            .data
            .column("driver_age")
            .unwrap()
            .f64()
            .unwrap()
            .into_no_null_iter()
            .collect();
        for pair in ages.windows(2) {
            assert!(pair[0] <= pair[1], "age bounds must ascend: {:?}", ages);
        }

        // And the whole thing must fit and score without dropping observations.
        let fitted = plan.fit(&df, "claims", options()).unwrap();
        let v = fitted.validate(&df, &ValidationOptions::default()).unwrap();
        assert_eq!(v.unmatched_rows, 0, "every row must land in the crossed table");
    }

    // ------------------------------------------------------------ checking

    #[test]
    fn check_reports_what_it_decided_before_anything_is_fitted() {
        let df = motor(480);
        let plan = Plan::frequency("exposure")
            .with(Term::banded("driver_age", Breaks::quantile(4)))
            .with(Term::categorical("region"));

        let check = plan.check(&df, "claims").unwrap();

        assert!(check.is_fittable(), "{:?}", check.issues);
        // Intercept plus two terms.
        assert_eq!(check.resolved.len(), 3);
        assert_eq!(check.resolved[0].kind, "intercept");

        let age = &check.resolved[1];
        assert_eq!(age.kind, "banded");
        let edges = age.edges.as_ref().expect("a quantile rule must state its edges");
        assert!(
            edges.last().unwrap().is_infinite(),
            "the top band is unbounded: {:?}",
            edges
        );
        assert!(edges.len() >= 2, "{:?}", edges);

        let region = &check.resolved[2];
        assert_eq!(region.kind, "categorical");
        assert_eq!(region.rows, 4);
        // Exposure rises with `i % 4`, and so does the region index, so `west` is the
        // most exposed level — not the alphabetically first, which would be `east`.
        assert_eq!(region.base_level.as_deref(), Some("west"));

        // Parameter count is the thing that decides whether a plan is affordable.
        assert_eq!(
            check.parameters,
            check.resolved.iter().map(|r| r.parameters).sum::<usize>()
        );
    }

    #[test]
    fn check_finds_the_data_problems_a_fit_would_have_hit_one_at_a_time() {
        let n = 200;
        let mut claims: Vec<Option<f64>> = (0..n).map(|i| Some((i % 3) as f64)).collect();
        claims[7] = None;
        let mut exposure: Vec<f64> = vec![1.0; n];
        exposure[11] = 0.0;
        let df = DataFrame::new(vec![
            Series::new("region".into(), (0..n).map(|i| if i % 2 == 0 { "a" } else { "b" }).collect::<Vec<&str>>()).into(),
            Series::new("flat".into(), vec![3.0f64; n]).into(),
            Series::new("claims".into(), claims).into(),
            Series::new("exposure".into(), exposure).into(),
        ])
        .unwrap();

        let plan = Plan::frequency("exposure")
            .with(Term::categorical("region"))
            .with(Term::banded("flat", Breaks::explicit(vec![5.0])));

        let check = plan.check(&df, "claims").unwrap();
        let codes: Vec<&str> = check.issues.iter().map(|i| i.code.as_str()).collect();

        assert!(codes.contains(&"target_has_nulls"), "{:?}", codes);
        assert!(codes.contains(&"non_positive_exposure"), "{:?}", codes);
        assert!(codes.contains(&"constant_feature"), "{:?}", codes);
        assert!(!check.is_fittable());

        // Ordered most severe first, so a caller reading the head sees the blockers.
        for pair in check.issues.windows(2) {
            assert!(pair[0].severity >= pair[1].severity);
        }
        // Every message names the column it is about.
        let nulls = check.issues.iter().find(|i| i.code == "target_has_nulls").unwrap();
        assert!(nulls.message.contains("claims"), "{}", nulls.message);
    }

    #[test]
    fn check_sees_two_tables_describing_the_same_driver() {
        let n = 480;
        // `band` is a coarsening of `age`: exactly the redundancy that makes a
        // backfit crawl and leaves the pair unidentified.
        let age: Vec<f64> = (0..n).map(|i| 20.0 + (i % 40) as f64).collect();
        let band: Vec<i32> = age.iter().map(|a| if *a < 40.0 { 0 } else { 1 }).collect();
        let df = DataFrame::new(vec![
            Series::new("age".into(), age).into(),
            Series::new("band".into(), band).into(),
            Series::new("claims".into(), (0..n).map(|i| (i % 3) as f64).collect::<Vec<f64>>()).into(),
            Series::new("exposure".into(), vec![1.0; n]).into(),
        ])
        .unwrap();

        let plan = Plan::frequency("exposure")
            .with(Term::banded("age", Breaks::explicit(vec![39.0])))
            .with(Term::categorical("band"));

        let check = plan.check(&df, "claims").unwrap();
        assert!(
            check.correlated_pairs.iter().any(|(a, b, rho)| {
                *rho > 0.99 && ((a == "age" && b == "band") || (a == "band" && b == "age"))
            }),
            "the aliased pair must be named: {:?}",
            check.correlated_pairs
        );
        assert!(check
            .issues
            .iter()
            .any(|i| i.code == "near_aliased_tables" && i.severity >= Severity::Medium));
    }

    #[test]
    fn check_rejects_a_variate_degree_the_bands_cannot_identify() {
        let df = motor(240);
        let plan = Plan::frequency("exposure")
            .with(Term::variate("driver_age", Breaks::explicit(vec![30.0, 50.0]), 3));
        // `check` reports a structural fault rather than throwing: a caller must be
        // able to learn everything wrong from one call.
        let check = plan.check(&df, "claims").unwrap();
        let issue = check
            .issues
            .iter()
            .find(|i| i.code == "variate_degree_too_high")
            .expect("degree 3 over 3 bands is not identified");
        assert_eq!(issue.severity, Severity::High);
        assert!(!check.is_fittable());
        assert!(
            issue.message.contains("degree 2 or lower"),
            "the message must carry the repair: {}",
            issue.message
        );

        // Building it directly is still an error — the plan really is invalid.
        assert!(plan.build(&plan.prepare(&df, None).unwrap()).is_err());
    }

    // ------------------------------------------------------------ round trip

    #[test]
    fn a_plan_round_trips_through_json_unchanged() {
        let plan = Plan::frequency("exposure")
            .with(Term::banded("driver_age", Breaks::explicit(vec![25.0, 40.0])))
            .with(Term::categorical_based_on(
                "region",
                Base::Level { value: "north".to_string() },
            ))
            .with(Term::variate("vehicle_value", Breaks::quantile(10), 2))
            .with(Term::interaction(
                vec!["driver_age", "region"],
                vec![Some(Breaks::equal_width(3)), None],
            ));

        let json = plan.to_json().unwrap();
        let back = Plan::from_json(&json).unwrap();
        assert_eq!(plan, back, "a plan is the model's source code and must survive a save");

        // And it is legible rather than an opaque blob, since a person may edit it.
        assert!(json.contains("\"driver_age\""), "{}", json);
        assert!(json.contains("\"poisson\""), "{}", json);
    }

    // ------------------------------------------------------------ end to end

    #[test]
    fn a_plan_fits_scores_and_validates_from_an_ordinary_dataframe() {
        let df = motor(480);
        let plan = Plan::frequency("exposure")
            .with(Term::banded("driver_age", Breaks::explicit(vec![30.0, 45.0, 60.0])))
            .with(Term::categorical("region"));

        let check = plan.check(&df, "claims").unwrap();
        assert!(check.is_fittable(), "{:?}", check.issues);

        let fitted = plan.fit(&df, "claims", options()).unwrap();
        assert_eq!(fitted.converged(), Some(true));
        assert_eq!(fitted.table_names, vec!["intercept", "driver_age", "region"]);

        // Scoring goes through the same encoding the fit used.
        let predictions = fitted.predict(&df).unwrap();
        assert_eq!(predictions.len(), 480);
        assert!(predictions.f64().unwrap().into_no_null_iter().all(|v| v.is_finite()));

        // A Poisson fit with a free intercept balances actual against expected.
        let v = fitted.validate(&df, &ValidationOptions::default()).unwrap();
        assert_eq!(v.unmatched_rows, 0);
        assert!((v.ae_ratio - 1.0).abs() < 1e-6, "ae_ratio {}", v.ae_ratio);

        // The rating tables carry the level text back, not just the codes.
        let tables = fitted.rating_tables().unwrap();
        let region = &tables[2];
        assert!(region.get_column_names().iter().any(|c| c.as_str() == "region_Level"));
        assert!(region.get_column_names().iter().any(|c| c.as_str() == "Relativity"));
        assert!(region.get_column_names().iter().any(|c| c.as_str() == "Standard_Error"));
        let status: Vec<&str> = region
            .column("Status")
            .unwrap()
            .str()
            .unwrap()
            .into_no_null_iter()
            .collect();
        assert_eq!(status[0], "reference", "row 0 is the anchored base level");
    }

    #[test]
    fn scoring_data_with_an_unseen_level_is_reported_rather_than_recoded() {
        let df = motor(240);
        let plan = Plan::frequency("exposure").with(Term::categorical("region"));
        let fitted = plan.fit(&df, "claims", options()).unwrap();

        let mut regions: Vec<&str> = (0..240).map(|i| ["north", "south", "east", "west"][i % 4]).collect();
        regions[0] = "atlantis";
        let holdout = DataFrame::new(vec![
            Series::new("region".into(), regions).into(),
            Series::new("driver_age".into(), (0..240).map(|i| 18 + ((i * 7) % 55) as i64).collect::<Vec<i64>>()).into(),
            Series::new("exposure".into(), vec![1.0f64; 240]).into(),
            Series::new("claims".into(), (0..240).map(|i| (i % 3) as f64).collect::<Vec<f64>>()).into(),
        ])
        .unwrap();

        // The unseen level must not silently collide with a real one.
        let v = fitted.validate(&holdout, &ValidationOptions::default()).unwrap();
        assert_eq!(v.unmatched_rows, 1);
        assert!(v
            .warnings
            .iter()
            .any(|w| w.code == "unmatched_observations" && w.severity == Severity::High));
    }

    #[test]
    fn the_family_on_the_plan_overrides_whatever_the_options_say() {
        let df = motor(240);
        let plan = Plan::pure_premium("exposure")
            .with_tweedie_power(1.7)
            .with(Term::categorical("region"));

        // Options arriving with a contradictory family must not win: the plan is the
        // single authority, so the likelihood and the link cannot disagree.
        let fitted = plan
            .fit(
                &df,
                "claims",
                GLMOptions {
                    objective: "gaussian".to_string(),
                    tweedie_power: 1.1,
                    ..options()
                },
            )
            .unwrap();
        assert_eq!(fitted.family, "tweedie");
        assert_eq!(fitted.model.get_link_function(), "log");
    }

    #[test]
    fn an_empty_plan_and_a_missing_column_both_explain_themselves() {
        let df = motor(120);

        let empty = Plan::frequency("exposure");
        let err = expect_err(empty.build(&empty.prepare(&df, None).unwrap()));
        assert!(err.contains("at least one term"), "{}", err);

        let missing = Plan::frequency("exposure").with(Term::categorical("postcode"));
        let err = expect_err(missing.prepare(&df, None));
        assert!(err.contains("postcode"), "{}", err);
        assert!(err.contains("Columns present"), "must list what is there: {}", err);
    }

    #[test]
    fn two_terms_on_one_column_are_refused_rather_than_silently_aliased() {
        let df = motor(240);
        let plan = Plan::frequency("exposure")
            .with(Term::banded("driver_age", Breaks::quantile(3)))
            .with(Term::banded("driver_age", Breaks::quantile(5)));
        let err = expect_err(plan.build(&plan.prepare(&df, None).unwrap()));
        assert!(err.contains("driver_age"), "{}", err);
    }
}
