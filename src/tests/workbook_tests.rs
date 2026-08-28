//! Tests for the model-as-a-file artifact.
//!
//! Two things are pinned. That a model survives the trip to disk and back *as the model
//! it was* — including everything the tables alone cannot say, which is where the
//! previous round trip quietly lost the offset flags, the row locks and the variates.
//! And that a hand-edited file which is wrong fails loudly, because every one of those
//! faults otherwise mis-prices in silence.

#[cfg(test)]
mod workbook_tests {
    use crate::plan::Encoding;
    use crate::rating_model::{LinkFunction, RatingModel, RatingTable};
    use crate::workbook::{Scale, Workbook};
    use polars::prelude::*;
    use std::collections::BTreeMap;

    /// `unwrap_err` needs `T: Debug`, and these carry a `RatingModel`, which has none.
    fn expect_err<T>(result: Result<T, PolarsError>) -> String {
        match result {
            Ok(_) => panic!("expected the call to be refused"),
            Err(e) => format!("{}", e),
        }
    }

    fn intercept(value: f64) -> DataFrame {
        DataFrame::new(vec![Series::new("Rating_Factor".into(), vec![value]).into()]).unwrap()
    }

    fn banded(bounds: Vec<f64>, factors: Vec<f64>) -> DataFrame {
        DataFrame::new(vec![
            Series::new("age".into(), bounds).into(),
            Series::new("Rating_Factor".into(), factors).into(),
        ])
        .unwrap()
    }

    fn categorical(codes: Vec<i32>, factors: Vec<f64>) -> DataFrame {
        DataFrame::new(vec![
            Series::new("region".into(), codes).into(),
            Series::new("Rating_Factor".into(), factors).into(),
        ])
        .unwrap()
    }

    /// A three-table log-link model: intercept, an age band, a region factor.
    fn model() -> RatingModel {
        RatingModel::new(
            vec![
                RatingTable::new(intercept(-0.7), None),
                RatingTable::new(
                    banded(vec![25.0, 45.0, 65.0, f64::INFINITY], vec![0.0, 0.2, 0.5, 0.6]),
                    None,
                ),
                RatingTable::new(categorical(vec![0, 1, 2], vec![0.0, 0.15, -0.1]), None),
            ],
            LinkFunction::Log,
        )
    }

    fn names() -> Vec<String> {
        ["intercept", "age", "region"]
            .iter()
            .map(|s| s.to_string())
            .collect()
    }

    fn encoding() -> Encoding {
        let mut maps = BTreeMap::new();
        maps.insert(
            "region".to_string(),
            vec![
                ("east".to_string(), 0),
                ("north".to_string(), 1),
                ("west".to_string(), 2),
            ],
        );
        Encoding { maps }
    }

    fn scoring() -> DataFrame {
        DataFrame::new(vec![
            Series::new("age".into(), vec![22.0f64, 35.0, 55.0, 80.0]).into(),
            Series::new("region".into(), vec![0i32, 1, 2, 1]).into(),
        ])
        .unwrap()
    }

    fn predictions(model: &RatingModel) -> Vec<f64> {
        model
            .predict(&scoring())
            .unwrap()
            .f64()
            .unwrap()
            .into_no_null_iter()
            .collect()
    }

    fn book(model: &RatingModel, scale: Option<Scale>) -> Workbook {
        Workbook::from_model(model, "poisson", &names(), &encoding(), 1.5, scale).unwrap()
    }

    // ------------------------------------------------------------ round trip

    #[test]
    fn a_model_survives_json_as_the_model_it_was() {
        let original = model();
        let before = predictions(&original);

        let json = book(&original, None).to_json().unwrap();
        let loaded = Workbook::from_json(&json).unwrap().to_model().unwrap();

        // The relativity scale writes exp(factor) and reads back ln(relativity), and
        // that pair is not bit-exact: predictions agree to a couple of units in the
        // last place, not to the bit. That is the price of writing the file in the
        // units a person edits, and it is far below anything a rate depends on.
        for (after, before) in predictions(&loaded.model).iter().zip(before.iter()) {
            assert!(
                (after - before).abs() <= 1e-12 * before.abs().max(1.0),
                "{} vs {}",
                after,
                before
            );
        }
        assert_eq!(loaded.table_names, names());
        assert_eq!(loaded.family, "poisson");
        assert_eq!(loaded.encoding.label_for("region", 1), Some("north"));
        assert!(loaded.issues.is_empty(), "{:?}", loaded.issues);
    }

    /// The factor scale has no such conversion, so it must be exact to the bit — which
    /// is what to use when two models are being compared or a fit is being reproduced.
    #[test]
    fn the_factor_scale_round_trips_bit_for_bit() {
        let original = model();
        let json = book(&original, Some(Scale::Factor)).to_json().unwrap();
        let loaded = Workbook::from_json(&json).unwrap().to_model().unwrap();

        for table in 0..original.tables.len() {
            let before: Vec<u64> = original.tables[table]
                .data
                .column("Rating_Factor")
                .unwrap()
                .f64()
                .unwrap()
                .into_no_null_iter()
                .map(f64::to_bits)
                .collect();
            let after: Vec<u64> = loaded.model.tables[table]
                .data
                .column("Rating_Factor")
                .unwrap()
                .f64()
                .unwrap()
                .into_no_null_iter()
                .map(f64::to_bits)
                .collect();
            assert_eq!(before, after, "table {} changed on the factor scale", table);
        }
        assert_eq!(predictions(&loaded.model), predictions(&original));
    }

    #[test]
    fn the_infinite_top_band_survives_both_formats() {
        let original = model();
        let json = book(&original, None).to_json().unwrap();
        let loaded = Workbook::from_json(&json).unwrap().to_model().unwrap();
        let bounds: Vec<f64> = loaded.model.tables[1]
            .data
            .column("age")
            .unwrap()
            .f64()
            .unwrap()
            .into_no_null_iter()
            .collect();
        assert!(
            bounds.last().unwrap().is_infinite(),
            "the unbounded top band must round-trip, got {:?}",
            bounds
        );
        // JSON has no representation for infinity, so it must travel as text.
        assert!(json.contains("\"inf\""), "infinity must be written explicitly");
    }

    #[test]
    fn offsets_locked_rows_and_variates_all_survive() {
        let mut original = model();
        original.tables[2] = original.tables[2].clone().as_offset();
        original.tables[1].set_row_offset(2, true);

        let json = book(&original, None).to_json().unwrap();
        let loaded = Workbook::from_json(&json).unwrap().to_model().unwrap();

        assert!(
            loaded.model.tables[2].metadata.is_offset,
            "an offset table must reload as an offset, or an existing rating plan cannot \
             be carried into a new model"
        );
        assert!(loaded.model.tables[1].is_row_offset(2), "a locked row must stay locked");
        assert!(!loaded.model.tables[1].is_row_offset(0));

        // A variate too, which the plain table round trip also used to drop.
        let mut with_variate = model();
        with_variate.tables[1] = with_variate.tables[1]
            .clone()
            .as_polynomial_variate(vec![20.0, 35.0, 55.0, 70.0], 1)
            .unwrap();
        // Put its factors on the curve so nothing is flagged as edited.
        let slope = 0.01;
        let onto: Vec<f64> = [20.0, 35.0, 55.0, 70.0]
            .iter()
            .map(|v| slope * (v - 20.0))
            .collect();
        let mut data = with_variate.tables[1].data.clone();
        data.with_column(Series::new("Rating_Factor".into(), onto)).unwrap();
        with_variate.tables[1] = RatingTable::new(data, None)
            .as_polynomial_variate(vec![20.0, 35.0, 55.0, 70.0], 1)
            .unwrap();

        let json = book(&with_variate, None).to_json().unwrap();
        let loaded = Workbook::from_json(&json).unwrap().to_model().unwrap();
        assert_eq!(loaded.model.tables[1].variate_degree(), Some(1));
        assert_eq!(
            loaded.model.tables[1].variate_values(),
            Some([20.0, 35.0, 55.0, 70.0].as_slice())
        );
        assert!(loaded.issues.is_empty(), "{:?}", loaded.issues);
    }

    #[test]
    fn a_csv_directory_round_trips_and_is_readable() {
        let dir = std::env::temp_dir().join(format!("avenue_wb_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);

        let original = model();
        let before = predictions(&original);
        book(&original, None).save_csv_dir(&dir).unwrap();

        // One file per table plus the manifest, named so a person can find them.
        let age_csv = std::fs::read_to_string(dir.join("01_age.csv")).unwrap();
        assert!(age_csv.starts_with("age,Relativity"), "header was: {}", age_csv.lines().next().unwrap());
        assert!(age_csv.contains("inf,"), "the top band must read as inf: {}", age_csv);
        assert!(dir.join("manifest.json").exists());

        let loaded = Workbook::load_csv_dir(&dir).unwrap().to_model().unwrap();
        for (a, b) in predictions(&loaded.model).iter().zip(before.iter()) {
            assert!((a - b).abs() < 1e-12, "{} vs {}", a, b);
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    // ------------------------------------------------------------ editing

    #[test]
    fn editing_a_relativity_in_the_file_changes_the_model_by_exactly_that_much() {
        let dir = std::env::temp_dir().join(format!("avenue_edit_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        book(&model(), None).save_csv_dir(&dir).unwrap();

        // What a person actually does: open the file, change one number.
        let path = dir.join("02_region.csv");
        let text = std::fs::read_to_string(&path).unwrap();
        let mut lines: Vec<String> = text.lines().map(|l| l.to_string()).collect();
        // Row for code 1 is the second data line; set its relativity to exactly 2.
        lines[2] = "1,2.0".to_string();
        std::fs::write(&path, lines.join("\n")).unwrap();

        let loaded = Workbook::load_csv_dir(&dir).unwrap().to_model().unwrap();
        let factor = loaded.model.tables[2]
            .data
            .column("Rating_Factor")
            .unwrap()
            .f64()
            .unwrap()
            .get(1)
            .unwrap();
        assert!(
            (factor - 2.0f64.ln()).abs() < 1e-12,
            "a relativity of 2 is a factor of ln(2); got {}",
            factor
        );

        // And the prediction moves by exactly the factor of two that was asked for.
        let before = predictions(&model());
        let after = predictions(&loaded.model);
        // Row 1 of the scoring frame is region 1.
        let expected = before[1] * 2.0 / model().tables[2].data
            .column("Rating_Factor").unwrap().f64().unwrap().get(1).unwrap().exp();
        assert!((after[1] - expected).abs() < 1e-12, "{} vs {}", after[1], expected);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_scale_is_declared_so_there_is_never_a_second_column_to_edit_by_mistake() {
        // A log-link workbook is written as relativities, which is what gets edited.
        let relativity = book(&model(), None);
        assert_eq!(relativity.manifest.scale, Scale::Relativity);
        let columns: Vec<String> = relativity.tables[2]
            .get_column_names()
            .iter()
            .map(|c| c.to_string())
            .collect();
        assert!(columns.contains(&"Relativity".to_string()));
        assert!(
            !columns.contains(&"Rating_Factor".to_string()),
            "two columns encoding one truth is how an edit gets silently ignored: {:?}",
            columns
        );
        // exp(0.15) for the second region.
        let written = relativity.tables[2].column("Relativity").unwrap().f64().unwrap().get(1).unwrap();
        assert!((written - 0.15f64.exp()).abs() < 1e-12);

        // The factor scale is available when asked for, and carries the other column.
        let factor = book(&model(), Some(Scale::Factor));
        assert_eq!(factor.manifest.scale, Scale::Factor);
        assert!(factor.tables[2]
            .get_column_names()
            .iter()
            .any(|c| c.as_str() == "Rating_Factor"));
    }

    #[test]
    fn relativities_are_refused_where_they_would_mean_nothing() {
        let gaussian = RatingModel::new(
            vec![
                RatingTable::new(intercept(0.0), None),
                RatingTable::new(categorical(vec![0, 1], vec![0.0, 1.0]), None),
            ],
            LinkFunction::Identity,
        );
        let error = expect_err(Workbook::from_model(
            &gaussian,
            "gaussian",
            &["intercept".to_string(), "region".to_string()],
            &Encoding::default(),
            1.5,
            Some(Scale::Relativity),
        ));
        assert!(error.contains("log link"), "{}", error);
        // And the default for such a model is the factor scale.
        assert_eq!(Scale::default_for("identity"), Scale::Factor);
        assert_eq!(Scale::default_for("log"), Scale::Relativity);
    }

    // ------------------------------------------------------------ bad edits

    /// Load a workbook whose table 1 has been replaced, and return the error text.
    fn load_with_table(table: DataFrame, scale: Scale) -> String {
        let mut workbook = book(&model(), Some(scale));
        workbook.tables[1] = table;
        match workbook.to_model() {
            Ok(_) => panic!("expected the load to be refused"),
            Err(error) => format!("{}", error),
        }
    }

    #[test]
    fn out_of_order_bands_are_refused_instead_of_mis_binning() {
        // Exactly the edit that used to give age 22 the factor for the 25-45 band.
        let scrambled = DataFrame::new(vec![
            Series::new("age".into(), vec![45.0f64, 25.0, 65.0, f64::INFINITY]).into(),
            Series::new("Rating_Factor".into(), vec![0.2f64, 0.0, 0.5, 0.6]).into(),
        ])
        .unwrap();
        let error = load_with_table(scrambled, Scale::Factor);
        assert!(error.contains("bounds_not_ascending"), "{}", error);
        assert!(error.contains("row 1"), "the message must locate the fault: {}", error);
        assert!(error.contains("Sort the rows"), "and carry the repair: {}", error);
    }

    #[test]
    fn a_missing_unbounded_band_is_refused_instead_of_scoring_nan() {
        let truncated = DataFrame::new(vec![
            Series::new("age".into(), vec![25.0f64, 45.0, 65.0]).into(),
            Series::new("Rating_Factor".into(), vec![0.0f64, 0.2, 0.5]).into(),
        ])
        .unwrap();
        let error = load_with_table(truncated, Scale::Factor);
        assert!(error.contains("no_unbounded_band"), "{}", error);
        assert!(error.contains("inf"), "the repair names the fix: {}", error);
    }

    #[test]
    fn duplicate_and_empty_rows_are_refused() {
        let duplicated = DataFrame::new(vec![
            Series::new("age".into(), vec![25.0f64, 25.0, 65.0, f64::INFINITY]).into(),
            Series::new("Rating_Factor".into(), vec![0.0f64, 0.2, 0.5, 0.6]).into(),
        ])
        .unwrap();
        let error = load_with_table(duplicated, Scale::Factor);
        assert!(
            error.contains("duplicate_bound") || error.contains("duplicate_row"),
            "{}",
            error
        );

        let blank = DataFrame::new(vec![
            Series::new("age".into(), vec![25.0f64, 45.0, f64::INFINITY]).into(),
            Series::new("Rating_Factor".into(), vec![Some(0.0f64), None, Some(0.6)]).into(),
        ])
        .unwrap();
        let error = load_with_table(blank, Scale::Factor);
        assert!(error.contains("null_factor"), "{}", error);
    }

    #[test]
    fn a_dtype_the_matcher_cannot_read_is_refused_by_name() {
        // Int64 is numpy's default integer, and the shape that used to be dropped
        // during construction — leaving the column constraining nothing.
        let wrong = DataFrame::new(vec![
            Series::new("age".into(), vec![25i64, 45, 65]).into(),
            Series::new("Rating_Factor".into(), vec![0.0f64, 0.2, 0.5]).into(),
        ])
        .unwrap();
        let error = load_with_table(wrong, Scale::Factor);
        assert!(error.contains("unreadable_dtype"), "{}", error);
        assert!(error.contains("Int64"), "the message must name the dtype: {}", error);
    }

    #[test]
    fn a_non_positive_relativity_is_refused() {
        let negative = DataFrame::new(vec![
            Series::new("age".into(), vec![25.0f64, 45.0, f64::INFINITY]).into(),
            Series::new("Relativity".into(), vec![1.0f64, -0.5, 1.2]).into(),
        ])
        .unwrap();
        let error = load_with_table(negative, Scale::Relativity);
        assert!(error.contains("non_positive_relativity"), "{}", error);
    }

    #[test]
    fn every_fault_is_reported_at_once_not_just_the_first() {
        // Three separate faults in one table.
        let broken = DataFrame::new(vec![
            Series::new("age".into(), vec![45.0f64, 25.0, 65.0]).into(),
            Series::new("Rating_Factor".into(), vec![Some(0.2f64), None, Some(f64::NAN)]).into(),
        ])
        .unwrap();
        let error = load_with_table(broken, Scale::Factor);
        for code in ["null_factor", "non_finite_factor", "bounds_not_ascending", "no_unbounded_band"] {
            assert!(error.contains(code), "missing {}: {}", code, error);
        }
        assert!(
            error.contains("4 problems found"),
            "the count must be stated so a caller knows the list is complete: {}",
            error
        );
    }

    #[test]
    fn an_intercept_table_that_is_not_one_row_is_refused() {
        let mut workbook = book(&model(), Some(Scale::Factor));
        workbook.tables[0] = DataFrame::new(vec![Series::new(
            "Rating_Factor".into(),
            vec![0.0f64, 1.0],
        )
        .into()])
        .unwrap();
        let error = expect_err(workbook.to_model());
        assert!(error.contains("intercept_not_single_row"), "{}", error);
    }

    // ------------------------------------------------------------ manifest

    #[test]
    fn a_workbook_from_the_future_is_refused_rather_than_misread() {
        let mut workbook = book(&model(), None);
        workbook.manifest.format_version = crate::workbook::FORMAT_VERSION + 1;
        let error = expect_err(workbook.to_model());
        assert!(error.contains("Upgrade avenue_model"), "{}", error);
    }

    #[test]
    fn the_manifest_records_its_provenance() {
        let workbook = book(&model(), None);
        assert_eq!(workbook.manifest.avenue_version, env!("CARGO_PKG_VERSION"));
        assert!(workbook.manifest.created.is_some());
        assert_eq!(workbook.manifest.link, "log");
        assert_eq!(workbook.manifest.family, "poisson");
        assert_eq!(workbook.manifest.tables.len(), 3);
        assert_eq!(workbook.manifest.tables[2].name, "region");
    }

    #[test]
    fn a_variate_whose_factors_were_edited_off_the_curve_says_so_without_refusing() {
        let mut original = model();
        original.tables[1] = original.tables[1]
            .clone()
            .as_polynomial_variate(vec![20.0, 35.0, 55.0, 70.0], 1)
            .unwrap();
        let mut workbook = book(&original, Some(Scale::Factor));

        // Bend one factor off the line the manifest records.
        let mut edited = workbook.tables[1].clone();
        edited
            .with_column(Series::new(
                "Rating_Factor".into(),
                vec![0.0f64, 0.2, 0.9, 0.6],
            ))
            .unwrap();
        workbook.tables[1] = edited;

        let loaded = workbook.to_model().unwrap();
        // Not refused: the table says what it says, and predictions follow it.
        let issue = loaded
            .issues
            .iter()
            .find(|i| i.code == "variate_factors_edited")
            .expect("an edited variate must be reported");
        assert!(!issue.blocking, "an edited curve is a note, not a refusal");
        assert!(issue.message.contains("refit"), "{}", issue.message);
    }

    /// A file that names its levels `3` forces the reader to the manifest before they
    /// can change anything. The label is the key, not a second copy of the factor.
    #[test]
    fn categorical_levels_are_written_by_name_and_read_back_by_name() {
        let workbook = book(&model(), None);
        let region = &workbook.tables[2];
        assert_eq!(
            region.column("region").unwrap().dtype(),
            &DataType::String,
            "levels must be written as text"
        );
        let labels: Vec<&str> = region
            .column("region")
            .unwrap()
            .str()
            .unwrap()
            .into_no_null_iter()
            .collect();
        assert_eq!(labels, vec!["east", "north", "west"]);

        // And they come back as the codes the matcher needs.
        let loaded = workbook.to_model().unwrap();
        let codes: Vec<i32> = loaded.model.tables[2]
            .data
            .column("region")
            .unwrap()
            .i32()
            .unwrap()
            .into_no_null_iter()
            .collect();
        assert_eq!(codes, vec![0, 1, 2]);
        assert_eq!(predictions(&loaded.model).len(), 4);
    }

    #[test]
    fn a_level_the_encoding_has_never_seen_is_reported_by_name() {
        let mut workbook = book(&model(), Some(Scale::Factor));
        let mut edited = workbook.tables[2].clone();
        edited
            .with_column(Series::new(
                "region".into(),
                vec!["east", "notaregion", "west"],
            ))
            .unwrap();
        workbook.tables[2] = edited;

        let error = expect_err(workbook.to_model());
        assert!(error.contains("unknown_level"), "{}", error);
        assert!(error.contains("notaregion"), "the message must quote it: {}", error);
        assert!(
            error.contains("east") && error.contains("north"),
            "and list the levels that do exist: {}",
            error
        );
    }

    /// A hand-written file that uses codes instead of names still works, so nobody is
    /// forced to look up a label they already know the code for.
    #[test]
    fn raw_codes_are_still_accepted_where_a_name_was_expected() {
        let mut workbook = book(&model(), Some(Scale::Factor));
        let mut edited = workbook.tables[2].clone();
        edited
            .with_column(Series::new("region".into(), vec!["0", "1", "2"]))
            .unwrap();
        workbook.tables[2] = edited;

        let loaded = workbook.to_model().unwrap();
        let codes: Vec<i32> = loaded.model.tables[2]
            .data
            .column("region")
            .unwrap()
            .i32()
            .unwrap()
            .into_no_null_iter()
            .collect();
        assert_eq!(codes, vec![0, 1, 2]);
    }
}
