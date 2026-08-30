"""Tests for the model-as-a-file workflow.

Run after ``maturin develop --release`` with
``python -m unittest tests.test_workbook_api``.

The Rust suite pins the arithmetic and the structural checks. What is exercised here is
the workflow itself: fit, write a spreadsheet, edit it the way a person would, read it
back, and carry it into a new model as a fixed offset.
"""

import csv
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import polars as pl

from avenue_model import GLMOptions, Plan, Workbook


def motor(n=720):
    regions = ["north", "south", "east", "west"]
    rows = []
    for i in range(n):
        exposure = 0.5 + (i % 4) / 4
        rate = (
            0.8
            * [1.0, 1.25, 1.5, 1.75][i % 4]
            * [0.8, 1.0, 1.2][(i // 4) % 3]
            * [0.9, 1.0, 1.1][(i // 12) % 3]
        )
        rows.append(
            {
                "region": regions[i % 4],
                "telematics": (i // 4) % 3,
                "exposure": exposure,
                "claims": rate * exposure,
                "frequency": rate,
            }
        )
    return pl.DataFrame(
        rows,
        schema={
            "region": pl.String,
            "telematics": pl.Int32,
            "exposure": pl.Float64,
            "claims": pl.Float64,
            "frequency": pl.Float64,
        },
    )


class WorkbookTests(unittest.TestCase):
    def setUp(self):
        self.df = motor()
        self.dir = Path(tempfile.mkdtemp())
        self.fitted = (
            Plan.frequency("exposure")
            .categorical("region")
            .fit(self.df, "frequency", GLMOptions())
        )

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_a_log_link_model_is_written_as_relativities(self):
        book = self.fitted.to_workbook()
        self.assertEqual(book.scale, "relativity")
        self.assertEqual(book.link, "log")

        region = book.tables[book.table_names.index("region")]
        self.assertIn("Relativity", region.columns)
        # One column, so there is never a second place for an edit to be ignored.
        self.assertNotIn("Rating_Factor", region.columns)
        self.assertNotIn("Standard_Error", region.columns)

    def test_a_csv_directory_is_something_a_person_can_open(self):
        self.fitted.to_workbook().save_csv_dir(str(self.dir))

        files = sorted(p.name for p in self.dir.iterdir())
        self.assertIn("manifest.json", files)
        self.assertTrue(any(f.endswith("_region.csv") for f in files))

        region_file = next(p for p in self.dir.iterdir() if p.name.endswith("_region.csv"))
        with region_file.open() as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(set(rows[0]), {"region", "Relativity"})
        self.assertEqual(len(rows), 4)

        # The manifest carries what the CSVs cannot say.
        manifest = json.loads((self.dir / "manifest.json").read_text())
        self.assertEqual(manifest["family"], "poisson")
        self.assertEqual(manifest["scale"], "relativity")
        self.assertIn("region", manifest["encodings"])

    def test_a_model_survives_the_round_trip(self):
        before = self.fitted.predict(self.df)["predictions"].to_list()
        self.fitted.to_workbook().save_csv_dir(str(self.dir))

        loaded = Workbook.load_csv_dir(str(self.dir)).to_model()
        self.assertEqual(loaded.family, "poisson")
        self.assertEqual(loaded.notes, [])

        after = loaded.predict(self.df)["predictions"].to_list()
        for a, b in zip(after, before):
            self.assertAlmostEqual(a, b, places=10)

    def test_editing_a_relativity_moves_the_model_by_exactly_that_much(self):
        self.fitted.to_workbook().save_csv_dir(str(self.dir))
        region_file = next(p for p in self.dir.iterdir() if p.name.endswith("_region.csv"))

        # Apply a 10% loading to every region, the way someone would in a spreadsheet.
        lines = region_file.read_text().strip().split("\n")
        header, body = lines[0], lines[1:]
        edited = [header]
        for line in body:
            cells = line.split(",")
            cells[-1] = str(float(cells[-1]) * 1.1)
            edited.append(",".join(cells))
        region_file.write_text("\n".join(edited))

        before = self.fitted.predict(self.df)["predictions"].to_list()
        loaded = Workbook.load_csv_dir(str(self.dir)).to_model()
        after = loaded.predict(self.df)["predictions"].to_list()
        for a, b in zip(after, before):
            self.assertAlmostEqual(a, b * 1.1, places=10)

    def test_a_bad_edit_is_refused_with_every_fault_named(self):
        self.fitted.to_workbook().save_csv_dir(str(self.dir))
        region_file = next(p for p in self.dir.iterdir() if p.name.endswith("_region.csv"))

        # Blank one relativity and make another negative.
        lines = region_file.read_text().strip().split("\n")
        lines[1] = lines[1].split(",")[0] + ","
        lines[2] = lines[2].split(",")[0] + ",-1.0"
        region_file.write_text("\n".join(lines))

        with self.assertRaises(ValueError) as caught:
            Workbook.load_csv_dir(str(self.dir)).to_model()
        message = str(caught.exception)
        self.assertIn("null_factor", message)
        self.assertIn("non_positive_relativity", message)
        self.assertIn("row", message)


class CompositionTests(unittest.TestCase):
    """An existing plan, held fixed, with a new factor fitted on top."""

    def setUp(self):
        self.df = motor()
        self.dir = Path(tempfile.mkdtemp())
        existing = (
            Plan.frequency("exposure")
            .categorical("region")
            .fit(self.df, "frequency", GLMOptions())
        )
        existing.to_workbook().save_csv_dir(str(self.dir))
        self.loaded = Workbook.load_csv_dir(str(self.dir)).to_model()

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_a_carried_plan_is_held_fixed_and_costs_no_parameters(self):
        plan = (
            Plan.frequency("exposure")
            .offset_model(self.loaded, prefix="prior")
            .categorical("telematics")
        )

        check = plan.check(self.df, "frequency")
        self.assertTrue(check.is_fittable, check.issues)
        by_name = {term["name"]: term for term in check.resolved}
        self.assertEqual(by_name["prior.region"]["kind"], "offset")
        self.assertEqual(by_name["prior.region"]["parameters"], 0)

        fitted = plan.fit(self.df, "frequency", GLMOptions())
        self.assertTrue(fitted.converged)

        # The carried table came out exactly as it went in.
        carried = fitted.rating_tables()[fitted.table_names.index("prior.region")]
        original = self.loaded.rating_model.model_tables()[self.loaded.table_names.index("region")]
        self.assertEqual(
            carried["Rating_Factor"].to_list(), original["Rating_Factor"].to_list()
        )

        # And the model still balances.
        v = fitted.validate(self.df)
        self.assertEqual(v.unmatched_rows, 0)
        self.assertAlmostEqual(v.ae_ratio, 1.0, places=6)

    def test_a_supplied_table_can_define_the_shape_instead(self):
        region = self.loaded.rating_model.model_tables()[self.loaded.table_names.index("region")]
        plan = Plan.frequency("exposure").given("region", region)

        fitted = plan.fit(self.df, "frequency", GLMOptions())
        by_name = {term["name"]: term for term in fitted.resolved}
        self.assertEqual(by_name["region"]["kind"], "given")
        # A given structure is estimated; an offset is not.
        self.assertEqual(by_name["region"]["parameters"], region.height - 1)

    def test_the_plan_still_serialises_with_the_tables_inside_it(self):
        plan = (
            Plan.frequency("exposure")
            .offset_model(self.loaded, prefix="prior")
            .categorical("telematics")
        )
        recovered = Plan.from_json(plan.to_json())
        self.assertEqual(recovered.to_json(), plan.to_json())
        self.assertIn("prior.region", recovered.term_names)


class OneTypeTests(unittest.TestCase):
    """Every route produces the same object, so every capability is reachable."""

    def setUp(self):
        self.df = motor()
        self.dir = Path(tempfile.mkdtemp())
        self.fitted = (
            Plan.frequency("exposure")
            .categorical("region")
            .fit(self.df, "frequency", GLMOptions())
        )
        self.fitted.to_workbook().save_csv_dir(str(self.dir))
        self.loaded = Workbook.load_csv_dir(str(self.dir)).to_model()

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_a_loaded_model_validates_reports_and_saves_like_a_fitted_one(self):
        self.assertFalse(self.loaded.was_fitted)
        # Not fitted here, so it must not claim a fit that did or did not converge.
        self.assertIsNone(self.loaded.converged)
        self.assertEqual(self.loaded.target, "frequency")

        a = self.fitted.validate(self.df)
        b = self.loaded.validate(self.df)
        self.assertAlmostEqual(a.ae_ratio, b.ae_ratio, places=9)
        self.assertAlmostEqual(a.gini, b.gini, places=9)

        report = self.loaded.report(self.df)
        self.assertEqual(report.fit_summary, {})
        self.assertIn("loaded or converted rather than fitted", report.markdown)

        self.loaded.to_workbook().save_json(str(self.dir / "again.json"))
        again = Workbook.load_json(str(self.dir / "again.json")).to_model()
        self.assertEqual(again.target, "frequency")

    def test_frequency_plus_severity_is_pure_premium(self):
        severity = (
            Plan.severity("claims")
            .categorical("telematics")
            .fit(self.df, "claims", GLMOptions())
        )
        pure_premium = self.fitted + severity

        f = self.fitted.predict(self.df)["predictions"].to_list()
        s = severity.predict(self.df)["predictions"].to_list()
        pp = pure_premium.predict(self.df)["predictions"].to_list()
        for got, want in zip(pp, (a * b for a, b in zip(f, s))):
            self.assertAlmostEqual(got, want, places=9)

        self.assertFalse(pure_premium.was_fitted)

    def test_the_check_is_carried_so_plan_findings_arrive_unasked(self):
        # `report` takes no check argument: forgetting it used to produce a cleaner
        # report, which is the wrong way round.
        report = self.fitted.report(self.df)
        self.assertTrue(all(f["stage"] in {"plan", "fit", "validation"} for f in report.findings))

    def test_an_unfittable_plan_is_refused_rather_than_fitted(self):
        broken = self.df.with_columns(
            pl.when(pl.int_range(pl.len()) == 0).then(None).otherwise(pl.col("frequency")).alias("frequency")
        )
        with self.assertRaises(ValueError) as caught:
            Plan.frequency("exposure").categorical("region").fit(broken, "frequency", GLMOptions())
        self.assertIn("cannot be fitted", str(caught.exception))


if __name__ == "__main__":
    unittest.main()


class ConvertedCategoryNamesTests(unittest.TestCase):
    """Naming a converted model's category codes, so its tables can be read.

    LightGBM is handed numbers, so a converted model knows codes and not what they
    stood for, and a rating table whose column reads `3` is not something anyone can
    file. `with_categories` supplies the names the workbook writer already knows how
    to print — and must do so without touching a single prediction, since the model
    matches on the code either way.
    """

    def _booster_and_levels(self):
        import numpy as np

        lgb = __import__("lightgbm")
        rng = np.random.default_rng(4)
        levels = ["north", "south", "east", "west"]
        codes = rng.integers(0, len(levels), size=800).astype(np.int32)
        driver = rng.uniform(18, 80, size=800)
        y = 0.05 + 0.02 * codes + 0.001 * driver + rng.normal(0, 0.01, 800)
        frame = np.column_stack([driver, codes]).astype(float)
        data = lgb.Dataset(frame, label=y, feature_name=["driver_age", "region"],
                           categorical_feature=["region"],
                           params={"feature_pre_filter": False})
        booster = lgb.train({"objective": "regression", "num_leaves": 8, "verbose": -1,
                             "min_data_in_leaf": 20, "seed": 1, "deterministic": True,
                             "num_threads": 1}, data, num_boost_round=25)
        return booster, frame, levels

    def test_naming_categories_renames_rows_without_moving_predictions(self):
        import json
        import os
        import tempfile

        import numpy as np

        from avenue_model import FittedModel, Workbook

        booster, frame, levels = self._booster_and_levels()
        converted = FittedModel.from_lgbm_json(json.dumps(booster.dump_model()),
                                               consolidation="max")
        named = converted.with_categories({"region": levels})

        scoring = pl.DataFrame({"driver_age": frame[:, 0], "region": frame[:, 1].astype("int32")})
        before = converted.predict(scoring).to_series(0).to_numpy()
        after = named.predict(scoring).to_series(0).to_numpy()
        # Names are presentation. The model still matches on the code, so this is not
        # "close enough" - it is the same arithmetic and must be bit-identical.
        self.assertTrue(np.array_equal(before, after))

        with tempfile.TemporaryDirectory() as directory:
            named.to_workbook().save_csv_dir(directory)
            written = "\n".join(
                Path(directory, name).read_text()
                for name in os.listdir(directory) if name.endswith(".csv"))
            self.assertIn("north", written)
            # The wildcard row is labelled rather than left as a bare -999, which reads
            # as a rating factor for a region numbered minus nine hundred and ninety nine.
            self.assertIn("(any other level)", written)
            self.assertNotIn("-999", written)

            reloaded = Workbook.load_csv_dir(directory).to_model()
            self.assertTrue(np.allclose(
                reloaded.predict(scoring).to_series(0).to_numpy(), after, atol=1e-12))

    def test_partial_and_explicit_code_mappings_are_both_accepted(self):
        import json

        from avenue_model import FittedModel

        booster, _, levels = self._booster_and_levels()
        converted = FittedModel.from_lgbm_json(json.dumps(booster.dump_model()),
                                               consolidation="max")
        # A dict keyed by code, for codes that are not contiguous.
        by_code = converted.with_categories({"region": {0: "north", 3: "west"}})
        self.assertIsNotNone(by_code)
        # A mapping that names only some codes must not blank the rest.
        partial = converted.with_categories({"region": ["north"]})
        self.assertIsNotNone(partial)
        with self.assertRaises(ValueError):
            converted.with_categories({"region": 17})
