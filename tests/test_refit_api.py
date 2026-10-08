"""Tables define both refitting and fixed-offset workflows, regardless of origin."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import polars as pl
from polars.testing import assert_frame_equal

from avenue_model import GLMOptions, Plan, Workbook, from_booster


class RefitTests(unittest.TestCase):
    def setUp(self):
        self.data = pl.DataFrame({
            "region": ["north", "south"] * 60,
            "age": [float(i % 30) for i in range(120)],
            "exposure": [1.0, 2.0, 3.0] * 40,
            "frequency": [0.1, 0.3] * 60,
        }).with_columns((pl.col("frequency") * pl.col("exposure")).alias("claims"))
        self.book = Workbook.from_tables(
            {"region": pl.DataFrame({"region": ["north", "south"], "Relativity": [1., 2.]})},
            family="poisson", base_value=0.2, target="frequency", exposure="exposure",
        )

    def values(self, model, data=None):
        return model.predict(self.data if data is None else data).to_series().to_numpy()

    def test_import_scores_base_times_relativities_and_reload_refits(self):
        source = self.book.to_model()
        np.testing.assert_allclose(self.values(source), [0.2, 0.4] * 60)
        with tempfile.TemporaryDirectory() as folder:
            self.book.save_csv_dir(folder)
            loaded = Workbook.load_csv_dir(folder).to_model()
            fitted = loaded.refit(self.data)  # recorded target and exposure
            self.assertTrue(fitted.converged)
            self.assertTrue(fitted.was_fitted)
            self.assertEqual(fitted.table_names, source.table_names)
            self.assertIsNone(fitted.plan)
            np.testing.assert_allclose(self.values(fitted), self.data["frequency"], rtol=1e-6)
            np.testing.assert_allclose(self.values(loaded), self.values(source))
            fitted.to_workbook().save_csv_dir(folder)
            np.testing.assert_allclose(self.values(Workbook.load_csv_dir(folder).to_model()), self.values(fitted))
            self.assertAlmostEqual(fitted.validate(self.data).ae_ratio, 1.0, places=6)

    def test_imported_plan_as_offset_preserves_all_factors(self):
        source = self.book.to_model()
        data = self.data.with_columns((pl.Series(self.values(source)) * 1.5).alias("frequency"))
        adjusted = Plan.frequency("exposure").offset_model(source).fit(data, "frequency")
        self.assertTrue(adjusted.converged)
        np.testing.assert_allclose(self.values(adjusted), self.values(source) * 1.5, rtol=1e-6)
        for name, table in zip(source.table_names, source.rating_model.model_tables()):
            actual = adjusted.rating_model.model_tables()[adjusted.table_names.index("prior." + name)]
            assert_frame_equal(actual, table, check_column_order=False)
        # Refitting the combined model must still leave prior factors fixed.
        refitted = adjusted.refit(data.with_columns((pl.col("frequency") * 2).alias("frequency")))
        np.testing.assert_allclose(self.values(refitted), self.values(source) * 3, rtol=1e-6)
        for name in source.table_names:
            i = adjusted.table_names.index("prior." + name)
            assert_frame_equal(refitted.rating_model.model_tables()[i], adjusted.rating_model.model_tables()[i])

    def test_response_override_counts_applies_exposure_once(self):
        source = self.book.to_model()
        fitted = source.refit(self.data, "claims", exposure_role="offset")
        np.testing.assert_allclose(self.values(fitted), self.data["claims"], rtol=1e-6)
        np.testing.assert_allclose(fitted.predict_rate(self.data).to_series(), self.data["frequency"], rtol=1e-6)
        self.assertEqual(source.prediction_kind, "rate")

    def test_quantile_bands_are_not_relearned(self):
        source = Plan.frequency("exposure").banded("age", quantile=3).fit(self.data, "frequency")
        changed = self.data.with_columns((pl.col("age") + 4).alias("age"))
        result = source.refit(changed)
        self.assertTrue(result.converged)
        before = source.rating_model.model_tables()[1].drop("Rating_Factor")
        after = result.rating_model.model_tables()[1].drop("Rating_Factor")
        assert_frame_equal(before, after)

    def test_locked_rows_and_table_order_survive(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "plan.json"
            self.book.save_json(str(path))
            document = json.loads(path.read_text())
            document["manifest"]["tables"][1]["locked_rows"] = [1]
            path.write_text(json.dumps(document))
            source = Workbook.load_json(str(path)).to_model()
            result = source.refit(self.data)
            self.assertTrue(result.converged)
            self.assertEqual(result.rating_model.model_tables()[1]["Rating_Factor"][1],
                             source.rating_model.model_tables()[1]["Rating_Factor"][1])
            result.to_workbook().save_json(str(path))
            self.assertEqual(json.loads(path.read_text())["manifest"]["tables"][1]["locked_rows"], [1])

    def test_spline_and_monotonic_metadata_survive_refit(self):
        for method in (lambda p: p.spline("age", knots=[0., 10., 20., 30.]),
                       lambda p: p.monotone("age", "increasing", breaks=[10., 20.])):
            with self.subTest(method=method):
                source = method(Plan.frequency("exposure")).fit(self.data, "frequency")
                with tempfile.TemporaryDirectory() as folder:
                    path = str(Path(folder) / "plan.json")
                    source.to_workbook().save_json(path)
                    before = json.loads(Path(path).read_text())["manifest"]["tables"]
                    result = Workbook.load_json(path).to_model().refit(self.data)
                    self.assertTrue(result.converged)
                    result.to_workbook().save_json(path)
                    self.assertEqual(before, json.loads(Path(path).read_text())["manifest"]["tables"])

    def test_invalid_refit_inputs_fail(self):
        source = self.book.to_model()
        for data in (self.data.drop("region"), self.data.with_columns(pl.lit("unknown").alias("region")),
                     self.data.with_columns(pl.lit(float("nan")).alias("frequency")),
                     self.data.with_columns(pl.lit(-1.).alias("frequency"))):
            with self.subTest(data=data.head(1)), self.assertRaises(ValueError):
                source.refit(data)
        unknown_target = Workbook.from_tables({}, family="poisson", base_value=1.).to_model()
        with self.assertRaisesRegex(ValueError, "target"):
            unknown_target.refit(self.data)
        with self.assertRaisesRegex(ValueError, "exposure"):
            unknown_target.refit(self.data, "frequency", exposure_role="offset")
        integer_target = self.data.with_columns(pl.lit(1).alias("claims"))
        self.assertTrue(unknown_target.refit(integer_target, "claims").converged)

    def test_factor_scale_and_band_import(self):
        book = Workbook.from_tables({"age": pl.DataFrame({
            "age": [10., float("inf")], "Rating_Factor": [0., 2.],
        })}, family="gaussian", base_value=3., scale="factor")
        np.testing.assert_allclose(book.to_model().predict(pl.DataFrame({"age": [5, 20]})).to_series(), [3., 5.])

    def test_shared_category_encodings_and_integer_categories(self):
        book = Workbook.from_tables({
            "region": pl.DataFrame({"region": ["south", "north"], "Relativity": [2., 1.]}),
            "region_tier": pl.DataFrame({"region": ["north", "south"], "tier": [7, 9],
                                          "Relativity": [3., 4.]}),
        }, family="poisson", base_value=0.1)
        data = pl.DataFrame({"region": ["north", "south"], "tier": [7, 9]})
        np.testing.assert_allclose(book.to_model().predict(data).to_series(), [0.3, 0.8])
        with tempfile.TemporaryDirectory() as folder:
            book.save_csv_dir(folder)
            np.testing.assert_allclose(Workbook.load_csv_dir(folder).to_model().predict(data).to_series(), [0.3, 0.8])

    def test_tweedie_power_and_new_options_are_recorded(self):
        source = Workbook.from_tables({}, family="tweedie", base_value=0.2,
                                      tweedie_power=1.7).to_model()
        fitted = source.refit(self.data, "frequency", GLMOptions(alpha=1e-4), exposure="exposure")
        self.assertEqual(fitted.fit_options["tweedie_power"], 1.7)
        self.assertEqual(fitted.fit_options["alpha"], 1e-4)
        fresh = fitted.refit(self.data)
        self.assertEqual(fresh.fit_options["alpha"], 0.0)
        self.assertTrue(fresh.converged)

    def test_bad_imports_rejected(self):
        for frame in (pl.DataFrame({"region": ["north"], "Relativity": [0.]}),
                      pl.DataFrame({"region": ["north", "north"], "Relativity": [1., 2.]}),
                      pl.DataFrame({"age": [10., 20.], "Relativity": [1., 2.]}),
                      pl.DataFrame({"region": ["north"], "Relativity": [1.], "Rating_Factor": [0.]})):
            with self.subTest(frame=frame), self.assertRaises(ValueError):
                Workbook.from_tables({"bad": frame}, family="poisson", base_value=0.2)

    def test_converted_model_direct_refit_and_regularization(self):
        import lightgbm as lgb
        x = self.data.select("age").to_numpy()
        booster = lgb.train({"objective": "poisson", "verbosity": -1, "num_threads": 2,
                             "num_leaves": 3, "min_data_in_leaf": 10},
                            lgb.Dataset(x, label=self.data["frequency"].to_numpy(), feature_name=["age"]),
                            num_boost_round=5)
        source = from_booster(booster, self.data).model
        for alpha in (0., 1e-4):
            with self.subTest(alpha=alpha):
                fitted = source.refit(self.data, "frequency", GLMOptions(alpha=alpha, max_iterations=500),
                                      exposure="exposure")
                self.assertTrue(fitted.converged)
                self.assertEqual(fitted.table_names, source.table_names)
                self.assertEqual(fitted.prediction_kind, "rate")
                np.testing.assert_allclose(fitted.predict_count(self.data).to_series(),
                                           fitted.predict(self.data).to_series() * self.data["exposure"])
                for before, after in zip(source.rating_model.model_tables(), fitted.rating_model.model_tables()):
                    assert_frame_equal(before.drop("Rating_Factor"), after.drop("Rating_Factor"))
                self.assertFalse(source.was_fitted)


if __name__ == "__main__":
    unittest.main()
