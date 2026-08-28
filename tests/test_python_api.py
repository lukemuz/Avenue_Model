"""Smoke tests for the user-facing Python modelling API.

Run after ``maturin develop`` with ``python -m unittest tests.test_python_api``.
"""

import unittest

import polars as pl

from avenue_model import GLMOptions, RatingModel, fit_glm_with_diagnostics


class PhaseOneApiTests(unittest.TestCase):
    def setUp(self):
        self.tables = [
            pl.DataFrame({"Rating_Factor": [0.0]}),
            pl.DataFrame(
                {
                    "group": pl.Series([0, 1], dtype=pl.Int32),
                    "Rating_Factor": [0.0, 0.0],
                }
            ),
        ]
        self.data = pl.DataFrame(
            {
                "group": pl.Series([0, 0, 1, 1], dtype=pl.Int32),
                "claims": [1.0, 2.0, 3.0, 4.0],
                "exposure": [1.0, 1.0, 1.0, 1.0],
            }
        )

    def test_family_names_result_tables_and_predictions_stay_together(self):
        model = RatingModel(
            self.tables,
            family="poisson",
            table_names=["base", "group"],
        )
        result = fit_glm_with_diagnostics(
            model,
            self.data,
            "claims",
            weight_col="exposure",
            options=GLMOptions(),
        )

        self.assertEqual(result.model.family, "poisson")
        self.assertEqual(result.model.table_names, ["base", "group"])
        self.assertTrue(result.converged)
        self.assertTrue(
            {"Coefficient", "Standard_Error", "Status", "Relativity"}
            <= set(result.rating_tables()[1].columns)
        )
        self.assertEqual(result.predict_components(self.data).columns, ["base", "group"])
        self.assertEqual(result.predict_expected(self.data, "exposure").columns, ["expected"])

    def test_table_names_are_unique_and_complete(self):
        with self.assertRaisesRegex(ValueError, "2 tables"):
            RatingModel(self.tables, "poisson", table_names=["base"])
        with self.assertRaisesRegex(ValueError, "non-empty and unique"):
            RatingModel(self.tables, "poisson", table_names=["same", "same"])

    def test_unknown_family_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown family"):
            RatingModel(self.tables, "not-a-family")


if __name__ == "__main__":
    unittest.main()
