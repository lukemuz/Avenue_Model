"""Tests for the plan, validation and report API.

Run after ``maturin develop --release`` with ``python -m unittest tests.test_plan_api``.

The Rust suite already pins the arithmetic. What is checked here is the *surface*: that
a plan can be stated, checked, fitted, validated and reported on from Python without
building a single table by hand, and that everything a caller needs to relay comes back
as data rather than as something printed.
"""

import json
import unittest

import polars as pl

from avenue_model import GLMOptions, Plan


def motor(n=480):
    """Motor-shaped data with a multiplicative truth, matching the log link.

    `region` is a string and `driver_age` an Int64 — numpy's default integer, and what
    ``pandas.Categorical(...).codes`` widens to. Neither is a dtype the matcher reads,
    so this frame is exactly what a plan has to normalise before anything can be fitted.
    """
    regions = ["north", "south", "east", "west"]
    rows = []
    for i in range(n):
        age = 18 + (i * 7) % 55
        exposure = 0.5 + (i % 4) / 4
        rate = 0.8 * [1.0, 1.25, 1.5, 1.75][i % 4] * (1.6 if age > 45 else 1.0)
        variation = [0.75, 1.0, 1.25][(i // 4) % 3]
        rows.append(
            {
                "region": regions[i % 4],
                "driver_age": age,
                "exposure": exposure,
                "claims": rate * exposure * variation,
                "frequency": rate * variation,
            }
        )
    return pl.DataFrame(
        rows,
        schema={
            "region": pl.String,
            "driver_age": pl.Int64,
            "exposure": pl.Float64,
            "claims": pl.Float64,
            "frequency": pl.Float64,
        },
    )


def frequency_plan():
    return (
        Plan.frequency("exposure")
        .banded("driver_age", breaks=[30.0, 45.0, 60.0])
        .categorical("region")
    )


class PlanTests(unittest.TestCase):
    def setUp(self):
        self.df = motor()

    def test_a_frequency_plan_states_its_own_idiom(self):
        plan = frequency_plan()
        self.assertEqual(plan.family, "poisson")
        self.assertEqual(plan.exposure, "exposure")
        # Frequency is a rate target, with exposure carrying its credibility.
        self.assertEqual(plan.exposure_role, "weight")
        self.assertEqual(plan.term_names, ["intercept", "driver_age", "region"])

        self.assertEqual(Plan.severity("claim_count").family, "gamma")
        self.assertEqual(Plan.severity("claim_count").exposure_role, "weight")
        self.assertEqual(Plan.pure_premium("exposure").family, "tweedie")

    def test_check_states_what_it_decided_before_anything_is_fitted(self):
        check = frequency_plan().check(self.df, "frequency")
        self.assertTrue(check.is_fittable, check.issues)

        by_name = {term["name"]: term for term in check.resolved}
        self.assertEqual(set(by_name), {"intercept", "driver_age", "region"})

        age = by_name["driver_age"]
        self.assertEqual(age["kind"], "banded")
        # Three cut points become four bands, the last unbounded.
        self.assertEqual(len(age["edges"]), 4)
        self.assertEqual(age["edges"][-1], float("inf"))

        region = by_name["region"]
        self.assertEqual(region["kind"], "categorical")
        self.assertEqual(region["rows"], 4)
        # The base defaults to the most exposed level, and is stated rather than assumed.
        self.assertIsNotNone(region["base_level"])

        self.assertGreater(check.parameters, 0)

    def test_check_reports_data_problems_instead_of_raising(self):
        broken = self.df.with_columns(
            pl.when(pl.int_range(pl.len()) == 3)
            .then(None)
            .otherwise(pl.col("frequency"))
            .alias("frequency")
        )
        check = frequency_plan().check(broken, "frequency")

        codes = [issue["code"] for issue in check.issues]
        self.assertIn("target_has_nulls", codes)
        self.assertFalse(check.is_fittable)

        # Findings carry a code to branch on and a message to show to a person.
        finding = next(i for i in check.issues if i["code"] == "target_has_nulls")
        self.assertEqual(finding["severity"], "high")
        self.assertIn("frequency", finding["message"])

        # Ordered most severe first, and JSON-serialisable so they can be relayed.
        severities = [i["severity"] for i in check.issues]
        rank = {"high": 2, "medium": 1, "low": 0}
        self.assertEqual(severities, sorted(severities, key=lambda s: -rank[s]))
        json.dumps(check.issues)

    def test_a_plan_round_trips_through_json(self):
        plan = (
            frequency_plan()
            .variate("driver_age_copy", quantile=8, degree=2)
            .interaction(["driver_age", "region"], [[30.0, 50.0], None])
        )
        recovered = Plan.from_json(plan.to_json())
        self.assertEqual(recovered.to_json(), plan.to_json())
        self.assertEqual(recovered.term_names, plan.term_names)

    def test_exactly_one_band_specification_is_required(self):
        with self.assertRaisesRegex(ValueError, "exactly one"):
            Plan.frequency("exposure").banded("driver_age")
        with self.assertRaisesRegex(ValueError, "exactly one"):
            Plan.frequency("exposure").banded("driver_age", breaks=[1.0], quantile=4)


class FitAndValidateTests(unittest.TestCase):
    def setUp(self):
        self.df = motor()
        self.plan = frequency_plan()
        self.check = self.plan.check(self.df, "frequency")
        self.fitted = self.plan.fit(self.df, "frequency", GLMOptions())

    def test_an_ordinary_dataframe_fits_without_a_table_built_by_hand(self):
        self.assertEqual(self.fitted.converged, True)
        self.assertEqual(
            self.fitted.table_names, ["intercept", "driver_age", "region"]
        )

        predictions = self.fitted.predict(self.df)
        self.assertEqual(predictions.height, self.df.height)

    def test_rating_tables_carry_inference_and_the_level_text(self):
        tables = self.fitted.rating_tables()
        region = tables[self.fitted.table_names.index("region")]

        for column in ("Coefficient", "Standard_Error", "Status", "Relativity"):
            self.assertIn(column, region.columns)
        # Codes are not what a person reads, so the level text comes back too.
        self.assertIn("region_Level", region.columns)
        self.assertEqual(region["Status"][0], "reference")

    def test_validation_answers_the_whole_question_in_one_call(self):
        v = self.fitted.validate(self.df)

        self.assertEqual(v.unmatched_rows, 0)
        self.assertAlmostEqual(v.ae_ratio, 1.0, places=6)
        self.assertGreater(v.gini, 0.0)
        self.assertTrue(v.is_usable, v.warnings)

        self.assertEqual(v.calibration.height, 10)
        for column in ("actual", "expected", "ae_ratio", "actual_rate"):
            self.assertIn(column, v.calibration.columns)

        # One actual-versus-expected frame per table, the exhibit pricing work needs.
        self.assertEqual(len(v.actual_vs_expected), len(self.fitted.table_names))
        region_ave = v.actual_vs_expected[self.fitted.table_names.index("region")]
        for column in ("Exposure", "Actual", "Expected", "AE_Ratio", "N"):
            self.assertIn(column, region_ave.columns)

    def test_an_unseen_level_is_reported_rather_than_scored_as_nan(self):
        holdout = self.df.with_columns(
            pl.when(pl.int_range(pl.len()) < 40)
            .then(pl.lit("atlantis"))
            .otherwise(pl.col("region"))
            .alias("region")
        )
        v = self.fitted.validate(holdout)

        self.assertEqual(v.unmatched_rows, 40)
        self.assertFalse(v.is_usable)
        codes = [w["code"] for w in v.warnings]
        self.assertIn("unmatched_observations", codes)
        json.dumps(v.warnings)


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.df = motor()
        self.plan = frequency_plan()
        self.check = self.plan.check(self.df, "frequency")
        self.fitted = self.plan.fit(self.df, "frequency", GLMOptions())

    def test_a_sound_model_reports_usable_and_carries_its_evidence(self):
        report = self.fitted.report(self.df)

        self.assertEqual(report.verdict, "usable")
        self.assertEqual(report.findings, [])
        self.assertTrue(report.fit_summary["converged"])
        self.assertIsNotNone(report.fit_summary["aic"])
        self.assertIsNotNone(report.validation)
        self.assertEqual(len(report.rating_tables()), 3)

        # The headline is meant to be shown to a person as written.
        self.assertIn("actual over expected", report.headline)

    def test_a_report_never_implies_a_validation_it_did_not_do(self):
        report = self.fitted.report()
        self.assertIsNone(report.validation)
        self.assertEqual(report.verdict, "usable_with_caveats")
        self.assertIn("not_validated", [f["code"] for f in report.findings])
        self.assertIn("has not been measured against held-out data", report.headline)

    def test_the_markdown_leads_with_the_verdict_and_carries_the_plan_back(self):
        report = self.fitted.report(self.df)
        markdown = report.markdown

        for section in ("## The model", "## Fit", "## Validation", "## Rating tables", "## Plan"):
            self.assertIn(section, markdown)

        # The embedded plan is the real thing, so the report reproduces the model.
        embedded = markdown.split("```json\n")[1].split("```")[0].strip()
        self.assertEqual(
            Plan.from_json(embedded).to_json(), self.plan.to_json()
        )

    def test_findings_are_branchable_and_the_verdict_is_one_value(self):
        holdout = self.df.with_columns(
            pl.when(pl.int_range(pl.len()) < 40)
            .then(pl.lit("atlantis"))
            .otherwise(pl.col("region"))
            .alias("region")
        )
        report = self.fitted.report(holdout)

        self.assertEqual(report.verdict, "not_usable")
        blocking = [f for f in report.findings if f["severity"] == "high"]
        self.assertTrue(blocking)
        self.assertTrue(all(f["stage"] in {"plan", "fit", "validation"} for f in report.findings))
        json.dumps(report.findings)

        # A reader who stops after the first screen already knows not to trust it.
        markdown = report.markdown
        self.assertLess(markdown.index("## Findings"), markdown.index("## Fit"))

    def test_the_fingerprint_identifies_the_plan(self):
        a = self.fitted.report()
        b = self.plan.fit(self.df, "frequency", GLMOptions()).report()
        self.assertEqual(a.fingerprint, b.fingerprint)

        other = (
            Plan.frequency("exposure")
            .banded("driver_age", breaks=[40.0])
            .categorical("region")
        )
        c = other.fit(self.df, "frequency", GLMOptions()).report()
        self.assertNotEqual(a.fingerprint, c.fingerprint)


if __name__ == "__main__":
    unittest.main()
