"""Insurance response units must survive scoring and workbook reload."""
import unittest

import polars as pl

from avenue_model import Plan


class ScoringContract(unittest.TestCase):
    def setUp(self):
        self.df = pl.DataFrame({
            'exposure': [.25, .5, 1., .25, .5, 1.],
            'claims': [1., 2., 4., 1., 2., 4.],
            'group': ['A'] * 6,
            'frequency': [4.] * 6,
            'severity': [100.] * 6,
        })

    def assert_predictions(self, model, frame, expected):
        for artifact in (model, model.to_workbook().to_model()):
            actual = artifact.predict(frame).to_series().to_list()
            self.assertEqual(len(actual), len(expected))
            for a, e in zip(actual, expected):
                self.assertAlmostEqual(a, e, places=7)

    def test_offset_counts_reconcile_to_validation(self):
        model = Plan('poisson', exposure='exposure', exposure_role='offset').categorical('group').fit(self.df, 'claims')
        self.assert_predictions(model, self.df.select('group', 'exposure'), [1., 2., 4.] * 2)
        self.assertAlmostEqual(sum(model.predict(self.df).to_series()), model.validate(self.df).total_expected)
        self.assert_predictions(model, pl.DataFrame({'group': ['A'], 'exposure': [0.]}), [0.])
        for value in (-1., float('nan'), float('inf'), None):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'Exposure.*row 0'):
                model.predict(pl.DataFrame({'group': ['A'], 'exposure': [value]}))

    def test_training_weights_are_not_quote_inputs(self):
        frequency = Plan.frequency('exposure').categorical('group').fit(self.df, 'frequency')
        severity = Plan.severity('claims').categorical('group').fit(self.df, 'severity')
        quotes = self.df.select('group')
        self.assert_predictions(frequency, quotes, [4.] * 6)
        self.assert_predictions(severity, quotes, [100.] * 6)

    def test_intercept_only_is_an_ordinary_model(self):
        model = Plan.frequency('exposure').fit(self.df, 'frequency')
        self.assertTrue(model.converged)
        self.assertEqual(model.table_names, ['intercept'])
        self.assert_predictions(model, self.df.select('group'), [4.] * 6)


if __name__ == '__main__':
    unittest.main()
