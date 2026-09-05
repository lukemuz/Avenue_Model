import unittest
import polars as pl
from avenue_model import prepare_pricing


class PreparationTests(unittest.TestCase):
    def test_populations_and_experience_reconcile(self):
        data = pl.DataFrame({'e': [1., .5, 0., 1.], 'c': [0, 2, 0, 1],
                              'loss': [0., 100., 0., 0.], 'region': ['a', 'b', 'b', 'a']})
        result = prepare_pricing(data, exposure='e', claims='c', loss='loss', large_loss=90.)
        self.assertEqual(result.frequency['avenue_row'].to_list(), [0, 1, 3])
        self.assertEqual(result.severity['avenue_row'].to_list(), [1])
        self.assertEqual(result.severity['avenue_severity'].to_list(), [50.])
        self.assertEqual(result.frequency['avenue_frequency'].to_list(), [0., 4., 1.])
        self.assertEqual(result.pure_premium['avenue_pure_premium'].to_list(), [0., 200., 0.])
        self.assertIn('large_loss', result.audit['flags'][1])
        self.assertEqual(result.frequency['loss'].sum(), 100.)
        exhibit = result.experience('region')
        self.assertEqual(exhibit['exposure'].sum(), 2.5)
        self.assertEqual(exhibit['claims'].sum(), 3.)
        self.assertEqual(exhibit['loss'].sum(), 100.)
        self.assertEqual(result.summary['excluded_rows'].to_list(), [1, 3, 1])
        self.assertEqual(data.columns, ['e', 'c', 'loss', 'region'])

    def test_invalid_rows_need_explicit_exclusion(self):
        data = pl.DataFrame({'e': [1., 0., 1., 1., 1.], 'c': [0., 1., .5, 1., 1.],
                             'loss': [5., 5., 5., -1., 1.], 'region': ['a', 'a', 'a', 'a', None]})
        with self.assertRaisesRegex(ValueError, 'Invalid pricing row 0'):
            prepare_pricing(data, exposure='e', claims='c', loss='loss', predictors=['region'])
        result = prepare_pricing(data, exposure='e', claims='c', loss='loss', predictors=['region'], invalid='exclude')
        self.assertEqual(result.frequency.height, 0)
        self.assertEqual(result.audit['valid'].to_list(), [False] * 5)
        self.assertIn('activity_without_exposure', result.audit['reasons'][1])
        self.assertIn('claims:not_integer', result.audit['reasons'][2])
        self.assertIn('loss:negative', result.audit['reasons'][3])
        self.assertIn('missing_predictor', result.audit['reasons'][4])

    def test_null_nonfinite_and_reserved_columns_fail_clearly(self):
        for exposure in (None, float('nan'), float('inf'), -1.):
            with self.assertRaisesRegex(ValueError, 'Invalid pricing row'):
                prepare_pricing(pl.DataFrame({'e': [exposure], 'c': [1], 'loss': [1.]},
                                             schema_overrides={'e': pl.Float64}),
                                exposure='e', claims='c', loss='loss')
