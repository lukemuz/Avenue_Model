import unittest

import pandas as pd
import polars as pl

from avenue_model import Plan, from_pandas


class AdapterTests(unittest.TestCase):
    def test_categories_are_labels_not_pandas_codes(self):
        frame = pd.DataFrame({'region': pd.Categorical(['b', 'a', 'b', 'a'] * 8,
                                                       categories=['unused', 'b', 'a']),
                              'exposure': [1.] * 32, 'rate': [4., 2., 4., 2.] * 8})
        native = pl.DataFrame({'region': ['b', 'a', 'b', 'a'] * 8,
                               'exposure': [1.] * 32, 'rate': [4., 2., 4., 2.] * 8})
        converted = from_pandas(frame)
        self.assertEqual(converted.to_dicts(), native.to_dicts())
        plan = Plan.frequency('exposure').categorical('region')
        a, b = plan.fit(converted, 'rate'), plan.fit(native, 'rate')
        self.assertEqual(a.predict(native).to_dicts(), b.predict(native).to_dicts())
        self.assertEqual(a.input_schema, b.input_schema)
        self.assertEqual(a.input_schema['predictors']['region']['levels'], [('a', 0), ('b', 1)])
        self.assertEqual(frame['region'].cat.categories.to_list(), ['unused', 'b', 'a'])
        quotes = from_pandas(pd.DataFrame({'region': pd.Categorical(['a', None, 'new'],
                                                                   categories=['new', 'a'])}))
        self.assertEqual(a.predict_diagnostics(quotes)['status'].to_list(), ['ok', 'unmatched', 'unmatched'])

    def test_nullable_numeric_and_integer_categories_preserve_values(self):
        frame = pd.DataFrame({'number': pd.Series([1, None], dtype='Int64'),
                              'category': pd.Categorical([100, None], categories=[200, 100]),
                              'flag': pd.Series([True, None], dtype='boolean'),
                              'value': [1., float('nan')]})
        converted = from_pandas(frame)
        self.assertEqual(converted['category'].to_list(), [100, None])
        self.assertEqual(converted['number'].to_list(), [1, None])
        self.assertEqual(converted['flag'].to_list(), [True, None])
        self.assertEqual(converted['value'].to_list(), [1., None])

    def test_index_is_explicit_and_unsupported_labels_fail(self):
        frame = pd.DataFrame({'value': [1., 2.]}, index=['p2', 'p1'])
        self.assertEqual(from_pandas(frame).columns, ['value'])
        self.assertEqual(from_pandas(frame, index_column='policy_id')['policy_id'].to_list(), ['p2', 'p1'])
        with self.assertRaisesRegex(ValueError, 'new nonempty'):
            from_pandas(frame, index_column='value')
        for values in ([1, '1'], [1.5, 2.5]):
            with self.assertRaisesRegex(TypeError, 'explicit encoding'):
                from_pandas(pd.DataFrame({'mixed': pd.Categorical(values)}))
        with self.assertRaisesRegex(TypeError, 'cast numerical'):
            from_pandas(pd.DataFrame({'mixed': [1, 'a']}))
