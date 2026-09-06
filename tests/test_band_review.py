import tempfile
import unittest

import polars as pl
from avenue_model import Plan, Workbook, coefficient_intervals


class BandReviewTests(unittest.TestCase):
    def test_category_level_labels_cannot_replace_an_interaction_predictor(self):
        data = pl.DataFrame({'g': ['a', 'a', 'b', 'b'] * 10,
                             'g_Level': [-1., 1., -1., 1.] * 10,
                             'y': [1., 2., 3., 5.] * 10})
        model = (Plan('gaussian').categorical('g', base='first').banded('g_Level', breaks=[0.])
                 .interaction(['g', 'g_Level'], [None, [0.]]).fit(data, 'y'))
        predictions = model.predict(data).to_dicts()
        for review in [model.rating_tables_by_name, lambda: model.validate(data)]:
            with self.assertRaisesRegex(ValueError, 'conflicts'):
                review()
        self.assertEqual(model.predict(data).to_dicts(), predictions)

    def test_derived_columns_cannot_overwrite_predictors_in_review(self):
        for name in ['Coefficient', 'Band_Interval_Status', 'N', 'Coefficient_Lower']:
            data = pl.DataFrame({name: [-1., 1.] * 20, 'y': [1., 2.] * 20})
            model = Plan('poisson').banded(name, breaks=[0.]).fit(data, 'y')
            predictions = model.predict(data).to_dicts()
            with self.assertRaisesRegex(ValueError, 'conflict'):
                if name == 'Coefficient_Lower':
                    coefficient_intervals(model)
                elif name == 'N':
                    model.validate(data)
                else:
                    model.rating_tables_by_name()
            self.assertEqual(model.predict(data).to_dicts(), predictions)
            self.assertIn(name, model.to_workbook().tables[model.table_names.index(name)].columns)

    def test_estimates_validation_intervals_and_reload_share_numeric_bounds(self):
        data = pl.DataFrame({'x': [-1., 0., 1., 5., 6.] * 20,
                             'y': [1., 1., 2., 2., 3.] * 20})
        model = Plan('poisson').banded('x', breaks=[0., 5.]).fit(data, 'y')
        tables = [model.rating_tables_by_name()['x'], coefficient_intervals(model).tables['x'],
                  model.validate(data).actual_vs_expected[model.table_names.index('x')]]
        with tempfile.TemporaryDirectory() as directory:
            model.to_workbook().save_csv_dir(directory)
            loaded = Workbook.load_csv_dir(directory).to_model()
            tables.append(loaded.rating_tables_by_name()['x'])
            self.assertEqual(model.predict(data).to_dicts(), loaded.predict(data).to_dicts())
            self.assertNotIn('x_Lower', loaded.to_workbook().tables[model.table_names.index('x')].columns)
        for table in tables:
            self.assertEqual(table['x_Lower'].to_list(), [float('-inf'), 0., 5.])
            self.assertEqual(table['x_Upper'].to_list(), [0., 5., float('inf')])
            self.assertEqual(table['x_Lower_Inclusive'].to_list(), [False] * 3)
            self.assertEqual(table['x_Upper_Inclusive'].to_list(), [True, True, False])
            self.assertEqual(table['Band_Interval_Status'].unique().to_list(), ['ordered_grid'])
        self.assertEqual(tables[2]['N'].to_list(), [40, 40, 20])
        self.assertEqual(tables[2]['Actual'].sum(), data['y'].sum())
