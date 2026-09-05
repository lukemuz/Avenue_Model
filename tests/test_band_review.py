import tempfile
import unittest

import polars as pl
from avenue_model import Plan, Workbook, coefficient_intervals


class BandReviewTests(unittest.TestCase):
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
