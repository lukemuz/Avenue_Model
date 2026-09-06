"""Final conversion complexity includes the actual missing-route cells."""
import json
from pathlib import Path
import tempfile
import unittest

import polars as pl
from avenue_model import from_booster


class FourCellBooster:
    def __init__(self):
        self.params = {}
        self.dump_calls = 0

    def dump_model(self):
        self.dump_calls += 1
        def split(feature, left, right):
            return {'split_index': 0, 'split_feature': feature, 'threshold': 0., 'decision_type': '<=',
                    'default_left': False, 'missing_type': 'None', 'internal_value': 0.,
                    'left_child': left, 'right_child': right}
        return {'objective': 'regression', 'feature_names': ['x', 'y'],
                'tree_info': [{'tree_structure': split(0,
                    split(1, {'leaf_index': 0, 'leaf_value': 1.}, {'leaf_index': 1, 'leaf_value': 2.}),
                    split(1, {'leaf_index': 2, 'leaf_value': 3.}, {'leaf_index': 3, 'leaf_value': 4.}))}]}

    def predict(self, data):
        return [1. + 2. * (x > 0) + (y > 0) for x, y in data]


class ConversionComplexityTests(unittest.TestCase):
    def test_final_complexity_includes_missing_routes(self):
        quotes = pl.DataFrame({'x': [-1., -1., 1., 1.], 'y': [-1., 1., -1., 1.]})
        # Each numeric axis also has an explicit missing route: 3 x 3 cells,
        # plus one intercept. Analysis mode also retains a three-row main effect.
        # Four finite leaves alone undercount either representation.
        for mode in ['max', 'analysis']:
            expected = {'tables': 2 if mode == 'max' else 3,
                        'total_rows': 10 if mode == 'max' else 13,
                        'largest_table': 9, 'largest_interaction_order': 2}
            result = from_booster(FourCellBooster(), quotes, consolidation=mode)
            self.assertEqual(result.metadata['complexity'], expected)
            self.assertEqual(result.parity['status'], 'passed')
            self.assertEqual(result.model.predict(quotes)['predictions'].to_list(), [1., 2., 3., 4.])
            table = result.model.to_workbook().tables[-1]
            self.assertEqual(table.filter(pl.col('x').is_nan() | pl.col('y').is_nan()).height, 5)
            with tempfile.TemporaryDirectory() as directory:
                result.save(directory)
                saved = json.loads((Path(directory) / 'conversion.json').read_text())
                self.assertEqual(saved['metadata']['complexity'], expected)
