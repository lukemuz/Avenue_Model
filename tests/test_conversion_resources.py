"""Final resource bounds use a known four-cell interaction, not CV estimates."""
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


class ResourceLimitTests(unittest.TestCase):
    def test_exact_final_limits_preserve_predictions_and_export_evidence(self):
        quotes = pl.DataFrame({'x': [-1., -1., 1., 1.], 'y': [-1., 1., -1., 1.]})
        # Each numeric axis also has an explicit missing route: 3 x 3 cells,
        # plus one intercept. Analysis mode also retains a three-row main effect.
        # Four finite leaves alone undercount either representation.
        for mode in ['max', 'analysis']:
            expected = {'tables': 2 if mode == 'max' else 3,
                        'total_rows': 10 if mode == 'max' else 13,
                        'largest_table': 9, 'largest_interaction_order': 2}
            limits = dict(expected)
            result = from_booster(FourCellBooster(), quotes, consolidation=mode, resource_limits=limits)
            self.assertEqual(result.metadata['complexity'], expected)
            self.assertEqual(result.metadata['resource_check'], {'status': 'passed', 'limits': expected})
            self.assertEqual(result.parity['status'], 'passed')
            self.assertEqual(result.model.predict(quotes)['predictions'].to_list(), [1., 2., 3., 4.])
            table = result.model.to_workbook().tables[-1]
            self.assertEqual(table.filter(pl.col('x').is_nan() | pl.col('y').is_nan()).height, 5)
            limits['tables'] = 0
            self.assertEqual(result.metadata['resource_check']['limits']['tables'], expected['tables'])
            with tempfile.TemporaryDirectory() as directory:
                result.save(directory)
                saved = json.loads((Path(directory) / 'conversion.json').read_text())
                self.assertEqual(saved['metadata']['resource_check'], result.metadata['resource_check'])

    def test_each_exceeded_limit_rejects_the_actual_artifact(self):
        for name, bound, actual in [('tables', 1, 2), ('total_rows', 9, 10),
                                    ('largest_table', 8, 9), ('largest_interaction_order', 1, 2)]:
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, f'{name}={actual}'):
                from_booster(FourCellBooster(), resource_limits={name: bound})
        result = from_booster(FourCellBooster())
        self.assertEqual(result.metadata['resource_check']['status'], 'not_requested')
        self.assertEqual(result.parity['status'], 'not_verified')

    def test_invalid_limit_requests_fail_before_conversion(self):
        for limits in [{'tables': -1}, {'tables': True}, {'tables': 2.0},
                       {'tables': None}, {'scoring_seconds': 1}, ['tables']]:
            booster = FourCellBooster()
            with self.subTest(limits=limits), self.assertRaises((TypeError, ValueError)):
                from_booster(booster, resource_limits=limits)
            self.assertEqual(booster.dump_calls, 0)
