"""Small routing fixtures, independent of Avenue's table construction."""
import json
import math
import random
from pathlib import Path
import tempfile
import unittest

import polars as pl
from avenue_model import FittedModel, Workbook


def leaf(value):
    return {'leaf_index': 0, 'leaf_value': value}


def split(feature, threshold, left, right, categorical=False):
    return {'split_index': 0, 'split_feature': feature, 'threshold': threshold,
            'decision_type': '==' if categorical else '<=', 'default_left': False,
            'missing_type': 'None', 'internal_value': 0.,
            'left_child': left, 'right_child': right}


def dump(tree, names):
    return {'objective': 'regression', 'feature_names': names,
            'tree_info': [{'tree_structure': tree}]}


def reference(node, row, names):
    if 'leaf_value' in node:
        return node['leaf_value']
    value = row[names[node['split_feature']]]
    go_left = (value in [int(v) for v in str(node['threshold']).split('||')]
               if node['decision_type'] == '==' else value <= node['threshold'])
    return reference(node['left_child'] if go_left else node['right_child'], row, names)


class ConversionContract(unittest.TestCase):
    def check_conversion(self, model_dump, frame):
        names = model_dump['feature_names']
        raw = [sum(reference(t['tree_structure'], row, names) for t in model_dump['tree_info'])
               for row in frame.to_dicts()]
        expected = [math.exp(v) for v in raw] if model_dump['objective'] == 'poisson' else raw
        for mode in ('analysis', 'max'):
            model = FittedModel.from_lgbm_json(json.dumps(model_dump), consolidation=mode)
            with tempfile.TemporaryDirectory() as path:
                model.to_workbook().save_csv_dir(path)
                for artifact in (model, Workbook.load_csv_dir(path).to_model()):
                    actual = artifact.predict(frame)['predictions'].to_list()
                    for row, (a, e) in enumerate(zip(actual, expected)):
                        self.assertTrue(math.isclose(a, e, abs_tol=1e-12, rel_tol=1e-12),
                                        (mode, row, a, e))

    def test_partial_categorical_wildcards(self):
        tree = split(0, '1', split(1, '2', leaf(3.), leaf(7.), True), leaf(11.), True)
        self.check_conversion(dump(tree, ['a', 'b']),
                              pl.DataFrame({'a': [1, 1, 9, 9], 'b': [2, 9, 2, 9]}))

    def test_adjacent_numeric_thresholds(self):
        for threshold in (95.50000000000001, 1.0000000180025095e-35):
            upper = math.nextafter(threshold, math.inf)
            tree = split(0, threshold, leaf(3.), split(0, upper, leaf(7.), leaf(11.)))
            self.check_conversion(dump(tree, ['x']), pl.DataFrame({'x': [
                math.nextafter(threshold, -math.inf), threshold, upper,
                math.nextafter(upper, math.inf)]}))

    def test_randomized_small_tree_routing(self):
        rng = random.Random(814)
        for trial in range(8):
            def grow(depth):
                if depth == 0:
                    return leaf(float(rng.randrange(-10, 10)))
                feature = rng.randrange(3)
                threshold = str(rng.randrange(3)) if feature < 2 else rng.uniform(-2., 2.)
                return split(feature, threshold, grow(depth - 1), grow(depth - 1), feature < 2)
            model_dump = dump(grow(3), ['a', 'b', 'x'])
            frame = pl.DataFrame({'a': [rng.randrange(5) for _ in range(30)],
                                  'b': [rng.randrange(5) for _ in range(30)],
                                  'x': [rng.uniform(-3., 3.) for _ in range(30)]})
            with self.subTest(trial=trial):
                self.check_conversion(model_dump, frame)

    def test_captured_fork_categorical_tree(self):
        model_dump = json.loads(Path(__file__).with_name('fixtures').joinpath('categorical_tree.json').read_text())
        rows = [{'age': 55., 'vehicle_age': v, 'bonus': b, 'region': r, 'fuel': f}
                for v in (0., 2.) for b in (50., 75., 100.)
                for r in (0, 4, 17) for f in (0, 1)]
        self.check_conversion(model_dump, pl.DataFrame(rows))


class BoosterConversionContract(unittest.TestCase):
    def test_installed_lightgbm_build(self):
        try:
            import lightgbm as lgb
            import numpy as np
        except ImportError:
            self.skipTest('optional LightGBM test dependencies are unavailable')
        rng = np.random.default_rng(31)
        data = np.column_stack([rng.integers(0, 5, 600), rng.integers(0, 4, 600),
                                rng.uniform(-2, 2, 600)]).astype(float)
        target = .2 + .3 * (data[:, 0] == 1) + .4 * (data[:, 1] == 2) + .1 * (data[:, 2] > 0)
        booster = lgb.train({'objective': 'poisson', 'verbosity': -1, 'num_threads': 1,
                             'num_leaves': 6, 'min_data_in_leaf': 5, 'min_data_per_group': 5,
                             'cat_smooth': 1, 'seed': 31},
                            lgb.Dataset(data, label=target, feature_name=['a', 'b', 'x'],
                                        categorical_feature=['a', 'b']), num_boost_round=5)
        model_dump = booster.dump_model()
        rows = data.tolist()
        def boundaries(node):
            if 'split_feature' not in node:
                return
            if node['decision_type'] == '<=':
                threshold = float(node['threshold'])
                for value in (math.nextafter(threshold, -math.inf), threshold,
                              math.nextafter(threshold, math.inf)):
                    rows.append([1., 2., value])
            boundaries(node['left_child'])
            boundaries(node['right_child'])
        for tree in model_dump['tree_info']:
            boundaries(tree['tree_structure'])
        points = np.asarray(rows)
        expected = booster.predict(points)
        frame = pl.DataFrame({'a': points[:, 0].astype('int32'),
                              'b': points[:, 1].astype('int32'), 'x': points[:, 2]})
        for mode in ('analysis', 'max'):
            converted = FittedModel.from_lgbm_json(json.dumps(model_dump), consolidation=mode)
            with tempfile.TemporaryDirectory() as path:
                converted.to_workbook().save_csv_dir(path)
                for artifact in (converted, Workbook.load_csv_dir(path).to_model()):
                    np.testing.assert_allclose(artifact.predict(frame)['predictions'].to_numpy(),
                                               expected, atol=1e-12, rtol=1e-12)
