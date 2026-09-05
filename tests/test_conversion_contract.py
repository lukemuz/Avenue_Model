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
        if model_dump['objective'].startswith('binary'):
            expected = [1 / (1 + math.exp(-v)) for v in raw]
        for mode in ('analysis', 'max'):
            model = FittedModel.from_lgbm_json(json.dumps(model_dump), consolidation=mode)
            with tempfile.TemporaryDirectory() as path:
                model.to_workbook().save_csv_dir(path)
                for artifact in (model, Workbook.load_csv_dir(path).to_model()):
                    actual = artifact.predict(frame)['predictions'].to_list()
                    for row, (a, e) in enumerate(zip(actual, expected)):
                        self.assertTrue(math.isclose(a, e, abs_tol=1e-12, rel_tol=1e-12),
                                        (mode, row, a, e))

    def test_constant_boosters_and_later_constant_trees(self):
        constant = {'leaf_value': 2.5, 'leaf_count': 100}
        base = dump(constant, ['x'])
        frame = pl.DataFrame({'x': [1., 2., 3.]})
        self.check_conversion(base, frame)
        for mode in ('analysis', 'max'):
            model = FittedModel.from_lgbm_json(json.dumps(base), consolidation=mode)
            self.assertEqual(len(model.table_names), 1)
        base['tree_info'].append({'tree_structure': {'leaf_value': .7}})
        self.check_conversion(base, frame)

    def test_binary_objective_options_keep_logit_link(self):
        base = dump(split(0, 0., leaf(-2.), leaf(2.)), ['x'])
        self.check_conversion({**base, 'objective': 'binary sigmoid:1'},
                              pl.DataFrame({'x': [-1., 1.]}))

    def test_unsupported_semantics_raise(self):
        base = dump(split(0, 0., leaf(1.), leaf(2.)), ['x'])
        for update in ({'objective': 'multiclass num_class:3'}, {'objective': 'custom'},
                       {'objective': 'binary sigmoid:2'}, {'average_output': True},
                       {'num_tree_per_iteration': 3}):
            with self.subTest(update=update), self.assertRaisesRegex(ValueError, 'Unsupported'):
                FittedModel.from_lgbm_json(json.dumps({**base, **update}))
        with self.assertRaisesRegex(ValueError, 'nonempty array'):
            FittedModel.from_lgbm_json(json.dumps({**base, 'tree_info': []}))
        linear = dump(split(0, 0., {**leaf(1.), 'leaf_coeff': [2.]}, leaf(2.)), ['x'])
        with self.assertRaisesRegex(ValueError, 'linear leaves'):
            FittedModel.from_lgbm_json(json.dumps(linear))

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
        from avenue_model import from_booster
        constant = lgb.train({'objective': 'poisson', 'verbosity': -1, 'num_threads': 1,
                              'min_data_in_leaf': 1000},
                             lgb.Dataset(data, label=target, feature_name=['a', 'b', 'x']),
                             num_boost_round=2)
        constant_frame = pl.DataFrame({'a': data[:, 0], 'b': data[:, 1], 'x': data[:, 2]})
        for mode in ('analysis', 'max'):
            constant_result = from_booster(constant, constant_frame, consolidation=mode)
            self.assertEqual(constant_result.parity['status'], 'passed')
            self.assertEqual(constant_result.model.table_names, ['intercept'])
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
        from avenue_model import from_booster
        result = from_booster(booster, frame)
        self.assertEqual(result.parity['status'], 'passed')
        self.assertEqual(result.parity['failed_rows'], [])
        self.assertEqual(result.metadata['version'], lgb.__version__)
        self.assertEqual(from_booster(booster).parity['status'], 'not_verified')
        class ShiftedReference:
            params = {}
            def dump_model(self):
                return booster.dump_model()
            def predict(self, values):
                return booster.predict(values) + 1.
        failed = from_booster(ShiftedReference(), frame)
        self.assertEqual(failed.parity['status'], 'failed')
        self.assertEqual(failed.parity['failed_rows'], list(range(frame.height)))
        self.assertAlmostEqual(failed.parity['max_absolute_error'], 1.)

        with tempfile.TemporaryDirectory() as path:
            result.save(path)
            evidence = json.loads(Path(path, 'conversion.json').read_text())
            self.assertEqual(evidence['metadata']['dump_sha256'], result.metadata['dump_sha256'])
            np.testing.assert_allclose(Workbook.load_csv_dir(path).to_model().predict(frame)['predictions'],
                                       expected, atol=1e-12, rtol=1e-12)
        with self.assertRaisesRegex(ValueError, 'Missing/default routing'):
            from_booster(booster, frame.with_columns(pl.lit(float('nan')).alias('x')))
        for mode in ('analysis', 'max'):
            converted = FittedModel.from_lgbm_json(json.dumps(model_dump), consolidation=mode)
            with tempfile.TemporaryDirectory() as path:
                converted.to_workbook().save_csv_dir(path)
                for artifact in (converted, Workbook.load_csv_dir(path).to_model()):
                    np.testing.assert_allclose(artifact.predict(frame)['predictions'].to_numpy(),
                                               expected, atol=1e-12, rtol=1e-12)
