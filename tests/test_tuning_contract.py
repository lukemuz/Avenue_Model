"""Tuning complexity describes the selected boosting prefix, on repeatable folds."""
import json
import unittest
from unittest.mock import patch

import numpy as np
import polars as pl
from avenue_model import FittedModel, estimate_num_tables, resolve_lightgbm, tune_lgbm
import avenue_model.tuning as tuning


class TuningContractTests(unittest.TestCase):
    def test_scoring_cost_rejects_failed_conversion_parity(self):
        lgb, _ = resolve_lightgbm()
        x = np.arange(80.).reshape(-1, 1)
        dataset = lgb.Dataset(x, label=np.full(80, 2.), feature_name=['x'])
        actual_cv = lgb.cv

        def inconsistent_reference(**kwargs):
            result = actual_cv(**kwargs)
            result['cvbooster'].boosters[0].predict = lambda data, **kw: np.full(len(data), 7.)
            return result

        with patch.object(lgb, 'cv', side_effect=inconsistent_reference), \
                patch.object(tuning, 'supports_interaction_penalties', return_value=False):
            with self.assertRaisesRegex(ValueError, 'parity'):
                tune_lgbm(dataset, {'objective': 'poisson', 'num_iterations': 3,
                    'min_data_in_leaf': 100, 'verbosity': -1, 'num_threads': 1},
                    tunable=[], n_trials=1, nfold=2, seed=7,
                    scoring_data=pl.DataFrame({'x': [1., 2.]}))

    def test_selected_round_complexity_and_generator_folds(self):
        lgb, _ = resolve_lightgbm()
        rng = np.random.default_rng(24)
        x = rng.normal(size=(120, 4))
        y = np.exp(.7 * x[:, 0] + .3 * x[:, 1] + .5 * x[:, 2] + .4 * x[:, 3])
        names = [f'x{i}' for i in range(x.shape[1])]
        dataset = lgb.Dataset(x, label=y, feature_name=names)
        quotes = pl.DataFrame({name: x[:17, i] for i, name in enumerate(names)})
        pairs = [(np.arange(60), np.arange(60, 120)), (np.arange(60, 120), np.arange(60))]
        actual_cv = lgb.cv
        captured = []

        def cv(**kwargs):
            self.assertEqual(len(kwargs['folds']), 2)
            self.assertEqual(kwargs['seed'], 29)
            result = actual_cv(**kwargs)
            # Force round one to isolate the bookkeeping contract from model
            # selection variability, while retaining real multi-round boosters.
            result['valid poisson-mean'] = [0.] + [1.] * (len(result['valid poisson-mean']) - 1)
            captured.append(result['cvbooster'].boosters)
            return result

        with patch.object(lgb, 'cv', side_effect=cv), patch.object(tuning, 'supports_interaction_penalties', return_value=False):
            result = tune_lgbm(dataset, {'objective': 'poisson', 'num_iterations': 15,
                'num_leaves': 2, 'min_data_in_leaf': 5, 'verbosity': -1, 'num_threads': 1},
                tunable=['learning_rate'], space={'learning_rate': (.2, .3)},
                n_trials=2, seed=29, folds=(pair for pair in pairs), scoring_data=quotes)
        for record, boosters in zip(result.trials, captured):
            self.assertEqual(record.num_iterations, 1)
            expected = [float(estimate_num_tables(json.dumps(b.dump_model(num_iteration=1)))) for b in boosters]
            self.assertEqual(record.fold_tables, expected)
            self.assertEqual(record.tables, sum(expected) / 2)
            for measured, booster in zip(record.fold_complexity, boosters):
                tables = FittedModel.from_lgbm_json(json.dumps(booster.dump_model(num_iteration=1))).to_workbook().tables
                self.assertEqual(measured['num_iterations'], 1)
                self.assertEqual(measured['total_rows'], sum(t.height for t in tables))
                self.assertEqual(measured['largest_table'], max(t.height for t in tables))
                self.assertEqual(measured['largest_interaction_order'], max(t.width - 1 for t in tables))
                self.assertIsNone(measured['statistical_rank'])
                self.assertEqual(measured['support_status'], 'not_measured')
                self.assertEqual(measured['scoring_rows'], quotes.height)
                self.assertEqual(len(measured['scoring_samples_seconds']), 3)
                self.assertEqual(measured['scoring_seconds'], np.median(measured['scoring_samples_seconds']))
                self.assertEqual(measured['scoring_parity']['status'], 'passed')
                self.assertLess(measured['scoring_parity']['max_absolute_error'], 1e-12)
                self.assertEqual(measured['scoring_feature_names'], names)
            self.assertTrue(any(b.num_trees() > 1 for b in boosters))
        self.assertTrue(any(estimate_num_tables(json.dumps(b.dump_model())) !=
                            estimate_num_tables(json.dumps(b.dump_model(num_iteration=1)))
                            for boosters in captured for b in boosters))
        self.assertIn('mean tables', result.summary())
        self.assertIn('mean rows', result.summary())
        self.assertIn('score ms', result.summary())

    def test_constant_booster_counts_as_one_table(self):
        lgb, _ = resolve_lightgbm()
        x = np.arange(80.).reshape(-1, 1)
        dataset = lgb.Dataset(x, label=np.full(80, 2.))
        with patch.object(tuning, 'supports_interaction_penalties', return_value=False):
            result = tune_lgbm(dataset, {'objective': 'poisson', 'num_iterations': 3,
                'min_data_in_leaf': 100, 'verbosity': -1, 'num_threads': 1},
                tunable=[], n_trials=1, nfold=2, seed=7)
        self.assertEqual(result.best_cv.fold_tables, [1., 1.])
        self.assertEqual(result.select().tables, 1.)
        self.assertEqual([c['total_rows'] for c in result.best_cv.fold_complexity], [1, 1])
        self.assertEqual([c['largest_interaction_order'] for c in result.best_cv.fold_complexity], [0, 0])
        self.assertTrue(all(c['scoring_seconds'] is None for c in result.best_cv.fold_complexity))

    def test_equal_table_counts_do_not_hide_large_row_counts(self):
        trials = [tuning.Trial({}, 1., 4., 10, fold_complexity=[
            {'total_rows': n, 'largest_table': n - 3, 'largest_interaction_order': 3}])
            for n in [40, 19181]]
        summary = tuning.TuningResult(trials, 'poisson', False).summary()
        self.assertIn('19181.0', summary)
        self.assertIn('19178', summary)
        self.assertIn('40.0', summary)
