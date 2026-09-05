"""Tuning complexity describes the selected boosting prefix, on repeatable folds."""
import json
import unittest
from unittest.mock import patch

import numpy as np
from avenue_model import estimate_num_tables, resolve_lightgbm, tune_lgbm
import avenue_model.tuning as tuning


class TuningContractTests(unittest.TestCase):
    def test_selected_round_complexity_and_generator_folds(self):
        lgb, _ = resolve_lightgbm()
        rng = np.random.default_rng(24)
        x = rng.normal(size=(120, 4))
        y = np.exp(.7 * x[:, 0] + .3 * x[:, 1] + .5 * x[:, 2] + .4 * x[:, 3])
        dataset = lgb.Dataset(x, label=y)
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
                n_trials=2, seed=29, folds=(pair for pair in pairs))
        for record, boosters in zip(result.trials, captured):
            self.assertEqual(record.num_iterations, 1)
            expected = [float(estimate_num_tables(json.dumps(b.dump_model(num_iteration=1)))) for b in boosters]
            self.assertEqual(record.fold_tables, expected)
            self.assertEqual(record.tables, sum(expected) / 2)
            self.assertTrue(any(b.num_trees() > 1 for b in boosters))
        self.assertTrue(any(estimate_num_tables(json.dumps(b.dump_model())) !=
                            estimate_num_tables(json.dumps(b.dump_model(num_iteration=1)))
                            for boosters in captured for b in boosters))
        self.assertIn('mean tables', result.summary())

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
