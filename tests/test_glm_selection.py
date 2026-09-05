import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import polars as pl
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_tweedie_deviance
from avenue_model import GLMTrial, Plan, SplitSpec, select_glm


class SelectionTests(unittest.TestCase):
    def test_penalties_and_selection_match_independent_elastic_net(self):
        rng = np.random.default_rng(29)
        category = np.tile(['a', 'b', 'c'], 100)
        design = np.column_stack([category == 'b', category == 'c']).astype(float)
        y = 1. + design @ np.array([.2, .6]) + rng.normal(0., 2., len(category))
        data = pl.DataFrame({'category': category, 'y': y})
        plan = Plan('gaussian').categorical('category', base='first')
        trials = {str(alpha): GLMTrial(plan, {'alpha': alpha, 'l1_ratio': .4,
                    'tolerance': 1e-10, 'max_iterations': 1000}) for alpha in (.001, .03, .3)}
        folds = SplitSpec.random(n_splits=3, seed=52).split(data)
        result = select_glm(data, trials, target='y', unit='response',
                            metric='squared_error', split=folds)
        reference_losses = {}
        for name, trial in trials.items():
            losses = []
            for fold in folds:
                reference = ElasticNet(alpha=trial.options['alpha'], l1_ratio=.4,
                                       tol=1e-12, max_iter=10000).fit(
                    design[list(fold.train_rows)], y[list(fold.train_rows)])
                mu = reference.predict(design[list(fold.validation_rows)])
                loss = np.mean((mu - y[list(fold.validation_rows)]) ** 2)
                actual = result.history.filter((pl.col('trial') == name) & (pl.col('split_id') == fold.split_id))
                self.assertTrue(actual['eligible'][0], actual.to_dicts())
                self.assertAlmostEqual(actual['mean_loss'][0], loss, places=7)
                losses.extend((mu - y[list(fold.validation_rows)]) ** 2)
            reference_losses[name] = np.mean(losses)
        self.assertEqual(result.recommended, min(reference_losses, key=reference_losses.get))
        self.assertTrue(result.refit(data).converged)
        with self.assertRaisesRegex(ValueError, 'differs'):
            result.refit(data.reverse())
        with tempfile.TemporaryDirectory() as tmp:
            result.save(Path(tmp) / 'selection')
            saved = json.loads((Path(tmp) / 'selection' / 'selection.json').read_text())
            self.assertEqual(saved['recommended'], result.recommended)
            self.assertEqual(saved['folds'][0]['fingerprint'], folds[0].fingerprint)

    def test_failed_and_nonconverged_trials_remain_visible(self):
        data = pl.DataFrame({'x': ['a', 'b'] * 50, 'y': [1., 100.] * 50})
        result = select_glm(data, {
            'good': GLMTrial(Plan('poisson')),
            'not_converged': GLMTrial(Plan('poisson').categorical('x'), {'max_iterations': 1, 'tolerance': 1e-14}),
            'missing_column': GLMTrial(Plan('poisson').categorical('absent')),
        }, target='y', unit='counts', metric='poisson', split=SplitSpec.random(2))
        self.assertEqual(result.recommended, 'good')
        summaries = {row['trial']: row for row in result.summary.to_dicts()}
        self.assertFalse(summaries['not_converged']['eligible'])
        self.assertEqual(summaries['missing_column']['scored_folds'], 0)
        self.assertEqual(result.history.filter(pl.col('trial') == 'missing_column')['error'].null_count(), 0)

    def test_tweedie_powers_use_common_weighted_validation_loss(self):
        data = pl.DataFrame({'time': [1, 1, 2, 2, 3, 3], 'y': [0., 2., 8., 1., 3., 12.],
                             'w': [.1, 1., 10., 2., .5, 3.]})
        folds = SplitSpec.random(n_splits=3, seed=17).split(data)
        trials = {str(power): GLMTrial(Plan('tweedie', exposure='w', exposure_role='weight',
                                           tweedie_power=power)) for power in (1.2, 1.8)}
        result = select_glm(data, trials, target='y', unit='loss/exposure', metric='tweedie',
                            tweedie_power=1.4, weight='w', split=folds)
        for name in trials:
            total_loss = total_weight = 0.
            for fold in folds:
                train, validation = fold.frames(data)
                # Intercept-only weighted Tweedie means solve sum(w * (y-mu))=0.
                mean = (train['y'] * train['w']).sum() / train['w'].sum()
                loss = mean_tweedie_deviance(validation['y'], [mean] * validation.height,
                                            sample_weight=validation['w'], power=1.4)
                total_loss += loss * validation['w'].sum()
                total_weight += validation['w'].sum()
            actual = result.summary.filter(pl.col('trial') == name)['mean_loss'][0]
            self.assertAlmostEqual(actual, total_loss / total_weight, places=7)
        with self.assertRaisesRegex(ValueError, 'trial Plan'):
            select_glm(data, {'bad': GLMTrial(Plan('tweedie'), {'tweedie_power': 1.8})},
                       target='y', unit='loss/exposure', metric='tweedie', split=folds)

    def test_training_only_encoding_and_overlap_guard(self):
        data = pl.DataFrame({'time': [1, 1, 2, 2], 'x': ['a', 'a', 'new', 'new'], 'y': [1., 2., 3., 4.]})
        folds = SplitSpec.out_of_time('time', 2).split(data)
        result = select_glm(data, {'category': GLMTrial(Plan('poisson').categorical('x'))},
                            target='y', unit='counts', metric='poisson', split=folds)
        self.assertIsNone(result.recommended)
        self.assertEqual(result.history['status'][0], 'failed')
        with self.assertRaisesRegex(ValueError, 'No trial'):
            result.refit(data)
        with self.assertRaisesRegex(ValueError, 'overlap'):
            select_glm(data, {'base': GLMTrial(Plan('poisson'))}, target='y', unit='counts',
                       metric='poisson', split=folds * 2)
