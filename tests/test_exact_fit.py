"""Exact cell fits must converge without dividing rounding noise by itself."""
import unittest

import polars as pl
from avenue_model import GLMOptions, Plan


class ExactFitTests(unittest.TestCase):
    def test_exact_cells_converge_across_families_and_units(self):
        for family in ('gaussian', 'poisson', 'gamma', 'tweedie'):
            for scale in (1e-6, 1., 1e6):
                data = pl.DataFrame({'group': ['a', 'b'] * 20,
                                     'weight': [.5, 1.] * 20,
                                     'y': [2. * scale, 4. * scale] * 20})
                with self.subTest(family=family, scale=scale):
                    model = Plan(family, exposure='weight', exposure_role='weight').categorical('group').fit(data, 'y')
                    self.assertTrue(model.converged, model.report().fit_summary)
                    prediction = model.predict(data)['predictions']
                    for actual, expected in zip(prediction, data['y']):
                        self.assertLess(abs(actual / expected - 1), 1e-7)

    def test_noisy_weighted_cells_match_closed_form_means(self):
        import random
        rng = random.Random(39)
        groups = ['a', 'b', 'c'] * 50
        weights = [rng.uniform(.1, 2.) for _ in groups]
        response = [rng.uniform(.2, 4.) * (1 + i % 3) for i in range(len(groups))]
        data = pl.DataFrame({'group': groups, 'weight': weights, 'y': response})
        # For constant cell means, each of these families has the same weighted
        # score equation sum(w * (y - mu)) = 0, giving this independent reference.
        means = {g: sum(w * y for k, w, y in zip(groups, weights, response) if k == g) /
                    sum(w for k, w in zip(groups, weights) if k == g) for g in set(groups)}
        for family in ('gaussian', 'poisson', 'gamma', 'tweedie'):
            model = Plan(family, exposure='weight', exposure_role='weight').categorical('group').fit(data, 'y')
            self.assertTrue(model.converged)
            for group, prediction in zip(groups, model.predict(data)['predictions']):
                self.assertLess(abs(prediction / means[group] - 1), 1e-7)

    def test_stopping_still_requires_score_convergence(self):
        data = pl.DataFrame({'group': ['a', 'b'] * 20, 'y': [1., 100.] * 20})
        model = Plan('poisson').categorical('group').fit(
            data, 'y', GLMOptions(max_iterations=1, tolerance=1e-14))
        self.assertFalse(model.converged)

    def test_exact_offset_counts_converge_and_reconcile(self):
        data = pl.DataFrame({'group': ['a', 'b'] * 20, 'exposure': [.5, 1.] * 20,
                             'count': [1., 4.] * 20})
        model = Plan('poisson', exposure='exposure', exposure_role='offset').categorical('group').fit(data, 'count')
        self.assertTrue(model.converged)
        self.assertAlmostEqual(model.predict(data)['predictions'].sum(), data['count'].sum(), places=6)
