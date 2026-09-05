"""HC0 covariance compared with independent dense score/information calculations."""
import tempfile
import unittest

import numpy as np
import polars as pl
from avenue_model import GLMOptions, Plan, coefficient_intervals, save_bundle


class RobustInferenceTests(unittest.TestCase):
    def test_weighted_family_scores_and_unchanged_estimates(self):
        rng = np.random.default_rng(131)
        group = np.tile(['a', 'b', 'c'], 200)
        x = np.column_stack([np.ones(len(group)), group == 'b', group == 'c']).astype(float)
        weight = rng.uniform(.2, 2., len(group))
        for family in ('gaussian', 'poisson', 'gamma', 'tweedie', 'binomial'):
            mean = np.exp(x @ np.array([.1, .3, -.2]))
            if family == 'gaussian':
                y = x @ np.array([1., .5, 2.]) + rng.normal(size=len(group)) * (1 + 2 * (group == 'c'))
            elif family == 'binomial':
                y = rng.binomial(1, mean / (1 + mean)).astype(float)
            elif family == 'poisson':
                y = rng.poisson(mean * rng.gamma(.7, 1/.7, len(group))).astype(float)
            else:
                y = mean * rng.gamma(.7, 1/.7, len(group))
            data = pl.DataFrame({'g': group, 'w': weight, 'y': y})
            plan = Plan(family, exposure='w', exposure_role='weight').categorical('g', base='first')
            ordinary = plan.fit(data, 'y')
            for solver in ('global', 'table'):
                with self.subTest(family=family, solver=solver):
                    model = plan.fit(data, 'y', GLMOptions(covariance='hc0', solver=solver))
                    self.assertTrue(model.converged)
                    mu = model.predict(data).to_series().to_numpy()
                    np.testing.assert_allclose(mu, ordinary.predict(data).to_series(), atol=1e-7, rtol=1e-7)
                    if family == 'gaussian':
                        derivative, variance = np.ones(len(mu)), np.ones(len(mu))
                    elif family == 'binomial':
                        derivative = variance = mu * (1-mu)
                    else:
                        derivative = mu
                        variance = mu ** {'poisson': 1., 'gamma': 2., 'tweedie': 1.5}[family]
                    scores = weight * (y - mu) * derivative / variance
                    information = x.T @ ((weight * derivative**2 / variance)[:, None] * x)
                    bread = np.linalg.inv(information)
                    covariance = bread @ (x.T @ (scores[:, None]**2 * x)) @ bread
                    expected = np.sqrt(np.diag(covariance))
                    actual = [model.rating_tables_by_name()['intercept']['Standard_Error'][0],
                              *model.rating_tables_by_name()['g']['Standard_Error'].to_list()[1:]]
                    np.testing.assert_allclose(actual, expected, rtol=1e-7, atol=1e-9)
                    self.assertEqual(model.inference_summary['covariance_method'], 'hc0')
                    self.assertIn('| Covariance | hc0 |', model.report().markdown)
                    intervals = coefficient_intervals(model)
                    self.assertEqual(intervals.metadata['covariance_method'], 'hc0')
                    self.assertIsNone(intervals.metadata['dispersion'])
                    if family == 'poisson':
                        with self.assertRaisesRegex(ValueError, 'rescale HC0'):
                            coefficient_intervals(model, dispersion='quasi_poisson')

    def test_count_offsets_and_hierarchical_contrasts(self):
        rng = np.random.default_rng(17)
        a = np.tile(['a', 'a', 'b', 'b'], 100)
        b = np.tile(['x', 'y', 'x', 'y'], 100)
        x = np.column_stack([np.ones(len(a)), a == 'b', b == 'y', (a == 'b') & (b == 'y')]).astype(float)
        exposure = rng.uniform(.1, 2., len(a))
        count = rng.poisson(exposure * np.exp(x @ np.array([.1, .4, -.2, .5]))).astype(float)
        data = pl.DataFrame({'a': a, 'b': b, 'e': exposure, 'y': count})
        plan = (Plan('poisson', exposure='e', exposure_role='offset')
                .categorical('a', base='first').categorical('b', base='first')
                .interaction(['a', 'b'], [None, None]))
        model = plan.fit(data, 'y', GLMOptions(covariance='hc0'))
        self.assertTrue(model.converged)
        mu = model.predict(data).to_series().to_numpy()
        bread = np.linalg.inv(x.T @ (mu[:, None] * x))
        covariance = bread @ (x.T @ ((count-mu)[:, None]**2 * x)) @ bread
        term = next(t for t in model.resolved if t['kind'] == 'interaction_contrast')
        table = model.rating_tables_by_name()[term['name']]
        interior = table.filter((pl.col('a_Level') == 'b') & (pl.col('b_Level') == 'y'))
        self.assertAlmostEqual(interior['Standard_Error'][0], np.sqrt(covariance[-1, -1]), places=8)
        with tempfile.TemporaryDirectory() as path:
            bundle = save_bundle(model, path + '/model')
            self.assertEqual(bundle.source_evidence['inference_summary']['covariance_method'], 'hc0')
            self.assertEqual(bundle.model.inference_summary, {})

    def test_weighted_mean_intercept_and_factor_contrasts(self):
        data = pl.DataFrame({'g': ['a'] * 3 + ['b'] * 4,
                             'y': [1., 2., 3., 2., 3., 6., 9.],
                             'w': [1., 2., 1., 1., 3., 1., 2.]})
        x = np.column_stack([np.ones(7), (data['g'].to_numpy() == 'b').astype(float)])
        w = data['w'].to_numpy()
        share = np.sum(w * x[:, 1]) / np.sum(w)
        contrasts = np.array([[1., share], [0., -share], [0., 1-share]])
        for method in ('model_based', 'hc0'):
            model = Plan('gaussian', exposure='w', exposure_role='weight').categorical('g', base='first').fit(
                data, 'y', GLMOptions(covariance=method, normalization='weighted_mean'))
            mu = model.predict(data).to_series().to_numpy()
            residual = data['y'].to_numpy() - mu
            bread = np.linalg.inv(x.T @ (w[:, None] * x))
            if method == 'hc0':
                covariance = bread @ (x.T @ ((w * residual)[:, None]**2 * x)) @ bread
            else:
                covariance = bread * np.sum(w * residual**2) / (len(w)-2)
            expected = np.sqrt(np.diag(contrasts @ covariance @ contrasts.T))
            actual = [model.rating_tables_by_name()['intercept']['Standard_Error'][0],
                      *model.rating_tables_by_name()['g']['Standard_Error'].to_list()]
            np.testing.assert_allclose(actual, expected, atol=1e-9, rtol=1e-7)

    def test_unsupported_requests_fail_clearly(self):
        data = pl.DataFrame({'y': [1., 2.] * 20})
        for options in (GLMOptions(covariance='hc0', alpha=.1),
                        GLMOptions(covariance='hc0', compute_standard_errors=False)):
            with self.assertRaisesRegex(ValueError, 'HC0 covariance requires'):
                Plan('poisson').fit(data, 'y', options)
        with self.assertRaisesRegex(ValueError, 'Unknown covariance'):
            GLMOptions(covariance='hc3')
