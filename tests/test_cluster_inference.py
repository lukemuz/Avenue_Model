"""One-way CR0 matches independently aggregated observation scores."""
import tempfile
import unittest

import numpy as np
import polars as pl
from avenue_model import GLMOptions, Plan, coefficient_intervals


class ClusterInferenceTests(unittest.TestCase):
    def test_family_cluster_scores_and_both_solvers(self):
        rng = np.random.default_rng(843)
        cluster = np.repeat(np.arange(80), 4)
        group = np.tile(['a', 'b', 'c', 'a'], 80)
        x = np.column_stack([np.ones(len(group)), group == 'b', group == 'c']).astype(float)
        weight = rng.uniform(.2, 2., len(group))
        shock = np.repeat(rng.gamma(.8, 1/.8, 80), 4)
        for family in ('gaussian', 'poisson', 'gamma', 'tweedie', 'binomial'):
            mean = np.exp(x @ np.array([.2, -.3, .5]))
            if family == 'gaussian':
                y = x @ np.array([1., -.3, .5]) + shock + rng.normal(0, .2, len(group))
            elif family == 'binomial':
                y = rng.binomial(1, mean * shock / (1 + mean * shock)).astype(float)
            elif family == 'poisson':
                y = rng.poisson(mean * shock).astype(float)
            else:
                y = mean * shock * rng.gamma(2., .5, len(group))
            data = pl.DataFrame({'group': group, 'w': weight, 'y': y,
                                  'policy': ['policy-' + str(c) for c in cluster]})
            plan = Plan(family, exposure='w', exposure_role='weight').categorical('group', base='first')
            ordinary = plan.fit(data, 'y')
            for solver in ('global', 'table'):
                with self.subTest(family=family, solver=solver):
                    model = plan.fit(data, 'y', GLMOptions(covariance='cluster', cluster='policy', solver=solver))
                    self.assertTrue(model.converged)
                    mu = model.predict(data.drop('policy')).to_series().to_numpy()
                    np.testing.assert_allclose(mu, ordinary.predict(data).to_series(), rtol=1e-7, atol=1e-7)
                    if family == 'gaussian':
                        derivative = variance = np.ones(len(mu))
                    elif family == 'binomial':
                        derivative = variance = mu * (1-mu)
                    else:
                        derivative = mu
                        variance = mu ** {'poisson': 1, 'gamma': 2, 'tweedie': 1.5}[family]
                    scores = x * (weight * (y-mu) * derivative / variance)[:, None]
                    cluster_scores = np.stack([scores[cluster == c].sum(axis=0) for c in range(80)])
                    bread = np.linalg.inv(x.T @ ((weight * derivative**2 / variance)[:, None] * x))
                    covariance = bread @ (cluster_scores.T @ cluster_scores) @ bread
                    expected = np.sqrt(np.diag(covariance))
                    tables = model.rating_tables_by_name()
                    actual = [tables['intercept']['Standard_Error'][0], *tables['group']['Standard_Error'].to_list()[1:]]
                    np.testing.assert_allclose(actual, expected, rtol=1e-7, atol=1e-9)
                    self.assertEqual(model.inference_summary['n_clusters'], 80)
                    self.assertEqual(model.inference_summary['cluster_column'], 'policy')
                    self.assertEqual(model.inference_summary['covariance_method'], 'cluster_cr0')
                    self.assertIn('| Positive-weight clusters | 80 |', model.report().markdown)
                    intervals = coefficient_intervals(model)
                    self.assertIn('independent clusters', intervals.metadata['interpretation'])
                    self.assertIsNone(intervals.metadata['dispersion'])

    def test_singletons_equal_hc0_and_zero_weight_groups_do_not_count(self):
        data = pl.DataFrame({'g': ['a', 'b'] * 40, 'y': [1., 4., 2., 8.] * 20,
                             'w': [1.] * 79 + [0.], 'cluster': list(range(80))})
        plan = Plan('poisson', exposure='w', exposure_role='weight').categorical('g', base='first')
        hc0 = plan.fit(data, 'y', GLMOptions(covariance='hc0'))
        clustered = plan.fit(data, 'y', GLMOptions(covariance='cluster', cluster='cluster'))
        reverse = plan.fit(data.reverse(), 'y', GLMOptions(covariance='cluster', cluster='cluster'))
        self.assertEqual(clustered.inference_summary['n_clusters'], 79)
        for name, table in hc0.rating_tables_by_name().items():
            np.testing.assert_allclose(clustered.rating_tables_by_name()[name]['Standard_Error'], table['Standard_Error'], atol=1e-9)
            np.testing.assert_allclose(reverse.rating_tables_by_name()[name]['Standard_Error'], table['Standard_Error'], atol=1e-9)
        loaded = clustered.to_workbook().to_model()
        self.assertEqual(loaded.inference_summary, {})
        np.testing.assert_allclose(loaded.predict(data.select('g')).to_series(), clustered.predict(data).to_series(), atol=1e-12)

    def test_offset_scores_use_count_means_and_preserve_row_order(self):
        rng = np.random.default_rng(141)
        ids = np.repeat(np.arange(60), 3)
        exposure = rng.uniform(.1, 2., len(ids))
        count = rng.poisson(exposure * np.repeat(rng.gamma(2., 1., 60), 3)).astype(float)
        data = pl.DataFrame({'cluster': ids, 'exposure': exposure, 'count': count})
        model = Plan('poisson', exposure='exposure', exposure_role='offset').fit(
            data, 'count', GLMOptions(covariance='cluster', cluster='cluster'))
        mu = model.predict(data).to_series().to_numpy()
        group_scores = np.array([(count-mu)[ids == i].sum() for i in range(60)])
        expected = np.sqrt(np.sum(group_scores**2)) / np.sum(mu)
        self.assertAlmostEqual(model.rating_tables_by_name()['intercept']['Standard_Error'][0], expected, places=9)
        with self.assertRaisesRegex(ValueError, 'cannot rescale'):
            coefficient_intervals(model, dispersion='quasi_poisson')

    def test_invalid_cluster_definitions_and_options(self):
        data = pl.DataFrame({'y': [1., 2., 1., 2.], 'cluster': [1, 1, 2, 2], 'w': [1., 1., 0., 0.]})
        with self.assertRaisesRegex(ValueError, 'two positive-weight clusters'):
            Plan('poisson', exposure='w', exposure_role='weight').fit(data, 'y', GLMOptions(covariance='cluster', cluster='cluster'))
        for invalid in (pl.Series('cluster', [1., 1., 2., 2.]), pl.Series('cluster', ['a', None, 'b', 'b'])):
            with self.assertRaisesRegex(ValueError, 'non-null integer or string'):
                Plan('poisson').fit(data.with_columns(invalid), 'y', GLMOptions(covariance='cluster', cluster='cluster'))
        with self.assertRaisesRegex(ValueError, 'Cluster covariance requires'):
            Plan('poisson').fit(data, 'y', GLMOptions(covariance='cluster', cluster='cluster', alpha=.1))
        with self.assertRaisesRegex(ValueError, 'requires a nonempty cluster'):
            GLMOptions(covariance='cluster')
        with self.assertRaisesRegex(ValueError, "cluster requires covariance"):
            GLMOptions(cluster='cluster')
