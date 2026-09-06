"""Joint contrasts checked against independent dense covariance calculations."""
import tempfile
import unittest

import numpy as np
import polars as pl
from scipy.stats import chi2

from avenue_model import GLMOptions, Plan, Workbook, term_tests
from avenue_model.inference import _chi_square_survival


class TermTestTests(unittest.TestCase):
    def test_weighted_family_tests_use_full_covariance_and_are_anchor_invariant(self):
        rng = np.random.default_rng(631)
        g = np.tile(np.array(['a', 'b', 'c']), 200)
        h = rng.integers(0, 2, len(g))
        x = np.column_stack([np.ones(len(g)), g == 'b', g == 'c', h]).astype(float)
        w = rng.uniform(.2, 2., len(g))
        eta = x @ np.array([.2, .3, -.2, .4])
        cluster = np.repeat(np.arange(100), 6)
        for family in ('gaussian', 'poisson', 'gamma', 'tweedie', 'binomial'):
            mu = np.exp(eta)
            if family == 'gaussian':
                y = eta + rng.normal(size=len(g))
            elif family == 'binomial':
                y = rng.binomial(1, mu/(1+mu)).astype(float)
            elif family == 'poisson':
                y = rng.poisson(mu*rng.gamma(2., .5, len(g))).astype(float)
            else:
                y = rng.gamma(2., mu/2.)
            data = pl.DataFrame({'g': g, 'h': h.astype(str), 'y': y, 'w': w, 'cluster': cluster})
            plan = Plan(family, exposure='w', exposure_role='weight').categorical('g', base='first').categorical('h', base='first')
            for covariance in ('model_based', 'hc0', 'cluster'):
                reference_statistic = None
                for normalization in ('base_level', 'weighted_mean', 'none'):
                    model = plan.fit(data, 'y', GLMOptions(covariance=covariance,
                        cluster='cluster' if covariance == 'cluster' else None,
                        normalization=normalization, tolerance=1e-10)) if normalization != 'none' or covariance == 'model_based' else None
                    if model is None:
                        continue  # Existing robust inference requires an identified reporting anchor.
                    self.assertTrue(model.converged)
                    fitted = model.predict(data).to_series().to_numpy()
                    link = fitted if family == 'gaussian' else np.log(fitted/(1-fitted)) if family == 'binomial' else np.log(fitted)
                    beta = np.linalg.lstsq(x, link, rcond=None)[0]
                    if family == 'gaussian':
                        derivative = variance = np.ones(len(g))
                    elif family == 'binomial':
                        derivative = variance = fitted*(1-fitted)
                    else:
                        derivative = fitted
                        variance = fitted**{'poisson': 1., 'gamma': 2., 'tweedie': 1.5}[family]
                    bread = np.linalg.inv(x.T @ ((w*derivative**2/variance)[:, None]*x))
                    if covariance == 'model_based':
                        scale = 1. if family in ('poisson', 'binomial') else np.sum(w*(y-fitted)**2/variance)/(len(g)-4)
                        cov = bread*scale
                    else:
                        scores = x*(w*(y-fitted)*derivative/variance)[:, None]
                        if covariance == 'cluster':
                            scores = np.array([scores[cluster == i].sum(axis=0) for i in np.unique(cluster)])
                        cov = bread @ (scores.T @ scores) @ bread
                    expected = beta[1:3] @ np.linalg.solve(cov[1:3, 1:3], beta[1:3])
                    result = term_tests(model)
                    row = result.table.filter(pl.col('term') == 'g').row(0, named=True)
                    self.assertEqual(row['df'], 2)
                    self.assertEqual(row['status'], 'available')
                    self.assertAlmostEqual(row['statistic'], expected, delta=1e-6*max(1., expected))
                    self.assertAlmostEqual(row['p_value'], chi2.sf(expected, 2), delta=1e-8)
                    if reference_statistic is not None:
                        self.assertAlmostEqual(row['statistic'], reference_statistic, delta=1e-6)
                    reference_statistic = row['statistic']
                    if family == 'poisson' and covariance == 'model_based':
                        quasi = term_tests(model, dispersion='quasi_poisson')
                        phi = np.sum(w*(y-fitted)**2/fitted)/(len(g)-4)
                        self.assertAlmostEqual(quasi.table.filter(pl.col('term') == 'g')['statistic'][0], expected/phi, delta=1e-6)

    def test_interaction_joint_null_and_polynomial_terms(self):
        rng = np.random.default_rng(45)
        a = np.repeat(['A', 'B'], 300)
        b = np.tile(np.repeat(['X', 'Y', 'Z'], 100), 2)
        x = np.column_stack([np.ones(600), a == 'B', b == 'Y', b == 'Z',
                             (a == 'B') & (b == 'Y'), (a == 'B') & (b == 'Z')]).astype(float)
        y = rng.poisson(np.exp(x @ np.array([.1, .3, -.2, .4, .2, -.4]))).astype(float)
        data = pl.DataFrame({'a': a, 'b': b, 'y': y})
        model = Plan('poisson').categorical('a', base='first').categorical('b', base='first').interaction(['a', 'b'], [None, None]).fit(data, 'y')
        mu = model.predict(data).to_series().to_numpy()
        beta = np.linalg.lstsq(x, np.log(mu), rcond=None)[0]
        cov = np.linalg.inv(x.T @ (mu[:, None]*x))
        row = term_tests(model).table.filter(pl.col('term') == 'a x b').row(0, named=True)
        self.assertEqual(row['df'], 2)
        self.assertIn('locked rows', row['null_hypothesis'])
        self.assertAlmostEqual(row['statistic'], beta[-2:] @ np.linalg.solve(cov[-2:, -2:], beta[-2:]), places=7)
        z = np.tile(np.arange(5, dtype=float), 100)
        data = pl.DataFrame({'z': z, 'y': 2.+.3*z+.2*z*z+rng.normal(size=len(z))})
        model = Plan('gaussian').variate('z', breaks=[0., 1., 2., 3.], values=[0., 1., 2., 3., 4.], degree=2).fit(data, 'y')
        x = np.column_stack([np.ones(len(z)), z, z*z])
        beta = np.linalg.lstsq(x, data['y'].to_numpy(), rcond=None)[0]
        phi = np.sum((data['y'].to_numpy()-x@beta)**2)/(len(z)-3)
        cov = phi*np.linalg.inv(x.T@x)
        row = term_tests(model).table.row(0, named=True)
        self.assertEqual(row['df'], 2)
        self.assertAlmostEqual(row['statistic'], beta[1:] @ np.linalg.solve(cov[1:, 1:], beta[1:]), places=6)

    def test_unavailable_models_and_empty_bands_are_explicit(self):
        data = pl.DataFrame({'x': [1., 3.] * 30, 'y': [2., 5., 1., 4.] * 15})
        plan = Plan('poisson').banded('x', breaks=[1., 2.])
        model = plan.fit(data, 'y')
        result = term_tests(model).table.row(0, named=True)
        self.assertEqual(result['df'], 1)
        self.assertEqual(result['excluded_rows'], [1])
        self.assertIn('supported', result['null_hypothesis'])
        leading = term_tests(Plan('poisson').banded('x', breaks=[0., 1., 2.]).fit(data, 'y')).table.row(0, named=True)
        self.assertEqual(leading['excluded_rows'], [0, 2])
        self.assertAlmostEqual(leading['statistic'], result['statistic'], places=7)
        self.assertEqual(term_tests(Plan('poisson').fit(data, 'y')).table.height, 0)
        for options in (GLMOptions(alpha=.1), GLMOptions(compute_standard_errors=False), GLMOptions(max_iterations=1, tolerance=1e-15)):
            with self.assertRaises(ValueError):
                term_tests(plan.fit(data, 'y', options))
        with self.assertRaisesRegex(ValueError, 'Monotonic'):
            term_tests(Plan('poisson').monotone('x', 'increasing', breaks=[2.]).fit(data, 'y'))
        with self.assertRaisesRegex(ValueError, 'HC0/cluster'):
            term_tests(plan.fit(data, 'y', GLMOptions(covariance='hc0')), dispersion='quasi_poisson')
        with tempfile.TemporaryDirectory() as path:
            model.to_workbook().save_json(path+'/model.json')
            with self.assertRaisesRegex(ValueError, 'original'):
                term_tests(Workbook.load_json(path+'/model.json').to_model())

    def test_insufficient_cluster_rank_and_locked_prior_terms(self):
        data = pl.DataFrame({'g': ['a', 'b', 'c'] * 40, 'y': [1., 3., 6., 2., 4., 8.] * 20,
                             'cluster': [0, 1] * 60})
        plan = Plan('poisson').categorical('g', base='first')
        model = plan.fit(data, 'y', GLMOptions(covariance='cluster', cluster='cluster'))
        row = term_tests(model).table.row(0, named=True)
        self.assertEqual(row['status'], 'unavailable')
        self.assertIsNone(row['p_value'])
        self.assertIn('cluster scores', row['note'])
        prior = plan.fit(data, 'y')
        updated = Plan('poisson').offset_model(prior, prefix='prior').fit(data, 'y')
        result = term_tests(updated).table
        self.assertTrue(all(status == 'unavailable' for status in result['status']))
        self.assertTrue(all('fixed' in note for note in result['note']))

    def test_aliased_variates_do_not_receive_separate_tests(self):
        x = np.tile(np.arange(4, dtype=float), 30)
        data = pl.DataFrame({'x': x, 'copy': x, 'y': 1.+x+np.sin(np.arange(len(x)))})
        plan = Plan('gaussian')
        for column in ('x', 'copy'):
            plan = plan.variate(column, breaks=[0., 1., 2.], values=[0., 1., 2., 3.])
        model = plan.fit(data, 'y')
        self.assertTrue(model.converged)
        tests = term_tests(model).table
        self.assertTrue(all(status == 'unavailable' for status in tests['status']))
        self.assertTrue(all('separately estimable' in note for note in tests['note']))
        for table in model.rating_tables_by_name().values():
            if 'x' in table.columns or 'copy' in table.columns:
                self.assertTrue(all(np.isnan(value) for value in table['Standard_Error'].to_list()[1:]))

    def test_chi_square_tails_match_independent_special_function(self):
        for df in (1, 2, 3, 4, 7, 100, 999, 5000):
            for statistic in (0., 1e-12, .1, 1., 10., float(df), 2.*df, 10000.):
                expected = chi2.sf(statistic, df)
                actual = _chi_square_survival(statistic, df)
                self.assertAlmostEqual(actual, expected, delta=2e-11*max(expected, 1e-290))


if __name__ == '__main__':
    unittest.main()
