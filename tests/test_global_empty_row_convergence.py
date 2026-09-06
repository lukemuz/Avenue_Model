"""Treatment-coded ridge KKT checks exclude fixed, unsupported reference directions."""
import unittest

import numpy as np
import polars as pl
from scipy.optimize import root

from avenue_model import GLMOptions, Plan


class GlobalEmptyRowConvergenceTests(unittest.TestCase):
    def test_ridge_with_nonzero_empty_row_matches_independent_free_score(self):
        x = np.tile([0., 1., 2.], 40)
        y = np.tile([1., 3., 7., 2., 4., 6.], 20)
        w = np.tile([.5, 1., 2.], 40)
        data = pl.DataFrame({'x': x, 'y': y, 'w': w})
        design = np.column_stack([np.ones(len(x)), x == 1, x == 2])
        for alpha in [.001, .1]:
            ridge = np.array([0., alpha, alpha])
            def score(beta):
                return design.T @ (w * (np.exp(design @ beta) - y)) / w.sum() + ridge * beta
            def hessian(beta):
                return design.T @ ((w * np.exp(design @ beta))[:, None] * design) / w.sum() + np.diag(ridge)
            independent = root(score, [0., 0., 0.], jac=hessian, tol=1e-11)
            self.assertLess(np.max(np.abs(score(independent.x))), 1e-9)
            expected = np.exp(design @ independent.x)
            for solver in ['auto', 'global', 'table']:
                for empty_factor in [.7, -2.]:
                    with self.subTest(alpha=alpha, solver=solver, empty_factor=empty_factor):
                        table = pl.DataFrame({'x': [0., 1., 2., float('inf'), float('nan')],
                                              'Rating_Factor': [.2, .4, .8, empty_factor, -empty_factor]})
                        plan = Plan('poisson', exposure='w', exposure_role='weight').given('shape', table)
                        fitted = plan.fit(data, 'y', GLMOptions(alpha=alpha, l1_ratio=0.,
                                                              solver=solver, tolerance=1e-10))
                        self.assertTrue(fitted.converged, fitted.report().fit_summary)
                        self.assertLess(fitted.report().fit_summary['max_gradient'], 1e-10)
                        np.testing.assert_allclose(fitted.predict(data)['predictions'], expected,
                                                   atol=1e-8, rtol=1e-8)
                        quotes = pl.DataFrame({'x': [3., float('nan'), None]})
                        np.testing.assert_allclose(fitted.predict(quotes)['predictions'],
                                                   np.exp(independent.x[0]), atol=1e-8, rtol=1e-8)
                        unsupported = fitted.rating_tables_by_name()['shape'].filter(pl.col('Status') == 'no_data')
                        self.assertEqual(unsupported.height, 2)
                        np.testing.assert_allclose(unsupported['Rating_Factor'], 0., atol=1e-10)
                        loaded = fitted.to_workbook().to_model()
                        np.testing.assert_array_equal(loaded.predict(quotes).to_numpy(),
                                                      fitted.predict(quotes).to_numpy())
                        stopped = plan.fit(data, 'y', GLMOptions(alpha=alpha, l1_ratio=0.,
                                          solver=solver, tolerance=1e-10, max_iterations=1))
                        self.assertFalse(stopped.converged)

    def test_paired_ridge_and_elastic_net_use_reference_for_unsupported_rows(self):
        data = pl.DataFrame({'x': [0., 1., 2.] * 40, 'z': [1., 2., 0.] + [0., 1., 2.] * 39,
                             'y': [1., 3., 7., 2., 4., 6.] * 20})
        plan = Plan('poisson')
        for column in ['x', 'z']:
            plan = plan.given(column, pl.DataFrame({column: [0., 1., 2., float('inf'), float('nan')],
                                                  'Rating_Factor': [.2, .4, .8, .7, -2.]}))
        quotes = pl.DataFrame({'x': [0., 3., None], 'z': [0., 3., None]})
        self.assertGreater(plan.check(data, 'y').correlated_pairs[0][2], .9)
        for ratio in [0., .5, 1.]:
            predictions = []
            for solver in ['table', 'global']:
                with self.subTest(ratio=ratio, solver=solver):
                    fitted = plan.fit(data, 'y', GLMOptions(alpha=.01, l1_ratio=ratio,
                                       solver=solver, tolerance=1e-9, max_iterations=500))
                    self.assertTrue(fitted.converged, fitted.report().fit_summary)
                    for table in fitted.rating_tables_by_name().values():
                        unsupported = table.filter(pl.col('Status') == 'no_data')
                        np.testing.assert_allclose(unsupported['Rating_Factor'], 0., atol=1e-9)
                    scored = fitted.predict(quotes)['predictions'].to_numpy()
                    np.testing.assert_allclose(scored, scored[0], atol=1e-8, rtol=1e-8)
                    predictions.append(fitted.predict(data)['predictions'].to_numpy())
            np.testing.assert_allclose(predictions[0], predictions[1], atol=1e-7, rtol=1e-7)

    def test_zero_weight_rows_across_families(self):
        for family in ['gaussian', 'poisson', 'gamma', 'tweedie', 'binomial']:
            with self.subTest(family=family):
                values = [.1, .3, .7] if family == 'binomial' else [1., 3., 7.]
                data = pl.DataFrame({'x': [0., 1., 2.] * 40 + [3., float('nan')],
                                     'y': values * 40 + [values[-1]] * 2,
                                     'w': [1.] * 120 + [0., 0.]})
                table = pl.DataFrame({'x': [0., 1., 2., float('inf'), float('nan')],
                                     'Rating_Factor': [0., .1, .2, .7, -2.]})
                plan = Plan(family, exposure='w', exposure_role='weight').given('shape', table)
                predictions = []
                for solver in ['table', 'global']:
                    fitted = plan.fit(data, 'y', GLMOptions(alpha=.05, l1_ratio=0.,
                                       solver=solver, tolerance=1e-9, max_iterations=500))
                    self.assertTrue(fitted.converged, fitted.report().fit_summary)
                    predictions.append(fitted.predict(data)['predictions'].to_numpy())
                    np.testing.assert_allclose(predictions[-1][-2:], predictions[-1][0],
                                               atol=1e-8, rtol=1e-8)
                    self.assertEqual(fitted.rating_tables_by_name()['shape'].filter(
                        pl.col('Status') == 'no_data').height, 2)
                np.testing.assert_allclose(predictions[0], predictions[1], atol=1e-7, rtol=1e-7)


if __name__ == '__main__':
    unittest.main()
