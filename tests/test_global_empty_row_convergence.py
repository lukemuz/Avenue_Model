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
            for solver in ['auto', 'global']:
                for empty_factor in [.7, -2.]:
                    with self.subTest(alpha=alpha, solver=solver, empty_factor=empty_factor):
                        table = pl.DataFrame({'x': [0., 1., float('inf'), float('nan')],
                                              'Rating_Factor': [.2, .4, .8, empty_factor]})
                        plan = Plan('poisson', exposure='w', exposure_role='weight').given('shape', table)
                        fitted = plan.fit(data, 'y', GLMOptions(alpha=alpha, l1_ratio=0.,
                                                              solver=solver, tolerance=1e-10))
                        self.assertTrue(fitted.converged, fitted.report().fit_summary)
                        self.assertLess(fitted.report().fit_summary['max_gradient'], 1e-10)
                        np.testing.assert_allclose(fitted.predict(data)['predictions'], expected,
                                                   atol=1e-8, rtol=1e-8)
                        stopped = plan.fit(data, 'y', GLMOptions(alpha=alpha, l1_ratio=0.,
                                          solver=solver, tolerance=1e-10, max_iterations=1))
                        self.assertFalse(stopped.converged)


if __name__ == '__main__':
    unittest.main()
