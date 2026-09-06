"""Constrained fits checked against a separate dense constrained optimizer."""
import json
import tempfile
import unittest

import numpy as np
import polars as pl
from scipy.optimize import LinearConstraint, minimize

from avenue_model import Plan, GLMOptions, Workbook, coefficient_intervals


class MonotonicTests(unittest.TestCase):
    def test_numeric_looking_category_labels_survive_json_and_csv(self):
        labels = ['001', '1', '1.0', 'inf', '-999', '10'] * 8
        data = pl.DataFrame({'label': labels, 'y': [1., 2., 3., 4., 5., 6.] * 8})
        model = Plan('poisson').categorical('label', base='first').fit(data, 'y')
        expected = model.predict(data).to_numpy()
        with tempfile.TemporaryDirectory() as path:
            model.to_workbook().save_json(path+'/model.json')
            model.to_workbook().save_csv_dir(path+'/csv')
            for workbook in [Workbook.load_json(path+'/model.json'), Workbook.load_csv_dir(path+'/csv')]:
                np.testing.assert_allclose(workbook.to_model().predict(data).to_numpy(), expected, rtol=1e-12)

    def test_multiterm_fits_match_independent_constrained_optimizer(self):
        rng = np.random.default_rng(741)
        n = 900
        band = rng.integers(0, 5, n)
        group = rng.integers(0, 2, n)
        weight = rng.uniform(.2, 2., n)
        design = np.column_stack([np.eye(5)[band], group])
        constraint = np.zeros((4, 6))
        for i in range(4):
            constraint[i, i:i+2] = [-1., 1.]
        for direction, sign in [('increasing', 1.), ('decreasing', -1.)]:
            mean = np.exp(.4 + sign*np.array([0., .9, .4, 1.2, 1.7])[band] + .3*group)
            for family, p in [('poisson', 1.), ('tweedie', 1.5), ('gamma', 2.)]:
                y = rng.poisson(mean).astype(float) if p < 2 else rng.gamma(3., mean/3.)
                data = pl.DataFrame({'x': band.astype(float), 'group': group.astype(str), 'w': weight, 'y': y})
                plan = Plan(family, exposure='w', exposure_role='weight', tweedie_power=p).monotone(
                    'x', direction, breaks=[0., 1., 2., 3.]).categorical('group', base='first')
                plan = Plan.from_json(plan.to_json())
                model = plan.fit(data, 'y', GLMOptions(tolerance=1e-10, max_iterations=1000))
                self.assertTrue(model.converged)
                self.assertEqual(model.solver_used, 'table')

                def objective(beta):
                    eta = design @ beta
                    mu = np.exp(eta)
                    if p == 1:
                        loss = mu-y*eta
                    elif p == 2:
                        loss = eta+y/mu
                    else:
                        loss = mu**(2-p)/(2-p)+y*mu**(1-p)/(p-1)
                    gradient = design.T @ (weight*(mu-y)*mu**(1-p))/weight.sum()
                    return np.dot(weight, loss)/weight.sum(), gradient

                reference = minimize(objective, np.zeros(6), jac=True, method='SLSQP',
                    constraints=LinearConstraint(sign*constraint, 0., np.inf),
                    options={'ftol': 1e-13, 'maxiter': 1000})
                self.assertTrue(reference.success, reference.message)
                actual = model.predict(data).to_numpy().reshape(-1)
                np.testing.assert_allclose(actual, np.exp(design @ reference.x), rtol=3e-6, atol=1e-7)
                factors = model.rating_tables_by_name()['x']['Coefficient'].to_numpy()
                self.assertTrue(np.all(sign*np.diff(factors) >= 0.))
                self.assertTrue(np.any(np.diff(factors) == 0.))  # Active pooling, not an already ordered fit.
                self.assertIsNone(model.inference_summary['n_parameters'])
                with self.assertRaisesRegex(ValueError, 'Monotonic'):
                    coefficient_intervals(model)
                with tempfile.TemporaryDirectory() as path:
                    model.to_workbook().save_csv_dir(path+'/csv')
                    loaded = Workbook.load_csv_dir(path+'/csv').to_model()
                    np.testing.assert_allclose(loaded.predict(data).to_numpy().reshape(-1), actual, rtol=1e-12)
                    self.assertEqual(json.loads(model.plan.to_json())['terms'][0]['direction'], direction)
                    self.assertIn('Monotonic', model.inference_summary['standard_errors_note'])

    def test_empty_bands_and_rejected_combinations(self):
        data = pl.DataFrame({'x': [1., 1., 3., 3., 5., 5.], 'y': [1., 2., 3., 4., 5., 6.]})
        data = pl.concat([data, data])
        plan = Plan('poisson').monotone('x', 'increasing', breaks=[0., 1., 2., 3., 4., 5.])
        model = plan.fit(data, 'y')
        self.assertTrue(model.converged)
        table = model.rating_tables_by_name()['x']
        self.assertEqual(table['Status'].to_list(), ['no_data', 'estimated', 'no_data', 'estimated', 'no_data', 'estimated', 'no_data'])
        for left, right in [(0, 1), (1, 2), (3, 4), (5, 6)]:
            self.assertEqual(table['Coefficient'][left], table['Coefficient'][right])
        with self.assertRaisesRegex(ValueError, 'direction'):
            Plan('poisson').monotone('x', 'auto', breaks=[2.])
        for options in [GLMOptions(solver='global'), GLMOptions(alpha=.1), GLMOptions(covariance='hc0')]:
            with self.assertRaisesRegex(ValueError, 'Monotonic'):
                plan.fit(data, 'y', options)
        with self.assertRaisesRegex(ValueError, 'Monotonic'):
            Plan('gaussian').monotone('x', 'increasing', breaks=[2.]).fit(data, 'y')
        with self.assertRaisesRegex(ValueError, 'monotonic predictor'):
            plan.banded('x', breaks=[2.]).fit(data, 'y')


if __name__ == '__main__':
    unittest.main()
