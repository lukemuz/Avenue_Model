"""Spline covariance and joint tests against independent dense SciPy basis algebra."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import polars as pl
from scipy.interpolate import CubicSpline

from avenue_model import Plan, GLMOptions, coefficient_intervals, term_tests, Workbook


class SplineInferenceTests(unittest.TestCase):
    def test_five_families_and_three_covariances_with_reporting_anchors(self):
        fixture = json.loads((Path(__file__).parent/'fixtures/spline_fit.json').read_text())
        knots = np.array(fixture['knots'])
        x = np.array(fixture['x'])
        weights = np.array(fixture['weights'])
        category = np.array(fixture['category'])
        offset = np.array(fixture['offset'])
        cubic = CubicSpline(knots, np.eye(len(knots)), bc_type='natural')
        basis = cubic(np.clip(x, knots[0], knots[-1]))
        for mask, boundary, endpoint in [(x < knots[0], knots[0], 0), (x > knots[-1], knots[-1], -1)]:
            basis[mask] = np.eye(len(knots))[endpoint] + (x[mask]-boundary)[:, None]*cubic(boundary, 1)
        design = np.column_stack([np.ones(len(x)), category, basis[:, 1:]])
        cluster = np.arange(len(x)) % 17
        band_bounds = np.arange(len(x), dtype=float)
        band_bounds[-1] = np.inf
        known_offset = pl.DataFrame({'index': band_bounds, 'Rating_Factor': offset})
        for case in fixture['cases']:
            family = case['family']
            target = np.array(case['target'])
            expected_mu = np.array(case['means'])
            beta = np.array(case['coefficients'])
            data = pl.DataFrame({'x': x, 'category': category.astype(np.int32), 'w': weights,
                                 'index': np.arange(len(x), dtype=float), 'cluster': cluster, 'y': target})
            plan = (Plan(family, exposure='w', exposure_role='weight')
                    .categorical('category', base='first').spline('x', knots=knots.tolist())
                    .offset('known', known_offset))
            if family == 'gaussian':
                derivative = variance = np.ones(len(x))
            elif family == 'binomial':
                derivative = variance = expected_mu*(1-expected_mu)
            else:
                derivative = expected_mu
                variance = expected_mu**case['power']
            bread = np.linalg.inv(design.T @ ((weights*derivative**2/variance)[:, None]*design))
            phi = 1. if family in ('poisson', 'binomial') else np.sum(weights*(target-expected_mu)**2/variance)/(np.count_nonzero(weights)-design.shape[1])
            score_rows = (weights*(target-expected_mu)*derivative/variance)[:, None]*design
            for covariance in ('model_based', 'hc0', 'cluster'):
                cov = phi*bread
                if covariance != 'model_based':
                    scores = score_rows if covariance == 'hc0' else np.array([score_rows[cluster == k].sum(axis=0) for k in range(17)])
                    cov = bread @ (scores.T@scores) @ bread
                for anchor in ('base_level', 'weighted_mean'):
                    model = plan.fit(data, 'y', GLMOptions(covariance=covariance,
                        cluster='cluster' if covariance == 'cluster' else None,
                        normalization=anchor, tolerance=1e-10, max_iterations=1000))
                    self.assertTrue(model.converged)
                    self.assertEqual(model.inference_summary['n_parameters'], 5)
                    np.testing.assert_allclose(model.predict(data).to_numpy().reshape(-1), expected_mu, rtol=2e-7)
                    tables = model.rating_tables_by_name()
                    contrasts = np.zeros((1+2+len(knots), design.shape[1]))
                    contrasts[0,0] = 1.
                    contrasts[2,1] = 1.
                    contrasts[4:,2:] = np.eye(len(knots)-1)
                    if anchor == 'weighted_mean':
                        category_mean = np.sum(weights*category)/weights.sum()
                        spline_mean = weights@basis[:, 1:]/weights.sum()
                        contrasts[0,1] += category_mean
                        contrasts[0,2:] += spline_mean
                        contrasts[1:3,1] -= category_mean
                        contrasts[3:,2:] -= spline_mean
                    expected_se = np.sqrt(np.maximum(0., np.einsum('ij,jk,ik->i', contrasts, cov, contrasts)))
                    actual_se = np.concatenate([tables[name]['Standard_Error'].to_numpy() for name in ('intercept','category','x')])
                    np.testing.assert_allclose(actual_se, expected_se, rtol=3e-6, atol=1e-9)
                    joint = term_tests(model).table.filter(pl.col('term') == 'x').row(0, named=True)
                    expected_stat = beta[2:] @ np.linalg.solve(cov[2:,2:], beta[2:])
                    self.assertEqual(joint['df'], len(knots)-1)
                    self.assertEqual(joint['excluded_rows'], [])
                    self.assertIn('constant spline', joint['null_hypothesis'])
                    np.testing.assert_allclose(joint['statistic'], expected_stat, rtol=4e-6)
                    intervals = coefficient_intervals(model)
                    self.assertIsNotNone(intervals)
            # Unanchored factors have no standalone SE, but knot contrasts still
            # define a parameterization-invariant joint null.
            unanchored = plan.fit(data, 'y', GLMOptions(normalization='none', tolerance=1e-10))
            joint = term_tests(unanchored).table.filter(pl.col('term') == 'x').row(0, named=True)
            np.testing.assert_allclose(joint['statistic'], beta[2:]@np.linalg.solve((phi*bread)[2:,2:], beta[2:]), rtol=4e-6)
            if family == 'poisson':
                quasi_phi = np.sum(weights*(target-expected_mu)**2/expected_mu)/(np.count_nonzero(weights)-5)
                quasi = term_tests(unanchored, dispersion='quasi_poisson').table.filter(pl.col('term') == 'x').row(0, named=True)
                np.testing.assert_allclose(quasi['statistic'], joint['statistic']/quasi_phi, rtol=4e-6)


    def test_empty_support_group_does_not_remove_a_knot_parameter(self):
        x = np.linspace(.5, 2.5, 100)
        basis = CubicSpline([0., 1., 2., 3.], np.eye(4), bc_type='natural')(x)
        design = np.column_stack([np.ones(len(x)), basis[:, 1:]])
        y = np.exp(design@np.array([.2, .15, -.2, .1])) * np.exp(.2*np.sin(np.arange(len(x))))
        data = pl.DataFrame({'x': x, 'y': y})
        model = Plan('poisson').spline('x', knots=[0., 1., 2., 3.]).fit(data, 'y', GLMOptions(tolerance=1e-10))
        mu = model.predict(data).to_numpy().reshape(-1)
        cov = np.linalg.inv(design.T@(mu[:, None]*design))
        beta = np.linalg.lstsq(design, np.log(mu), rcond=None)[0]
        self.assertEqual(model.inference_summary['n_parameters'], 4)
        table = model.rating_tables_by_name()['x']
        self.assertNotIn('no_data', table['Status'].to_list())
        np.testing.assert_allclose(table['Standard_Error'].to_numpy(), np.r_[0., np.sqrt(np.diag(cov)[1:])], rtol=2e-7)
        joint = term_tests(model).table.row(0, named=True)
        self.assertEqual(joint['df'], 3)
        self.assertEqual(joint['excluded_rows'], [])
        np.testing.assert_allclose(joint['statistic'], beta[1:]@np.linalg.solve(cov[1:,1:], beta[1:]), rtol=2e-7)

    def test_cross_spline_aliases_have_no_false_standard_errors_or_joint_test(self):
        x = np.linspace(0., 3., 150)
        data = pl.DataFrame({'x': x, 'copy': x, 'y': np.exp(.2 + .1*np.sin(x))})
        model = Plan('poisson').spline('x', knots=[0.,1.,2.,3.]).spline('copy', knots=[0.,1.,2.,3.]).fit(data, 'y')
        self.assertEqual(model.inference_summary['n_parameters'], 4)
        tests = term_tests(model).table
        self.assertTrue(all(status != 'available' for status in tests['status']))
        for name in ('x','copy'):
            table = model.rating_tables_by_name()[name]
            self.assertTrue(all(value is None or np.isnan(value) for value in table['Standard_Error'].to_list()[1:]))
        loaded = model.to_workbook().to_model()
        self.assertIsNone(loaded.converged)
        with self.assertRaises(ValueError):
            term_tests(loaded)



if __name__ == '__main__':
    unittest.main()
