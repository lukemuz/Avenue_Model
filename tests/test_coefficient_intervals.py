import tempfile
import unittest
from statistics import NormalDist

import numpy as np
import polars as pl
from avenue_model import GLMOptions, Plan, Workbook, coefficient_intervals


class CoefficientIntervalTests(unittest.TestCase):
    def test_quasi_poisson_matches_independent_information_and_pearson(self):
        rng = np.random.default_rng(94)
        group = np.tile(['a', 'b'], 300)
        exposure = rng.uniform(.2, 2., len(group))
        mean = np.exp(.4 + .6 * (group == 'b')) * exposure
        counts = rng.poisson(mean * rng.gamma(shape=.5, scale=2., size=len(group))).astype(float)
        data = pl.DataFrame({'group': group, 'exposure': exposure, 'counts': counts,
                             'rate': counts / exposure})
        models = [Plan.frequency('exposure').categorical('group', base='first').fit(data, 'rate'),
                  Plan('poisson', exposure='exposure', exposure_role='offset').categorical('group', base='first').fit(data, 'counts')]
        for model in models:
            before = model.predict(data).to_numpy()
            intervals = coefficient_intervals(model, dispersion='quasi_poisson')
            ordinary = coefficient_intervals(model)
            mu = model.predict_count(data).to_series().to_numpy()
            design = np.column_stack([np.ones(len(group)), (group == 'b').astype(float)])
            pearson = np.sum((counts - mu) ** 2 / mu)
            scale = pearson / (len(group) - 2)
            covariance = scale * np.linalg.inv(design.T @ (mu[:, None] * design))
            self.assertGreater(scale, 2.)
            self.assertAlmostEqual(intervals.metadata['dispersion'], scale, places=8)
            table = intervals.tables['group']
            expected_se = np.sqrt(covariance[1, 1])
            self.assertAlmostEqual(table['Interval_Standard_Error'][1], expected_se, places=8)
            self.assertEqual(table['Interval_Status'][0], 'fixed')
            lower = table['Coefficient'][1] - NormalDist().inv_cdf(.975) * expected_se
            self.assertAlmostEqual(table['Coefficient_Lower'][1], lower, places=8)
            self.assertAlmostEqual(table['Relativity_Lower'][1], np.exp(lower), places=8)
            self.assertGreater(expected_se, ordinary.tables['group']['Interval_Standard_Error'][1])
            np.testing.assert_array_equal(before, model.predict(data).to_numpy())

    def test_unavailable_evidence_is_not_an_interval(self):
        data = pl.DataFrame({'group': ['a', 'b'] * 40, 'y': [1., 100.] * 40})
        plan = Plan('poisson').categorical('group')
        nonconverged = plan.fit(data, 'y', GLMOptions(max_iterations=1, tolerance=1e-14))
        with self.assertRaisesRegex(ValueError, 'converged'):
            coefficient_intervals(nonconverged)
        penalized = plan.fit(data, 'y', GLMOptions(alpha=.1))
        with self.assertRaises(ValueError):
            coefficient_intervals(penalized)
        no_inference = plan.fit(data, 'y', GLMOptions(compute_standard_errors=False))
        with self.assertRaisesRegex(ValueError, 'not computed'):
            coefficient_intervals(no_inference)
        model = plan.fit(data, 'y')
        with tempfile.TemporaryDirectory() as path:
            model.to_workbook().save_csv_dir(path)
            loaded = Workbook.load_csv_dir(path).to_model()
            self.assertEqual(loaded.inference_summary, {})
            with self.assertRaisesRegex(ValueError, 'original'):
                coefficient_intervals(loaded)
        for confidence in (0, 1, float('nan')):
            with self.assertRaisesRegex(ValueError, 'confidence'):
                coefficient_intervals(model, confidence=confidence)

    def test_gaussian_intervals_match_independent_ols(self):
        data = pl.DataFrame({'g': ['a', 'a', 'a', 'b', 'b', 'b'], 'y': [1., 2., 3., 2., 4., 6.]})
        model = Plan('gaussian').categorical('g', base='first').fit(data, 'y')
        result = coefficient_intervals(model, confidence=.9)
        # Residual SSE = 2 + 8, n-p = 4; variance of difference of two means = phi*(1/3+1/3).
        expected_se = np.sqrt(2.5 * 2 / 3)
        self.assertAlmostEqual(result.tables['g']['Interval_Standard_Error'][1], expected_se, places=8)
        self.assertAlmostEqual(result.tables['g']['Coefficient_Upper'][1],
                               2 + NormalDist().inv_cdf(.95) * expected_se, places=8)
        self.assertNotIn('Relativity_Upper', result.tables['g'].columns)
        with self.assertRaisesRegex(ValueError, 'Poisson'):
            coefficient_intervals(model, dispersion='quasi_poisson')
