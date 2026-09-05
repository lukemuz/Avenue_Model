import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
import polars as pl
from scipy.integrate import quad

from avenue_model import (Plan, CredibilityResult, poisson_credibility,
                          coefficient_intervals, frequency_severity)


class CredibilityTests(unittest.TestCase):
    def test_risk_adjusted_baseline_encodings_and_new_quote_exposures_survive(self):
        seed = pl.DataFrame({'risk': ['low', 'high'] * 5, 'exposure': [1., 2.] * 5, 'rate': [1., 2.] * 5})
        baseline = Plan.frequency('exposure').categorical('risk').fit(seed, 'rate')
        data = pl.DataFrame({'territory': ['a', 'a', 'b'], 'risk': ['low', 'high', 'high'],
                             'exposure': [1., 3., 2.], 'claims': [0, 7, 8]})
        result = poisson_credibility(data, baseline, group='territory', counts='claims', prior_strength=2.)
        np.testing.assert_allclose(result.posterior['expected'], [7., 4.], rtol=1e-8)
        quotes = pl.DataFrame({'territory': ['a', 'b'], 'risk': ['high', 'low'], 'exposure': [2., 3.]})
        np.testing.assert_allclose(result.model.predict(quotes).to_series(), [4., 5.], rtol=1e-8)
        np.testing.assert_allclose(result.model.predict_rate(quotes.drop('exposure')).to_series(), [2., 5./3.], rtol=1e-8)

    def baseline(self, role='weight'):
        data = pl.DataFrame({'exposure': [1., 2.], 'y': [1., 1. if role == 'weight' else 2.]})
        return Plan('poisson', exposure='exposure', exposure_role=role).fit(data, 'y')

    def test_posterior_matches_likelihood_quadrature_and_exposure_is_applied_once(self):
        data = pl.DataFrame({'territory': ['001', '001', 'thin', 'empty'],
                             'exposure': [1., 2., .5, 0.], 'claims': [1, 2, 0, 0]})
        for role in ['weight', 'offset']:
            result = poisson_credibility(data, self.baseline(role), group='territory', counts='claims', prior_strength=2.)
            np.testing.assert_allclose(result.model.predict(data).to_series(), [1., 2., .4, 0.])
            np.testing.assert_allclose(result.model.predict_rate(data.drop('exposure')).to_series(), [1., 1., .8, 1.])
            for row in result.posterior.to_dicts():
                # Direct prior times Poisson likelihood kernel, independently
                # normalized by quadrature, rather than another Gamma formula.
                k, e = row['counts'], row['expected']
                def kernel(theta):
                    return theta * math.exp(-2*theta) * theta**k * math.exp(-e*theta)
                normalizer = quad(kernel, 0, np.inf)[0]
                mean = quad(lambda t: t*kernel(t), 0, np.inf)[0] / normalizer
                second = quad(lambda t: t*t*kernel(t), 0, np.inf)[0] / normalizer
                mass = quad(kernel, row['relativity_lower'], row['relativity_upper'])[0] / normalizer
                self.assertAlmostEqual(row['relativity'], mean, places=9)
                self.assertAlmostEqual(row['posterior_sd'], math.sqrt(second - mean**2), places=9)
                self.assertAlmostEqual(mass, .95, places=8)
            self.assertIsNone(result.model.converged)
            with self.assertRaises(ValueError):
                coefficient_intervals(result.model)
            with self.assertRaises(ValueError):
                result.model.predict_rate(pl.DataFrame({'territory': ['unseen']}))
            severity = Plan('gamma').fit(pl.DataFrame({'loss': [10., 10.]}), 'loss')
            np.testing.assert_allclose(frequency_severity(result.model, severity).predict(data).to_series(), [10., 10., 8., 10.])
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'result'
                result.save(path)
                loaded = CredibilityResult.load(path)
                self.assertEqual(loaded.posterior['group'].to_list(), result.posterior['group'].to_list())
                np.testing.assert_allclose(loaded.model.predict(data).to_numpy(), result.model.predict(data).to_numpy())
                (path / 'posterior.csv').write_text('changed')
                with self.assertRaisesRegex(ValueError, 'changed'):
                    CredibilityResult.load(path)

    def test_invalid_count_likelihood_and_duplicate_group_effect_are_rejected(self):
        baseline = self.baseline()
        for count, exposure in [(1., 0.), (-1., 1.), (.5, 1.), (float('nan'), 1.)]:
            with self.assertRaises(ValueError):
                poisson_credibility(pl.DataFrame({'g': ['a'], 'n': [count], 'exposure': [exposure]}),
                                    baseline, group='g', counts='n', prior_strength=2.)
        data = pl.DataFrame({'g': ['a', 'b'], 'n': [1, 2], 'exposure': [1., 1.]})
        for strength in [0., -1., float('inf')]:
            with self.assertRaises(ValueError):
                poisson_credibility(data, baseline, group='g', counts='n', prior_strength=strength)
        fitted_group = Plan.frequency('exposure').categorical('g').fit(
            pl.concat([data] * 3).with_columns(pl.col('n').cast(pl.Float64)), 'n')
        with self.assertRaisesRegex(ValueError, 'already'):
            poisson_credibility(data, fitted_group, group='g', counts='n', prior_strength=2.)
