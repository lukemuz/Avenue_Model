import tempfile
import unittest

import polars as pl
from avenue_model import ComposedModel, Plan, frequency_severity, sum_loss_costs


class NamedCompositionTests(unittest.TestCase):
    def setUp(self):
        self.data = pl.DataFrame({'region': ['a', 'b'] * 20, 'exposure': [.5, 1.] * 20,
                                  'count': [1., 4.] * 20, 'rate': [2., 4.] * 20,
                                  'severity': [100., 200.] * 20, 'premium': [200., 800.] * 20})
        self.frequency = Plan.frequency('exposure').categorical('region').fit(self.data, 'rate')
        self.severity = Plan.severity('count').categorical('region').fit(self.data, 'severity')
        self.premium = Plan.pure_premium('exposure').categorical('region').fit(self.data, 'premium')

    def test_product_and_peril_sum_reconcile_and_reload(self):
        product = frequency_severity(self.frequency, self.severity)
        total = sum_loss_costs({'collision': product, 'other': self.premium})
        quotes = self.data.select('region')
        expected = (self.frequency.predict(quotes).to_series() * self.severity.predict(quotes).to_series()
                    + self.premium.predict(quotes).to_series())
        self.assertIsNone(total.family)
        self.assertEqual(total.converged, all(m.converged for m in (self.frequency, self.severity, self.premium)))
        self.assertEqual(total.unit, 'loss_per_exposure')
        self.assertEqual(total.predict_components(quotes).columns, ['collision', 'other'])
        with tempfile.TemporaryDirectory() as directory:
            total.save(directory)
            loaded = ComposedModel.load(directory)
            self.assertIsNone(loaded.converged)  # no fabricated fit evidence after workbook load
            self.assertEqual(list(loaded.components), ['collision', 'other'])
            for artifact in (total, loaded):
                actual = artifact.predict(quotes).to_series()
                for a, e in zip(actual, expected):
                    self.assertAlmostEqual(a, e)
                with self.assertRaises(TypeError):
                    artifact.validate(self.data, target='premium')
                validation = artifact.validate(self.data, target='premium', metric='tweedie', weight='exposure')
                self.assertEqual(validation.metadata['metric'], 'tweedie')

    def test_offset_frequency_is_converted_to_rate_once(self):
        counts = Plan('poisson', exposure='exposure', exposure_role='offset').categorical('region').fit(self.data, 'count')
        model = frequency_severity(counts, self.severity)
        expected = frequency_severity(self.frequency, self.severity).predict(self.data.select('region')).to_series()
        actual = model.predict(self.data.select('region')).to_series()
        for a, e in zip(actual, expected):
            self.assertAlmostEqual(a, e)
        with self.assertRaisesRegex(ValueError, 'not Poisson'):
            sum_loss_costs({'frequency': self.frequency})
        with self.assertRaisesRegex(ValueError, 'units differ'):
            sum_loss_costs({'one': model}, unit='different_currency')
