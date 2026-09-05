import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
import polars as pl
from avenue_model import (ComposedBundle, Plan, frequency_severity, sum_loss_costs,
                          load_bundle, save_bundle)
from avenue_model.bundle import _hashes


class ComposedBundleTests(unittest.TestCase):
    def setUp(self):
        self.data = pl.DataFrame({'region': ['a', 'b'] * 20, 'exposure': [.5, 1.] * 20,
            'count': [1., 4.] * 20, 'rate': [2., 4.] * 20,
            'severity': [100., 200.] * 20, 'premium': [200., 800.] * 20})
        self.frequency = Plan('poisson', exposure='exposure', exposure_role='offset').categorical('region').fit(self.data, 'count')
        self.severity = Plan.severity('count').categorical('region').fit(self.data, 'severity')
        self.other = Plan.pure_premium('exposure').categorical('region').fit(self.data, 'premium')
        self.product = frequency_severity(self.frequency, self.severity)
        self.total = sum_loss_costs({'collision': self.product, 'other': self.other})

    def test_nested_evidence_units_validation_and_edit_isolation(self):
        quotes = self.data.select('region')
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'bundle'
            bundle = save_bundle(self.total, path, training_id='portfolio-v1',
                validation_data=self.data, validation_id='diagnostic-only',
                validation_options={'target': 'premium', 'metric': 'tweedie', 'weight': 'exposure'},
                component_context={'collision': {'component_context': {
                    'frequency': {'training_id': 'frequency-v1', 'validation_data': self.data},
                    'severity': {'training_id': 'claims-v1'}}}})
            self.assertIsInstance(bundle, ComposedBundle)
            self.assertFalse(bundle.edited)
            self.assertIsNone(bundle.model.family)
            self.assertIsNone(bundle.model.converged)
            self.assertEqual(bundle.model.unit, self.total.unit)
            self.assertEqual(list(bundle.components), ['collision', 'other'])
            leaf = bundle.components['collision'].components['frequency']
            self.assertTrue(leaf.source_evidence['fit_summary']['converged'])
            self.assertEqual(leaf.source_evidence['caller_context']['training_id'], 'frequency-v1')
            self.assertTrue(leaf.source_evidence['validation_recorded'])
            self.assertIsNotNone(leaf.source_plan)
            self.assertTrue(bundle.source_evidence['validation_recorded'])
            self.assertNotIn('predictions', bundle.source_evidence['validation'])
            np.testing.assert_allclose(bundle.model.predict(quotes).to_numpy(), self.total.predict(quotes).to_numpy())
            csv = next((path / 'components/component_0/components/component_0/scoring').glob('*.csv'))
            frame = pl.read_csv(csv)
            frame.with_columns((pl.col('Rating_Factor') + math.log(1.1)).alias('Rating_Factor')).write_csv(csv)
            changed = load_bundle(path)
            self.assertTrue(changed.edited)
            self.assertEqual(changed.changed_files, (csv.relative_to(path).as_posix(),))
            np.testing.assert_allclose(changed.source_model.predict(quotes).to_numpy(), self.total.predict(quotes).to_numpy())
            expected = 1.1 * self.product.predict(quotes).to_numpy() + self.other.predict(quotes).to_numpy()
            np.testing.assert_allclose(changed.model.predict(quotes).to_numpy(), expected)
            self.assertEqual(changed.source_evidence, bundle.source_evidence)
            self.assertIsNone(changed.model.converged)

    def test_parent_protects_child_evidence_and_manifests(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'bundle'
            save_bundle(self.product, path)
            child = path / 'components/component_0'
            evidence = child / 'source/evidence.json'
            evidence.write_text(evidence.read_text() + '\n')
            manifest = json.loads((child / 'bundle.json').read_text())
            manifest['source_hashes'] = _hashes(child / 'source')
            (child / 'bundle.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'integrity'):
                load_bundle(path)

    def test_invalid_graph_context_and_failed_child_leave_no_bundle(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'bundle'
            cases = [{'component_context': {'unknown': {}}},
                     {'component_context': {'frequency': {'training_id': object()}}},
                     {'unit': 'different'}, {'validation_data': self.data}]
            for kwargs in cases:
                with self.assertRaises((ValueError, TypeError)):
                    save_bundle(self.product, path, **kwargs)
                self.assertFalse(path.exists())
                self.assertEqual(list(Path(tmp).iterdir()), [])
            self.total.components['loop'] = self.total
            with self.assertRaisesRegex(ValueError, 'cycle'):
                save_bundle(self.total, path)
            self.assertFalse(path.exists())
