import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import polars as pl
from avenue_model import Plan, SplitSpec, load_bundle, save_bundle


class BundleTests(unittest.TestCase):
    def setUp(self):
        self.data = pl.DataFrame({'region': ['a', 'b'] * 20, 'w': [1.] * 40,
                                  'y': [1., 2., 2., 4.] * 10})
        self.model = Plan.frequency('w').categorical('region').fit(self.data, 'y')

    def test_evidence_roundtrip_and_edit_isolation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'bundle'
            bundle = save_bundle(self.model, path, training_id='synthetic-v1',
                                 validation_data=self.data, validation_id='diagnostic-only',
                                 fit_options={'solver': 'auto'}, unit='claims/exposure')
            self.assertFalse(bundle.edited)
            self.assertTrue(bundle.source_evidence['fit_summary']['converged'])
            self.assertTrue(bundle.source_evidence['validation_recorded'])
            self.assertIsNotNone(bundle.source_plan)
            self.assertEqual(bundle.source_evidence['input_schema'], json.loads(json.dumps(self.model.input_schema)))
            self.assertIsNone(bundle.model.converged)
            np.testing.assert_allclose(bundle.model.predict(self.data).to_numpy(), self.model.predict(self.data).to_numpy())
            csv = next((path / 'scoring').glob('*.csv'))
            frame = pl.read_csv(csv)
            self.assertIn('Rating_Factor', frame.columns)
            frame.with_columns((pl.col('Rating_Factor') + .1).alias('Rating_Factor')).write_csv(csv)
            edited = load_bundle(path)
            self.assertTrue(edited.edited)
            self.assertEqual(edited.changed_files, (csv.name,))
            self.assertEqual(edited.model.report().fit_summary, {})
            np.testing.assert_allclose(edited.source_model.predict(self.data).to_numpy(), self.model.predict(self.data).to_numpy())
            self.assertFalse(np.allclose(edited.model.predict(self.data).to_numpy(), self.model.predict(self.data).to_numpy()))
            with self.assertRaises(FileExistsError):
                save_bundle(self.model, path)

    def test_fold_and_plan_are_reusable(self):
        fold = SplitSpec.random(n_splits=2, seed=19).split(self.data)[0]
        fitted = fold.fit(Plan.frequency('w').categorical('region'), self.data, 'y')
        _, holdout = fold.frames(self.data)
        with tempfile.TemporaryDirectory() as tmp:
            bundle = save_bundle(fitted.model, Path(tmp) / 'bundle', fold=fold,
                                 validation_data=holdout)
            self.assertEqual(bundle.source_evidence['split_id'], fold.split_id)
            self.assertEqual(bundle.source_evidence['fold'], json.loads(fold.to_json()))
            self.assertEqual(bundle.source_evidence['validation']['n_rows'], holdout.height)
            refit = fold.fit(bundle.source_plan, self.data, 'y').model
            np.testing.assert_allclose(refit.predict(holdout).to_numpy(), bundle.model.predict(holdout).to_numpy())

    def test_integrity_version_and_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'bundle'
            with self.assertRaises(TypeError):
                save_bundle(self.model, path, preprocessing=lambda x: x)
            self.assertFalse(path.exists())
            save_bundle(self.model, path)
            evidence = path / 'source' / 'evidence.json'
            evidence.write_text(evidence.read_text() + ' ')
            with self.assertRaisesRegex(ValueError, 'integrity'):
                load_bundle(path)
            manifest = path / 'bundle.json'
            value = json.loads(manifest.read_text())
            value['schema_version'] = 999
            manifest.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, 'version'):
                load_bundle(path)
