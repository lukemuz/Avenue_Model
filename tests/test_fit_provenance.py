"""Fitting configuration and row provenance survive review without becoming new fits."""
import tempfile
import unittest

import numpy as np
import polars as pl
from avenue_model import GLMOptions, Plan, coefficient_intervals, save_bundle


class FitProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.data = pl.DataFrame({'g': ['a', 'b'] * 40, 'y': [1., 4., 2., 8.] * 20,
                                  'w': [1.] * 80, 'cluster': list(range(80))})

    def test_capture_effective_options_and_reproduce_cluster_fit(self):
        plan = Plan('tweedie', exposure='w', exposure_role='weight', tweedie_power=1.7).categorical('g')
        model = plan.fit(self.data, 'y', GLMOptions(max_iterations=250, tolerance=1e-8,
                          covariance='cluster', cluster='cluster', tweedie_power=1.2))
        self.assertTrue(model.converged)
        options = model.fit_options
        self.assertEqual(options['tweedie_power'], 1.7)  # Plan owns fitting power.
        self.assertEqual(options['max_iterations'], 250)
        self.assertEqual(options['covariance'], 'cluster')
        self.assertEqual(options['cluster'], 'cluster')
        self.assertEqual(options['solver'], 'auto')
        self.assertEqual(model.solver_used, 'global')
        reproduced = model.plan.fit(self.data, 'y', GLMOptions(**options))
        np.testing.assert_array_equal(model.predict(self.data).to_numpy(), reproduced.predict(self.data).to_numpy())
        with tempfile.TemporaryDirectory() as path:
            bundle = save_bundle(model, path + '/bundle', fit_options={'max_iterations': 999})
            self.assertEqual(bundle.source_evidence['effective_fit_options'], options)
            self.assertEqual(bundle.source_evidence['solver_used'], 'global')
            self.assertEqual(bundle.source_evidence['caller_context']['fit_options'], {'max_iterations': 999})
            self.assertEqual(bundle.model.fit_options, {})
            self.assertIsNone(bundle.model.solver_used)

    def test_unanchored_intervals_have_an_explanation_and_preserve_means(self):
        plan = Plan('poisson').categorical('g')
        model = plan.fit(self.data, 'y', GLMOptions(normalization='none'))
        self.assertTrue(model.converged)
        self.assertEqual(model.solver_used, 'table')
        self.assertIn("normalization='none'", model.inference_summary['standard_errors_note'])
        with self.assertRaisesRegex(ValueError, 'not identified'):
            coefficient_intervals(model)
        ordinary = plan.fit(self.data, 'y')
        np.testing.assert_allclose(model.predict(self.data).to_numpy(), ordinary.predict(self.data).to_numpy(), atol=1e-8)

    def test_loaded_and_locked_factors_are_not_labeled_estimated(self):
        original = Plan('poisson').categorical('g').fit(self.data, 'y')
        loaded = original.to_workbook().to_model()
        for table in loaded.rating_tables_by_name().values():
            self.assertEqual(table['Status'].unique().to_list(), ['scoring_only'])
        updated = Plan('poisson').offset_model(loaded, prefix='prior').fit(self.data, 'y')
        self.assertTrue(updated.converged)
        for name, table in updated.rating_tables_by_name().items():
            if name.startswith('prior.'):
                self.assertEqual(table['Status'].unique().to_list(), ['locked'])
                self.assertTrue(table['Standard_Error'].is_nan().all())
        np.testing.assert_allclose(updated.predict(self.data).to_numpy(), original.predict(self.data).to_numpy(), atol=1e-8)
