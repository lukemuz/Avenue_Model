import random
import tempfile
from pathlib import Path
import unittest

import numpy as np
import polars as pl

from avenue_model import Plan, bootstrap_stability


class BootstrapStabilityTests(unittest.TestCase):
    def test_row_and_cluster_draws_match_independent_weighted_means(self):
        data = pl.DataFrame({'cluster': ['a', 'a', 'b', 'c', 'c', 'c'],
                             'w': [.5, 2., 1., 3., 2., .5], 'y': [0., 1., 4., 2., 8., 1.]}).with_columns(pl.col('cluster').alias(''))
        for group, units in [(None, [[i] for i in range(6)]), ('cluster', [[0, 1], [2], [3, 4, 5]]),
                             ('', [[0, 1], [2], [3, 4, 5]])]:
            result = bootstrap_stability(data, Plan.frequency('w'), pl.DataFrame({'quote': [1, 2]}),
                                         target='y', resamples=30, seed=7, group=group)
            self.assertTrue(result.metadata['bands_available'])
            if group:
                self.assertGreater(result.history['sample_rows'].n_unique(), 1)
            references = []
            for row in result.history.to_dicts():
                rng = random.Random(row['sample_seed'])
                selected = [rng.randrange(len(units)) for _ in units]
                indices = [i for unit in selected for i in units[unit]]
                expected = sum(data['w'][i]*data['y'][i] for i in indices) / sum(data['w'][i] for i in indices)
                self.assertEqual(row['sample_rows'], len(indices))
                np.testing.assert_allclose(result.draws[f"draw_{row['replicate']}"], expected, rtol=1e-9)
                references.append(expected)
            np.testing.assert_allclose(result.summary['stability_lower'], np.quantile(references, .025))
            np.testing.assert_allclose(result.summary['stability_upper'], np.quantile(references, .975))
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'stability'
                result.save(path)
                self.assertEqual(pl.read_parquet(path / 'draws.parquet').to_dicts(), result.draws.to_dicts())

    def test_ridge_refits_reproduce_and_do_not_claim_confidence_coverage(self):
        data = pl.DataFrame({'x': ['a', 'b'] * 30, 'y': [1., 3., 2., 4.] * 15})
        plan = Plan('gaussian').categorical('x', base='first')
        options = {'alpha': .3, 'l1_ratio': 0., 'tolerance': 1e-10}
        kwargs = dict(target='y', resamples=20, seed=18, options=options)
        a = bootstrap_stability(data, plan, pl.DataFrame({'x': ['a', 'b']}), **kwargs)
        b = bootstrap_stability(data, plan, pl.DataFrame({'x': ['a', 'b']}), **kwargs)
        self.assertEqual(a.draws.to_dicts(), b.draws.to_dicts())
        self.assertEqual(a.history.to_dicts(), b.history.to_dicts())
        self.assertIn('no guaranteed', a.metadata['interpretation'])
        options['alpha'] = 999
        self.assertEqual(a.metadata['options']['alpha'], .3)
        self.assertFalse(a.metadata['options']['compute_standard_errors'])
        for record in a.history.to_dicts():
            rng = random.Random(record['sample_seed'])
            indices = [rng.randrange(data.height) for _ in range(data.height)]
            x = np.array([float(data['x'][i] == 'b') for i in indices])
            design = np.column_stack([np.ones(len(x)), x])
            beta = np.linalg.solve(design.T @ design/len(x) + np.diag([0., .3]),
                                   design.T @ data['y'].to_numpy()[indices]/len(x))
            np.testing.assert_allclose(a.draws[f"draw_{record['replicate']}"], [beta[0], beta.sum()], rtol=1e-8)

    def test_missing_resampled_level_is_retained_and_withholds_bands(self):
        data = pl.DataFrame({'x': ['a'] * 11 + ['b'], 'y': [1.] * 11 + [2.]})
        result = bootstrap_stability(data, Plan('gaussian').categorical('x'), pl.DataFrame({'x': ['a', 'b']}),
                                     target='y', resamples=30, seed=4)
        self.assertFalse(result.metadata['bands_available'])
        self.assertGreater(result.metadata['successful_resamples'], 0)
        self.assertEqual(result.summary['stability_lower'].null_count(), 2)
        failed = result.history.filter(pl.col('status') == 'failed')
        self.assertGreater(failed.height, 0)
        self.assertEqual(failed['error'].null_count(), 0)
        for replicate in failed['replicate']:
            self.assertEqual(result.draws[f'draw_{replicate}'].null_count(), 2)
