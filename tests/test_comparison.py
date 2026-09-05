import unittest

import polars as pl
from avenue_model import Candidate, Plan, compare_models


class ComparisonTests(unittest.TestCase):
    def test_nearly_constant_targets_have_stable_normalized_discrimination(self):
        for epsilon in [1e-12, 1e-14, 1e-15]:
            y = [1., 1. + epsilon, 1. + 2 * epsilon]
            result = compare_models(pl.DataFrame({'y': y}),
                                    {'oracle': Candidate(y, 'mean', True),
                                     'reverse': Candidate(y[::-1], 'mean', True),
                                     'constant': Candidate([1.] * 3, 'mean', True)},
                                    target='y', unit='mean', metric='squared_error')
            pairwise = sum(abs(a - b) for a in y for b in y) / (2 * len(y) * sum(y))
            self.assertAlmostEqual(result.summary['gini'][0] / pairwise, 1.)
            for actual, expected in zip(result.summary['normalized_gini'], [1., -1., 0.]):
                self.assertAlmostEqual(actual, expected, places=14)

    def test_discrimination_matches_pairwise_reference_and_reconciles(self):
        y, w = [0., 1., 4., 8., 100.], [.5, 2., 1., 3., 0.]
        data = pl.DataFrame({'y': y, 'w': w})
        vectors = {'tied': [1., 1., 3., 2., 99.], 'oracle': y,
                   'reverse': [-v for v in y], 'constant': [1.] * len(y)}
        result = compare_models(data, {k: Candidate(v, 'rate', True) for k, v in vectors.items()},
                                target='y', weight='w', unit='rate', metric='squared_error')
        def reference(scores):
            return sum(w[i] * w[j] * (y[i] - y[j]) * ((scores[i] > scores[j]) - (scores[i] < scores[j]))
                       for i in range(len(y)) for j in range(len(y))) / (2 * sum(w) * sum(a*b for a,b in zip(y,w)))
        for row in result.summary.to_dicts():
            name = row['candidate']
            self.assertAlmostEqual(row['gini'], reference(vectors[name]))
            self.assertAlmostEqual(row['normalized_gini'], reference(vectors[name]) / reference(y))
            curve = result.discrimination[name]
            self.assertEqual(curve['rows'].sum(), 4)
            self.assertAlmostEqual(curve['weight'].sum(), row['weight'])
            self.assertAlmostEqual(curve['actual'].sum(), row['actual'])
            self.assertEqual(curve['weight_share'][0], 0.)
            self.assertEqual(curve['actual_share'][-1], 1.)
        self.assertEqual(result.discrimination['tied'].height, 4)  # origin + three supported scores
        self.assertEqual(result.recommended, 'oracle')

    def test_discrimination_ties_are_order_invariant_and_scaling_does_not_change_ranks(self):
        data = pl.DataFrame({'y': [0., 5., 2., 8.], 'w': [1., 2., 3., 1.], 'score': [1., 1., 3., 4.]})
        results = []
        for frame in [data, data.reverse()]:
            results.append(compare_models(frame, {'base': Candidate(frame['score'], 'rate', True),
                                                  'scaled': Candidate(frame['score'] * 10., 'rate', True)},
                                          target='y', weight='w', unit='rate', metric='poisson'))
        for result in results:
            self.assertAlmostEqual(result.summary['gini'][0], results[0].summary['gini'][0])
            self.assertAlmostEqual(result.summary['normalized_gini'][0], result.summary['normalized_gini'][1])
            self.assertNotEqual(result.summary['ae_ratio'][0], result.summary['ae_ratio'][1])

    def test_undefined_discrimination_is_explicit_and_does_not_disable_loss_comparison(self):
        for y, status in [([0., 0.], 'zero_actual'), ([-1., 2.], 'negative_target'), ([2., 2.], 'constant_target')]:
            result = compare_models(pl.DataFrame({'y': y}),
                                    {'valid': Candidate([1., 3.], 'mean', True), 'failed': Candidate([None, 1.], 'mean')},
                                    target='y', unit='mean', metric='squared_error')
            row = result.summary.row(0, named=True)
            self.assertEqual(row['discrimination_status'], status)
            self.assertIsNone(row['normalized_gini'])
            self.assertEqual(row['gini'], 0. if status == 'constant_target' else None)
            self.assertEqual(result.recommended, 'valid')
            self.assertEqual(result.summary['discrimination_status'][1], 'scoring_failed')
            self.assertNotIn('failed', result.discrimination)

    def setUp(self):
        self.data = pl.DataFrame({'y': [0., 1., 2., 5.], 'w': [.5, 1., 2., 1.],
                                  'region': ['a', 'a', 'b', 'b'], 'policy': [1, 1, 2, 2]})

    def compare(self, candidates, **kwargs):
        return compare_models(self.data, candidates, target='y', unit='rate',
                              metric='poisson', weight='w', **kwargs)

    def test_independent_weighted_deviance_and_reconciling_exhibits(self):
        from sklearn.metrics import mean_poisson_deviance
        prediction = [1., 1., 2., 4.]
        result = self.compare({'external': Candidate(prediction, 'rate', converged=True)}, segments=['region'])
        summary = result.summary.row(0, named=True)
        expected = mean_poisson_deviance(self.data['y'], prediction, sample_weight=self.data['w'])
        self.assertAlmostEqual(summary['mean_loss'], expected)
        self.assertEqual(result.recommended, 'external')
        self.assertEqual(result.metadata['excluded_rows'], 0)
        exhibit = result.segments['region']
        for column in ('actual', 'expected', 'weight'):
            self.assertAlmostEqual(exhibit[column].sum(), summary[column])
        self.assertEqual(result.predictions['row'].to_list(), [0, 1, 2, 3])

    def test_fitting_status_cannot_be_overridden_by_better_loss(self):
        class Nonconverged:
            converged = False
            def predict(self, data):
                return pl.DataFrame({'predictions': [.01, 1., 2., 5.]})
        result = self.compare({'failed_fit': Candidate(Nonconverged(), 'rate', converged=True),
                               'unknown': Candidate([.1, 1., 2., 5.], 'rate'),
                               'reviewed': Candidate([1.] * 4, 'rate', converged=True)})
        self.assertEqual(result.recommended, 'reviewed')
        self.assertFalse(result.summary.filter(pl.col('candidate') == 'failed_fit')['eligible'][0])

    def test_candidate_failures_are_retained_without_changing_population(self):
        result = self.compare({'good': Candidate([1.] * 4, 'rate', True),
                               'missing': Candidate([1., None, 2., 3.], 'rate'),
                               'short': Candidate([1.], 'rate')})
        self.assertEqual(result.summary['status'].to_list(), ['scored', 'failed', 'failed'])
        self.assertEqual(result.predictions.height, 4)
        self.assertEqual(result.predictions['missing'].null_count(), 4)
        self.assertEqual(result.summary['rows'].to_list(), [4, 4, 4])
        with self.assertRaisesRegex(ValueError, 'common unit'):
            self.compare({'wrong': Candidate([1.] * 4, 'count')})

    def test_paired_bootstrap_is_reproducible_and_identical_predictions_have_zero_difference(self):
        candidates = {'base': Candidate([1.] * 4, 'rate', True),
                      'identical': Candidate([1.] * 4, 'rate', True)}
        a = self.compare(candidates, bootstrap=100, seed=19, bootstrap_group='policy')
        b = self.compare(candidates, bootstrap=100, seed=19, bootstrap_group='policy')
        self.assertEqual(a.summary.to_dicts(), b.summary.to_dicts())
        self.assertEqual(a.summary['loss_difference_lower'].to_list(), [0., 0.])
        self.assertEqual(a.summary['loss_difference_upper'].to_list(), [0., 0.])

    def test_gamma_and_tweedie_use_independent_common_loss(self):
        from sklearn.metrics import mean_gamma_deviance, mean_tweedie_deviance
        data = self.data.with_columns((pl.col('y') + 1).alias('y'))
        mu = [1., 2., 4., 4.]
        for metric, power in [('gamma', 2.), ('tweedie', 1.4)]:
            result = compare_models(data, {'external': Candidate(mu, 'loss_per_exposure', True)},
                                    target='y', weight='w', unit='loss_per_exposure', metric=metric,
                                    tweedie_power=power)
            reference = (mean_gamma_deviance(data['y'], mu, sample_weight=data['w']) if metric == 'gamma'
                         else mean_tweedie_deviance(data['y'], mu, sample_weight=data['w'], power=power))
            self.assertAlmostEqual(result.summary['mean_loss'][0], reference)

    def test_avenue_and_external_predictions_share_review(self):
        data = self.data.with_columns(pl.lit(1.).alias('exposure'))
        fitted = Plan.frequency('exposure').fit(data, 'y')
        result = compare_models(data, {'avenue': Candidate(fitted, 'rate'),
                                      'external': Candidate(fitted.predict(data).to_series(), 'rate', True)},
                                target='y', unit='rate', metric='poisson')
        self.assertEqual(result.summary['mean_loss'][0], result.summary['mean_loss'][1])
        self.assertEqual(result.recommended, 'avenue')
