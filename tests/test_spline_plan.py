"""Public natural-cubic Plan fitting, fold resolution and delivery."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import polars as pl
from scipy.interpolate import CubicSpline

from avenue_model import (Plan, GLMOptions, SplitSpec, GLMTrial, select_glm,
                          save_bundle, load_bundle, coefficient_intervals)


class SplinePlanTests(unittest.TestCase):
    def test_public_families_recover_continuous_scipy_curve_and_bundle(self):
        x = np.linspace(0., 3., 120)
        knots = [0., .4, 1.7, 3.]
        curve = CubicSpline(knots, [0., .2, -.3, .15], bc_type='natural')
        eta = .3 + curve(x)
        weights = .2 + np.arange(len(x)) % 7 / 3
        for family in ('gaussian', 'poisson', 'gamma', 'tweedie', 'binomial'):
            y = eta if family == 'gaussian' else 1/(1+np.exp(-eta)) if family == 'binomial' else np.exp(eta)
            data = pl.DataFrame({'x': x, 'y': y, 'w': weights})
            plan = Plan(family, exposure='w', exposure_role='weight').spline('x', knots=knots)
            serialized = plan.to_json()
            self.assertEqual(json.loads(serialized)['terms'][0]['knots'], {'kind': 'explicit', 'values': knots})
            model = Plan.from_json(serialized).fit(data, 'y', GLMOptions(tolerance=1e-10))
            self.assertTrue(model.converged)
            resolved = model.resolved[1]
            self.assertEqual(resolved['kind'], 'natural_cubic_spline')
            self.assertEqual(resolved['knots'], knots)
            self.assertIsNone(resolved['edges'])
            np.testing.assert_allclose(model.predict(data).to_numpy().reshape(-1), y, rtol=2e-7, atol=2e-8)
            self.assertNotIn('no_data', model.rating_tables_by_name()['x']['Status'].to_list())
            self.assertIsNotNone(coefficient_intervals(model))
            with tempfile.TemporaryDirectory() as tmp:
                save_bundle(model, Path(tmp)/'bundle', validation_data=data)
                bundle = load_bundle(Path(tmp)/'bundle')
                self.assertEqual(bundle.source_plan.to_json(), serialized)
                np.testing.assert_allclose(bundle.model.predict(data).to_numpy(), model.predict(data).to_numpy(), rtol=1e-12)
                refit = bundle.source_plan.fit(data, 'y', GLMOptions(tolerance=1e-10))
                np.testing.assert_allclose(refit.predict(data).to_numpy(), model.predict(data).to_numpy(), rtol=1e-12)

    def test_fold_local_knots_ties_zero_weights_and_selection(self):
        # Holdout extremes and zero-weight training extremes must not choose knots.
        x = list(np.linspace(0., 3., 100)) + [40.] + [50., 60., 70.]
        w = [1.] * 100 + [0.] + [1.] * 3
        data = pl.DataFrame({'x': x, 'y': [1.] * len(x), 'w': w,
                             'time': [0] * 101 + [1] * 3})
        fold = SplitSpec.out_of_time('time', 1).split(data)[0]
        for kwargs in ({'quantile': 5}, {'equal_width': 4}, {}):
            plan = Plan('poisson', exposure='w', exposure_role='weight').spline('x', **kwargs)
            model = fold.fit(plan, data, 'y').model
            knots = model.resolved[1]['knots']
            self.assertEqual((knots[0], knots[-1]), (0., 3.))
            np.testing.assert_allclose(model.predict(data).to_numpy(), 1., rtol=1e-9)
        tied = pl.DataFrame({'x': [0., 1., 2.] * 20, 'y': [1., 1., 1.] * 20})
        model = Plan('poisson').spline('x', quantile=8).fit(tied, 'y')
        self.assertEqual(model.resolved[1]['knots'], [0., 1., 2.])
        result = select_glm(data, {'smooth': GLMTrial(Plan('poisson', exposure='w', exposure_role='weight').spline('x', quantile=4))},
                            target='y', unit='rate', metric='poisson', weight='w', split=[fold])
        self.assertEqual(result.recommended, 'smooth')
        self.assertEqual(result.history['n_parameters'].to_list(), [4])

    def test_invalid_geometry_and_predictors_fail_without_silent_repair(self):
        data = pl.DataFrame({'x': np.linspace(0., 3., 30), 'y': [1.] * 30})
        for kwargs in ({'knots': [0., 0., 3.]}, {'knots': [0., 3., 1.]},
                       {'knots': [0., float('inf')]}, {'knots': [0.]},
                       {'quantile': 1}, {'equal_width': 0}):
            with self.assertRaises(ValueError):
                Plan('poisson').spline('x', **kwargs).fit(data, 'y')
        with self.assertRaisesRegex(ValueError, 'only one'):
            Plan('poisson').spline('x', knots=[0., 3.], quantile=4)
        for value in (None, float('nan'), float('inf')):
            bad = data.with_columns(pl.Series('x', [value] + data['x'].to_list()[1:]))
            with self.assertRaisesRegex(ValueError, 'finite'):
                Plan('poisson').spline('x').fit(bad, 'y')
        with self.assertRaises(ValueError):
            Plan('poisson').spline('x').fit(data.with_columns(pl.lit(1.).alias('x')), 'y')
        with self.assertRaises(ValueError):
            Plan('poisson').spline('x').banded('x', breaks=[1.]).fit(data, 'y')


if __name__ == '__main__':
    unittest.main()
