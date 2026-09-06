"""Fixed continuous priors and conditional inference on newly fitted factors."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import polars as pl
from scipy.interpolate import CubicSpline

from avenue_model import Plan, GLMOptions, Workbook, term_tests


class SplinePriorTests(unittest.TestCase):
    def test_prior_curve_is_fixed_and_exposure_is_applied_once(self):
        x = np.linspace(-.5, 3.5, 120)
        # Fit on the boundary range; probe both tails during the update.
        train_x = np.linspace(0., 3., 120)
        cubic = CubicSpline([0., 1., 2., 3.], [0., .3, -.2, .1], bc_type='natural')
        exposure = .2 + (np.arange(120) % 7) / 4
        for role in ('weight', 'offset'):
            seed = pl.DataFrame({'x': train_x, 'w': exposure,
                'y': np.exp(.2 + cubic(train_x)) * (exposure if role == 'offset' else 1.)})
            prior = Plan('poisson', exposure='w', exposure_role=role).spline('x', knots=[0., 1., 2., 3.]).fit(seed, 'y', GLMOptions(tolerance=1e-10))
            data = pl.DataFrame({'x': x, 'w': exposure, 'region': ['a', 'b'] * 60,
                                 'cluster': np.arange(120) % 13})
            base = prior.predict(data).to_numpy().reshape(-1)
            y = base * np.tile([1.1, 1.4], 60) * np.exp(.15*np.sin(np.arange(120)))
            data = data.with_columns(pl.Series('y', y))
            plan = Plan('poisson', exposure='w', exposure_role=role).offset_model(prior, prefix='prior').categorical('region', base='first')
            encoded = json.loads(plan.to_json())
            carried = next(t for t in encoded['terms'] if t['name'] == 'prior.x')
            self.assertEqual(carried['spline'], 'natural_cubic_linear_tails')
            # Integer JSON knot spellings must stay numeric coordinates.
            for row in carried['table']:
                row['x'] = int(row['x'])
            plan = Plan.from_json(json.dumps(encoded))
            weights = exposure if role == 'weight' else np.ones(120)
            group = np.arange(120) % 2
            multipliers = np.array([np.sum(weights[group == g]*y[group == g]) /
                                    np.sum(weights[group == g]*base[group == g]) for g in (0, 1)])
            mu = base * multipliers[group]
            design = np.column_stack([np.ones(120), group])
            bread = np.linalg.inv(design.T @ ((weights*mu)[:, None]*design))
            beta = np.array([np.log(multipliers[0]), np.log(multipliers[1]/multipliers[0])])
            for mode in ('model', 'HC0', 'cluster'):
                options = GLMOptions(tolerance=1e-10, covariance={'model': 'model_based', 'HC0': 'hc0', 'cluster': 'cluster'}[mode],
                                     cluster='cluster' if mode == 'cluster' else None)
                updated = plan.fit(data, 'y', options)
                self.assertTrue(updated.converged)
                np.testing.assert_allclose(updated.predict(data).to_numpy().reshape(-1), mu, rtol=2e-8)
                tables = updated.rating_tables_by_name()
                np.testing.assert_array_equal(tables['prior.x']['Coefficient'].to_numpy(), prior.rating_tables_by_name()['x']['Coefficient'].to_numpy())
                self.assertEqual(tables['prior.x']['Status'].unique().to_list(), ['locked'])
                covariance = bread
                if mode != 'model':
                    scores = (weights*(y-mu))[:, None]*design
                    if mode == 'cluster':
                        scores = np.array([scores[np.arange(120) % 13 == k].sum(axis=0) for k in range(13)])
                    covariance = bread @ (scores.T @ scores) @ bread
                np.testing.assert_allclose([tables['intercept']['Standard_Error'][0], tables['region']['Standard_Error'][1]], np.sqrt(np.diag(covariance)), rtol=2e-7)
                test = term_tests(updated).table.filter(pl.col('term') == 'region')
                np.testing.assert_allclose(test['statistic'][0], beta[1]**2/covariance[1,1], rtol=3e-7)
                with tempfile.TemporaryDirectory() as tmp:
                    updated.to_workbook().save_json(str(Path(tmp)/'model.json'))
                    loaded = Workbook.load_json(str(Path(tmp)/'model.json')).to_model()
                    np.testing.assert_allclose(loaded.predict(data).to_numpy().reshape(-1), mu, rtol=2e-8)
                    self.assertIn('natural_cubic_linear_tails', updated.plan.to_json())


    def test_carried_geometry_is_validated_and_prior_encodings_survive(self):
        data = pl.DataFrame({'x': np.linspace(0., 3., 60), 'region': ['a', 'b'] * 30, 'y': [1., 2.] * 30})
        prior = Plan('poisson').spline('x', knots=[0., 1., 2., 3.]).categorical('region').fit(data, 'y')
        with tempfile.TemporaryDirectory() as tmp:
            prior.to_workbook().save_csv_dir(tmp)
            prior = Workbook.load_csv_dir(tmp).to_model()
            plan = Plan('poisson').offset_model(prior, prefix='prior')
            updated = Plan.from_json(plan.to_json()).fit(data, 'y')
            np.testing.assert_allclose(updated.predict(data).to_numpy(), prior.predict(data).to_numpy(), rtol=1e-7)
            workbook_path = Path(tmp)/'updated.json'
            updated.to_workbook().save_json(str(workbook_path))
            malformed = json.loads(workbook_path.read_text())
            index = next(i for i, term in enumerate(malformed['manifest']['tables']) if term['name'] == 'prior.intercept')
            malformed['tables'][index].append(malformed['tables'][index][0].copy())
            workbook_path.write_text(json.dumps(malformed))
            with self.assertRaises(ValueError):
                Workbook.load_json(str(workbook_path)).to_model()
            bad = json.loads(plan.to_json())
            curve = next(t for t in bad['terms'] if t['name'] == 'prior.x')
            curve['table'][1]['x'] = curve['table'][0]['x']
            with self.assertRaises(ValueError):
                Plan.from_json(json.dumps(bad)).fit(data, 'y')


if __name__ == '__main__':
    unittest.main()
