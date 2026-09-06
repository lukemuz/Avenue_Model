import math
import tempfile
import unittest

import numpy as np
import polars as pl
from sklearn.linear_model import PoissonRegressor

from avenue_model import Plan, Workbook


class HierarchicalInteractionTests(unittest.TestCase):
    def data(self):
        rng = np.random.default_rng(83)
        a = np.repeat(['A', 'B'], 300)
        b = np.tile(np.repeat(['X', 'Y', 'Z'], 100), 2)
        weight = rng.uniform(.3, 2., len(a))
        mean = np.exp(.2 + .3 * (a == 'B') + .2 * (b == 'Y') - .1 * (b == 'Z')
                      + .6 * ((a == 'B') & (b == 'Y')) - .4 * ((a == 'B') & (b == 'Z')))
        y = rng.poisson(mean * weight) / weight
        return pl.DataFrame({'a': a, 'b': b, 'w': weight, 'y': y})

    def test_means_and_contrasts_match_treatment_coded_reference(self):
        data = self.data()
        plan = (Plan.frequency('w').categorical('a', base='first').categorical('b', base='first')
                .interaction(['a', 'b'], [None, None]))
        check = plan.check(data, 'y')
        self.assertTrue(check.is_fittable, check.issues)
        model = plan.fit(data, 'y')
        self.assertTrue(model.converged, model.report().fit_summary)
        a = data['a'].to_numpy() == 'B'
        y = data['b'].to_numpy() == 'Y'
        z = data['b'].to_numpy() == 'Z'
        design = np.column_stack([a, y, z, a & y, a & z]).astype(float)
        reference = PoissonRegressor(alpha=0., tol=1e-11, max_iter=1000).fit(
            design, data['y'].to_numpy(), sample_weight=data['w'].to_numpy())
        np.testing.assert_allclose(model.predict(data).to_series(), reference.predict(design), atol=1e-7, rtol=1e-7)
        info = next(term for term in model.resolved if term['kind'] == 'interaction_contrast')
        self.assertEqual(info['parameters'], 2)
        table = model.rating_tables_by_name()[info['name']]
        interior = table.filter((pl.col('a_Level') == 'B') & (pl.col('b_Level') != 'X'))
        np.testing.assert_allclose(interior['Coefficient'].to_numpy(), reference.coef_[-2:], atol=1e-7, rtol=1e-7)
        full_design = np.column_stack([np.ones(data.height), design])
        information = full_design.T @ ((data['w'].to_numpy() * reference.predict(design))[:, None] * full_design)
        reference_se = np.sqrt(np.diag(np.linalg.inv(information)))
        np.testing.assert_allclose(interior['Standard_Error'].to_numpy(), reference_se[-2:], atol=1e-7, rtol=1e-7)

        boundary = table.filter((pl.col('a_Level') == 'A') | (pl.col('b_Level') == 'X'))
        self.assertEqual(boundary['Coefficient'].to_list(), [0.] * 4)
        with tempfile.TemporaryDirectory() as path:
            model.to_workbook().save_csv_dir(path)
            loaded = Workbook.load_csv_dir(path).to_model()
            np.testing.assert_allclose(loaded.predict(data).to_series(), reference.predict(design), atol=1e-7, rtol=1e-7)

    def test_reference_constraints_do_not_depend_on_term_order(self):
        data = self.data().with_columns((pl.col('w') * pl.when(pl.col('a') == 'B').then(10.).otherwise(1.)).alias('w'))
        plan = (Plan.frequency('w').interaction(['a', 'b'], [None, None])
                .categorical('a').categorical('b'))
        model = plan.fit(data, 'y')
        self.assertTrue(model.converged)
        self.assertEqual(model.rating_tables_by_name()['a']['a_Level'][0], 'B')
        info = next(term for term in model.resolved if term['kind'] == 'interaction_contrast')
        table = model.rating_tables_by_name()[info['name']]
        self.assertEqual(table.filter(pl.col('a_Level') == 'B')['Coefficient'].to_list(), [0.] * 3)
        means = data.group_by(['a', 'b']).agg(((pl.col('y') * pl.col('w')).sum() / pl.col('w').sum()).alias('expected'))
        joined = data.with_columns(model.predict(data)['predictions']).join(means, on=['a', 'b'])
        np.testing.assert_allclose(joined['predictions'], joined['expected'], atol=1e-7, rtol=1e-7)

    def test_banded_main_effect_and_interaction_share_edges(self):
        data = self.data().with_columns(pl.when(pl.col('a') == 'A').then(20.).otherwise(50.).alias('age'))
        plan = (Plan.frequency('w').banded('age', breaks=[30.]).categorical('b')
                .interaction(['age', 'b'], [[30.], None]))
        model = plan.fit(data, 'y')
        self.assertTrue(model.converged)
        info = next(term for term in model.resolved if term['kind'] == 'interaction_contrast')
        self.assertEqual(info['parameters'], 2)
        self.assertEqual(model.report().fit_summary['n_parameters'], 6)
