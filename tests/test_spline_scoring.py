"""Continuous scoring artifacts; fitting remains an explicit integration gate."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import polars as pl
from avenue_model import Plan, Workbook


class SplineScoringTests(unittest.TestCase):
    def artifact(self, directory, values=(0., .4, 0.), *, exposure=False):
        data = pl.DataFrame({'y': [1., 2., 3.], 'w': [1., 1., 1.]})
        seed = (Plan('poisson', exposure='w', exposure_role='offset') if exposure else Plan('poisson')).fit(data, 'y')
        path = Path(directory) / 'spline.json'
        seed.to_workbook(scale='factor').save_json(str(path))
        artifact = json.loads(path.read_text())
        artifact['manifest']['format_version'] = 4
        artifact['manifest']['tables'].append({
            'name': 'curve', 'spline': 'natural_cubic_linear_tails'})
        artifact['tables'][0] = [{'Rating_Factor': .2}]
        artifact['tables'].append([
            {'x': float(x), 'Rating_Factor': value} for x, value in enumerate(values)])
        path.write_text(json.dumps(artifact))
        return Workbook.load_json(str(path)).to_model()

    def test_workbook_and_explanations_preserve_the_curve(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = self.artifact(tmp)
            changed = self.artifact(tmp, (0., .8, 0.))
            data = pl.DataFrame({'x': [-1., 0., .5, 1., 1.5, 2., 3.]})
            curve = np.array([-.6, 0., .275, .4, .275, 0., -.6])
            expected = np.exp(.2 + curve)
            np.testing.assert_allclose(model.predict(data).to_numpy().reshape(-1), expected)
            model.to_workbook().save_json(str(Path(tmp)/'saved.json'))
            loaded = Workbook.load_json(str(Path(tmp)/'saved.json')).to_model()
            np.testing.assert_allclose(loaded.predict(data).to_numpy().reshape(-1), expected)
            before = loaded.explain(data)['contributions'].filter(pl.col('kind') == 'spline')
            after = changed.explain(data)['contributions'].filter(pl.col('kind') == 'spline')
            np.testing.assert_allclose(after['coefficient'] - before['coefficient'], curve)
            self.assertEqual(before['table_row'].null_count(), len(curve))
            np.testing.assert_allclose(changed.predict(data).to_numpy().reshape(-1) - expected,
                                       np.exp(.2+2*curve)-expected)
            self.assertIsNone(loaded.converged)
            self.assertEqual(loaded.rating_tables_by_name()['curve']['Status'].unique().to_list(), ['scoring_only'])

    def test_offsets_invalid_quotes_and_csv_relativity_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = self.artifact(tmp, exposure=True)
            data = pl.DataFrame({'x': [-1., .5, 3.], 'w': [0., 2., .5]})
            rate = np.exp(.2 + np.array([-.6, .275, -.6]))
            np.testing.assert_allclose(model.predict_rate(data.drop('w')).to_numpy().reshape(-1), rate)
            np.testing.assert_allclose(model.predict(data).to_numpy().reshape(-1), rate * data['w'].to_numpy())
            path = str(Path(tmp) / 'csv')
            model.to_workbook(scale='relativity').save_csv_dir(path)
            loaded = Workbook.load_csv_dir(path).to_model()
            np.testing.assert_allclose(loaded.predict(data).to_numpy(), model.predict(data).to_numpy())
            invalid = pl.DataFrame({'x': [.5, None, float('nan'), float('inf')], 'w': [1.] * 4})
            self.assertEqual(loaded.predict_diagnostics(invalid)['predictions'].null_count(), 3)
            with self.assertRaises(ValueError):
                loaded.predict(invalid)


if __name__ == '__main__':
    unittest.main()
