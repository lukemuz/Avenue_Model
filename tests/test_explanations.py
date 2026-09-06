import math
import unittest

import polars as pl
from avenue_model import Plan


class ExplanationTests(unittest.TestCase):
    def data(self):
        return pl.DataFrame({'region': ['a', 'b', 'a', 'b'] * 10,
                             'age': [20., 20., 40., 40.] * 10,
                             'exposure': [.5, 1., .5, 1.] * 10,
                             'count': [1., 4., 2., 8.] * 10})

    def test_factors_reconstruct_counts_rates_and_reloaded_predictions(self):
        data = self.data().with_columns((pl.col('count') / pl.col('exposure')).alias('rate'))
        for role, target in (('offset', 'count'), ('weight', 'rate')):
            model = (Plan('poisson', exposure='exposure', exposure_role=role)
                     .categorical('region').banded('age', breaks=[30.]).fit(data, target))
            for artifact in (model, model.to_workbook().to_model()):
                details = artifact.explain(data)
                summary, parts = details['summary'], details['contributions']
                self.assertEqual(summary['row'].to_list(), list(range(data.height)))
                self.assertEqual(set(parts['term']), {'intercept', 'region', 'age'} |
                                 ({'exposure'} if role == 'offset' else set()))
                for row in range(data.height):
                    terms = parts.filter(pl.col('row') == row)
                    reconstructed = math.exp(terms['coefficient'].sum())
                    self.assertAlmostEqual(reconstructed, summary['predictions'][row])
                    self.assertAlmostEqual(math.prod(terms['multiplier']), summary['predictions'][row])
                    for name in ('region', 'age'):
                        index = terms.filter(pl.col('term') == name)['table_row'][0]
                        factor = artifact.rating_tables_by_name()[name]['Coefficient'][index]
                        self.assertAlmostEqual(factor, terms.filter(pl.col('term') == name)['coefficient'][0])
                quotes = data.select('region', 'age')
                if role == 'weight':
                    self.assertEqual(artifact.explain(quotes)['summary']['predictions'].to_list(),
                                     artifact.predict(quotes)['predictions'].to_list())

    def test_zero_exposure_and_unmatched_rows_keep_scoring_contract(self):
        data = self.data()
        model = Plan('poisson', exposure='exposure', exposure_role='offset').categorical('region').fit(data, 'count')
        quote = pl.DataFrame({'region': ['a'], 'exposure': [0.]})
        explanation = model.explain(quote)
        self.assertEqual(explanation['summary']['predictions'][0], 0.)
        offset = explanation['contributions'].filter(pl.col('kind') == 'exposure')
        self.assertEqual(offset['coefficient'][0], -math.inf)
        self.assertEqual(offset['multiplier'][0], 0.)
        with self.assertRaisesRegex(ValueError, 'unmatched'):
            model.explain(quote.with_columns(pl.lit('unknown').alias('region')))

    def test_identity_link_exposes_additive_effects(self):
        data = self.data()
        model = Plan('gaussian').categorical('region').fit(data, 'count')
        explanation = model.explain(data)
        self.assertEqual(explanation['contributions']['multiplier'].null_count(),
                         explanation['contributions'].height)
        for row in range(data.height):
            parts = explanation['contributions'].filter(pl.col('row') == row)
            self.assertAlmostEqual(parts['coefficient'].sum(), explanation['summary']['predictions'][row])

    def test_known_workbook_edit_preserves_explanation_arithmetic(self):
        import tempfile
        from pathlib import Path
        from avenue_model import Workbook
        data = self.data()
        original = Plan.frequency('exposure').categorical('region').fit(data, 'count')
        with tempfile.TemporaryDirectory() as directory:
            original.to_workbook().save_csv_dir(directory)
            path = next(Path(directory).glob('*region.csv'))
            table = pl.read_csv(path)
            table.with_columns((pl.col('Relativity') * 1.1).alias('Relativity')).write_csv(path)
            edited = Workbook.load_csv_dir(directory).to_model()
            before, after = original.predict(data).to_series(), edited.predict(data).to_series()
            for a, b in zip(before, after):
                self.assertAlmostEqual(b, a * 1.1)
            old = original.explain(data)['contributions'].filter(pl.col('term') == 'region')
            new = edited.explain(data)['contributions'].filter(pl.col('term') == 'region')
            for delta in new['coefficient'] - old['coefficient']:
                self.assertAlmostEqual(delta, math.log(1.1))
