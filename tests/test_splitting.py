import datetime
import json
import unittest

import polars as pl
from avenue_model import Fold, Plan, SplitSpec


class SplitTests(unittest.TestCase):
    def setUp(self):
        self.data = pl.DataFrame({'policy': [i // 3 for i in range(60)],
                                  'period': [2020 + i // 20 for i in range(60)],
                                  'age': [20. + i for i in range(60)],
                                  'region': ['A'] * 60, 'exposure': [1.] * 60,
                                  'frequency': [2.] * 60})

    def test_random_folds_cover_each_row_once_and_roundtrip(self):
        spec = SplitSpec.random(n_splits=4, seed=71)
        folds = spec.split(self.data)
        self.assertEqual(folds, spec.split(self.data))
        self.assertEqual(sorted(i for fold in folds for i in fold.validation_rows), list(range(60)))
        self.assertNotEqual(folds, SplitSpec.random(4, seed=72).split(self.data))
        for fold in folds:
            self.assertFalse(set(fold.train_rows) & set(fold.validation_rows))
            self.assertEqual(Fold.from_json(fold.to_json()), fold)
            self.assertEqual(Fold.from_json(fold.to_json()).split_id, fold.split_id)
            train, valid = fold.frames(self.data)
            self.assertEqual(train.height + valid.height, self.data.height)
        with self.assertRaisesRegex(ValueError, 'different data'):
            folds[0].frames(self.data.reverse())

    def test_grouped_folds_do_not_split_policies(self):
        for fold in SplitSpec.grouped('policy', n_splits=4, seed=2).split(self.data):
            train, valid = fold.frames(self.data)
            self.assertFalse(set(train['policy']) & set(valid['policy']))
            self.assertEqual(valid.height, 15)

    def test_time_boundary_and_fold_local_preparation(self):
        # An extreme holdout age and unseen category must never enter fitted terms.
        data = self.data.with_columns(
            pl.when(pl.col('period') == 2022).then(10000.).otherwise(pl.col('age')).alias('age'),
            pl.when(pl.col('period') == 2022).then(pl.lit('holdout_only')).otherwise(pl.col('region')).alias('region'))
        fold, = SplitSpec.out_of_time('period', 2022).split(data)
        train, valid = fold.frames(data)
        self.assertEqual(train['period'].max(), 2021)
        self.assertEqual(valid['period'].min(), 2022)
        plan = Plan.frequency('exposure').banded('age', quantile=3).categorical('region')
        result = fold.fit(plan, data, 'frequency')
        direct = plan.fit(train, 'frequency')
        self.assertEqual(result.model.resolved, direct.resolved)
        self.assertEqual(result.model.input_schema['predictors']['region']['levels'], [('A', 0)])
        age = next(term for term in result.model.resolved if term['name'] == 'age')
        self.assertNotIn(10000., age['edges'])
        self.assertEqual(result.fold, fold)

    def test_validation_uses_only_retained_holdout(self):
        fold, = SplitSpec.out_of_time('period', 2022).split(self.data)
        result = fold.fit(Plan.frequency('exposure'), self.data, 'frequency')
        self.assertAlmostEqual(result.validate(self.data).total_expected, 40.)

    def test_saved_membership_rejects_overlap_and_future_versions(self):
        fold = SplitSpec.random(3).split(self.data)[0]
        payload = json.loads(fold.to_json())
        with self.assertRaisesRegex(ValueError, 'schema version'):
            Fold.from_json(json.dumps({**payload, 'schema_version': 2}))
        with self.assertRaisesRegex(ValueError, 'overlap'):
            Fold.from_json(json.dumps({**payload, 'validation_rows': payload['train_rows']}))
        with self.assertRaisesRegex(ValueError, 'different data'):
            fold.frames(self.data.with_columns((pl.col('age') + 1).alias('age')))

    def test_dates_and_invalid_specs(self):
        data = pl.DataFrame({'date': [datetime.date(2020, 1, 1), datetime.date(2021, 1, 1)]})
        fold, = SplitSpec.out_of_time('date', datetime.date(2021, 1, 1)).split(data)
        self.assertEqual(json.loads(fold.to_json())['specification']['cutoff'], '2021-01-01')
        for spec in (SplitSpec.random(1), SplitSpec.grouped('policy', 21),
                     SplitSpec.out_of_time('period', 2030)):
            with self.assertRaises(ValueError):
                spec.split(self.data)
        with self.assertRaisesRegex(ValueError, 'missing'):
            SplitSpec.grouped('policy').split(self.data.with_columns(pl.lit(None).alias('policy')))
