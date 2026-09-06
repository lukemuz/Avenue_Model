"""Tune/convert the installed stock or fork LightGBM; optionally refit its shapes.

Uses numpy/scikit-learn for checks and study-level comparisons. Input definitions
and final-year holdout match auto_pricing_study.py; no losses or exposures are capped.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
import polars as pl
from sklearn.metrics import mean_poisson_deviance
from sklearn.model_selection import GroupKFold
from avenue_model import GLMOptions, Plan, Workbook, from_booster, resolve_lightgbm, tune_lgbm
from auto_pricing_study import synthetic


def run(output, data_path=None, refit_glm=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    if data_path is None:
        data_path = output / 'synthetic_auto.csv'
        synthetic(data_path)
    data = pl.read_csv(data_path)
    if data['exposure'].null_count() or not data['exposure'].is_finite().all() or (data['exposure'] <= 0).any():
        raise ValueError('This study requires finite positive exposure')
    data = data.with_columns((pl.col('claims') / pl.col('exposure')).alias('frequency'))
    train, holdout = data.filter(pl.col('year') < 2022), data.filter(pl.col('year') >= 2022)
    levels = sorted(train['region'].unique().to_list())
    codes = {name: i for i, name in enumerate(levels)}
    def predictors(frame):
        return frame.select(pl.col('age').cast(pl.Float64),
                            pl.col('region').replace_strict(codes, default=-1).cast(pl.Int32))
    lgb, module_name = resolve_lightgbm()
    dataset = lgb.Dataset(predictors(train).to_numpy(), label=train['frequency'].to_numpy(),
                          weight=train['exposure'].to_numpy(), feature_name=['age', 'region'],
                          categorical_feature=['region'])
    folds = list(GroupKFold(3).split(train, groups=train['policy_id'].to_numpy()))
    result = tune_lgbm(dataset, {'objective': 'poisson', 'num_iterations': 20,
        'num_threads': 1, 'verbosity': -1, 'min_data_in_leaf': 80, 'max_depth': 2,
        'num_leaves': 3, 'seed': 47}, n_trials=3, seed=47, folds=folds,
        tunable=['learning_rate', 'interaction_penalty', 'interaction_complexity'],
        space={'learning_rate': (.05, .2), 'interaction_penalty': (0., .05),
               'interaction_complexity': (0., .1)})
    (output / 'trials.json').write_text(json.dumps([asdict(t) for t in result.trials], indent=2))
    (output / 'tuning.txt').write_text(result.summary())
    selected = result.best_cv
    booster = lgb.train({**selected.params, 'num_iterations': selected.num_iterations}, dataset)
    booster.save_model(str(output / 'booster.txt'))
    converted = from_booster(booster, predictors(holdout))
    if converted.parity['status'] != 'passed':
        raise RuntimeError(converted.parity)
    model = (converted.model.with_categories({'region': levels})
             .with_response('frequency', exposure='exposure', exposure_role='weight'))
    converted.model = model
    converted.save(output / 'converted')
    quotes = holdout.select('age', 'region')
    loaded = Workbook.load_csv_dir(str(output / 'converted')).to_model()
    np.testing.assert_allclose(loaded.predict(quotes).to_series(),
                               booster.predict(predictors(holdout).to_numpy()), atol=1e-12, rtol=1e-12)
    (output / 'booster_review.md').write_text(model.report(holdout).markdown)
    baseline = Plan.frequency('exposure').banded('age', breaks=[25., 50.]).categorical('region').fit(train, 'frequency')
    if not baseline.converged:
        raise RuntimeError('Baseline did not converge')
    models = {'glm': baseline, 'booster': model}
    if refit_glm:
        plan = Plan.frequency('exposure')
        for name, table in model.rating_tables_by_name().items():
            columns = [c for c in table.columns if c in model.input_schema['predictors']]
            if columns:
                plan = plan.given(name, table.select(columns + ['Rating_Factor']))
        # Prespecified penalty; selecting it on the final holdout would leak.
        refit = plan.fit(train, 'frequency', GLMOptions(alpha=1e-4, l1_ratio=0., max_iterations=500))
        if not refit.converged:
            raise RuntimeError('Refit did not converge')
        refit.to_workbook().save_csv_dir(str(output / 'refit'))
        (output / 'refit_review.md').write_text(refit.report(holdout).markdown)
        models['booster_structure_glm'] = refit
    rows = []
    for name, fitted in models.items():
        predictions = fitted.predict(quotes).to_series()
        rows.append({'model': name, 'ae_ratio': holdout['claims'].sum() / (predictions * holdout['exposure']).sum(),
                     'mean_poisson_deviance': mean_poisson_deviance(holdout['frequency'], predictions,
                                                                  sample_weight=holdout['exposure'])})
    comparison = pl.DataFrame(rows)
    comparison.write_csv(output / 'comparison.csv')
    model.explain(quotes.head(10))['contributions'].write_csv(output / 'quote_explanations.csv')
    edited_dir = output / 'edited'
    model.to_workbook().save_csv_dir(str(edited_dir))
    intercept = next(edited_dir.glob('*intercept.csv'))
    pl.read_csv(intercept).with_columns((pl.col('Relativity') * 1.05).alias('Relativity')).write_csv(intercept)
    edited = Workbook.load_csv_dir(str(edited_dir)).to_model()
    np.testing.assert_allclose(edited.predict(quotes).to_series(), model.predict(quotes).to_series() * 1.05, atol=1e-12, rtol=1e-12)
    (output / 'edited_review.md').write_text(edited.report(holdout).markdown)
    print(module_name, lgb.__version__)
    print(comparison)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--data')
    parser.add_argument('--refit-glm', action='store_true')
    args = parser.parse_args()
    run(args.output, args.data, args.refit_glm)
