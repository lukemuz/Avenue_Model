"""Synthetic frequency challenger: tune the installed stock/fork CPU build and review tables.

Run with a fresh output directory. Input CSV schema matches auto_pricing_study.py.
Claims/exposure is the response, exposure the weight; no capping is performed.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np
import polars as pl
from avenue_model import (Candidate, GLMOptions, Plan, SplitSpec, Workbook, compare_changes,
                          compare_models, from_booster, prepare_pricing,
                          resolve_lightgbm, save_bundle, tune_lgbm)
from auto_pricing_study import synthetic


def run(output, data_path=None, refit_glm=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    if data_path is None:
        data_path = output / 'synthetic_auto.csv'
        synthetic(data_path)
    prepared = prepare_pricing(pl.read_csv(data_path), exposure='exposure', claims='claims',
                               loss='loss', predictors=['age', 'region'])
    prepared.audit.write_csv(output / 'preparation_audit.csv')
    fold, = SplitSpec.out_of_time('year', 2022).split(prepared.frequency)
    train, holdout = fold.frames(prepared.frequency)
    (output / 'final_split.json').write_text(fold.to_json())
    # Learn code identity on training data only, and retain it in the workbook.
    levels = sorted(train['region'].unique().to_list())
    codes = {name: i for i, name in enumerate(levels)}
    def predictors(frame):
        return frame.select(pl.col('age').cast(pl.Float64),
                            pl.col('region').replace_strict(codes, default=-1).cast(pl.Int32))
    lgb, module_name = resolve_lightgbm()
    dataset = lgb.Dataset(predictors(train).to_numpy(), label=train['avenue_frequency'].to_numpy(),
                          weight=train['exposure'].to_numpy(), feature_name=['age', 'region'],
                          categorical_feature=['region'])
    folds = SplitSpec.grouped('policy_id', n_splits=3, seed=47).split(train)
    (output / 'cv_splits.json').write_text(json.dumps([json.loads(f.to_json()) for f in folds]))
    result = tune_lgbm(dataset, {'objective': 'poisson', 'num_iterations': 20,
        'num_threads': 1, 'verbosity': -1, 'min_data_in_leaf': 80, 'max_depth': 2,
        'num_leaves': 3, 'seed': 47}, n_trials=3, seed=47,
        folds=[(list(f.train_rows), list(f.validation_rows)) for f in folds],
        tunable=['learning_rate', 'interaction_penalty', 'interaction_complexity'],
        space={'learning_rate': (.05, .2), 'interaction_penalty': (0., .05),
               'interaction_complexity': (0., .1)},
        scoring_data=predictors(train.head(256)))
    (output / 'trials.json').write_text(json.dumps([asdict(t) for t in result.trials], indent=2))
    (output / 'tuning.txt').write_text(result.summary())
    selected = result.best_cv
    booster = lgb.train({**selected.params, 'num_iterations': selected.num_iterations}, dataset)
    booster.save_model(str(output / 'booster.txt'))
    converted = from_booster(booster, predictors(holdout))
    if converted.parity['status'] != 'passed':
        raise RuntimeError(converted.parity)
    model = (converted.model.with_categories({'region': levels})
             .with_response('avenue_frequency', exposure='exposure', exposure_role='weight'))
    converted.model = model
    converted.save(output / 'converted')
    quotes = holdout.select('age', 'region')
    loaded = Workbook.load_csv_dir(str(output / 'converted')).to_model()
    reference = booster.predict(predictors(holdout).to_numpy())
    np.testing.assert_allclose(loaded.predict(quotes)['predictions'], reference, atol=1e-12, rtol=1e-12)
    model.explain(quotes.head(10))['contributions'].write_csv(output / 'quote_explanations.csv')
    (output / 'booster_review.md').write_text(model.report(holdout).markdown)
    baseline = Plan.frequency('exposure').banded('age', breaks=[25., 50.]).categorical('region').fit(train, 'avenue_frequency')
    if baseline.converged is not True:
        raise RuntimeError('Baseline GLM did not converge')
    candidates = {
        'glm': Candidate(baseline, 'claims/exposure'),
        # The selected schedule completed and conversion/reload parity passed above.
        # This is training evidence, not a GLM score-convergence certificate.
        'booster': Candidate(model, 'claims/exposure', training_status='completed'),
    }
    if refit_glm:
        # Re-estimate supported rows of the converted shapes. The penalty sends
        # unsupported rows to the reference; check/report flags them. Category labels in
        # rating_tables are already decoded for this raw-data Plan.
        plan = Plan.frequency('exposure')
        features = set(model.input_schema['predictors'])
        for name, table in model.rating_tables_by_name().items():
            columns = [column for column in table.columns if column in features]
            if columns:  # Plan supplies a new, freely fitted intercept.
                plan = plan.given(name, table.select(columns + ['Rating_Factor']))
        check = plan.check(train, 'avenue_frequency')
        (output / 'refit_check.json').write_text(json.dumps({
            'is_fittable': check.is_fittable, 'parameters': check.parameters,
            'rows': check.rows, 'issues': check.issues,
            'table_conditioning': check.table_conditioning}, indent=2))
        if not check.is_fittable:
            raise RuntimeError('Booster structure is not fittable; inspect refit_check.json')
        # Fixed before seeing the final holdout. Tuning on this already selected
        # structure would not evaluate the full structure-selection procedure.
        refitted = plan.fit(train, 'avenue_frequency',
                            GLMOptions(alpha=1e-4, l1_ratio=0., max_iterations=500))
        if refitted.converged is not True:
            raise RuntimeError('Booster-structure GLM did not converge')
        (output / 'refit_review.md').write_text(refitted.report(holdout).markdown)
        bundle = save_bundle(refitted, output / 'refit_bundle', fold=fold,
            training_id='booster-training-pre-2022', validation_data=holdout,
            validation_id='booster-holdout-2022',
            lineage={'structure': 'converted', 'booster': 'booster.txt',
                     'method': 'Poisson GLM refit with prespecified ridge alpha=1e-4',
                     'inference_scope': 'No post-selection confidence claim; penalized errors withheld'})
        np.testing.assert_allclose(bundle.model.predict(quotes)['predictions'],
                                   refitted.predict(quotes)['predictions'], atol=1e-12, rtol=1e-12)
        change = compare_changes(model, refitted, holdout, unit='claims/exposure',
                                 weight='exposure', segments=['region'])
        change.totals.write_csv(output / 'refit_change_totals.csv')
        change.segments['region'].write_csv(output / 'refit_change_region.csv')
        refitted.explain(quotes.head(10))['contributions'].write_csv(output / 'refit_explanations.csv')
        candidates['booster_structure_glm'] = Candidate(refitted, 'claims/exposure')
    comparison = compare_models(holdout, candidates,
       target='avenue_frequency', unit='claims/exposure', metric='poisson', weight='exposure',
       segments=['region', 'year'], bootstrap=50, seed=47)
    comparison.summary.write_csv(output / 'comparison.csv')
    booster_status = comparison.summary.filter(pl.col('candidate') == 'booster').row(0, named=True)
    assert booster_status['training_status'] == 'completed' and booster_status['eligible']
    assert booster_status['converged'] is None
    tables = model.to_workbook(scale='factor').tables
    started = time.perf_counter()
    loaded.predict(quotes)
    elapsed = time.perf_counter() - started
    (output / 'complexity.json').write_text(json.dumps({
        'module': module_name, 'version': lgb.__version__, 'device': 'cpu',
        'cv_mean_tables': selected.tables, 'cv_fold_tables': selected.fold_tables,
        'selected_iterations': selected.num_iterations, 'final_tables': len(tables),
        'total_rows': sum(t.height for t in tables), 'largest_table': max(t.height for t in tables),
        'scoring_rows': quotes.height, 'observed_scoring_seconds': elapsed,
        'timing_note': 'one warm batch observation, not a benchmark',
    }, indent=2))
    edited_dir = output / 'edited'
    model.to_workbook().save_csv_dir(str(edited_dir))
    intercept = next(edited_dir.glob('*intercept.csv'))
    pl.read_csv(intercept).with_columns((pl.col('Relativity') * 1.05).alias('Relativity')).write_csv(intercept)
    edited = Workbook.load_csv_dir(str(edited_dir)).to_model()
    change = compare_changes(model, edited, holdout, unit='claims/exposure', weight='exposure', segments=['region'])
    change.totals.write_csv(output / 'change_totals.csv')
    change.contributions.write_csv(output / 'change_contributions.csv')
    (output / 'edited_review.md').write_text(edited.report(holdout).markdown)
    print(result.summary())
    print(comparison.summary)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--data')
    parser.add_argument('--refit-glm', action='store_true',
                        help='Refit booster table shapes with a prespecified ridge Poisson GLM')
    args = parser.parse_args()
    run(args.output, args.data, args.refit_glm)
