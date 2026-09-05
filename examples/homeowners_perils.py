"""Synthetic attritional water/theft study; deliberately excludes catastrophe losses.

Run: python examples/homeowners_perils.py --output /tmp/homeowners-study
Use --data for a CSV with the same columns. Explicit supplied development/trend
factors are applied and audited; coverage is assumed to be at a common limit and
deductible basis. No catastrophe, limit, deductible or inflation model is inferred.
"""
import argparse
import json
from pathlib import Path
import random

import polars as pl
from avenue_model import (GLMOptions, coefficient_intervals, term_tests, Candidate, ComposedModel, Plan, SplitSpec, Workbook,
                          compare_models, prepare_pricing, sum_loss_costs, save_bundle)


def synthetic(path):
    rng = random.Random(61)
    rows = []
    for home in range(1500):
        territory = rng.choice(['rural', 'urban'])
        age = rng.randrange(1, 80)
        for year in (2020, 2021, 2022):
            exposure = rng.uniform(.5, 1.)
            water = int(rng.random() < exposure * .12 * (1.5 if age > 40 else 1.))
            theft = int(rng.random() < exposure * .07 * (1.5 if territory == 'urban' else 1.))
            rows.append({'home_id': home, 'year': year, 'territory': territory, 'home_age': age,
                         'exposure': exposure, 'water_claims': water, 'theft_claims': theft,
                         'water_loss': rng.gammavariate(2., 2500.) if water else 0.,
                         'theft_loss': rng.gammavariate(2., 1000.) if theft else 0.,
                         'development_factor': 1., 'trend_factor': 1.})
    pl.DataFrame(rows).write_csv(path)


def run(output, data_path=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if data_path is None:
        data_path = output / 'synthetic_homeowners.csv'
        synthetic(data_path)
    raw = pl.read_csv(data_path)
    for column in ('development_factor', 'trend_factor'):
        values = raw[column]
        if values.null_count() or not values.is_finite().all() or (values <= 0).any():
            raise ValueError(f'{column} must contain supplied finite positive adjustments')
    models, baselines, adjustments, bundle_contexts = {}, {}, [], {}
    for peril in ('water', 'theft'):
        developed = f'{peril}_prepared_loss'
        raw = raw.with_columns((pl.col(f'{peril}_loss') * pl.col('development_factor') *
                                pl.col('trend_factor')).alias(developed))
        adjustments.append({'peril': peril, 'input_loss': raw[f'{peril}_loss'].sum(),
                            'prepared_loss': raw[developed].sum()})
    pl.DataFrame(adjustments).write_csv(output / 'loss_adjustments.csv')
    fold = SplitSpec.grouped('home_id', n_splits=3, seed=61).split(raw)[0]
    (output / 'split.json').write_text(fold.to_json())
    train, holdout = fold.frames(raw)
    for peril in ('water', 'theft'):
        prepared = prepare_pricing(train, exposure='exposure', claims=f'{peril}_claims',
                                   loss=f'{peril}_prepared_loss', predictors=['territory', 'home_age'])
        validation = prepare_pricing(holdout, exposure='exposure', claims=f'{peril}_claims',
                                     loss=f'{peril}_prepared_loss', predictors=['territory', 'home_age'])
        prepared.audit.write_csv(output / f'{peril}_training_audit.csv')
        validation.audit.write_csv(output / f'{peril}_validation_audit.csv')
        prepared.experience('territory').write_csv(output / f'{peril}_experience.csv')
        model = (Plan.pure_premium('exposure').banded('home_age', breaks=[20., 40., 60.])
                 .categorical('territory').fit(prepared.pure_premium, 'avenue_pure_premium',
                      GLMOptions(covariance='cluster', cluster='home_id')))
        baseline = Plan.pure_premium('exposure').fit(prepared.pure_premium, 'avenue_pure_premium',
                      GLMOptions(covariance='cluster', cluster='home_id'))
        if not model.converged or not baseline.converged:
            raise RuntimeError(f'{peril} did not converge')
        models[peril], baselines[peril] = model, baseline
        bundle_contexts[peril] = {'training_id': f'home-training-{peril}',
                                 'validation_data': validation.pure_premium,
                                 'validation_id': f'home-holdout-{peril}'}
        (output / f'{peril}_review.md').write_text(model.report(validation.pure_premium).markdown)
        joint = term_tests(model)
        (output / f'{peril}_term_tests.json').write_text(json.dumps(
            {'table': joint.table.to_dicts(), 'metadata': joint.metadata}, indent=2, allow_nan=False))
        for name, table in coefficient_intervals(model).tables.items():
            table.write_csv(output / f'{peril}_{name}_cluster_intervals.csv')
        for name, table in model.rating_tables_by_name().items():
            table.write_csv(output / f'{peril}_{name}_factors.csv')
    total = sum_loss_costs(models)
    baseline = sum_loss_costs(baselines)
    holdout = holdout.with_columns(((pl.col('water_prepared_loss') + pl.col('theft_prepared_loss')) /
                                    pl.col('exposure')).alias('combined_loss_cost'))
    comparison = compare_models(holdout, {'peril_models': Candidate(total, total.unit),
                                         'intercept_baseline': Candidate(baseline, baseline.unit)},
                                target='combined_loss_cost', weight='exposure', unit=total.unit,
                                metric='tweedie', tweedie_power=1.5, segments=['territory', 'year'],
                                bootstrap=100, bootstrap_group='home_id', seed=61)
    comparison.summary.write_csv(output / 'comparison.csv')
    for name, table in comparison.segments.items():
        table.write_csv(output / f'comparison_{name}.csv')
    total.save(output / 'peril_plan')
    analytical = save_bundle(total, output / 'peril_bundle', fold=fold,
        validation_data=holdout, validation_id='home-grouped-holdout',
        validation_options={'target': 'combined_loss_cost', 'metric': 'tweedie',
                            'weight': 'exposure', 'segments': ['territory', 'year']},
        component_context=bundle_contexts,
        lineage={'adjustment_audit': 'loss_adjustments.csv', 'scope': 'attritional water and theft'})
    reloaded = ComposedModel.load(output / 'peril_plan')
    quotes = holdout.select('territory', 'home_age')
    original = total.predict(quotes).to_series()
    restored = reloaded.predict(quotes).to_series()
    if (original - restored).abs().max() > 1e-8:
        raise RuntimeError('Reload changed composed quote predictions')
    if (original - analytical.model.predict(quotes).to_series()).abs().max() > 1e-8:
        raise RuntimeError('Analytical peril bundle changed quote predictions')
    components = reloaded.predict_components(quotes)
    if (components['water'] + components['theft'] - restored).abs().max() > 1e-8:
        raise RuntimeError('Peril contributions do not reconcile')
    components.head(20).write_csv(output / 'quote_peril_contributions.csv')
    # Edit one component and retain a fresh, explicit common-metric validation.
    edited_directory = output / 'edited_water'
    models['water'].to_workbook().save_csv_dir(str(edited_directory))
    factor_path = next(edited_directory.glob('*territory.csv'))
    factors = pl.read_csv(factor_path)
    factors.with_columns((pl.col('Relativity') * 1.05).alias('Relativity')).write_csv(factor_path)
    edited_water = Workbook.load_csv_dir(str(edited_directory)).to_model()
    edited_total = sum_loss_costs({'water': edited_water, 'theft': models['theft']})
    edited_total.save(output / 'edited_peril_plan')
    edited_total.validate(holdout, target='combined_loss_cost', metric='tweedie',
                          weight='exposure', segments=['territory']).summary.write_csv(output / 'edited_validation.csv')
    delta = edited_total.predict(quotes).to_series() - original
    if (delta - components['water'] * .05).abs().max() > 1e-8:
        raise RuntimeError('Known water factor edit did not reconcile')
    pl.DataFrame({'old': original, 'change': delta, 'exposure': holdout['exposure']}).write_csv(output / 'policy_changes.csv')
    print(comparison.summary)
    return comparison


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--data')
    args = parser.parse_args()
    run(args.output, args.data)
