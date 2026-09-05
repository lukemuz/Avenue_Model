"""Executable synthetic auto study. Run: python examples/auto_pricing_study.py --output /tmp/auto-study

Generates and loads CSV data when --data is omitted. Supplied CSVs must have the same
columns. Loss is assumed already developed/trended to the selected cost level; no
limits, deductibles, trend or development adjustment is inferred here.
"""
import argparse
from pathlib import Path
import random

import polars as pl
from avenue_model import GLMTrial, select_glm, save_bundle, Candidate, Plan, SplitSpec, Workbook, compare_models, prepare_pricing, compare_changes, frequency_severity


def synthetic(path):
    rng = random.Random(47)
    rows = []
    for i in range(3000):
        exposure = rng.uniform(.25, 1.)
        age = rng.randrange(18, 80)
        region = rng.choice(['north', 'south'])
        # A deliberately simple low-frequency synthetic portfolio, not real loss data.
        claims = int(rng.random() < exposure * .2 * (1.5 if age < 25 else 1.))
        loss = rng.gammavariate(2., 1500. * (1.2 if region == 'south' else 1.)) if claims else 0.
        rows.append({'policy_id': i, 'year': 2020 + i % 3, 'age': age, 'region': region,
                     'exposure': exposure, 'claims': claims, 'loss': loss})
    pl.DataFrame(rows).write_csv(path)


def run(output, data_path=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if data_path is None:
        data_path = output / 'synthetic_auto.csv'
        synthetic(data_path)
    raw = pl.read_csv(data_path)
    prepared = prepare_pricing(raw, exposure='exposure', claims='claims', loss='loss',
                               predictors=['age', 'region'], large_loss=10000.)
    prepared.audit.write_csv(output / 'preparation_audit.csv')
    prepared.summary.write_csv(output / 'populations.csv')
    prepared.experience('region').write_csv(output / 'region_experience.csv')
    data = prepared.frequency
    fold, = SplitSpec.out_of_time('year', 2022).split(data)
    (output / 'split.json').write_text(fold.to_json())
    train, holdout = fold.frames(data)
    shape = lambda plan: plan.banded('age', breaks=[25., 50.]).categorical('region')
    frequency = shape(Plan.frequency('exposure')).fit(train, 'avenue_frequency')
    severity_train = prepared.severity.filter(pl.col('year') < 2022)
    severity = shape(Plan.severity('claims')).fit(severity_train, 'avenue_severity')
    premium = shape(Plan.pure_premium('exposure')).fit(train, 'avenue_pure_premium')
    # Selection uses only pre-2022 data. Every fitting power is evaluated at the
    # same prespecified power; the final year is never used to choose this grid.
    selection = select_glm(train, {
        f'power={power},alpha={alpha}': GLMTrial(
            shape(Plan('tweedie', exposure='exposure', exposure_role='weight', tweedie_power=power)),
            {'alpha': alpha, 'l1_ratio': .5})
        for power in (1.3, 1.7) for alpha in (0., .01)
    }, target='avenue_pure_premium', unit='loss_per_exposure', metric='tweedie',
       tweedie_power=1.5, weight='exposure', split=SplitSpec.grouped('policy_id', 3, seed=47))
    selection.save(output / 'glm_selection')
    selected_premium = selection.refit(train)
    save_bundle(selected_premium, output / 'selected_premium_bundle',
                fit_options=selection.metadata['trials'][selection.recommended]['options'],
                training_id='auto-pre-2022', validation_data=holdout, validation_id='auto-2022',
                unit='loss_per_exposure', lineage={'selection': 'glm_selection/selection.json',
                                                   'trial': selection.recommended})
    for name, model, validation in (
        ('frequency', frequency, holdout),
        ('severity', severity, prepared.severity.filter(pl.col('year') >= 2022)),
        ('pure_premium', premium, holdout),
    ):
        if not model.converged:
            raise RuntimeError(f'{name} did not converge')
        report = model.report(validation)
        (output / f'{name}_review.md').write_text(report.markdown)
        model.to_workbook().save_csv_dir(str(output / name))
        for table_name, table in model.rating_tables_by_name().items():
            table.write_csv(output / f'{name}_{table_name}_estimates.csv')
        reloaded = Workbook.load_csv_dir(str(output / name)).to_model()
        quotes = holdout.select('age', 'region')
        actual = reloaded.predict(quotes).to_series()
        expected = model.predict(quotes).to_series()
        if (actual - expected).abs().max() > 1e-8:
            raise RuntimeError('Workbook predictions changed')
    product = frequency_severity(frequency, severity)
    product.save(output / 'frequency_severity')
    comparison = compare_models(
        holdout, {'frequency_times_severity': Candidate(product, 'loss_per_exposure'),
                  'tweedie': Candidate(premium, 'loss_per_exposure'),
                  'selected_tweedie': Candidate(selected_premium, 'loss_per_exposure')},
        target='avenue_pure_premium', unit='loss_per_exposure', metric='tweedie',
        tweedie_power=1.5, weight='exposure', segments=['region', 'year'], bootstrap=50, seed=47)
    comparison.summary.write_csv(output / 'comparison.csv')
    for name, table in comparison.segments.items():
        table.write_csv(output / f'comparison_{name}.csv')
    # A manual factor edit creates a new scoring artifact; its evidence is separate.
    edit_directory = output / 'edited_premium'
    premium.to_workbook().save_csv_dir(str(edit_directory))
    factor_path = next(edit_directory.glob('*region.csv'))
    factors = pl.read_csv(factor_path)
    factors.with_columns((pl.col('Relativity') * 1.05).alias('Relativity')).write_csv(factor_path)
    edited = Workbook.load_csv_dir(str(edit_directory)).to_model()
    changes = compare_changes(premium, edited, holdout, unit='loss_per_exposure',
                              weight='exposure', segments=['region'])
    changes.totals.write_csv(output / 'edit_totals.csv')
    changes.policies.sort('weighted_change', descending=True).head(20).write_csv(output / 'largest_changes.csv')
    changes.contributions.write_csv(output / 'factor_changes.csv')
    explained = edited.explain(holdout.head(5))
    explained['contributions'].write_csv(output / 'quote_explanations.csv')
    # Fresh validation evidence belongs to the edited artifact, not the original fit.
    (output / 'edited_review.md').write_text(edited.report(holdout).markdown)
    print(comparison.summary)
    return comparison


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--data')
    args = parser.parse_args()
    run(args.output, args.data)
