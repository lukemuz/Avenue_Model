"""Executable synthetic auto study. Run: python examples/auto_pricing_study.py --output /tmp/auto-study

Generates and loads CSV data when --data is omitted. Supplied CSVs must have the same
columns. Loss is assumed already developed/trended to the selected cost level; no
limits, deductibles, trend or development adjustment is inferred here.
"""
import argparse
from pathlib import Path
import random

import polars as pl
from avenue_model import Candidate, Plan, SplitSpec, Workbook, compare_models, prepare_pricing


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
    product = frequency.predict(holdout).to_series() * severity.predict(holdout).to_series()
    comparison = compare_models(
        holdout, {'frequency_times_severity': Candidate(product, 'loss_per_exposure', True),
                  'tweedie': Candidate(premium, 'loss_per_exposure')},
        target='avenue_pure_premium', unit='loss_per_exposure', metric='tweedie',
        tweedie_power=1.5, weight='exposure', segments=['region', 'year'], bootstrap=50, seed=47)
    comparison.summary.write_csv(output / 'comparison.csv')
    for name, table in comparison.segments.items():
        table.write_csv(output / f'comparison_{name}.csv')
    print(comparison.summary)
    return comparison


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--data')
    args = parser.parse_args()
    run(args.output, args.data)
