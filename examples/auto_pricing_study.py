"""Synthetic auto pricing using Plan, validation and editable workbooks.

Requires numpy and scikit-learn for example checks/comparisons. Supplied losses
must already have the intended development, trend, limits and deductible basis.
"""
import argparse
from pathlib import Path
import random
import numpy as np
import polars as pl
from sklearn.metrics import mean_tweedie_deviance
from avenue_model import Plan, Workbook


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
    output.mkdir(parents=True, exist_ok=False)
    if data_path is None:
        data_path = output / 'synthetic_auto.csv'
        synthetic(data_path)
    raw = pl.read_csv(data_path)
    # Study definitions are explicit dataframe operations, not a library pipeline.
    if raw.filter((pl.col('exposure') <= 0) | (pl.col('claims') < 0) | (pl.col('loss') < 0)).height:
        raise ValueError('This example requires positive exposures and nonnegative claims/losses')
    data = raw.with_columns((pl.col('claims') / pl.col('exposure')).alias('frequency'),
                           (pl.col('loss') / pl.col('exposure')).alias('premium'))
    severity_data = data.filter((pl.col('claims') > 0) & (pl.col('loss') > 0)).with_columns(
        (pl.col('loss') / pl.col('claims')).alias('severity'))
    train, holdout = data.filter(pl.col('year') < 2022), data.filter(pl.col('year') >= 2022)
    data.select('policy_id', (pl.col('year') >= 2022).alias('holdout')).write_csv(output / 'split.csv')
    shape = lambda p: p.banded('age', breaks=[25., 50.]).categorical('region')
    models = {}
    for name, plan, training, validation in [
        ('frequency', Plan.frequency('exposure'), train, holdout),
        ('severity', Plan.severity('claims'), severity_data.filter(pl.col('year') < 2022),
         severity_data.filter(pl.col('year') >= 2022)),
        ('premium', Plan.pure_premium('exposure'), train, holdout),
    ]:
        fitted = shape(plan).fit(training, name)
        if not fitted.converged:
            raise RuntimeError(f'{name} did not converge')
        (output / f'{name}_review.md').write_text(fitted.report(validation).markdown)
        (output / f'{name}_plan.json').write_text(fitted.plan.to_json())
        fitted.to_workbook().save_csv_dir(str(output / name))
        loaded = Workbook.load_csv_dir(str(output / name)).to_model()
        np.testing.assert_allclose(loaded.predict(holdout.select('age', 'region')).to_numpy(),
                                   fitted.predict(holdout).to_numpy(), atol=1e-12, rtol=1e-12)
        models[name] = fitted
    # Components are rate and mean severity: multiply their predicted means.
    product = models['frequency'].predict_rate(holdout).to_series() * models['severity'].predict(holdout).to_series()
    rows = []
    for name, prediction in [('frequency_severity', product), ('tweedie', models['premium'].predict(holdout).to_series())]:
        rows.append({'model': name, 'ae_ratio': holdout['loss'].sum() / (prediction * holdout['exposure']).sum(),
                     'mean_tweedie_deviance': mean_tweedie_deviance(holdout['premium'], prediction,
                         power=1.5, sample_weight=holdout['exposure'])})
    comparison = pl.DataFrame(rows)
    comparison.write_csv(output / 'comparison.csv')
    # A rate revision is an ordinary workbook edit and a fresh validation.
    edited_dir = output / 'edited_premium'
    models['premium'].to_workbook().save_csv_dir(str(edited_dir))
    factors = next(edited_dir.glob('*region.csv'))
    pl.read_csv(factors).with_columns((pl.col('Relativity') * 1.05).alias('Relativity')).write_csv(factors)
    edited = Workbook.load_csv_dir(str(edited_dir)).to_model()
    before, after = models['premium'].predict(holdout).to_series(), edited.predict(holdout).to_series()
    np.testing.assert_allclose(after, before * 1.05, atol=1e-12, rtol=1e-12)
    holdout.select('policy_id', 'region', 'exposure').with_columns(before.alias('old'), after.alias('new'),
        (after - before).alias('change')).write_csv(output / 'policy_changes.csv')
    edited.explain(holdout.head(5))['contributions'].write_csv(output / 'quote_explanations.csv')
    (output / 'edited_review.md').write_text(edited.report(holdout).markdown)
    print(comparison)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--data')
    args = parser.parse_args()
    run(args.output, args.data)
