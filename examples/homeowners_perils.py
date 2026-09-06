"""Synthetic attritional water/theft; no catastrophe model or inferred adjustments.

Requires numpy and scikit-learn. Supplied development/trend factors are applied
explicitly; coverage, limits and deductibles are assumed on a common basis.
"""
import argparse
from pathlib import Path
import random
import numpy as np
import polars as pl
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_tweedie_deviance
from avenue_model import GLMOptions, Plan, Workbook, coefficient_intervals


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
    output.mkdir(parents=True, exist_ok=False)
    if data_path is None:
        data_path = output / 'synthetic_homeowners.csv'
        synthetic(data_path)
    data = pl.read_csv(data_path)
    for column in ('exposure', 'development_factor', 'trend_factor'):
        if data[column].null_count() or not data[column].is_finite().all() or (data[column] <= 0).any():
            raise ValueError(f'{column} must be finite and positive')
    # Keep every renewal for a home on the same side of the split.
    training, testing = next(GroupShuffleSplit(n_splits=1, test_size=1/3, random_state=61).split(
        data, groups=data['home_id'].to_numpy()))
    models, predictions, adjustments = {}, {}, []
    for peril in ('water', 'theft'):
        loss = data[f'{peril}_loss'] * data['development_factor'] * data['trend_factor']
        adjustments.append({'peril': peril, 'input_loss': data[f'{peril}_loss'].sum(), 'prepared_loss': loss.sum()})
        frame = data.with_columns((loss / data['exposure']).alias('premium'))
        train, holdout = frame[training], frame[testing]
        model = (Plan.pure_premium('exposure').banded('home_age', breaks=[20., 40., 60.])
                 .categorical('territory').fit(train, 'premium', GLMOptions(covariance='cluster', cluster='home_id')))
        if not model.converged:
            raise RuntimeError(f'{peril} did not converge')
        (output / f'{peril}_review.md').write_text(model.report(holdout).markdown)
        for name, table in coefficient_intervals(model).tables.items():
            table.write_csv(output / f'{peril}_{name}_intervals.csv')
        model.to_workbook().save_csv_dir(str(output / peril))
        loaded = Workbook.load_csv_dir(str(output / peril)).to_model()
        quotes = holdout.select('home_age', 'territory')
        predictions[peril] = loaded.predict(quotes).to_series()
        np.testing.assert_allclose(predictions[peril], model.predict(quotes).to_series(), atol=1e-12, rtol=1e-12)
        models[peril] = model
    pl.DataFrame(adjustments).write_csv(output / 'loss_adjustments.csv')
    data[testing].select('home_id').unique().write_csv(output / 'holdout_homes.csv')
    holdout = data[testing]
    total = predictions['water'] + predictions['theft']
    actual = (holdout['water_loss'] + holdout['theft_loss']) * holdout['development_factor'] * holdout['trend_factor']
    comparison = pl.DataFrame({'ae_ratio': [actual.sum() / (total * holdout['exposure']).sum()],
        'mean_tweedie_deviance': [mean_tweedie_deviance(actual / holdout['exposure'], total,
                                                     power=1.5, sample_weight=holdout['exposure'])]})
    comparison.write_csv(output / 'comparison.csv')
    pl.DataFrame(predictions).with_columns(total.alias('total')).write_csv(output / 'peril_predictions.csv')
    edited_dir = output / 'edited_water'
    models['water'].to_workbook().save_csv_dir(str(edited_dir))
    factor = next(edited_dir.glob('*territory.csv'))
    pl.read_csv(factor).with_columns((pl.col('Relativity') * 1.05).alias('Relativity')).write_csv(factor)
    edited = Workbook.load_csv_dir(str(edited_dir)).to_model()
    changed = edited.predict(holdout).to_series() + predictions['theft']
    np.testing.assert_allclose(changed - total, predictions['water'] * .05, atol=1e-10, rtol=1e-10)
    pl.DataFrame({'old': total, 'new': changed, 'change': changed-total}).write_csv(output / 'policy_changes.csv')
    print(comparison)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--data')
    args = parser.parse_args()
    run(args.output, args.data)
