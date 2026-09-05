"""Select and deliver a continuous age effect on synthetic claim frequency."""
import argparse
from pathlib import Path

import numpy as np
import polars as pl

from avenue_model import Plan, GLMTrial, SplitSpec, select_glm, save_bundle, load_bundle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rng = np.random.default_rng(20260905)
    age = rng.uniform(18., 85., 8000)
    exposure = rng.uniform(.1, 1., len(age))
    rate = .12 * np.exp(.0009 * (age-48.)**2)
    claims = rng.poisson(exposure*rate)
    data = pl.DataFrame({'age': age, 'exposure': exposure,
                         'frequency': claims/exposure})
    outer = SplitSpec.random(n_splits=5, seed=31).split(data)[0]
    train, holdout = outer.frames(data)
    trials = {f'knots_{n}': GLMTrial(Plan.frequency('exposure').spline('age', quantile=n))
              for n in (5, 8)}
    selection = select_glm(train, trials, target='frequency', unit='claims/exposure',
                           metric='poisson', weight='exposure', split=SplitSpec.random(3, seed=19))
    model = selection.refit(train)
    args.output.mkdir(parents=True, exist_ok=False)
    selection.save(args.output/'selection')
    save_bundle(model, args.output/'model', training_id='synthetic-development',
                validation_data=holdout, validation_id='untouched-final-holdout',
                unit='claims/exposure', fold=outer)
    loaded = load_bundle(args.output/'model').model
    np.testing.assert_allclose(loaded.predict(holdout).to_numpy(), model.predict(holdout).to_numpy(), rtol=1e-12)
    grid = pl.DataFrame({'age': np.linspace(10., 95., 200)})
    grid.with_columns(pl.Series('frequency', loaded.predict_rate(grid).to_numpy().reshape(-1))).write_csv(args.output/'continuous_curve.csv')
    print(selection.summary)
    print('Selected:', selection.recommended)
    print('Resolved knots:', model.resolved[1]['knots'])
    print('Covariance:', model.inference_summary['covariance_method'])
    print('Estimated parameters:', model.inference_summary['n_parameters'])
    print('Bundle and continuous quote curve:', args.output)


if __name__ == '__main__':
    main()
