"""Fit and deliver a continuous age effect on synthetic claim frequency."""
import argparse
from pathlib import Path

import numpy as np
import polars as pl

from avenue_model import Plan, Workbook
from sklearn.model_selection import train_test_split


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
    training, testing = train_test_split(np.arange(data.height), test_size=.2, random_state=31)
    train, holdout = data[training], data[testing]
    model = Plan.frequency('exposure').spline('age', quantile=5).fit(train, 'frequency')
    if not model.converged:
        raise RuntimeError('Spline fit did not converge')
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'review.md').write_text(model.report(holdout).markdown)
    model.to_workbook().save_csv_dir(str(args.output/'model'))
    loaded = Workbook.load_csv_dir(str(args.output/'model')).to_model()
    np.testing.assert_allclose(loaded.predict(holdout).to_numpy(), model.predict(holdout).to_numpy(), rtol=1e-12)
    grid = pl.DataFrame({'age': np.linspace(10., 95., 200)})
    grid.with_columns(pl.Series('frequency', loaded.predict_rate(grid).to_numpy().reshape(-1))).write_csv(args.output/'continuous_curve.csv')
    print('Resolved knots:', model.resolved[1]['knots'])
    print('Covariance:', model.inference_summary['covariance_method'])
    print('Estimated parameters:', model.inference_summary['n_parameters'])
    print('Workbook and continuous quote curve:', args.output)


if __name__ == '__main__':
    main()
