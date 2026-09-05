"""Coverage pilot for fixed-penalty pairs bootstrap; not a public inference API."""
import argparse
import hashlib
import importlib
import importlib.metadata as metadata
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import polars as pl

from avenue_model import GLMOptions, Plan


def ridge_reference(x, y, alpha):
    design = np.column_stack([np.ones(len(x)), x])
    information = design.T @ design / len(x) + np.diag([0., alpha])
    beta = np.linalg.solve(information, design.T @ y / len(x))
    return np.array([beta[0], beta.sum()])


def run(output, datasets=100, resamples=100):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(20260906)
    n, intercept, effect = 300, .2, .8
    plan = Plan('gaussian').categorical('group', base='first')
    quotes = pl.DataFrame({'group': ['a', 'b']})
    records, failures = [], []
    max_error = 0.
    started = time.perf_counter()
    options = {a: GLMOptions(alpha=a, l1_ratio=0., tolerance=1e-10,
                              max_iterations=1000, compute_standard_errors=False)
               for a in [0., .01, .3]}
    for sample in range(datasets):
        x = rng.binomial(1, .5, n)
        y = intercept + effect*x + rng.normal(size=n)
        frame = pl.DataFrame({'group': np.where(x == 0, 'a', 'b'), 'y': y})
        # Every penalty sees the same resamples, independent of their outcomes.
        draws = rng.integers(0, n, size=(resamples, n))
        for alpha, fit_options in options.items():
            predictions = []
            for replicate, indices in enumerate([np.arange(n), *draws]):
                try:
                    fitted = plan.fit(frame[indices.tolist()], 'y', fit_options)
                    if fitted.converged is not True:
                        raise ValueError('Fit did not converge')
                    mu = fitted.predict(quotes).to_numpy().reshape(-1)
                    reference = ridge_reference(x[indices], y[indices], alpha)
                    error = float(np.max(np.abs(mu-reference)))
                    max_error = max(max_error, error)
                    np.testing.assert_allclose(mu, reference, rtol=1e-7, atol=1e-8)
                    predictions.append(mu)
                except (ValueError, AssertionError) as error:
                    failures.append({'dataset': sample, 'alpha': alpha, 'replicate': replicate, 'error': str(error)})
            if len(predictions) != resamples+1:
                continue  # No intervals from a silently reduced successful subset.
            point = predictions[0]
            lower, upper = np.quantile(predictions[1:], [.025, .975], axis=0)
            # Population penalized optimum under P(group=b)=1/2 and fixed alpha.
            penalized_effect = effect * .25 / (.25 + alpha)
            penalized_mean = np.array([intercept + .5*(effect-penalized_effect),
                                      intercept + .5*(effect+penalized_effect)])
            true_mean = np.array([intercept, intercept+effect])
            for group in [0, 1]:
                records.append({'dataset': sample, 'alpha': alpha, 'group': group,
                                'point': float(point[group]), 'lower': float(lower[group]), 'upper': float(upper[group]),
                                'true_mean': float(true_mean[group]), 'penalized_mean': float(penalized_mean[group]),
                                'covers_true_mean': bool(lower[group] <= true_mean[group] <= upper[group]),
                                'covers_penalized_mean': bool(lower[group] <= penalized_mean[group] <= upper[group])})
        if (sample+1) % 10 == 0:
            print(f'Completed {sample+1}/{datasets} datasets', flush=True)
    history = pl.DataFrame(records)
    history.write_csv(output / 'history.csv')
    summary = history.group_by('alpha', 'group').agg(pl.len().alias('datasets'),
        pl.col('covers_true_mean').mean().alias('true_mean_coverage'),
        pl.col('covers_penalized_mean').mean().alias('penalized_mean_coverage'),
        (pl.col('point')-pl.col('true_mean')).mean().alias('mean_bias'),
        (pl.col('upper')-pl.col('lower')).mean().alias('mean_width')).sort('alpha', 'group')
    summary.write_csv(output / 'summary.csv')
    result = {'status': 'failed' if failures else 'passed_mechanics',
              'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'extension_sha256': hashlib.sha256(Path(importlib.import_module('avenue_model.avenue_model').__file__).read_bytes()).hexdigest(),
              'versions': {name: metadata.version(name) for name in ['numpy', 'polars']},
              'seed': 20260906, 'datasets': datasets, 'rows_per_dataset': n, 'resamples': resamples,
              'max_independent_prediction_error': max_error, 'failures': failures,
              'wall_seconds': time.perf_counter()-started,
              'interpretation': 'fixed-penalty percentile pairs bootstrap; true and penalized population targets evaluated separately; no model selection or lasso coverage claim'}
    (output / 'result.json').write_text(json.dumps(result, indent=2)+'\n')
    print(summary)
    if failures:
        raise RuntimeError('Bootstrap mechanics failed; inspect retained failure records')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--datasets', type=int, default=100)
    parser.add_argument('--resamples', type=int, default=100)
    args = parser.parse_args()
    if args.datasets < 2 or args.resamples < 20:
        parser.error('Use at least two datasets and twenty resamples')
    run(args.output, args.datasets, args.resamples)
