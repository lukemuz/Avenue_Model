"""Reproducible refit variability, without a true-mean confidence-interval claim."""
from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
import sys

import polars as pl

from .avenue_model import GLMOptions, Plan
from .comparison import _numbers, _quantile
from .splitting import _fingerprint


@dataclass
class BootstrapStability:
    summary: pl.DataFrame
    draws: pl.DataFrame
    history: pl.DataFrame
    metadata: dict

    def save(self, directory):
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=False)
        self.summary.write_csv(path / 'summary.csv')
        self.draws.write_parquet(path / 'draws.parquet')
        self.history.write_csv(path / 'history.csv')
        (path / 'stability.json').write_text(json.dumps(self.metadata, indent=2, allow_nan=False)+'\n')


def bootstrap_stability(data, plan, quotes, *, target, options=None, resamples=200,
                        mass=.95, seed=0, group=None):
    """Pairs-bootstrap prediction stability for an existing Plan recipe.

    Rows, or whole groups, are sampled uniformly with replacement. Original exposure
    weights remain attached to their rows. Plan preprocessing is resolved anew in
    every refit; upstream model/penalty selection is not repeated. The point fit and
    every replicate must converge and score all quotes before bands are available.
    Bands describe the resampled fitting procedure, not guaranteed true-mean coverage,
    coefficient confidence intervals, or future-observation prediction intervals.
    """
    if not isinstance(data, pl.DataFrame) or not data.height:
        raise ValueError('Bootstrap requires nonempty Polars training data')
    if not isinstance(quotes, pl.DataFrame) or not quotes.height:
        raise ValueError('Bootstrap requires nonempty Polars quote data')
    if not isinstance(plan, Plan):
        raise TypeError('Supply a Plan recipe, not a fitted model')
    if type(resamples) is not int or resamples < 20:
        raise ValueError('resamples must be an integer of at least 20')
    if type(seed) is not int:
        raise ValueError('seed must be an integer')
    if group is not None and not isinstance(group, str):
        raise TypeError('group must be a column name or None')
    if isinstance(mass, bool) or not isinstance(mass, (int, float)) or not math.isfinite(mass) or not 0 < mass < 1:
        raise ValueError('mass must be strictly between zero and one')
    settings = json.loads(json.dumps({} if options is None else options, allow_nan=False))
    if not isinstance(settings, dict):
        raise TypeError('options must be a dictionary of GLMOptions arguments')
    # This diagnostic uses refitted means, never ordinary covariance from a penalized fit.
    settings['compute_standard_errors'] = False
    fit_options = GLMOptions(**settings)
    recipe = plan.to_json()
    units = {}
    for i, value in enumerate(data[group].to_list() if group is not None else range(data.height)):
        if value is None or isinstance(value, float) and not math.isfinite(value):
            raise ValueError('Resampling group values must be non-null and finite')
        try:
            units.setdefault(value, []).append(i)
        except TypeError as error:
            raise ValueError('Resampling group values must be scalar and hashable') from error
    units = list(units.values())
    if len(units) < 2:
        raise ValueError('At least two resampling units are required')
    fitted = Plan.from_json(recipe).fit(data, target, fit_options)
    if fitted.converged is not True:
        raise ValueError('Original fit did not converge; revise it before resampling')
    point = _numbers(fitted.predict(quotes).to_series(), 'original predictions', quotes.height)
    kind = fitted.prediction_kind
    del fitted
    rng = random.Random(seed)
    history, draws, successful = [], {}, []
    for replicate in range(resamples):
        sample_seed = rng.getrandbits(63)
        sampler = random.Random(sample_seed)
        selected = [sampler.randrange(len(units)) for _ in units]
        indices = [i for unit in selected for i in units[unit]]
        record = {'replicate': replicate, 'sample_seed': sample_seed, 'sample_rows': len(indices),
                  'distinct_units': len(set(selected)), 'status': 'failed', 'converged': None, 'error': None}
        values = [None] * quotes.height
        model = None
        try:
            model = Plan.from_json(recipe).fit(data[indices], target, fit_options)
            record['converged'] = model.converged
            if model.converged is not True:
                raise ValueError('Replicate fit did not converge')
            values = _numbers(model.predict(quotes).to_series(), 'replicate predictions', quotes.height)
            successful.append(values)
            record['status'] = 'scored'
        except (ValueError, TypeError, OverflowError) as error:
            record['error'] = str(error)
        finally:
            model = None
        draws[f'draw_{replicate}'] = pl.Series(values, dtype=pl.Float64)
        history.append(record)
    complete = len(successful) == resamples
    tail = (1-mass)/2
    summary = pl.DataFrame({
        'row': list(range(quotes.height)), 'prediction': point,
        'stability_lower': pl.Series([_quantile([v[i] for v in successful], tail) for i in range(quotes.height)]
                                     if complete else [None]*quotes.height, dtype=pl.Float64),
        'stability_upper': pl.Series([_quantile([v[i] for v in successful], 1-tail) for i in range(quotes.height)]
                                     if complete else [None]*quotes.height, dtype=pl.Float64),
        'status': ['complete' if complete else 'incomplete'] * quotes.height})
    return BootstrapStability(summary, pl.DataFrame(draws), pl.DataFrame(history, schema_overrides={
        'error': pl.String, 'converged': pl.Boolean}), {
        'schema_version': 1, 'method': 'percentile pairs-bootstrap refit stability',
        'plan_json': recipe, 'options': settings, 'target': target, 'prediction_kind': kind,
        'training_fingerprint': _fingerprint(data), 'quote_fingerprint': _fingerprint(quotes),
        'resamples': resamples, 'successful_resamples': len(successful), 'bands_available': complete,
        'mass': mass, 'seed': seed, 'group': group, 'resampling_units': len(units),
        'random_generator': 'Python random.Random; per-replicate 63-bit seeds retained in history',
        'python_version': sys.version,
        'interpretation': 'refit stability only; no guaranteed true-mean, coefficient, or future-observation coverage',
        'preprocessing': 'resolved within each resample; upstream selection and fixed-prior uncertainty not repeated',
        'failures': 'all failures retained; any failure withholds all percentile bands',
    })
