"""Compare response means on one explicit population, unit and evaluation loss."""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import random
from typing import Any

import polars as pl

from .splitting import _fingerprint


@dataclass(frozen=True)
class Candidate:
    """A model or positional prediction vector with explicitly declared response units.

    ``converged`` records caller evidence for external predictions. A model's own
    False convergence status always wins over a caller-supplied value.
    """
    source: Any
    unit: str
    converged: bool | None = None

    def __post_init__(self):
        if self.converged is not None and type(self.converged) is not bool:
            raise ValueError('converged must be True, False or None')


@dataclass
class Comparison:
    summary: pl.DataFrame
    predictions: pl.DataFrame
    segments: dict[str, pl.DataFrame]
    metadata: dict
    recommended: str | None
    discrimination: dict[str, pl.DataFrame] = field(default_factory=dict)


def _concentration(y, w, scores):
    """Ascending-score concentration curve; aggregate ties before integration."""
    curve = (pl.DataFrame({'score': scores, 'weight': w,
                           'actual': [a * b for a, b in zip(y, w)]})
             .filter(pl.col('weight') > 0)
             .group_by('score').agg(pl.len().alias('rows'), pl.col('weight').sum(),
                                    pl.col('actual').sum())
             .sort('score')
             .with_columns((pl.col('weight').cum_sum() / pl.col('weight').sum()).alias('weight_share'),
                           (pl.col('actual').cum_sum() / pl.col('actual').sum()).alias('actual_share')))
    # The area under the piecewise-linear curve treats equal scores as one block,
    # so row order cannot award discrimination for arbitrary ordering within ties.
    twice_area = curve.select(((pl.col('actual_share') + pl.col('actual_share').shift(1).fill_null(0))
                               * (pl.col('weight_share') - pl.col('weight_share').shift(1).fill_null(0)))
                              .sum()).item()
    origin = pl.DataFrame({'score': [None], 'rows': [0], 'weight': [0.], 'actual': [0.],
                           'weight_share': [0.], 'actual_share': [0.]}, schema=curve.schema)
    return 1. - twice_area, pl.concat([origin, curve])


def _numbers(values, name, n):
    values = list(values)
    if len(values) != n:
        raise ValueError(f'{name} has {len(values)} rows; expected {n}')
    if any(v is None or not math.isfinite(float(v)) for v in values):
        raise ValueError(f'{name} contains null/nonfinite values')
    return [float(v) for v in values]


def _loss(y, mu, metric, power):
    if metric == 'squared_error':
        return (y - mu) ** 2
    if mu == 0 and y == 0 and metric in ('poisson', 'tweedie'):
        return 0.
    if mu <= 0 or y < 0 or (metric == 'gamma' and y == 0):
        raise ValueError(f'{metric} requires positive predictions and appropriate nonnegative targets (Gamma targets strictly positive)')
    if metric == 'poisson':
        return 2 * ((y * math.log(y / mu) if y else 0.) - y + mu)
    if metric == 'gamma':
        return 2 * (y / mu - 1 - math.log(y / mu))
    return 2 * (y ** (2 - power) / ((1 - power) * (2 - power))
                - y * mu ** (1 - power) / (1 - power) + mu ** (2 - power) / (2 - power))


def _quantile(values, p):
    values = sorted(values)
    position = (len(values) - 1) * p
    lo = math.floor(position)
    hi = math.ceil(position)
    return values[lo] + (values[hi] - values[lo]) * (position - lo)


def compare_models(data, candidates, *, target, unit, metric, weight=None,
                   tweedie_power=1.5, segments=(), bootstrap=0, seed=0,
                   bootstrap_group=None):
    """Compare Candidate models/vectors using the same rows, weights and explicit loss.

    No rows are silently excluded. Candidate scoring failures are retained in the
    summary; they cannot be recommended. Recommendation requires known convergence.
    Bootstrap intervals are paired differences from the first candidate, with fixed
    predictions (no refitting). Set bootstrap_group for cluster resampling.
    Discrimination uses ascending predicted means and the same weights: Gini is
    one minus twice the area under the concentration curve, normalized by ordering
    on the observed target. Ties form one block. Negative supported targets or zero
    actual totals make it unavailable; constant targets have no normalized Gini.
    These empirical rank measures do not alter the loss-based recommendation.
    """
    if not isinstance(data, pl.DataFrame) or not data.height:
        raise ValueError('Comparison requires a nonempty Polars DataFrame')
    if not unit or not isinstance(unit, str):
        raise ValueError('Declare a nonempty response unit')
    if metric not in ('poisson', 'gamma', 'tweedie', 'squared_error'):
        raise ValueError('metric must be poisson, gamma, tweedie or squared_error')
    if metric == 'tweedie' and not 1 < tweedie_power < 2:
        raise ValueError('Tweedie evaluation power must be between 1 and 2')
    if type(bootstrap) is not int or bootstrap < 0 or bootstrap == 1:
        raise ValueError('bootstrap must be zero or an integer of at least two')
    if type(seed) is not int:
        raise ValueError('seed must be an integer')
    if isinstance(segments, str):
        segments = (segments,)
    if not candidates:
        raise ValueError('At least one candidate is required')
    for name, candidate in candidates.items():
        if not isinstance(name, str) or not name or name in ('row', 'actual', 'weight'):
            raise ValueError('Candidate names must be nonempty and cannot be row, actual or weight')
        if not isinstance(candidate, Candidate) or candidate.unit != unit:
            raise ValueError(f'Candidate {name!r} must declare the common unit {unit!r}')
    n = data.height
    y = _numbers(data[target], target, n)
    w = _numbers(data[weight], weight, n) if weight else [1.] * n
    if any(v < 0 for v in w) or sum(w) <= 0:
        raise ValueError('Weights must be nonnegative with positive total support')
    # Validate target domains independently of candidate failures.
    for value in y:
        _loss(value, 1., metric, tweedie_power)
    total_weight = math.fsum(w)
    total_actual = math.fsum(a * b for a, b in zip(y, w))
    supported_targets = [a for a, b in zip(y, w) if b > 0]
    discrimination_status = ('negative_target' if min(supported_targets) < 0 else
                             'zero_actual' if total_actual == 0 else
                             'constant_target' if min(supported_targets) == max(supported_targets) else 'available')
    oracle_gini = (_concentration(y, w, y)[0] if discrimination_status == 'available' else None)
    discrimination = {}
    predictions = {'row': list(range(n)), 'actual': y, 'weight': w}
    records, losses, valid_predictions = [], {}, {}
    for name, candidate in candidates.items():
        own_convergence = getattr(candidate.source, 'converged', None)
        convergence = own_convergence if own_convergence is not None else candidate.converged
        record = {'candidate': name, 'status': 'failed', 'converged': convergence,
                  'eligible': False, 'rows': n, 'weight': total_weight,
                  'actual': total_actual, 'expected': None, 'ae_ratio': None,
                  'mean_loss': None, 'error': None, 'gini': None, 'normalized_gini': None,
                  'discrimination_status': 'scoring_failed',
                  'loss_difference_lower': None, 'loss_difference_upper': None}
        try:
            raw = candidate.source.predict(data) if hasattr(candidate.source, 'predict') else candidate.source
            if isinstance(raw, pl.DataFrame):
                if raw.width != 1:
                    raise ValueError('Prediction frame must contain exactly one response-mean column')
                raw = raw.to_series()
            mu = _numbers(raw, name, n)
            row_losses = [_loss(a, b, metric, tweedie_power) for a, b in zip(y, mu)]
            if any(not math.isfinite(v) for v in row_losses):
                raise ValueError('Evaluation loss is nonfinite')
            expected = math.fsum(a * b for a, b in zip(mu, w))
            loss = math.fsum(a * b for a, b in zip(row_losses, w)) / total_weight
            record.update(status='scored', eligible=convergence is True, expected=expected,
                          ae_ratio=total_actual / expected if expected else None, mean_loss=loss)
            predictions[name] = mu
            valid_predictions[name] = mu
            losses[name] = row_losses
            record['discrimination_status'] = discrimination_status
            if discrimination_status in ('available', 'constant_target'):
                gini, curve = _concentration(y, w, mu)
                record['gini'] = gini if discrimination_status == 'available' else 0.
                record['normalized_gini'] = gini / oracle_gini if oracle_gini and oracle_gini > 0 else None
                discrimination[name] = curve
        except (ValueError, TypeError, OverflowError) as error:
            record['error'] = str(error)
            predictions[name] = [None] * n
        records.append(record)
    baseline = next(iter(candidates))
    effective_bootstrap = {name: 0 for name in candidates}
    if bootstrap:
        group_values = data[bootstrap_group].to_list() if bootstrap_group else range(n)
        groups = {}
        for row, group in enumerate(group_values):
            if group is None or (isinstance(group, float) and not math.isfinite(group)):
                raise ValueError('Bootstrap group contains missing/nonfinite values')
            groups.setdefault(group, []).append(row)
        groups = list(groups.values())
        rng = random.Random(seed)
        differences = {name: [] for name in losses}
        if baseline in losses:
            for _ in range(bootstrap):
                indices = [i for _ in groups for i in rng.choice(groups)]
                support = math.fsum(w[i] for i in indices)
                if not support:
                    continue
                for name in losses:
                    delta = math.fsum(w[i] * (losses[name][i] - losses[baseline][i]) for i in indices) / support
                    differences[name].append(delta)
            for record in records:
                samples = differences.get(record['candidate'], [])
                effective_bootstrap[record['candidate']] = len(samples)
                if len(samples) >= 2:
                    record['loss_difference_lower'] = _quantile(samples, .025)
                    record['loss_difference_upper'] = _quantile(samples, .975)
    exhibits = {}
    for column in segments:
        rows = []
        grouped = {}
        for i, value in enumerate(data[column].to_list()):
            grouped.setdefault(value, []).append(i)
        for name, mu in valid_predictions.items():
            for level, indices in grouped.items():
                actual = math.fsum(w[i] * y[i] for i in indices)
                expected = math.fsum(w[i] * mu[i] for i in indices)
                rows.append({'candidate': name, 'level': level, 'rows': len(indices),
                             'weight': math.fsum(w[i] for i in indices), 'actual': actual,
                             'expected': expected, 'ae_ratio': actual / expected if expected else None})
        exhibits[column] = pl.DataFrame(rows)
    eligible = [r for r in records if r['eligible']]
    recommended = min(eligible, key=lambda r: r['mean_loss'])['candidate'] if eligible else None
    numeric_summary = ('weight', 'actual', 'expected', 'ae_ratio', 'mean_loss',
                       'gini', 'normalized_gini',
                       'loss_difference_lower', 'loss_difference_upper')
    summary = pl.DataFrame(records, schema_overrides={
        **{name: pl.Float64 for name in numeric_summary}, 'converged': pl.Boolean, 'error': pl.String})
    return Comparison(summary, pl.DataFrame(predictions), exhibits,
                      {'unit': unit, 'target': target, 'weight': weight, 'metric': metric,
                       'tweedie_power': tweedie_power if metric == 'tweedie' else None,
                       'population_fingerprint': _fingerprint(data), 'excluded_rows': 0,
                       'bootstrap': bootstrap, 'effective_bootstrap': effective_bootstrap, 'seed': seed, 'bootstrap_group': bootstrap_group,
                       'difference_baseline': baseline,
                       'discrimination': {'ranking': 'ascending predicted response mean',
                           'weight': weight, 'ties': 'aggregate equal scores before trapezoidal integration',
                           'zero_weight_rows': 'retained in predictions; contribute no rank support',
                           'oracle_gini': oracle_gini, 'status': discrimination_status,
                           'uncertainty': 'empirical fixed-prediction estimates; no discrimination intervals'},
                       'uncertainty': 'paired percentile intervals; fixed predictions, no refitting'},
                      recommended, discrimination)
