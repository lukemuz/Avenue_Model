"""Reconciled policy and factor effects of changing a scoring artifact."""
from dataclasses import dataclass
import math

import polars as pl

from .comparison import _numbers
from .splitting import _fingerprint


@dataclass
class ModelChange:
    policies: pl.DataFrame
    contributions: pl.DataFrame
    totals: pl.DataFrame
    segments: dict[str, pl.DataFrame]
    metadata: dict


def compare_changes(old, new, data, *, unit, weight=None, segments=()):
    """Compare declared like-unit means. Coefficient differences are on the link scale.

    Both models must score every row and share a link. No original fit inference is
    attributed to edited factors. Weight means exposure for loss-per-exposure models;
    for expected losses/counts, avoid applying exposure twice.
    """
    if not isinstance(unit, str) or not unit:
        raise ValueError('Declare a common response unit for both artifacts')
    if not data.height:
        raise ValueError('Change analysis requires nonempty data')
    before, after = old.explain(data), new.explain(data)
    if not before['summary']['link'].equals(after['summary']['link']):
        raise ValueError('Factor change analysis requires a shared link')
    old_kind, new_kind = old.prediction_kind, new.prediction_kind
    if old_kind != new_kind:
        raise ValueError('Artifacts have different prediction conventions; convert to common means first')
    weights = _numbers(data[weight], weight, data.height) if weight else [1.] * data.height
    if any(w < 0 for w in weights) or math.fsum(weights) <= 0:
        raise ValueError('Change weights must be nonnegative with positive total support')
    a = before['summary']['predictions'].to_list()
    b = after['summary']['predictions'].to_list()
    policies = pl.DataFrame({'row': list(range(data.height)), 'weight': weights,
                             'old': a, 'new': b, 'change': [y - x for x, y in zip(a, b)],
                             'relative_change': [(y - x) / x if x else None for x, y in zip(a, b)],
                             'weighted_change': [w * (y - x) for w, x, y in zip(weights, a, b)]})
    # Keep the long exhibit columnar: row dictionaries and a union of tuple keys
    # previously used several GiB on ordinary six-table portfolios.
    keys = ['row', 'kind', 'term']
    def side(frame, prefix):
        return frame.select(
            *keys,
            pl.col('table_row').cast(pl.Int64).alias(prefix + '_table_row'),
            pl.col('coefficient').alias(prefix + '_coefficient'),
            pl.lit(True).alias('_' + prefix + '_present'))
    contributions = (
        side(before['contributions'], 'old')
        .join(side(after['contributions'], 'new'), on=keys, how='full', coalesce=True)
        .with_columns(
            pl.col('row').cast(pl.Int64),
            pl.col('old_coefficient').fill_null(0.),
            pl.col('new_coefficient').fill_null(0.),
            pl.when(pl.col('_old_present').is_null()).then(pl.lit('added'))
              .when(pl.col('_new_present').is_null()).then(pl.lit('removed'))
              .otherwise(pl.lit('both')).alias('presence'))
        .with_columns(
            # Equal -inf exposure offsets at zero exposure have no change.
            pl.when(pl.col('old_coefficient') == pl.col('new_coefficient')).then(0.)
              .otherwise(pl.col('new_coefficient') - pl.col('old_coefficient'))
              .alias('coefficient_change'))
        .select(*keys, 'old_table_row', 'new_table_row', 'old_coefficient',
                'new_coefficient', 'coefficient_change', 'presence')
        .sort(keys))
    def totals(indices):
        old_total = math.fsum(weights[i] * a[i] for i in indices)
        new_total = math.fsum(weights[i] * b[i] for i in indices)
        return {'rows': len(indices), 'weight': math.fsum(weights[i] for i in indices),
                'old': old_total, 'new': new_total, 'change': new_total - old_total,
                'relative_change': (new_total - old_total) / old_total if old_total else None}
    exhibits = {}
    if isinstance(segments, str):
        segments = (segments,)
    for column in segments:
        groups = {}
        for row, level in enumerate(data[column]):
            groups.setdefault(level, []).append(row)
        exhibits[column] = pl.DataFrame([{'level': level, **totals(indices)} for level, indices in groups.items()])
    return ModelChange(policies, contributions, pl.DataFrame([totals(list(range(data.height)))]),
                       exhibits, {'unit': unit, 'weight': weight, 'excluded_rows': 0,
                                  'population_fingerprint': _fingerprint(data),
                                  'link': before['summary']['link'][0],
                                  'interpretation': 'Artifact changes on supplied rows; no inference for edited coefficients.'})
