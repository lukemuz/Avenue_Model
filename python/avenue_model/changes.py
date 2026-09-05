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
    if before['summary']['link'].to_list() != after['summary']['link'].to_list():
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
    def indexed(frame):
        return {(r['row'], r['kind'], r['term']): r for r in frame.to_dicts()}
    left, right = indexed(before['contributions']), indexed(after['contributions'])
    contributions = []
    for key in sorted(left.keys() | right.keys()):
        old_part, new_part = left.get(key), right.get(key)
        old_value = old_part['coefficient'] if old_part else 0.
        new_value = new_part['coefficient'] if new_part else 0.
        # Identical -inf offsets at zero exposure have no change.
        delta = 0. if old_value == new_value else new_value - old_value
        contributions.append({'row': key[0], 'kind': key[1], 'term': key[2],
                              'old_table_row': old_part['table_row'] if old_part else None,
                              'new_table_row': new_part['table_row'] if new_part else None,
                              'old_coefficient': old_value, 'new_coefficient': new_value,
                              'coefficient_change': delta,
                              'presence': 'added' if old_part is None else 'removed' if new_part is None else 'both'})
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
    return ModelChange(policies, pl.DataFrame(contributions), pl.DataFrame([totals(list(range(data.height)))]),
                       exhibits, {'unit': unit, 'weight': weight, 'excluded_rows': 0,
                                  'population_fingerprint': _fingerprint(data),
                                  'link': before['summary']['link'][0],
                                  'interpretation': 'Artifact changes on supplied rows; no inference for edited coefficients.'})
