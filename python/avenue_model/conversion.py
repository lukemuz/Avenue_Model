"""Conversion with explicit, input-specific evidence of numerical agreement."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib
import json
import math
from pathlib import Path
from typing import Any

import polars as pl

from .avenue_model import FittedModel


@dataclass
class ConversionResult:
    """A scoring model and its conversion evidence; parity applies only to tested rows."""

    model: Any
    metadata: dict
    parity: dict

    def save(self, directory):
        """Save the editable workbook and a separate conversion evidence JSON file."""
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        self.model.to_workbook().save_csv_dir(str(path))
        (path / 'conversion.json').write_text(
            json.dumps({'metadata': self.metadata, 'parity': self.parity}, indent=2,
                       allow_nan=False) + '\n')


def from_booster(booster, data=None, *, consolidation='max', atol=1e-12, rtol=1e-12,
                 resource_limits=None):
    """Convert a Booster, optionally checking means against a Polars predictor frame.

    Returns a ConversionResult with model, metadata and parity. Failed parity is
    retained as evidence (``parity['status'] == 'failed'``), not silently accepted.
    Without data the status is ``not_verified``. Input column order is resolved from
    the booster feature names. Categorical booster inputs must be integer codes;
    use ``model.with_categories`` to attach an external code-to-label mapping.
    Numerical null/NaN routes are retained in explicit missing rows. LightGBM
    zero_as_missing semantics remain unsupported and are rejected by conversion.

    `resource_limits` optionally maps `tables`, `total_rows`, `largest_table` or
    `largest_interaction_order` to nonnegative integer upper bounds. Limits are
    checked on the actual consolidated artifact, including its intercept, and an
    exceeded limit raises ValueError. They do not limit conversion working memory
    or change the converted structure. Measured complexity and accepted limits are
    retained in metadata; absent limits impose no resource constraint.
    """
    if any(not math.isfinite(v) or v < 0 for v in (atol, rtol)):
        raise ValueError('atol and rtol must be finite and nonnegative')
    supported_limits = {'tables', 'total_rows', 'largest_table', 'largest_interaction_order'}
    if resource_limits is not None and not isinstance(resource_limits, dict):
        raise TypeError('resource_limits must be a dictionary of integer upper bounds')
    limits = dict(resource_limits or {})
    if any(key not in supported_limits for key in limits):
        raise ValueError(f'resource_limits keys must be among {sorted(supported_limits)}')
    if any(type(value) is not int or value < 0 for value in limits.values()):
        raise ValueError('resource_limits values must be nonnegative integers, not booleans or floats')
    dump = booster.dump_model()
    serialized = json.dumps(dump, sort_keys=True, separators=(',', ':'))
    model = FittedModel.from_lgbm_json(serialized, consolidation=consolidation)
    tables = model.to_workbook(scale='factor').tables
    complexity = {'tables': len(tables), 'total_rows': sum(table.height for table in tables),
                  'largest_table': max(table.height for table in tables),
                  'largest_interaction_order': max(table.width - 1 for table in tables)}
    del tables
    violations = [f'{key}={complexity[key]} (limit {bound})' for key, bound in limits.items()
                  if complexity[key] > bound]
    if violations:
        raise ValueError('Final converted artifact exceeds resource_limits: ' + ', '.join(violations))
    module_name = type(booster).__module__.split('.')[0]
    module = importlib.import_module(module_name)
    metadata = {'schema_version': 1, 'module': module_name,
                'version': getattr(module, '__version__', None),
                'objective': dump.get('objective'), 'consolidation': consolidation,
                'feature_names': dump['feature_names'],
                'tree_count': len(dump['tree_info']),
                'dump_sha256': hashlib.sha256(serialized.encode()).hexdigest(),
                'complexity': complexity,
                'resource_check': {'status': 'passed' if limits else 'not_requested', 'limits': limits},
                'params': json.loads(json.dumps(getattr(booster, 'params', {}), default=str)),
                'limitations': ['LightGBM zero_as_missing routing is not yet supported.']}
    parity = {'status': 'not_verified', 'rows': 0, 'atol': atol, 'rtol': rtol,
              'message': 'Numerical verification was not performed.'}
    if data is not None:
        if not isinstance(data, pl.DataFrame):
            raise TypeError('Parity data must be a Polars DataFrame of numerical booster inputs.')
        if not data.height:
            raise ValueError('Parity data must contain at least one row')
        predictors = data.select(dump['feature_names'])
        # Keep booster input types explicit; nulls become NaNs for reference scoring.
        for column in predictors:
            if not column.dtype.is_numeric():
                raise TypeError(f'Parity predictor {column.name!r} must contain numerical booster codes/values')
        reference = booster.predict(predictors.to_numpy())
        details = model.predict_diagnostics(predictors)
        failed = []
        max_absolute = 0.
        max_relative = 0.
        unmatched = 0
        nonfinite = 0
        for row, (expected, actual, status) in enumerate(zip(
                reference, details['predictions'], details['status'])):
            expected = float(expected)
            unmatched += status == 'unmatched'
            if actual is None or not math.isfinite(expected):
                nonfinite += status == 'nonfinite' or not math.isfinite(expected)
                failed.append(row)
                continue
            error = abs(actual - expected)
            max_absolute = max(max_absolute, error)
            if expected != 0:
                max_relative = max(max_relative, error / abs(expected))
            if error > atol + rtol * abs(expected):
                failed.append(row)
        parity = {'status': 'failed' if failed else 'passed', 'rows': data.height,
                  'atol': atol, 'rtol': rtol, 'max_absolute_error': max_absolute,
                  'max_relative_error_nonzero_reference': max_relative,
                  'unmatched_rows': unmatched, 'nonfinite_rows': nonfinite,
                  'failed_rows': failed,
                  'message': 'Evidence applies only to the supplied rows.'}
    return ConversionResult(model, metadata, parity)
