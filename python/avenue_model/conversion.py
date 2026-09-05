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


def from_booster(booster, data=None, *, consolidation='max', atol=1e-12, rtol=1e-12):
    """Convert a Booster, optionally checking means against a Polars predictor frame.

    Returns a ConversionResult with model, metadata and parity. Failed parity is
    retained as evidence (``parity['status'] == 'failed'``), not silently accepted.
    Without data the status is ``not_verified``. Input column order is resolved from
    the booster feature names. Categorical booster inputs must be integer codes;
    use ``model.with_categories`` to attach an external code-to-label mapping.
    Numerical missing/default routing is still unsupported and missing test inputs
    are refused until it is represented in the portable rating tables.
    """
    if any(not math.isfinite(v) or v < 0 for v in (atol, rtol)):
        raise ValueError('atol and rtol must be finite and nonnegative')
    dump = booster.dump_model()
    serialized = json.dumps(dump, sort_keys=True, separators=(',', ':'))
    model = FittedModel.from_lgbm_json(serialized, consolidation=consolidation)
    module_name = type(booster).__module__.split('.')[0]
    module = importlib.import_module(module_name)
    metadata = {'schema_version': 1, 'module': module_name,
                'version': getattr(module, '__version__', None),
                'objective': dump.get('objective'), 'consolidation': consolidation,
                'feature_names': dump['feature_names'],
                'tree_count': len(dump['tree_info']),
                'dump_sha256': hashlib.sha256(serialized.encode()).hexdigest(),
                'params': json.loads(json.dumps(getattr(booster, 'params', {}), default=str)),
                'limitations': ['Numerical missing/default routing is not yet supported.']}
    parity = {'status': 'not_verified', 'rows': 0, 'atol': atol, 'rtol': rtol,
              'message': 'Numerical verification was not performed.'}
    if data is not None:
        if not isinstance(data, pl.DataFrame):
            raise TypeError('Parity data must be a Polars DataFrame of numerical booster inputs.')
        if not data.height:
            raise ValueError('Parity data must contain at least one row')
        predictors = data.select(dump['feature_names'])
        # Make unsupported missing routes explicit instead of manufacturing parity.
        for column in predictors:
            if not column.dtype.is_numeric():
                raise TypeError(f'Parity predictor {column.name!r} must contain numerical booster codes/values')
            if column.null_count() or column.cast(pl.Float64).is_nan().any():
                raise ValueError(f'Missing/default routing is not yet supported: {column.name!r}')
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
