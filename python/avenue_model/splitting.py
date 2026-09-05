"""Reproducible train/validation membership and fold-local Plan fitting."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import random
from typing import Any

import polars as pl


def _fingerprint(data):
    if not isinstance(data, pl.DataFrame):
        raise TypeError('Split data must be a Polars DataFrame')
    digest = hashlib.sha256(str(data.schema).encode())
    for value in data.hash_rows(seed=0):
        digest.update(value.to_bytes(8, 'little'))
    return digest.hexdigest()


@dataclass(frozen=True)
class Fold:
    """Zero-based membership bound to the original ordered dataset fingerprint."""
    name: str
    train_rows: tuple[int, ...]
    validation_rows: tuple[int, ...]
    fingerprint: str
    specification: dict
    polars_version: str
    schema_version: int = 1

    def __post_init__(self):
        if self.schema_version != 1:
            raise ValueError('Unsupported fold schema version')
        for rows in (self.train_rows, self.validation_rows):
            if not rows or any(type(i) is not int or i < 0 for i in rows) or len(set(rows)) != len(rows):
                raise ValueError('Fold indices must be nonempty, unique nonnegative integers')
        if set(self.train_rows) & set(self.validation_rows):
            raise ValueError('Training and validation rows overlap')

    @property
    def split_id(self):
        return hashlib.sha256(self.to_json().encode()).hexdigest()

    def frames(self, data):
        if self.polars_version != pl.__version__ or _fingerprint(data) != self.fingerprint:
            raise ValueError('Fold belongs to different data, row order, schema or Polars version; resolve the split again')
        if max(self.train_rows + self.validation_rows) >= data.height:
            raise ValueError('Fold index exceeds the dataset row count')
        return data[list(self.train_rows)], data[list(self.validation_rows)]

    def fit(self, plan, data, target, options=None):
        """Fit an unresolved Plan on training rows only; holdout never resolves terms."""
        train, _ = self.frames(data)
        model = plan.fit(train, target) if options is None else plan.fit(train, target, options)
        return FoldFit(model=model, fold=self)

    def to_json(self):
        return json.dumps(asdict(self), sort_keys=True, allow_nan=False)

    @classmethod
    def from_json(cls, text):
        values = json.loads(text)
        values['train_rows'] = tuple(values['train_rows'])
        values['validation_rows'] = tuple(values['validation_rows'])
        return cls(**values)


@dataclass
class FoldFit:
    model: Any
    fold: Fold

    def validate(self, data, **options):
        """Validate on exactly the retained holdout rows."""
        _, validation = self.fold.frames(data)
        return self.model.validate(validation, **options)


@dataclass(frozen=True)
class SplitSpec:
    """Random/grouped K-fold or one out-of-time holdout; no automatic preprocessing."""
    kind: str
    n_splits: int = 5
    seed: int = 0
    column: str | None = None
    cutoff: Any = None

    @classmethod
    def random(cls, n_splits=5, seed=0):
        return cls('random', n_splits=n_splits, seed=seed)

    @classmethod
    def grouped(cls, column, n_splits=5, seed=0):
        return cls('grouped', n_splits=n_splits, seed=seed, column=column)

    @classmethod
    def out_of_time(cls, column, cutoff):
        """Train strictly before cutoff; validate at/after cutoff. No shuffling."""
        return cls('time', column=column, cutoff=cutoff)

    def split(self, data):
        fingerprint = _fingerprint(data)
        if data.height < 2:
            raise ValueError('Splitting requires at least two rows')
        spec = asdict(self)
        # Persist dates declaratively, retaining their type in the split specification.
        if hasattr(self.cutoff, 'isoformat'):
            spec['cutoff'] = self.cutoff.isoformat()
            spec['cutoff_type'] = type(self.cutoff).__name__
        if self.kind == 'time':
            if self.column is None or self.cutoff is None:
                raise ValueError('Out-of-time splitting requires a column and cutoff')
            values = data[self.column]
            if values.null_count() or (values.dtype.is_float() and values.is_nan().any()):
                raise ValueError('Time column contains missing values')
            mask = (values < self.cutoff).to_list()
            memberships = [(tuple(i for i, before in enumerate(mask) if before),
                            tuple(i for i, before in enumerate(mask) if not before))]
        elif self.kind in ('random', 'grouped'):
            if type(self.n_splits) is not int or self.n_splits < 2:
                raise ValueError('n_splits must be an integer of at least two')
            if type(self.seed) is not int:
                raise ValueError('seed must be an integer')
            groups = {}
            values = range(data.height) if self.kind == 'random' else data[self.column].to_list()
            for row, group in enumerate(values):
                if group is None or (isinstance(group, float) and not math.isfinite(group)):
                    raise ValueError('Group column contains missing/nonfinite values')
                groups.setdefault(group, []).append(row)
            if self.n_splits > len(groups):
                raise ValueError('n_splits exceeds the number of independent rows/groups')
            ordered = list(groups.values())
            random.Random(self.seed).shuffle(ordered)
            # Keep large groups intact while balancing observation counts. Stable sort
            # preserves seeded order among equal-sized groups.
            ordered.sort(key=len, reverse=True)
            buckets = [[] for _ in range(self.n_splits)]
            for group in ordered:
                min(buckets, key=len).extend(group)
            all_rows = set(range(data.height))
            memberships = [(tuple(sorted(all_rows - set(bucket))), tuple(sorted(bucket)))
                           for bucket in buckets]
        else:
            raise ValueError('Split kind must be random, grouped or time')
        folds = []
        for index, (train, validation) in enumerate(memberships):
            if not train or not validation:
                raise ValueError('Split must have nonempty training and validation populations')
            folds.append(Fold(f'{self.kind}_{index}', train, validation, fingerprint,
                              spec.copy(), pl.__version__))
        return tuple(folds)
