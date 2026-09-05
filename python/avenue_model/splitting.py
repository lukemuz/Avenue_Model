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
    """Reusable row membership bound to one ordered dataset and Polars version.

    Normally obtained from `SplitSpec.split`. Training and validation indices are
    zero-based, unique, nonempty and disjoint. The fingerprint covers every input
    column, its dtype and row order, including columns not used as predictors.

    Attributes:
        name: Human-readable fold label.
        train_rows: Positional indices used for fitting.
        validation_rows: Positional indices held out from fitting.
        fingerprint: Fingerprint of the original complete input frame.
        specification: Declarative split settings retained for review.
        polars_version: Version used to fingerprint the input frame.
        schema_version: Persisted membership format, currently 1.
    """
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
        """SHA-256 identity of the serialized membership and its metadata."""
        return hashlib.sha256(self.to_json().encode()).hexdigest()

    def frames(self, data):
        """Return `(training, validation)` frames in retained positional order.

        Args:
            data: The original complete Polars frame passed to `SplitSpec.split`.

        Raises:
            ValueError: The data values, order, schema or Polars version differ,
                or a retained position exceeds the frame's row count.

        Pass the full original frame, not an already sliced training or holdout
        frame. Even changes to unused columns invalidate the fingerprint.
        """
        if self.polars_version != pl.__version__ or _fingerprint(data) != self.fingerprint:
            raise ValueError('Fold belongs to different data, row order, schema or Polars version; resolve the split again')
        if max(self.train_rows + self.validation_rows) >= data.height:
            raise ValueError('Fold index exceeds the dataset row count')
        return data[list(self.train_rows)], data[list(self.validation_rows)]

    def fit(self, plan, data, target, options=None):
        """Fit a Plan on the retained training rows and return a `FoldFit`.

        Args:
            plan: A Plan whose data-dependent categories, bins and knots should
                resolve during fitting on this fold's training data.
            data: The original full Polars frame bound to this fold.
            target: Training response column name.
            options: Optional fitting options accepted by `Plan.fit`.

        Validation rows are not passed to `Plan.fit`. Explicit breaks, supplied
        tables, fixed priors and preprocessing already chosen by the caller remain
        fixed; this method does not undo upstream information leakage or rerun model
        selection. The returned model can be nonconverged: inspect its diagnostics
        before treating it as a successful fit.
        """
        train, _ = self.frames(data)
        model = plan.fit(train, target) if options is None else plan.fit(train, target, options)
        return FoldFit(model=model, fold=self)

    def to_json(self):
        """Serialize membership and metadata, without storing observations or a model."""
        return json.dumps(asdict(self), sort_keys=True, allow_nan=False)

    @classmethod
    def from_json(cls, text):
        """Restore a serialized fold; `frames` checks its binding when data are supplied.

        Unsupported schema versions and invalid/overlapping memberships are rejected.
        Loading does not regenerate a split from the saved specification.
        """
        values = json.loads(text)
        values['train_rows'] = tuple(values['train_rows'])
        values['validation_rows'] = tuple(values['validation_rows'])
        return cls(**values)


@dataclass
class FoldFit:
    """A fitted model paired with the fold that supplied its training population.

    Attributes:
        model: The result of `Plan.fit`, including its convergence diagnostics.
        fold: The original membership and data binding.
    """
    model: Any
    fold: Fold

    def validate(self, data, **options):
        """Validate on exactly the retained holdout rows of the original full frame.

        `data` must pass `Fold.frames`' fingerprint/version check. Keyword options
        are forwarded to the fitted model's `validate`; no new split or refit is
        performed. Validation/scoring requirements and unmatched-level errors remain
        those of the model.
        """
        _, validation = self.fold.frames(data)
        return self.model.validate(validation, **options)


@dataclass(frozen=True)
class SplitSpec:
    """A recipe for row/group K-fold membership or one out-of-time holdout.

    Use `random`, `grouped` or `out_of_time`, then call `split` on the complete
    modeling frame. Splitting does not fit a model or preprocess predictors.

    Attributes:
        kind: `random`, `grouped` or `time`.
        n_splits: Number of random/grouped validation folds; unused for time splits.
        seed: Integer shuffle seed for random/grouped splits; unused for time splits.
        column: Group identifier or time column, when required by the split kind.
        cutoff: Boundary for a time split, comparable with the time column's dtype.
    """
    kind: str
    n_splits: int = 5
    seed: int = 0
    column: str | None = None
    cutoff: Any = None

    @classmethod
    def random(cls, n_splits=5, seed=0):
        """Create shuffled row K-fold membership with balanced row counts.

        `n_splits` must be an integer from 2 through the input row count; `seed`
        must be an integer. These are checked by `split`. Each row appears in
        validation exactly once. Rows are treated as separate sampling units:
        repeated policies can cross folds. No target stratification is applied.
        """
        return cls('random', n_splits=n_splits, seed=seed)

    @classmethod
    def grouped(cls, column, n_splits=5, seed=0):
        """Keep every row of a group together while balancing fold row counts.

        Args:
            column: Column of non-null, finite, hashable group identifiers.
            n_splits: Integer from 2 through the number of distinct groups.
            seed: Integer controlling shuffled group order before balancing.

        Large groups are assigned first to the currently smallest fold. The seed
        breaks ties between equal-sized groups; changing it need not change groups
        of different sizes. Unequal groups need not yield equal fold sizes. This
        prevents shared identifiers crossing folds, not temporal leakage or all
        possible dependence. There is no target or exposure stratification.
        """
        return cls('grouped', n_splits=n_splits, seed=seed, column=column)

    @classmethod
    def out_of_time(cls, column, cutoff):
        """Create one split: train strictly before cutoff, validate at/after it.

        `column` must have a dtype comparable with `cutoff`, without null or NaN
        values. Both populations must be nonempty. No shuffling, rolling windows,
        embargo, group isolation or automatic date/timezone conversion is applied.
        A policy spanning the cutoff can occur in both populations.
        """
        return cls('time', column=column, cutoff=cutoff)

    def split(self, data):
        """Resolve the recipe into a tuple of data-bound `Fold` objects.

        Args:
            data: Complete Polars modeling frame with at least two rows.

        Returns:
            `n_splits` folds for random/grouped recipes, or one fold for a time
            recipe. Each partition retains the original input order. Time splits
            therefore require unpacking a one-element tuple, for example
            `fold, = SplitSpec.out_of_time('year', 2022).split(data)`.

        Raises:
            ValueError: Invalid split settings, missing/nonfinite group labels,
                missing time values, too few sampling units, or an empty partition.

        The complete frame is fingerprinted. Resolve splits after any intended
        row filtering and deterministic preparation. Resolve learned predictor
        transformations on training rows using `Fold.fit` or an equivalent explicit
        training-only workflow.
        """
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
