# Reproducible evaluation populations

```python
from avenue_model import Plan, SplitSpec

plan = Plan.frequency("exposure").banded("age", quantile=5).categorical("region")
folds = SplitSpec.grouped("policy_id", n_splits=5, seed=42).split(data)
for fold in folds:
    fitted = fold.fit(plan, data, "frequency")
    print(fold.split_id, fitted.model.converged)
    print(fitted.validate(data).ae_ratio)
    # Retains membership, specification, seed and dataset fingerprint, not raw data.
    saved_membership = fold.to_json()
```

`SplitSpec.random(n_splits=5, seed=42)` partitions rows into validation folds. Each
row appears in exactly one validation fold. `grouped(column, ...)` keeps each group
entirely on one side of each split. Group assignment balances row counts greedily,
with seeded ordering of equally sized groups. It does not stratify responses.

`SplitSpec.out_of_time("period", cutoff)` creates one holdout: training strictly
before the cutoff, validation at or after it. Use a cutoff comparable to the column's
dtype, such as a `datetime.date` for a Date column. Ties remain together. Neither
training nor validation may be empty. Null time or group values are rejected.

`Fold.frames(data)` returns the two frames in original row order. `Fold.fit()` passes
only the training frame to `Plan.fit`, so learned bands, categories and reference
levels resolve there. The result retains both the model and fold; `validate(data)`
evaluates exactly that fold's holdout. Inspect convergence and validation findings;
a split does not make a failed fit acceptable. Holdout-only categories remain unseen
unless the plan deliberately provides a supported wildcard or prior structure.

`Fold.from_json(text)` restores membership. A content/schema/order fingerprint and
the Polars version prevent accidental reuse on a changed or reordered dataset.
Re-resolve the specification after a Polars upgrade. Unknown future schema versions,
overlapping membership and invalid indices fail explicitly. `split_id` identifies
the complete serialized membership and specification. Persist the source dataset's
own stable business/version identifier alongside this evidence when available.

Externally prepared train/validation datasets remain supported directly:
`plan.fit(training, target).validate(validation)`. There is no mandatory random
split. These helpers do not infer temporal gaps, renewal leakage, loss development,
trend or coverage changes; encode those study assumptions in the input populations.
Use [common model comparison](MODEL_COMPARISON.md) for candidate predictions on the
same held-out frame, and [GLM grid selection](GLM_SELECTION.md) for fitting named
specifications across retained folds. Its selection loss uses a common prespecified
metric; a selected model still needs a separate final holdout.

Random splits do not keep repeated policies together. Grouped splits keep the selected
identifier together but do not enforce chronological order. A time split can place the
same policy on both sides of its cutoff. Choose membership to match the intended
generalization question; none of these helpers automatically detects every dependence
or implements an embargo. Greedy grouped balancing targets row counts, not total
exposure, and changing the seed may leave assignments of differently sized groups
unchanged.

Pass the original **full frame** to `Fold.frames`, `Fold.fit` and `FoldFit.validate`.
They select their own populations. Changing even an unused column invalidates the
frame fingerprint. Fold JSON preserves indices and metadata, not observations or a
recipe that is automatically rerun when loaded.

Only transformations still unresolved when `Plan.fit` runs are learned within the
training fold. Explicit breaks, precomputed predictors and supplied prior tables are
kept as provided. A fold helper cannot undo information leakage from preprocessing or
model choices already made using the full dataset.
