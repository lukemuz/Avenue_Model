# Analytical bundles

`save_bundle` preserves the evidence behind an individual fitted model alongside
an editable scoring workbook. It writes a new directory and refuses to overwrite
an existing one.

```python
from avenue_model import save_bundle, load_bundle

bundle = save_bundle(
    model, 'review/model-v1',
    training_id='warehouse-extract-2026-09-01',
    validation_data=holdout, validation_id='2025-holdout',
    fold=fold, unit='claims/exposure',
    preprocessing={'source': 'pricing-preparation-v1'},
)
# Review/edit the CSV files under review/model-v1/scoring, then reload.
bundle = load_bundle('review/model-v1')
print(bundle.edited, bundle.changed_files)
quotes = bundle.model.predict(raw_quotes)
original_quotes = bundle.source_model.predict(raw_quotes)
plan = bundle.source_plan  # None for converted/workbook-derived source models
```

The source snapshot retains the plan, schema/category mappings, resolved terms,
convergence summary, effective fitting options, actual solver, coefficient estimates
and the selected inference method, findings,
and Markdown report. Supplied validation data produces aggregate metrics,
calibration and factor actual/expected exhibits. Fold membership and identifiers
are recorded when supplied. No training or validation observations are stored;
aggregate exhibits can still contain sensitive category labels.

`effective_fit_options` automatically records every effective `GLMOptions` argument,
including defaults and the Plan-owned Tweedie power. `solver_used` records the actual
global/table path when `solver='auto'` was requested. These come from the original
fit and are also exposed as `model.fit_options` and `model.solver_used`. Loaded or
converted scorers have empty options and no actual solver record. Earlier version-1
bundles may lack these additive fields; their absence means unknown.

Dataset identifiers, units, preprocessing descriptions and lineage remain
**caller-supplied context**. The optional legacy `fit_options=` argument is retained
under `caller_context` as an annotation; it cannot replace the automatically captured
source configuration, even if the two disagree.
Likewise supplying a fold does not prove the model was fitted on that fold; use
`fold.fit` to enforce membership during fitting. Missing context remains null.
Preprocessing is declarative JSON metadata, not an executable transformation.
Nonfinite numerical evidence uses `{"nonfinite": "nan"}` (or `inf`/`-inf`).

`source/workbook.json` and `source/evidence.json` are checked against SHA-256 hashes
in the versioned manifest. Changes to source files cause loading to fail. These
checks detect accidental modification, not authenticity against an attacker who
can also replace the manifest. Unknown schema versions are rejected.

The editable `scoring/` directory uses exact factor scale. Any byte change there,
including formatting, is conservatively flagged; `changed_files` is a file list,
not a semantic coefficient attribution. Use `compare_changes` with the source and
current models for policy and factor reconciliation, then validate the current
model on fresh data. Neither loaded scorer inherits the original fit diagnostics:
those remain explicitly named `source_evidence`. Saving an already loaded model
cannot recreate lost fitting evidence.

Version 1 supports individual `FittedModel` objects. Bundle composition components
separately; automatic composition-graph evidence, executable preprocessing and exact
environment reconstruction remain outside this format.

To refit the source plan with its recorded effective options, use the original
training population and response definition:

```python
from avenue_model import GLMOptions
refitted = bundle.source_plan.fit(
    training, target,
    GLMOptions(**bundle.source_evidence['effective_fit_options']),
)
```

This is a new fit; changing data or software versions can change the result. The
bundle's original convergence/inference remains source evidence, not a guarantee
about the new fit. A new fit must be checked independently.
