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
    fit_options={'solver': 'auto'},
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
convergence summary, coefficient estimates and classical inference, findings,
and Markdown report. Supplied validation data produces aggregate metrics,
calibration and factor actual/expected exhibits. Fold membership and identifiers
are recorded when supplied. No training or validation observations are stored;
aggregate exhibits can still contain sensitive category labels.

Fitting options, dataset identifiers, units, preprocessing descriptions and lineage
are **caller-supplied context**. They are not automatically recovered or verified.
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
separately; automatic composition-graph evidence, executable preprocessing, automatic
fit-option capture and exact environment reconstruction remain outside this format.
