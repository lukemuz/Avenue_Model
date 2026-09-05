# Analytical bundles

`save_bundle` preserves the evidence behind an individual fitted model or nested
composition alongside editable scoring workbooks. It writes a new directory and refuses to overwrite
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

Bundles also retain [whole-term tests](TERM_TESTS.md) under `term_tests`, including
their covariance and conditional/asymptotic interpretation. Unavailable models record
a reason; unavailable individual terms remain in the joint table. These are default
source-covariance tests, not post-selection evidence. Older version-1 bundles may lack
this additive field. Edited/loaded scorers do not inherit the ability to run tests.

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

Version 1 supports individual `FittedModel` objects. Version 2 adds nested analytical
composition graphs while retaining version-1 bundles at their leaves. Executable
preprocessing and exact environment reconstruction remain outside these formats.

## Analytical composition graphs

```python
product = frequency_severity(frequency, severity)
bundle = save_bundle(
    product, "frequency_severity_bundle",
    validation_data=policy_holdout, validation_id="policies-2022",
    validation_options={"target": "pure_premium", "metric": "tweedie", "weight": "exposure"},
    component_context={
        "frequency": {"training_id": "policies-pre-2022", "validation_data": policy_holdout},
        "severity": {"training_id": "claims-pre-2022", "validation_data": claim_holdout},
    },
)
frequency_evidence = bundle.components["frequency"].source_evidence
```

Import `frequency_severity` from `avenue_model` for this example. Graph validation
requires its own explicit target and metric, because composition has no inherited
likelihood. The root records aggregate comparison summaries, segments and metadata;
raw input frames, row-level prediction vectors and discrimination-curve points are
not persisted. Children can use distinct response/weight columns and validation
populations. Root context is not automatically copied into child evidence.

`component_context` maps existing child names to keyword arguments for `save_bundle`.
For a nested peril sum, a child's options may themselves contain `component_context`.
Unknown child names, unsupported leaf types and cyclic graphs are rejected. Shared
components can occur in more than one branch; each occurrence is stored independently.
The current graph structure and declared units are checked before saving.

The returned `ComposedBundle` exposes `model`, `source_model`, `source_evidence`,
named child `components`, `changed_files` and `edited`. Both loaded graphs are
scoring-only: recorded convergence and inference belong to original child evidence.
There is **no joint composition covariance or uncertainty interval** implied by
bundling component standard errors. Saving a loaded scorer again cannot reconstruct
its original fit evidence; keep the original bundle for that evidence.

Component bundles live under `components/component_N`, recursively. Their `scoring/`
workbooks remain editable. The parent protects source evidence and child bundle
manifests; changing them causes an integrity error even if a child's checksums are
rewritten. Scoring edits instead appear as root-relative `changed_files`. Source
predictions and the recorded original validation remain accessible separately from
the changed graph. Root operation/unit/child declarations are source evidence, not an
editable graph editor. Revalidate the current graph after changes.

Saving uses a temporary staging directory and publishes the new directory only after
every component is written successfully. A failed child leaves no partial destination.
As with individual bundles, hashes protect against accidental modification, not an
attacker who can rewrite the entire root manifest and all its evidence.

To refit an individual model's source plan with its recorded effective options, use
the original training population and response definition. For a `ComposedBundle`,
first select the relevant leaf bundle (for example, `bundle.components['frequency']`);
there is no single fitting Plan for the entire graph:

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
