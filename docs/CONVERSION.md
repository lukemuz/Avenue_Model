# Booster conversion and numerical evidence

```python
from avenue_model import from_booster

# quote_predictors is a Polars frame, using the numeric inputs/codes of the booster.
result = from_booster(booster, quote_predictors, consolidation="max")
print(result.metadata)  # module/version, objective, parameters, dump fingerprint
print(result.parity)    # status, errors, failed rows, unmatched/nonfinite counts
if result.parity["status"] != "passed":
    raise ValueError("Review conversion discrepancies before using this model")
result.save("converted_plan")  # editable workbook plus conversion.json evidence
model = result.model
```

`from_booster(booster)` also works without a dataset. Its parity status is
`not_verified` and explicitly says numerical verification was not performed.
`FittedModel.from_lgbm_json()` remains available as a lower-level constructor; it
performs no parity check. A successful report establishes agreement on the supplied
inputs only. Include representative business and threshold/rare-category probes.

The report uses `abs(actual - expected) <= atol + rtol * abs(expected)`, with both
tolerances defaulting to 1e-12. Relative error is summarized over nonzero reference
means. Failed rows are zero-based positions in the supplied frame. Conversion metadata
and parity are saved separately from the editable scoring workbook; the evidence
belongs to the original artifact, not to later manual edits. The saved dump fingerprint
identifies the source model without storing training data.

## Explicit limits on the final artifact

CV table counts are a selection proxy. To enforce a resource limit on the final
conversion, request it explicitly:

```python
result = from_booster(booster, quote_predictors, consolidation="max",
                      resource_limits={"tables": 6, "total_rows": 20000,
                                       "largest_interaction_order": 2})
```

Accepted keys are `tables`, `total_rows`, `largest_table` and
`largest_interaction_order`. Values must be nonnegative Python integers (not booleans
or floats); limits are inclusive. Unspecified quantities are unconstrained. The
intercept counts as a table and a row, with interaction order zero. Row counts include
explicit missing/default routes. Interaction order counts the predictor columns of
the stored table, not a fitted statistical rank.

The check uses the actual artifact **after** the requested consolidation mode. An
exceeded limit raises `ValueError` showing observed counts and requested bounds;
it never silently truncates, simplifies or substitutes another model. Passing limits
does not establish conversion parity, fit convergence or statistical adequacy. With
no requested limits, conversion has no resource cap.

`metadata['complexity']` records all four measurements. `metadata['resource_check']`
records a snapshot of the requested limits and `passed` or `not_requested` status.
These refer to the original converted artifact; subsequent structural edits require
a fresh check. They are saved with the other conversion evidence.

These are final-artifact structural constraints, **not peak-memory or time limits**:
conversion must complete to measure the artifact. The low-level
`FittedModel.from_lgbm_json` entry point does not enforce these optional limits.

Supported objectives are regression/gaussian, Poisson, Gamma, Tweedie, and binary
with unit sigmoid. Multiclass, averaged ensembles, linear leaves and unsupported
split operators are rejected. The binary objective's serialized options preserve
its logit link. Inputs for this entry point are numeric booster values/codes;
`with_categories` can attach external labels to the resulting model.

Numerical null/NaN routes are represented by explicit missing-only rows, whose numeric
bound is `NaN`. Finite inputs cannot match those rows. LightGBM `missing_type="NaN"`
uses its declared default direction; `missing_type="None"` treats missing numerical
input as zero, as the booster does. Categorical nulls in Int32 input columns follow
the complement/wildcard route. Integer codes remain the boundary requirement for
categorical boosters.

`zero_as_missing` is explicitly rejected pending support for its separate near-zero
routing rule. Constant-only boosters produce an intercept artifact in both modes.
Complete declarative preprocessing remains pending. See IMPLEMENTATION_STATUS.md
for the current acceptance record.

New workbooks use format version 2 so older readers cannot silently interpret a
missing-only bound as an ordinary unconstrained bound. This build continues to read
version-1 artifacts. JSON encodes the bound as the string `"NaN"`; CSV writes `NaN`.
These are explicit rating-table matching rows, not nonfinite prediction values.
