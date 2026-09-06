# Scoring response means

`FittedModel.predict(frame)` returns the declared response mean, with the fitted
category encoding. The target column is never a scoring input.

- `Plan.frequency("exposure")` fits a Poisson **rate** target with exposure weights.
  Supply claims / exposure as the training target. Predictions are rates; quoting
  requires rating predictors, without exposure or observed claims.
- `Plan.severity("claims")` fits a Gamma average severity with claim-count weights.
  Quotes require rating predictors, without observed claims.
- `Plan.pure_premium("exposure")` fits loss / exposure with exposure weights.
  Predictions are loss per exposure.
- `Plan("poisson", exposure="exposure", exposure_role="offset")` fits claim counts.
  Predictions are expected counts: the fitted rate times the supplied exposure.
  Exposure is required for prediction and is applied exactly once. Fractional and
  zero scoring exposures are supported; zero produces zero under the log link.
  Negative, null, NaN and infinite scoring exposures raise with the column and row.

Training and validation still require their response and weight/offset columns.
These scoring rules also apply to a workbook converted back to a model.

An empty term list means an intercept-only model. For example,
`Plan.frequency("exposure").fit(data, "frequency")` estimates the exposure-weighted
portfolio frequency and exports an ordinary one-table workbook.

## Unsupported rows in a penalized refit

An existing table may contain rows with zero training exposure, including explicit
missing-value routes imported from a booster. In an unlocked ordinary step table
with an active L1 or L2 penalty, these rows take the reference relativity (log factor
zero after base-level normalization). Their data contribution is zero, so this is
the penalty-only optimum. Both table and global solvers use this rule, including
paired ridge updates. Review still marks these rows `no_data`; the value is not a
data-supported estimate or an uncertainty statement.

This does not add a route for an unmatched quote: strict matching still applies.
Locked factors remain fixed. Unpenalized unused rows retain their starting factors
subject to normalization; monotonic and smooth terms have their own documented
extrapolation rules. Existing saved scoring workbooks retain their saved values;
the new rule applies when fitting again.

## Invalid quote rows

`predict()` raises on unmatched rows or nonfinite results. For batch review, use
`predict_diagnostics(frame)`: it returns `row` (zero-based input position),
`predictions`, `status` (`ok`, `unmatched`, or `nonfinite`) and `unmatched_tables`.
Failed predictions are Polars nulls, not NaNs or invented neutral factors. Valid
wildcard table rows remain valid matches. Missing predictor columns and invalid
exposures raise even in diagnostic mode. Validation continues to report unmatched
rows and mark the result unusable; it does not silently accept the reduced population.

## Explicit Poisson predictions and input schema

Both the frequency preset and a Poisson offset model support:

```python
rates = model.predict_rate(quotes)             # column: rate; exposure not required
counts = model.predict_count(quotes_with_exposure)  # column: expected_count
print(model.prediction_kind)                   # rate, count, or response
print(model.input_schema)
```

`predict_count()` uses the recorded exposure column and applies it exactly once.
`predict_rate()` evaluates the rate directly, including at zero exposure; it does
not divide a count by zero. These methods require a Poisson response with a recorded
target, exposure column and weight/offset convention. Severity, composed and
unspecified response means raise instead of guessing a count interpretation.
`prediction_kind` describes this limited recorded convention; `response` leaves
physical units unspecified. Physical units are defined by the caller.

`input_schema` contains predictor kinds and internal dtypes, encoded `(label, code)`
pairs where available, separate `prediction_columns` and `validation_columns`,
target, exposure, and prediction kind. A categorical predictor with a mapping accepts
those strings or their numerical codes. Internal dtypes describe the table matcher;
the boundary normalizes supported input types as usual. The schema reflects the
current artifact and survives workbook reload.

Negative, null, NaN and infinite exposure raise during fitting and validation
as well as count scoring. Weight-zero rows carry no fitting weight. Offset models
still require positive training exposure; zero scoring exposure yields zero counts.

Numeric nulls and NaNs do not match ordinary finite bands. Ordinary GLM tables
without an explicit missing-only bound report them as unmatched. Converted booster
tables may contain an explicit `NaN` bound encoding the booster's valid missing
route. These rows are preserved by format-2 CSV/JSON workbooks.
