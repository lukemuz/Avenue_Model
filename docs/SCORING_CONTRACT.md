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

An empty term list now means an intercept-only model. For example,
`Plan.frequency("exposure").fit(data, "frequency")` estimates the exposure-weighted
portfolio frequency and exports an ordinary one-table workbook.

## Migration from 0.1.0 evaluation behavior

Offset models previously returned rates from `predict()` while validation used counts.
Remove manual multiplication by exposure after predicting with an offset model.
Frequency presets retain their existing rate-plus-weight convention; multiply their
rates by exposure when computing counts. Severity and rate quote frames no longer
need dummy training-weight columns.

This is the first scoring-contract increment. Explicit rate/count convenience methods,
response-unit metadata and strict unmatched-row diagnostics remain planned. Training
and validation exposure checks also need to be unified with scoring. Until those
changes land, inspect validation findings and avoid scoring unmatched predictors.
