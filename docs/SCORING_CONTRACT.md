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

`predict()` now raises on unmatched rows or nonfinite results. For batch review, use
`predict_diagnostics(frame)`: it returns `row` (zero-based input position),
`predictions`, `status` (`ok`, `unmatched`, or `nonfinite`) and `unmatched_tables`.
Failed predictions are Polars nulls, not NaNs or invented neutral factors. Valid
wildcard table rows remain valid matches. Missing predictor columns and invalid
exposures raise even in diagnostic mode. Validation continues to report unmatched
rows and mark the result unusable; it does not silently accept the reduced population.

Explicit rate/count convenience methods and response-unit metadata remain planned.
Training and validation exposure checks also need to be unified with scoring.
