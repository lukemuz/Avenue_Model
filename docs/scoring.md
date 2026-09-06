# Scoring and model review

Use the same rating tables to quote, inspect individual contributions and review edits. Predictions retain the response and exposure convention declared when fitting.

[Response and exposure](#response-and-exposure) · [Pandas input](#pandas-input) · [Numeric band bounds](#numeric-band-bounds) · [Quote explanations](#quote-explanations)

## Response and exposure

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

### Unsupported rows in a penalized refit

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

### Invalid quote rows

`predict()` raises on unmatched rows or nonfinite results. For batch review, use
`predict_diagnostics(frame)`: it returns `row` (zero-based input position),
`predictions`, `status` (`ok`, `unmatched`, or `nonfinite`) and `unmatched_tables`.
Failed predictions are Polars nulls, not NaNs or invented neutral factors. Valid
wildcard table rows remain valid matches. Missing predictor columns and invalid
exposures raise even in diagnostic mode. Validation continues to report unmatched
rows and mark the result unusable; it does not silently accept the reduced population.

### Explicit Poisson predictions and input schema

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

## Pandas input

Install `avenue_model[pandas]`, then convert once at the boundary:

```python
from avenue_model import Plan, from_pandas

training = from_pandas(pandas_training, index_column="source_row")
model = Plan.frequency("exposure").categorical("region").fit(training, "frequency")
quotes = from_pandas(pandas_quotes)
predictions = model.predict(quotes)
```

A pandas category's actual labels are retained, not its positional `.cat.codes`.
String, integer and boolean category labels are supported. Reordering the category
list does not change label identity. Unused levels do not become training observations
or enter Avenue's fitted encoding. Nulls, `pd.NA` and floating NaNs become Polars nulls;
nullable integer/boolean columns preserve their values. The input frame is not mutated.

The adapter excludes the pandas index by default. Supply `index_column` to preserve a
single index in a new named column. Reset a MultiIndex into explicit columns yourself.
Output predictions are positional; retain this source identifier when joining back to
other datasets. Column names must be unique strings.

Object columns must contain strings/nulls; cast numerical/date object columns to a
concrete dtype before conversion. Mixed-type and float-valued category labels require
an explicit encoding rather than automatic stringification. That avoids collisions
such as integer `1` and string `"1"`. General Avenue term requirements still apply:
for example a float driver needs a numeric term, not an integer-category interpretation.
Polars remains native; existing methods do not silently convert arbitrary objects.

## Numeric band bounds

`model.rating_tables_by_name()`, `coefficient_intervals(model).tables`, and validation
A/E tables share numeric interval labels. For a banded predictor `age`, the additional
columns are `age_Lower`, `age_Upper`, `age_Lower_Inclusive` and `age_Upper_Inclusive`.
The original threshold, coefficient, reference/status and support columns remain.

For breaks `[25, 50]`, the displayed intervals are `(-inf, 25]`, `(25, 50]` and
`(50, inf)`. Finite upper endpoints are inclusive; lower endpoints are exclusive.
An infinite endpoint denotes unboundedness and its inclusion flag is false. These
labels describe finite quote values. Missing/default routes require separate review.

`Band_Interval_Status="ordered_grid"` means the numeric and categorical coordinates
form a complete grid with rows in an order that proves these bounds under Avenue's
first-match rule. Numeric interactions receive one set of bounds per predictor.
No category encoding is mistaken for a numeric band.

Irregular, incomplete, duplicate, unordered, wildcard or missing-route tables can
describe regions that these simple rectangles cannot safely express. They retain
their thresholds and have `Band_Interval_Status="requires_match_review"`, without
invented interval bounds. Inspect their matching behavior and quote explanations.
A generated review column that collides with an existing predictor raises an
actionable error instead of replacing that predictor in the exhibit. This includes
numeric bounds and band status, category labels such as `region_Level`, coefficient
and interval fields, and A/E fields such as `N` and `Exposure`. Rename the conflicting
predictor before fitting to produce the affected exhibit. Scoring and workbook
export continue to use the original predictor names and values.

Splines retain their separate semantics: coefficient rows are knot values, while
validation rows show spline support intervals. They are not labeled as constant bands.
Pure categorical and intercept tables need no numeric interval labels.

These are derived review columns. Editable workbook files keep the original scoring
schema; loading a workbook regenerates the labels. Tests check multi-axis matching at
boundaries and in tails, unsupported geometry, name collisions, A/E reconciliation,
coefficient intervals and unchanged workbook predictions. Inspect `model.rating_tables_by_name()` to access these review labels.

## Quote explanations

`model.explain(quotes)` returns named `summary` and `contributions` Polars tables.
The summary retains original row position, linear predictor, link and final prediction.
The long contribution table names each term, its kind (intercept, table or exposure),
matched table row, coefficient and log-link multiplier. Table row positions are
zero-based within the current artifact and can be looked up in
`model.rating_tables_by_name()`.

Sum coefficients and apply the inverse link to reconstruct each response mean. Under
a log link, multiplying the base-rate and factor multipliers also reconstructs it.
For identity/logit links multipliers are null: effects add on the linear-predictor
scale. Exposure appears only when the model's declared mean uses an offset. Zero
exposure contributes negative infinity on that scale and a zero log-link multiplier.
Scoring errors remain strict; explanations never invent contributions for unmatched
rows. These are exact model mechanics, not causal attributions.

```python
changes = quotes.select("policy_id").with_columns(
    original.predict(quotes).to_series().alias("before"),
    edited.predict(quotes).to_series().alias("after"),
).with_columns((pl.col("after") - pl.col("before")).alias("change"))
```

Compare means in the same physical units and exposure convention. Aggregate with
Polars using exposure weights for rates, or sum totals directly. Inspect each
model's contributions when term-level explanations are needed.

An edited workbook is a new scoring artifact; it does not inherit standard errors
for its edited coefficients. Obtain fresh validation with `edited.report(data)`.
The auto example verifies a known 5% edit and saves policy movements and explanations.
