# Numeric bounds in review tables

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
A generated interval label that collides with an existing predictor raises an
actionable error instead of replacing that predictor in the exhibit.

Splines retain their separate semantics: coefficient rows are knot values, while
validation rows show spline support intervals. They are not labeled as constant bands.
Pure categorical and intercept tables need no numeric interval labels.

These are derived review columns. Editable workbook files keep the original scoring
schema; loading a workbook regenerates the labels. Tests check multi-axis matching at
boundaries and in tails, unsupported geometry, name collisions, A/E reconciliation,
coefficient intervals and unchanged workbook predictions. The auto-pricing example
exports these labels in its factor review files.
