# Avenue documentation

Avenue centers on `Plan` for fitting, `FittedModel` for prediction and inspection,
and `Workbook` for editable delivery. Data preparation and study orchestration
belong in the calling application.

Use Polars expressions for targets and adjustments, scikit-learn for splits and
metrics, and Python loops for candidate evaluation. Multiply frequency and severity
predictions or add peril predictions in consistent response units. These operations
do not require another model type or serialization format.

Executable examples: [auto](../examples/auto_pricing_study.py),
[homeowners perils](../examples/homeowners_perils.py),
[booster conversion/refit](../examples/booster_pricing_study.py), and
[continuous effects](../examples/smooth_pricing_study.py).

Workbooks preserve scoring, not a new fitting certificate. Keep the original model's
reports, `plan.to_json()`, `fit_options` and requested inference results in the
application's existing experiment records. Revalidate manually edited artifacts.

## Start here

The [repository quickstart](../README.md) fits and exports a model. For details, use:

- [Scoring and exposure conventions](SCORING_CONTRACT.md)
- [Interactions](HIERARCHICAL_INTERACTIONS.md), [monotonic bands](MONOTONIC_EFFECTS.md), and [splines](SPLINES.md)
- [Factor review bounds](BAND_REVIEW.md) and [quote explanations](EXPLANATIONS.md)
- [Coefficient intervals](COEFFICIENT_INTERVALS.md), [HC0](ROBUST_INFERENCE.md), [clustered covariance](CLUSTER_INFERENCE.md), and [joint tests](TERM_TESTS.md)
- [Booster conversion](CONVERSION.md), [LightGBM tuning](lightgbm.md), and [pandas input](PANDAS.md)

## Development and evaluation

[Evaluation instructions](../studies/README.md) cover the installed-wheel checks,
real-data comparisons, benchmark methodology and independent reference fixtures.
[API reference generation](API_REFERENCE.md) covers searchable documentation and CI.
Historical review plans and progress reports are not maintained as product guides.
