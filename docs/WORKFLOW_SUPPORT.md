# Supported fundamentals

Avenue centers on `Plan` for fitting, `FittedModel` for prediction and inspection,
and `Workbook` for editable delivery. Data preparation and study orchestration
belong in the calling application.

| Need | Core interface |
|---|---|
| Frequency, severity and pure premium | `Plan.frequency`, `Plan.severity`, `Plan.pure_premium` |
| Banded and categorical factors, interactions | Plan term specifications |
| Continuous numeric effects | `Plan.spline`; see [scope](SPLINES.md) |
| Monotonic bands | [Monotonic effects](MONOTONIC_EFFECTS.md) |
| Penalties, fixed priors and solver controls | `GLMOptions`, `Plan.given`, `Plan.offset_model` |
| Quote means, rates and exposure | [Scoring contract](SCORING_CONTRACT.md) |
| Diagnostics and exact factor contributions | `model.report`, `model.explain` |
| Coefficient intervals and joint tests | [Intervals](COEFFICIENT_INTERVALS.md), [term tests](TERM_TESTS.md) |
| Model-based, HC0 and one-way CR0 covariance | [HC0](ROBUST_INFERENCE.md), [CR0](CLUSTER_INFERENCE.md) |
| Editable scoring artifact | Workbook JSON and CSV directories |
| Booster conversion with numerical parity | [Conversion](CONVERSION.md) |
| Existing LightGBM tuning and structural counts | [Tuning](lightgbm.md) |
| Explicit pandas input conversion | [Pandas adapter](PANDAS.md) |

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
