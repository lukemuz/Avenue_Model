# Avenue Model

**Fast, defensible insurance models built directly from rating tables.**

Avenue turns an ordinary dataframe into a model that remains understandable after it is
fitted. State the rating plan, inspect what it will do, fit it, validate it, and hand it
to someone as editable tables. The same table representation also converts LightGBM
models exactly, without changing their predictions.

- **The plan is data.** Levels, bands, interactions, reference levels and exposure
  treatment are explicit, serializable and reproducible.
- **Problems are found before fitting.** `check()` reports data faults, thin levels,
  unidentified terms and redundant tables together, with actionable messages.
- **The fitted model is an artifact.** Save it as JSON or readable CSV tables, edit it,
  load it again, validate it and compose it with other models.
- **The engine is fast.** Avenue fits on rating tables without materializing a dummy-coded
  observation-by-parameter matrix, and includes global and table-native solvers.
- **LightGBM becomes inspectable.** Tree ensembles convert exactly into the same rating
  table representation used by GLMs.

## Installation

From a source checkout:

```bash
pip install .
```

After the first package release, the registry installs will be:

```bash
pip install avenue-model
```

```toml
[dependencies]
avenue_model = "0.1.0"
```

Python requires 3.12 or newer. The compatible Python Polars version is installed with
the package.

## Quick start

Frequency is modelled as claims per unit exposure, with exposure as its prior weight.
That makes predictions and composition read naturally: frequency predicts claims per
exposure, severity predicts loss per claim, and their product is loss per exposure.

```python
from avenue_model import Plan

plan = (
    Plan.frequency("exposure")
    .banded("driver_age", breaks=[21, 25, 35, 50, 70])
    .banded("vehicle_age", quantile=10)
    .categorical("region")
    .variate("vehicle_value", quantile=20, degree=2)
)

# See every decision and data problem before fitting.
check = plan.check(train, "frequency")
for issue in check.issues:
    print(issue["severity"], issue["code"], issue["message"])

fitted = plan.fit(train, "frequency")
predicted_frequency = fitted.predict(new_business)

# One call gives calibration, discrimination, A/E exhibits and graded findings.
validation = fitted.validate(holdout)
print(validation.ae_ratio, validation.gini)
print(validation.warnings)

# A report combines the plan, fit, validation and rating tables.
report = fitted.report(holdout)
print(report.verdict)
print(report.headline)
print(report.markdown)
```

Here `frequency = claim_count / exposure`. Multiply the prediction by exposure for
expected claim counts. To model raw counts instead, use the explicit Poisson formulation:

```python
count_plan = Plan(
    "poisson",
    exposure="exposure",
    exposure_role="offset",
)
fitted_counts = count_plan.fit(train, "claim_count")
```

Plans normalize ordinary integer, boolean, string and categorical columns at the
boundary. Category encodings remain attached to the model, so scoring cannot silently
assign a familiar level a different code.

### What `check()` adds

`plan.check(df, target)` reports rather than stopping at the first error. It returns the
band edges selected by quantile rules, chosen base levels, rows and parameters per term,
and issues such as:

- missing, null or invalid target and exposure values;
- observations that match no rating row;
- empty, constant or thin levels;
- unidentified variate degrees;
- near-aliased table pairs; and
- plans spending more parameters than the data can support.

The findings have stable codes for applications and messages suitable for showing to a
person. `fit()` runs the check itself and keeps it, so the findings automatically travel
into the final report.

## Models you can open and edit

The rating tables are the model. Export them as one JSON file or as CSVs with a manifest:

```python
fitted.to_workbook().save_csv_dir("plan_2026")
```

```text
plan_2026/
  manifest.json
  00_intercept.csv
  01_driver_age.csv
  02_region.csv
```

The CSVs use category names rather than internal codes and contain one editable factor
column, named `Relativity` for log-link models or `Rating_Factor` otherwise. The manifest
records the family, scale, category encoding, offset tables, locked rows and variates.

```python
from avenue_model import Workbook

loaded = Workbook.load_csv_dir("plan_2026").to_model()
loaded.predict(new_business)
loaded.validate(holdout)
loaded.report(holdout)
```

Loading checks every table before it becomes a model. Blank or non-positive factors,
duplicate levels, out-of-order bands, missing unbounded bands and unknown category names
are reported together, with the table, row and suggested repair.

### Carry an existing plan forward

An existing model can be held fixed while new factors are fitted on top:

```python
prior = Workbook.load_csv_dir("plan_2025").to_model()

updated = (
    Plan.frequency("exposure")
    .offset_model(prior, prefix="prior")
    .categorical("telematics")
    .fit(train, "frequency")
)
```

The prior tables contribute to every prediction and spend no new parameters. The new
intercept can still express a rate-level change. Use `.given(name, table)` instead when
the old table should provide the shape while its factors are re-estimated.

### Compose frequency and severity

All routes produce the same `FittedModel` type, so fitted, loaded and converted models
can predict, validate, report, save and compose.

```python
frequency = (
    Plan.frequency("exposure")
    .categorical("region")
    .fit(train, "frequency")
)
severity = (
    Plan.severity("claim_count")
    .categorical("region")
    .fit(claims, "severity")
)

pure_premium = frequency + severity
pure_premium.to_workbook().save_csv_dir("technical_price")
```

Under the shared log link, linear predictors add and fitted means multiply. Category
encodings are reconciled by level name when the component models saw different subsets.

## Exact LightGBM conversion

```python
import json
from avenue_model import FittedModel

converted = FittedModel.from_lgbm_json(
    json.dumps(booster.dump_model()),
    consolidation="max",
)
predictions = converted.predict(new_business)
```

Conversion changes the representation, not the prediction. `consolidation="max"`
produces the smallest consolidated set of tables; `"analysis"` preserves one table per
tree node for inspection. Shallow trees convert most compactly because table size grows
with the number of feature combinations represented by a path.

Converted models do not inherently know which observed response they explain. Add that
metadata before validation:

```python
converted = converted.with_response(
    "frequency",
    exposure="exposure",
    exposure_role="weight",
)
```

## Why fit on tables?

A conventional rating workflow expresses feature structure once in preprocessing and
again in the deployed tables:

```text
Conventional:
rating structure -> encoding -> model matrix -> coefficients -> rebuilt tables

Avenue:
rating tables + observations -> fitted rating tables
```

In Avenue, fitting and prediction use the same matching code. Levels, bands, interactions
and wildcard rules therefore cannot drift between estimation and deployment. Avoiding a
dense dummy-coded matrix is also valuable for high-cardinality factors and interactions.

For direct control, the lower-level `RatingModel`, `fit_glm` and
`fit_glm_with_diagnostics` APIs remain available.

## Capabilities

- Gaussian, Poisson, Gamma, Tweedie and Binomial families
- Prior weights, observation offsets, fixed offset tables and locked rows
- Categorical variables without dummy encoding
- Numeric bands, interactions and wildcard matching
- Polynomial variates with inference on the top degree
- Ridge, lasso and elastic-net penalties
- Standard errors, dispersion, deviance, AIC and BIC where supported
- Automatic choice between global and table-native solvers
- Calibration, lift, Gini and actual-versus-expected validation exhibits
- Structured model reports with a single severity-graded verdict
- Editable JSON and CSV workbook formats
- Exact LightGBM conversion and model composition

## Performance at a glance

Release benchmarks check prediction agreement before reporting time. On the real-data
suite Avenue is faster on four of six cases and uses less incremental memory on five of
six. Representative results include:

| Dataset | Avenue fit | Avenue peak | glum fit | glum peak |
|---|---:|---:|---:|---:|
| freMTPL2, 678k rows, 79 parameters, Poisson | **0.26 s** | **87 MB** | 0.49 s | 165 MB |
| census income, 45k rows, 116 parameters, Binomial | **0.15 s** | **6 MB** | 0.21 s | 11 MB |
| NYC taxi, 2.75M rows, 577 parameters, Gamma | 5.22 s | **272 MB** | **3.82 s** | 479 MB |
| house sales, 21.6k rows, 92 parameters, Gamma | **0.046 s** | **9 MB** | 0.055 s | 81 MB |

glum is the hard comparison, not the naive one — it avoids a dense dummy-coded design
matrix too. Against the rest of the field, on the same three designs (a smaller machine,
so read the ratios rather than the seconds):

| fit time, Avenue = 1.00x | Avenue | glum | scikit-learn | H2O |
|---|---:|---:|---:|---:|
| freMTPL2, Poisson, unpenalised | **1.00x** | 2.5x | 2.9x | 3.2x |
| census income, Binomial, ridge | **1.00x** | 1.8x | 1.3x | 4.3x |
| freMTPL2, Poisson, lasso | **1.00x** | 2.6x | no L1 for a Poisson GLM | different answer |

A penalty is close to free: an L1 makes glum abandon its Cholesky factorisation for
coordinate descent, while Avenue's algorithm already is coordinate descent. Where the
problem is small enough that a few factorisations beat tens of passes over the data,
scikit-learn's `newton-cholesky` wins, and the 21.6k-row Gamma fit goes to it.

The table solver is strongest when a model has many rows and ordinary rating-factor
structure. A direct/global solver is preferable for unpenalized Gaussian models and for
plans where many tables share one strongly correlated direction. `solver="auto"` selects
the global path when it supports the model.

Absolute timings are machine-dependent, and the two tables above were measured on
different machines. The full methodology, synthetic and 20-million-row results, memory
measurements, correctness gates and reproduction commands are in the
[GLM documentation](src/glm/README.md#benchmarks); the wider comparison is
[here](src/glm/README.md#the-rest-of-the-field).

## Known gaps

- Variates, locked rows, non-base normalization and models above 6,000 parameters fall
  back to table descent when `solver="auto"`.
- Penalized fits omit standard errors; selective or debiased inference is not implemented.
- Inference currently provides Wald standard errors, not likelihood-ratio tests, profile
  intervals or robust covariance.
- Numeric tables use step lookup and do not interpolate between rows.
- The matching path accepts `Int32` category codes and `Float64` numeric bounds, but not
  narrower integer dtypes.

## Development

```bash
cargo test --no-default-features
maturin develop --release
python -m unittest discover -s tests
```

The extension uses Python's stable ABI for Python 3.12 and newer. `pyo3-polars`, Rust
Polars and Python Polars must be upgraded together; the compatible Python package is
pinned in `pyproject.toml`.

## Documentation

- [GLM internals and benchmarks](src/glm/README.md)
- [Rating tables, matching and LightGBM conversion](src/rating_model/README.md)
- Rust API documentation: `cargo doc --open`
- Python API documentation is available through `help(avenue_model)` and
  `help(avenue_model.Plan)`; a hosted API reference is planned.

## Built with

- [Polars](https://www.pola.rs/) — fast DataFrames
- [PyO3](https://pyo3.rs/) — Python bindings
- [Rayon](https://github.com/rayon-rs/rayon) — parallelism
