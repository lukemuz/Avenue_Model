# Avenue Model

**Models represented as rating tables, from a GLM or from LightGBM.**

The rating table is the model. Avenue fits one — directly on the tables, with no
dummy-coded design matrix anywhere in the process — or converts a LightGBM booster into
one, exactly, without changing a single prediction. Either way what comes out is a set of
CSVs a person can read, edit, file and load back.

- **LightGBM becomes inspectable.** A tree ensemble converts into the same rating tables a
  GLM produces, with predictions preserved to floating-point noise.
  [Interaction-aware tuning](docs/lightgbm.md) reduces a booster to a handful of readable
  tables for little loss.
- **The engine is fast.** Avenue fits on rating tables without materializing an
  observation-by-parameter matrix — fastest in five of the six scenarios where every
  engine returned a comparable solution, against glum, scikit-learn and H2O across three
  families and three penalty settings
  ([methodology](src/glm/README.md#the-rest-of-the-field)).
- **The plan is data.** Levels, bands, interactions, reference levels and exposure
  treatment are explicit, serializable and reproducible.
- **Problems are found before fitting.** `check()` reports data faults, thin levels,
  unidentified terms and redundant tables together, with actionable messages.
- **The fitted model is an artifact.** Save it as JSON or readable CSV tables, edit it,
  load it again, validate it and compose it with other models.

The method, the penalties and the case studies are described in
[*GBMs as Factor Tables: Achieving Both Transparency and Interpretability Without
Approximation*](https://avenue-analytics.com/research/avenue-analytics-methodology.pdf).

## Installation

From a source checkout:

```bash
pip install .
```

After the first release:

```bash
pip install avenue-model
```

```toml
[dependencies]
avenue_model = "0.1.0"
```

Python 3.12 or newer is required. For LightGBM conversion and tuning:

```bash
pip install "avenue-model[tuning]"  # adds LightGBM and Optuna
pip install avenue-lightgbm         # optional interaction penalties
```

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

### Check before fitting

`plan.check(df, target)` returns resolved bands, base levels and parameter counts along
with all detected issues rather than stopping at the first:

- missing, null or invalid target and exposure values;
- observations that match no rating row;
- empty, constant or thin levels;
- unidentified variate degrees;
- near-aliased table pairs; and
- plans spending more parameters than the data can support.

Findings have stable codes and readable messages. `fit()` runs and retains the same
check, so its findings travel into the final report.

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

The CSVs use category names rather than internal codes and expose one editable factor
column. The manifest retains everything needed to reconstruct the model.

```python
from avenue_model import Workbook

loaded = Workbook.load_csv_dir("plan_2026").to_model()
loaded.predict(new_business)
loaded.validate(holdout)
loaded.report(holdout)
```

Loading validates every table and reports all problems together before constructing a
model.

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

What comes back is rating tables, not an explanation of a model that stays a black box:

```text
02_prior_charge_count__sex.csv        03_sex__age.csv
  Prior_Charge_Count  Sex  Factor       Sex   Age   Factor
                 0.0    F  -1.1001        F  20.5   1.4170
                 1.5    F  -0.4956        F  21.5   1.0817
                 2.5    F   0.0441        F  22.5   0.8567
                 ...                      ...
```

Add the intercept to one factor from each table and apply the inverse link. That is the
whole model — the same arithmetic a filed rating plan uses.

Conversion changes representation, not prediction: fitted means agree with the booster
to floating-point noise. On French motor claim frequency, the end-to-end example tunes
for accuracy and table count, converts the result and writes it as CSV:

```bash
python examples/french_motor.py
```

With [`avenue-lightgbm`](https://github.com/lukemuz/avenue-lightgbm), it produces five
tables at 0.591 held-out mean Poisson deviance. Table-size controls, category handling,
consolidation and GLM refitting are covered in [the LightGBM guide](docs/lightgbm.md).

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

Avenue supports Gaussian, Poisson, Gamma, Tweedie and Binomial GLMs; categorical,
banded, polynomial and interaction terms; ridge, lasso and elastic net; weights, offsets
and locked factors; statistical inference; validation and model reports; editable JSON
and CSV workbooks; model composition; and exact LightGBM conversion.

## Performance at a glance

**Avenue is usually the fastest GLM engine tested, rivaled only by glum, and is almost
always the most memory-efficient.** Every reported benchmark first checks that the
engines produce comparable fitted means. On the six-case real-data suite Avenue wins
four fits and five memory comparisons:

| Dataset | Avenue fit | Avenue peak | glum fit | glum peak |
|---|---:|---:|---:|---:|
| freMTPL2, 678k rows, 79 parameters, Poisson | **0.26 s** | **87 MB** | 0.49 s | 165 MB |
| census income, 45k rows, 116 parameters, Binomial | **0.15 s** | **6 MB** | 0.21 s | 11 MB |
| NYC taxi, 2.75M rows, 577 parameters, Gamma | 5.22 s | **272 MB** | **3.82 s** | 479 MB |
| house sales, 21.6k rows, 92 parameters, Gamma | **0.046 s** | **9 MB** | 0.055 s | 81 MB |

glum is the closest general-purpose competitor: like Avenue, it avoids a dense
dummy-coded design matrix. Avenue also leads the wider comparison with scikit-learn and
H2O, winning five of the six cases in which every engine returned a comparable solution:

| fit time, Avenue = 1.00x | Avenue | glum | scikit-learn | H2O |
|---|---:|---:|---:|---:|
| freMTPL2, Poisson, unpenalised | **1.00x** | 2.5x | 2.9x | 3.2x |
| census income, Binomial, ridge | **1.00x** | 1.8x | 1.3x | 4.3x |
| freMTPL2, Poisson, lasso | **1.00x** | 2.6x | n/a — no L1 for a Poisson GLM | n/a — fitted means disagree |

The exceptions are informative. glum wins the high-cardinality NYC taxi fit;
scikit-learn's `newton-cholesky` wins the small house-sales Gamma fit, where a few direct
factorisations cost less than repeated passes over the rows. `n/a` means an engine could
not express the model or did not return comparable fitted means—not that it ran slowly.

Timings vary by machine, so compare results within each table. Full methodology,
synthetic and 20-million-row results, memory measurements, correctness gates, and
reproduction commands are in the
[GLM documentation](src/glm/README.md#benchmarks); the wider comparison is
[here](src/glm/README.md#the-rest-of-the-field).

## Known gaps

- `predict()` returns a null for an observation matching no rating row — an unseen
  category level, most often — rather than raising. `validate()` reports the same
  situation at high severity and excludes those rows, so a scoring path that never
  validates is the one to watch.
- Converted models preserve LightGBM thresholds verbatim; rounding them can change which
  rows match.

## Development

```bash
cargo test --no-default-features
maturin develop --release
python -m unittest discover -s tests
```

Python code lives in `python/avenue_model/`; the Rust engine is re-exported from the
package root. Compatible Polars versions are pinned in `pyproject.toml`.

## Documentation

- [GLM internals and benchmarks](src/glm/README.md)
- [LightGBM as rating tables](docs/lightgbm.md) — table size, tuning, category names
  and refitting as a GLM
- [Rating tables, matching and LightGBM conversion](src/rating_model/README.md)
- Rust API documentation: `cargo doc --open`
- Python API documentation is available through `help(avenue_model)` and
  `help(avenue_model.Plan)`; a hosted API reference is planned.

## Built with

- [Polars](https://www.pola.rs/) — fast DataFrames
- [PyO3](https://pyo3.rs/) — Python bindings
- [Rayon](https://github.com/rayon-rs/rayon) — parallelism
