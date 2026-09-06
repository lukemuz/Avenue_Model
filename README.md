# Avenue Model

**Fit on rating tables. Convert boosted trees into them. Keep the model editable.**

Avenue is a Python package with a Rust engine for statistical models represented as
rating tables. It grew out of research into a practical insurance pricing question:
how much predictive accuracy can we retain in a model that someone can inspect,
change and deploy as a set of tables?

The tables are the model throughout the workflow. Define levels, bands and
interactions, fit their factors, then export them as readable CSVs. An existing
rating plan can supply the starting structure or stay fixed while new factors are
estimated. A LightGBM model can enter the same workflow through exact conversion.

Three contributions drive the project:

- **Direct estimation on rating tables.** The GLM table solver works with cached
  observation-to-row matches, avoiding an observation-by-parameter design matrix.
  Fitting and scoring share the same matching rules, so the estimated structure
  carries directly into the delivered model.
- **Competitive speed with a small memory footprint.** In the recorded six-case
  real-data comparison with glum, Avenue used less memory in five cases and fitted
  faster in four. At 20 million observations, one benchmark fitted in 3.1 seconds
  using 1.2 GB, against glum's 16.5 seconds and 3.6 GB. The
  [solver notes](src/glm/README.md#benchmarks) include the methods and cases where
  other engines win.
- **Exact booster conversion, with sparsity encouraged during training.** Supported
  LightGBM ensembles convert directly into rating tables without a surrogate fit.
  The companion [avenue-lightgbm](https://github.com/lukemuz/avenue-lightgbm) fork
  adds penalties for new feature combinations and within-tree interaction complexity.
  These let training favor a compact table structure alongside predictive accuracy.
  In the French motor experiment, an interaction penalty reduced 39 tables to five
  for a 0.7% increase in cross-validated loss.

The [LightGBM guide](docs/lightgbm.md) develops the conversion and sparsity work,
with reproducible examples. The accompanying research paper is
[*GBMs as Factor Tables: Achieving Both Transparency and Interpretability Without
Approximation*](https://avenue-analytics.com/research/avenue-analytics-methodology.pdf).

## Installation

From a source checkout, with Python 3.12 or newer and a Rust toolchain:

```bash
pip install .
```

For LightGBM conversion and tuning, install the optional dependencies:

```bash
pip install '.[tuning]'
pip install avenue-lightgbm  # optional fork with interaction penalties
```

## Fit, inspect and export

This example uses Polars frames `train`, `holdout` and `new_business`.
The training target `frequency` is claim count divided by exposure.

```python
from avenue_model import Plan, Workbook

plan = (
    Plan.frequency("exposure")
    .banded("driver_age", breaks=[21, 25, 35, 50, 70])
    .categorical("region")
)

check = plan.check(train, "frequency")
for issue in check.issues:
    print(issue["severity"], issue["message"])

fitted = plan.fit(train, "frequency")
print(fitted.report(holdout).markdown)
predicted_frequency = fitted.predict(new_business)

fitted.to_workbook().save_csv_dir("rating_plan")
loaded = Workbook.load_csv_dir("rating_plan").to_model()
loaded.predict(new_business)
```

`Plan.frequency` fits a Poisson rate using exposure weights; its predictions are
claims per unit exposure. Quotes need the rating predictors, without observed claims
or exposure. Explicit count/offset models are also supported; see
[response and exposure conventions](docs/scoring.md#response-and-exposure).

The export contains a manifest and one CSV per table, with category labels and
editable factors. Reloading reconstructs the scorer. After editing factors, use
`loaded.report(holdout)` to assess the revised model.

To carry an existing plan forward, hold it fixed and estimate an additional effect:

```python
updated = (
    Plan.frequency("exposure")
    .offset_model(loaded, prefix="prior")
    .categorical("telematics")
    .fit(train, "frequency")
)
```

Use `Plan.given()` when existing tables should supply the structure and their factors
should be re-estimated. This also lets the GLM solver refit a booster-selected structure.

## Convert a booster

```python
from avenue_model import from_booster

conversion = from_booster(booster, quote_predictors, consolidation="max")
print(conversion.parity)
converted = conversion.model
converted.to_workbook().save_csv_dir("converted_plan")
```

Conversion groups the ensemble's contributions into tables, sums their factors and
applies the inverse link. The result uses the same scoring and workbook interfaces
as a fitted GLM. The parity report checks prediction agreement on the supplied rows;
[conversion support](docs/lightgbm.md#conversion-support-and-verification) describes
supported objectives, missing-value routes and input requirements.

Exactness and readability are separate concerns: a complex ensemble can produce
large tables. `tune_lgbm` searches cross-validated loss and table count together;
the interaction penalties guide training toward fewer feature combinations. Review
total rows and interaction order as well as table count before choosing a model.

## Performance at a glance

These are recorded fit benchmarks for banded GLMs, with fitted-mean agreement checked
before comparing speed. They demonstrate the solver's strengths on rating models;
they are not a survey of every GLM implementation or a timing guarantee for each release.

| Dataset | Avenue fit | Avenue memory | glum fit | glum memory |
|---|---:|---:|---:|---:|
| French motor, 678k rows, Poisson | **0.26 s** | **87 MB** | 0.49 s | 165 MB |
| Census income, 45k rows, Binomial | **0.15 s** | **6 MB** | 0.21 s | 11 MB |
| NYC taxi, 2.75M rows, Gamma | 5.22 s | **272 MB** | **3.82 s** | 479 MB |
| House sales, 21.6k rows, Gamma | **0.046 s** | **9 MB** | 0.055 s | 81 MB |

A separate comparison with glum, scikit-learn and H2O found Avenue fastest in five of
six cases where all engines returned comparable fitted means. glum wins the taxi
fit above; direct factorization is also particularly effective on small Gaussian
models. Avenue provides both table descent and global IRLS to accommodate these
different workloads.

See [benchmarks and reproduction commands](src/glm/README.md#benchmarks) for the full
results, memory measurement conventions, synthetic cases and conditioning limits.
The [evaluation guide](studies/README.md) covers correctness checks and newer studies.

## Explore the package

Avenue supports Gaussian, Poisson, Gamma, Tweedie and Binomial GLMs, with categorical,
banded, polynomial, interaction and continuous spline effects. It includes monotonic
bands, ridge/lasso/elastic net, weights, offsets and locked factors. Supported
unpenalized fits provide coefficient intervals and joint term tests, with model-based,
HC0 or clustered covariance. The guides describe which combinations are available.

- [User guides](docs/README.md): specification, scoring, inference and LightGBM.
- [Auto pricing example](examples/auto_pricing_study.py): frequency/severity fitting,
  validation, export and factor edits.
- [Homeowners example](examples/homeowners_perils.py): separate peril models and
  clustered inference.
- [Booster example](examples/booster_pricing_study.py): tuning, conversion and GLM refitting.
- [Continuous effects example](examples/smooth_pricing_study.py): natural cubic splines.
- [GLM internals](src/glm/README.md) and [rating-table representation](src/rating_model/README.md).

The examples use synthetic data unless stated otherwise. The
[real motor study](studies/results/real_motor/README.md) records comparisons on public data.

## Development

```bash
cargo test --no-default-features
maturin develop --release
python -m unittest discover -s tests
```

Python code lives in `python/avenue_model/`; Rust fitting and scoring live in `src/`.
The engine uses Polars, PyO3 and Rayon. See the [evaluation guide](studies/README.md)
for installed-wheel checks and the [API reference guide](docs/API_REFERENCE.md) to
build searchable documentation. Rust API documentation is available with `cargo doc --open`.
