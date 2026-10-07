# Avenue Model

**GLMs and interpretable machine learning, built from editable tables.**

Avenue makes the tables the model: explicit levels, bands, interactions and factors
you can inspect, edit and deploy through a small Python API.

- **Fit a GLM directly on tables.** Define your factors, fit their coefficients and
  export the result. The table representation is the fitted GLM, with the same
  predictions and statistical interpretation.
- **Turn boosted trees into interpretable tables.** Supported LightGBM ensembles
  convert exactly, preserving predictions without a surrogate fit. Inspect and edit
  them through the same workflow, or refit their table structure as a GLM.
- **Keep models small enough to understand.** Avenue tunes predictive loss and
  table count together. Its
  [avenue-lightgbm](https://github.com/lukemuz/avenue-lightgbm) training backend penalizes new
  feature combinations during training. In the French motor experiment, this reduced
  **39 tables to five for a 0.7% increase in cross-validated loss**.
  [LightGBM integration](docs/lightgbm.md).

Fitting speed is **state of the art—competitive with and sometimes faster than glum**,
with state-of-the-art memory efficiency. [Performance examples](#performance-at-a-glance).
Install a wheel and [try it below](#try-it); no Rust toolchain is required.

## Installation

Download the `.whl` file for your operating system and processor from
[GitHub Releases](https://github.com/lukemuz/Avenue_Model/releases), then install it
with Python 3.12 or 3.13. Replace `WHEEL_FILENAME.whl` with the downloaded filename:

```bash
python -m pip install ./WHEEL_FILENAME.whl
```

You can also pass a wheel's GitHub release download URL directly to `pip install`.

For LightGBM conversion and tuning, install the wheel with optional dependencies:

```bash
python -m pip install "./WHEEL_FILENAME.whl[tuning]"
```

For help choosing a wheel, source installation and release instructions, see the
[installation and development guide](docs/installation.md).

## Try it

Fit a frequency model, predict new business and export the model as editable CSVs:

```python
import polars as pl
from avenue_model import Plan

train = pl.DataFrame({
    "region": ["north", "north", "south", "south"],
    "exposure": [100.0, 200.0, 100.0, 200.0],
    "frequency": [0.10, 0.12, 0.04, 0.06],
})

fitted = Plan.frequency("exposure").categorical("region").fit(train, "frequency")
quotes = pl.DataFrame({"region": ["north", "south"]})
print(fitted.predict(quotes))
fitted.to_workbook().save_csv_dir("rating_plan")
```

`frequency` is claim count divided by exposure; predictions are claims per unit
exposure. Add age bands, interactions or smooth effects as your model grows.
[The modeling guide](docs/modeling.md) covers these and fitting from an existing plan.
[The scoring guide](docs/scoring.md) shows how to reload and edit the exported tables.

## From boosting to editable tables

Trees learn useful cut points and interactions. Avenue combines their contributions
into tables grouped by feature combinations. Looking up those tables reproduces the
boosted model's predictions; you can then inspect, edit and export the model.

**[Run the complete example](examples/interpretable_boosting.py):** create sample data
→ tune for accuracy and table count → train → convert → inspect and export.
It also shows an optional GLM refit of the learned table structure. Conversion preserves
predictions; refitting estimates new coefficients and can change them.

```bash
python examples/interpretable_boosting.py --output /tmp/avenue-tables
```

With a trained LightGBM `booster` and a frame of `quote_predictors`, conversion is:

```python
from avenue_model import from_booster

conversion = from_booster(booster, quote_predictors, consolidation="max")
print(conversion.parity)
converted = conversion.model
converted.to_workbook().save_csv_dir("converted_plan")
```

The converted model preserves the supported booster's predictions and uses the same
scoring and export interface as a GLM. `tune_lgbm` searches predictive loss and table
count together. The
[avenue-lightgbm](https://github.com/lukemuz/avenue-lightgbm) training backend enables
the additional interaction penalties. The example also runs with stock LightGBM
without those penalties; it prints which capabilities are active.

See the [LightGBM guide](docs/lightgbm.md) for the workflow, supported models,
training results and conversion checks.

## Performance at a glance

On 678,013 French motor policies, Avenue fitted the 79-parameter Poisson model in
**0.16 seconds versus glum's 0.25 seconds**, and the 270-parameter model in
**0.49 seconds versus 1.05 seconds**, using each package's best tested thread setting.
On 20 million synthetic observations with five tables, Avenue's default global solver
used **2.73 GiB peak RAM versus glum's 3.43 GiB**, both at 32 threads.

These compare the Avenue 0.1.2 Linux wheel with glum 3.4.1 on a Ryzen 9 9950X desktop.
[Results and reproduction](docs/performance.md).

## Explore the package

Fit Gaussian, Poisson, Gamma, Tweedie and Binomial models with regularization,
weights, offsets, interactions and smooth effects. Export, edit and reload the same
model throughout the workflow.

- [User guides](docs/README.md): specification, scoring, inference and LightGBM.
- [Auto pricing example](examples/auto_pricing_study.py): frequency/severity fitting,
  validation, export and factor edits.
- [Homeowners example](examples/homeowners_perils.py): separate peril models and
  clustered inference.
- [Booster example](examples/booster_pricing_study.py): tuning, conversion and GLM refitting.
- [Continuous effects example](examples/smooth_pricing_study.py): natural cubic splines.

The examples use synthetic data unless stated otherwise. The
[real motor study](studies/results/real_motor/README.md) records comparisons on public data.

## Under the hood

See the [GLM solver](src/glm/README.md),
[rating-table representation](src/rating_model/README.md), and
[installation and development guide](docs/installation.md).
The research behind booster conversion is described in
[*GBMs as Factor Tables: Achieving Both Transparency and Interpretability Without
Approximation*](https://avenue-analytics.com/research/avenue-analytics-methodology.pdf).
