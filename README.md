# Avenue Model

**Fast GLMs. Interpretable LightGBM. Models you can edit.**

Avenue fits statistical models directly on readable rating tables and converts
LightGBM ensembles into the same format. Fit, inspect, edit and deploy your model
with a small Python API.

- **Fast.** GLM fitting speeds broadly competitive with state-of-the-art implementations.
- **Easy to use.** Install a wheel, define your factors and fit. No Rust toolchain required.
- **Interpretable.** Work with explicit levels, bands, interactions and factors.
  Convert supported LightGBM models without a surrogate fit, and use sparsity-aware
  training to keep the resulting tables compact.

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

## Convert a booster

Start with a trained LightGBM `booster` and a frame of `quote_predictors`:

```python
from avenue_model import from_booster

conversion = from_booster(booster, quote_predictors, consolidation="max")
print(conversion.parity)
converted = conversion.model
converted.to_workbook().save_csv_dir("converted_plan")
```

The converted model preserves the supported booster's predictions and uses the same
scoring and export interface as a GLM. `tune_lgbm` searches predictive loss and table
count together; the companion
[avenue-lightgbm](https://github.com/lukemuz/avenue-lightgbm) fork adds interaction
penalties to encourage simpler structures during training.

In the French motor experiment, an interaction penalty reduced 39 tables to five
for a 0.7% increase in cross-validated loss. See the [LightGBM guide](docs/lightgbm.md)
for the workflow, supported models and conversion checks.

## Performance at a glance

The published Linux wheel fits a **678,013-policy Poisson model in 0.16 seconds**
and a **20-million-row, five-table Poisson model in 3.0 seconds** on a Ryzen 9
9950X desktop, using 32 threads.

See the [benchmarks](src/glm/README.md#benchmarks) for real-data and 20-million-row
comparisons against glum, with reproduction details in the supporting results.

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

See the [GLM solver and benchmarks](src/glm/README.md),
[rating-table representation](src/rating_model/README.md), and
[installation and development guide](docs/installation.md).
The research behind booster conversion is described in
[*GBMs as Factor Tables: Achieving Both Transparency and Interpretability Without
Approximation*](https://avenue-analytics.com/research/avenue-analytics-methodology.pdf).
