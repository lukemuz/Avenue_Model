# Avenue Model

**GLM speed competitive with the state of the art. Interpretable boosted models.**

Avenue fits GLMs directly on rating tables and converts LightGBM ensembles into
the same representation, preserving their predictions. The tables are the model
you inspect, edit and deploy: explicit levels, bands, interactions and factors.

Three achievements make this practical:

- **GLM fitting broadly competitive with state-of-the-art implementations.**
  In published-wheel benchmarks on a Ryzen 9 9950X desktop, Avenue outperformed glum
  in **eight of nine real-data fits**, with both packages using 32 threads.
  On a 20-million-row synthetic portfolio with 100 tables, Avenue's table solver fitted in
  **36.6 seconds versus glum's 763 seconds—about 21× faster**.
  [Benchmarks and supporting evidence](src/glm/README.md#benchmarks).
- **Exact conversion of boosted trees into editable rating tables.** Supported
  LightGBM ensembles become tables that reproduce the ensemble's predictions,
  without approximation or a surrogate fit. The converted model uses the same
  inspection, editing, scoring and export workflow as a fitted GLM.
- **Simpler models through sparsity-aware training.** The companion
  [avenue-lightgbm](https://github.com/lukemuz/avenue-lightgbm) fork penalizes new
  feature combinations and interaction complexity during training. In the French
  motor experiment, this reduced **39 tables to five for a 0.7% increase in
  cross-validated loss**. Exact conversion preserves the resulting model's
  predictions. [LightGBM integration and results](docs/lightgbm.md).

Install a wheel, define your factors and fit with a small Python API. No Rust
toolchain is required. The [quickstart below](#try-it) fits, predicts and exports
an editable model in a few lines.

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

See the [LightGBM guide](docs/lightgbm.md) for the workflow, supported models,
training results and conversion checks.

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
