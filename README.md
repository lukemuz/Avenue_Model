# Avenue Model

**Fast GLMs built around rating tables, plus exact conversion of LightGBM models into
the same representation.**

A rating table is a lookup: a set of levels or numeric bands, each carrying a factor.
Insurance pricing, and a good deal of actuarial and risk modelling generally, is built out
of them. Avenue treats that structure as the model itself rather than as something to be
translated into a design matrix and back again.

Two things follow from that:

- **GLMs fit on the tables.** The tables define the features, levels, bands,
  interactions, reference levels and matching rules. There is no model matrix to build
  and no coefficient vector to map back. See [Fitting GLMs on tables](#fitting-glms-on-tables).
- **LightGBM models convert into tables exactly.** The representation changes; the
  predictions do not. See [Exact LightGBM conversion](#exact-lightgbm-conversion).

The fitter is competitive with the fastest GLM libraries available — usually faster, on
about half the memory — and the [Performance](#performance) section gives the numbers,
the methodology, and the cases where it is the wrong tool.

---

## Installation

**Python:**

```bash
maturin develop --release
```

**Rust:**

```toml
[dependencies]
avenue_model = "0.1.0"
```

The Python package requires Python 3.12+ and `polars`; the crate builds on Rust edition 2021.

---

## Quick start

### State a plan, check it, fit it, report on it

The shortest path from an ordinary dataframe to a model you can defend. Nothing here
builds a table by hand.

```python
from avenue_model import Plan, GLMOptions

plan = (
    Plan.frequency("exposure")                       # Poisson, log(exposure) as offset
    .banded("driver_age", breaks=[21, 25, 35, 50, 70])
    .banded("vehicle_age", quantile=10)
    .categorical("region")                            # base = most exposed level
    .variate("vehicle_value", quantile=20, degree=2)  # a curve, two parameters
)

# What would this do, and what is wrong with the data? Before a fit is burned.
check = plan.check(train, "claim_count")
for issue in check.issues:
    print(issue["severity"], issue["code"], issue["message"])

fitted = plan.fit(train, "claim_count")     # runs the check itself, and keeps it
report = fitted.report(holdout)             # so plan findings arrive unasked

print(report.verdict)     # "usable" | "usable_with_caveats" | "not_usable"
print(report.headline)    # one sentence, written to be shown to a person
print(report.markdown)    # the whole document
```

Ordinary dataframes work: integer widths, booleans, strings and categoricals are
normalised at the boundary, and the level mapping is kept on the fitted model so scoring
assigns the same codes. `Int64` matters most — it is numpy's default integer and what
`pandas.Categorical(...).codes` widens to.

**`plan.check(df, target)`** reports rather than raises. A check that stopped at the
first fault would leave you finding problems one failed attempt at a time, which is the
loop it exists to replace. It returns what the plan *decided* — the band edges a
quantile rule picked, the base level chosen, rows and parameters per term — alongside
unmatched rows, empty and thin levels, nulls, constant features, non-positive exposure
under a log offset, unidentified variate degrees, near-aliased table pairs, and plans
spending more parameters than there are rows.

**The plan is data.** `plan.to_json()` round-trips through `Plan.from_json`, so it can
be saved, diffed, shown to someone for approval, edited and re-run. It is the model's
source code, and it travels inside the report.

### Measure a model against data

```python
v = fitted.validate(holdout)

v.is_usable          # False if anything found should stop it being used
v.warnings           # dicts: severity, code, message, rows
v.ae_ratio           # actual over expected, 1.0 is calibrated
v.gini, v.lift       # how well it orders risk
v.calibration        # equal-exposure buckets: actual, expected, A/E per bucket
v.actual_vs_expected # one frame per rating factor
```

One call, complete verdict. Every judgement it can make, it makes — a caller reading
nothing but `warnings` should still never report a broken model as fine. Unmatched rows
are the case that motivates it: scoring alone turns them into `NaN` predictions that
average into a metric without complaint, so `validate` counts them, excludes them, and
says so.

Warnings carry a stable `code` to branch on and a `message` written to be relayed
unchanged. That is deliberate — the goal is not that someone feels confident about a
model, it is that their confidence is *calibrated*, so the caveats travel with the
numbers rather than depending on whoever reports them to think of the caveats.

### Hand the model to someone as a file

The tables are the model, so the model is a file — one you can open, change, and load
back as the model it now says it is.

```python
fitted.to_workbook().save_csv_dir("plan_2026")
```

```
plan_2026/
  manifest.json    family, link, scale, offsets, locks, variates, category codes
  00_intercept.csv
  01_driver_age.csv          02_region.csv
    driver_age,Relativity      region,Relativity
    25,1                       west,1
    45,1.0118                  east,0.8692
    65,1.6118                  north,0.5729
    inf,1.6318                 south,0.7188
```

Levels are written by **name**, not by code — a file whose region column reads `3` sends
the reader to the manifest before they can change anything. Names resolve back through
the manifest's encoding on load, and a name it has never seen is refused by name rather
than dropped. Raw codes are still accepted, so nobody is forced to look up a label they
already know.

```python
from avenue_model import Workbook

loaded = Workbook.load_csv_dir("plan_2026").to_model()   # after someone edits it
loaded.predict(new_business)
loaded.validate(holdout)      # a loaded model is not a lesser one
loaded.report(holdout)
```

**One factor column, named for its scale.** Log-link models write `Relativity`, because
that is the number an actuary edits; everything else writes `Rating_Factor`. Two columns
encoding one truth is how an edit gets silently ignored, so a workbook never carries
both — the manifest records which scale it is on. (`Rating_Factor` round-trips bit for
bit; `Relativity` goes through `exp`/`ln` and agrees to a couple of units in the last
place. Use the factor scale when reproducing a fit exactly.)

**The manifest is not decoration.** It carries what a table cannot say about itself:
which tables are offsets, which rows are locked, which are variates, and how category
levels map to codes. A directory of CSVs without it is not a model.

**A bad edit is refused, and every fault is named at once.** Out-of-order bands, a
deleted `inf` row, a blank factor, a duplicated level, a dtype the matcher cannot read —
each of those otherwise mis-prices in silence rather than failing:

```
This workbook cannot be loaded as a model. 2 problems found:
  [bounds_not_ascending] table 'driver_age' row 1: Band bound 'driver_age' is 25 but the
  row above it is 45. Bounds must ascend down the table: lookup takes the first row whose
  bound is not below the value, so an out-of-order row silently returns the wrong band.
  Sort the rows by 'driver_age'.
  [no_unbounded_band] table 'vehicle_value': Band bounds stop at 40000. The largest band
  must be unbounded (inf), or anything above 40000 matches no row and scores as NaN.
```

### Fit on top of a plan that is already in force

```python
prior = Workbook.load_csv_dir("plan_2025").to_model()

plan = (
    Plan.frequency("exposure")
    .offset_model(prior, prefix="prior")   # carried, fixed, costs no parameters
    .categorical("telematics")             # the new factor, fitted on top
)
fitted = plan.fit(train, "claim_count")
```

An offset table contributes to every prediction and is never updated, so the new factors
are estimated *against* what is already filed. The plan's own intercept is still fitted,
which is what lets the refit express a rate-level change. Use `.given(name, table)`
instead to keep a table's levels and bands while re-estimating its numbers.

Supplied tables travel inside the plan's JSON, so a plan stays a complete description of
its model, and they are checked exactly as strictly as one loaded from disk.

### One type, whichever way you got here

Everything returns a `FittedModel`, so every capability is reachable from every route.

```python
fitted = plan.fit(train, "claim_count")                    # fitted from a plan
loaded = Workbook.load_csv_dir("plan_2025").to_model()     # loaded from a file
converted = FittedModel.from_lgbm_json(booster.dump_model())  # converted from LightGBM
```

All three `predict`, `validate`, `report`, `to_workbook` and compose. What differs is
only how much is *known*: `was_fitted` is False for the last two, `converged` is `None`
rather than False, and a report omits its fit section instead of inventing one.

A model that was not fitted here does not know what its response is, so say once:

```python
converted = converted.with_response("claim_count", exposure="exposure",
                                    exposure_role="offset")
```

A workbook records that in its manifest, so a loaded model never needs telling.

**Frequency times severity is pure premium.** Under a log link the factors add, so the
means multiply:

```python
frequency = Plan.frequency("exposure").categorical("region").fit(train, "claim_count")
severity  = Plan.severity("claim_count").categorical("region").fit(claims, "severity")

pure_premium = frequency + severity      # one consolidated rating plan
pure_premium.to_workbook().save_csv_dir("technical_price")
```

### Fit a GLM on rating tables

```python
from avenue_model import RatingModel, fit_glm_with_diagnostics, GLMOptions
import polars as pl

# Tables define the structure: levels, bands, interactions, matching rules.
model = RatingModel(base_tables, family="poisson")

# Frequency models normally carry exposure as an offset, not a weight.
training_df = training_df.with_columns(pl.col("exposure").log().alias("log_exposure"))

options = GLMOptions(max_iterations=100, tolerance=1e-9)
result = fit_glm_with_diagnostics(
    model, training_df, "claim_count",
    offset_col="log_exposure",
    options=options,
)

print(result.diagnostics)    # iterations, convergence, deviance, dispersion, AIC
print(result.rating_tables()) # coefficients, relativities, SEs, and row status

predictions = result.predict_rate(new_data)
expected_claims = result.predict_expected(new_data, "exposure")
```

### Convert a LightGBM model

```python
from avenue_model import RatingModel
import json, lightgbm as lgb

lgbm_model = lgb.train(params, train_data)
model = RatingModel.from_lgbm_json(json.dumps(lgbm_model.dump_model()), "max")

# Predictions match lgbm_model.predict exactly.
predictions = model.predict(new_data)

# Now the ensemble is a set of inspectable tables, and they compose.
combined = model + manual_adjustments + territory_factors
```

### Rust

```rust
use avenue_model::rating_model::RatingModel;
use avenue_model::glm::{fit_glm, GLMOptions};

let model = RatingModel::from_lgbm_json(&lgbm_json, "max")?;

for table in &model.tables {
    println!("{:?}", table.data);   // Polars DataFrames
}

let options = GLMOptions {
    objective: "poisson".to_string(),
    max_iterations: 100,
    ..Default::default()
};
let fitted = fit_glm(&model, &data, "target", Some("weight"), None, options)?;
```

---

## Fitting GLMs on tables

A conventional GLM workflow translates rating tables into a model matrix, fits
coefficients, and maps those coefficients back into tables. Feature levels, numeric bands,
interactions, reference levels and missing-value rules end up expressed twice: once in the
preprocessing code, once in the rating structure. Keeping the two in agreement is
ordinary, unglamorous, and a reliable source of production incidents.

```text
Conventional:
rating tables → feature encoding → model matrix → coefficients → rebuilt tables

Avenue:
rating tables + observations → fitted rating tables
```

Fitting and prediction use the same matching code, so they cannot disagree about what a
level means. No observation-by-parameter matrix is materialised, which matters most
exactly where design matrices hurt: high-cardinality factors and interactions.

### Choosing a solver

`GLMOptions` defaults to `solver="auto"`:

| Solver | Best use | Method | Memory |
|---|---|---|---|
| `auto` | Most fits | Prefer global; fall back to table when needed | Chosen automatically |
| `global` | Up to a few thousand parameters | Global IRLS; direct solve or Gram coordinate descent | `O(p^2)` |
| `table` | Very wide or specialized rating models | Block coordinate descent over tables | Scales with table rows |

Global is usually fastest for unpenalized, Ridge, Lasso, and Elastic-Net fits. Table
descent remains available for variates, locked rows, non-base normalization, models over
6,000 parameters, and cases where predictable low memory matters more than latency.

Neither solver materializes an observation-by-parameter dummy matrix. Global builds the
much smaller `p x p` Gram matrix; table descent works directly from row matches.

### What table descent does

Avenue fits by **block coordinate descent over tables** — a backfit. Each sweep visits
every table in turn and updates all of its rows at once, holding the rest of the model
fixed.

- **Log-link families** (Poisson, Gamma, Tweedie) get an exact closed form per level:
  the update is `ln(actual / expected)`, which is the exact minimiser along that
  coordinate, not a step towards it.
- **Identity and logit links** take an IRLS step per table.
- **SQUAREM** three-point extrapolation accelerates the sequence, guarded so a bad
  extrapolation is at worst a few wasted passes.
- **Near-aliased table pairs** — a density band and an area code, an age band and a
  birth-year band — are detected before fitting and updated as a single block. This is
  the case that otherwise brings a backfit to a crawl.

A sweep touches each observation once per table: `O(n · T)` for `T` tables. An
IRLS solver instead forms `X'WX` from every *pair* of blocks: `O(n · T²)`. That
difference is most of the performance story below, and its limits are the rest of it.

### Diagnostics

`fit_glm_with_diagnostics` returns fitted tables and a diagnostics object:

| | |
|---|---|
| `converged`, `iterations`, `max_gradient` | convergence, on the score scale |
| `deviance`, `null_deviance`, `pseudo_r2` | fit quality |
| `standard_errors` | per table, per row; aligns with `model_tables()` |
| `aliased_rows`, `unfitted_rows` | rows whose effect is not identified, or that saw no exposure |
| `dispersion`, `pearson_chi2`, `df_residual` | scale and residual degrees of freedom |
| `log_likelihood`, `aic`, `bic` | model comparison |
| `table_conditioning` | how hard this plan is for table descent — see [below](#when-table-descent-is-the-wrong-tool) |
| `variate_terms` | fitted polynomial, standard errors and top-degree z for each variate |

Standard errors follow R's pivoted-QR convention: an anchoring reference row is exactly
`0`, and a row that is aliased or carries no exposure is `NaN` rather than a large
meaningless number. A level that is perfectly separated, or whose Fisher information
collapses during the fit, is detected and reported as aliased instead of being given a
standard error of `1e7`. Penalized fits intentionally report no standard errors because
ordinary Wald intervals are not defensible for shrinkage-biased estimates.

---

## Exact LightGBM conversion

Avenue can rewrite a trained LightGBM model as an additive collection of rating tables
without approximating or retraining it.

```text
training data → LightGBM → exact rating-table representation
                                ↓
                    inspect, consolidate, analyze,
                    predict, or refine with a GLM
```

The workflow this enables:

1. Use LightGBM to discover nonlinear effects and interactions.
2. Convert the fitted ensemble exactly into rating tables.
3. Inspect or consolidate the resulting structure.
4. Optionally use those tables as the starting structure for direct GLM fitting.

Two consolidation modes: `"max"` produces minimal tables, `"analysis"` produces one table
per tree node for interpretability.

Conversion works best with shallow trees (maximum depth ≤ 4) — deeper trees create
exponentially more table rows. The [Avenue LightGBM fork](https://github.com/avenue-model/LightGBM)
adds penalties that encourage the sparse, shallow trees this workflow wants.

---

## Capabilities

- Gaussian, Poisson, Gamma, Tweedie and Binomial families
- Prior weights, offset columns, offset tables, and locked rows
- Categorical variables without dummy encoding
- Multi-dimensional tables for interactions
- Wildcard matching (`-999`) for sparse representations
- Continuous drivers as polynomial *variates*, with a z statistic on the top degree
- Standard errors per level, plus dispersion, deviance, AIC and BIC
- Additive model structure, or multiplicative effects under a log link
- LightGBM conversion with exact prediction parity
- Models save and load as an editable workbook: JSON, or CSVs plus a manifest
- Existing tables compose in as fixed offsets or as model structure
- Declarative plans that build the tables, state every default, and round-trip as JSON
- One-call holdout validation: calibration, lift, Gini, actual-versus-expected, and
  severity-graded findings
- An assembled model report with a single verdict, in Markdown or as data

---

## Performance

### Solver and penalty comparison

Release builds, best of three runs, with coefficients checked against glum. Times are
seconds; lower is better.

| Dataset / penalty | Table | Global | glum |
|---|---:|---:|---:|
| Census, unpenalized | 0.144 | **0.094** | 0.198 |
| Census, Ridge | 0.083 | **0.044** | 0.076 |
| Census, Elastic Net | 0.912 | **0.076** | 0.081 |
| Census, Lasso | 0.500 | **0.064** | 0.093 |
| Housing, unpenalized | 0.048 | **0.022** | 0.058 |
| Housing, Ridge | 0.059 | **0.024** | 0.052 |
| Housing, Elastic Net | 0.171 | 0.064 | **0.061** |
| Housing, Lasso | 0.143 | **0.033** | 0.057 |
| Correlated synthetic, unpenalized | 0.332 | **0.183** | 0.324 |
| Correlated synthetic, Ridge | 0.334 | **0.178** | 0.316 |
| Correlated synthetic, Elastic Net | 1.077 | 0.379 | **0.372** |
| Correlated synthetic, Lasso | 1.181 | **0.348** | 0.403 |

On the 200,000-row correlated design, process-isolated incremental peak RSS was roughly
0.2-2.0 MB for table, 0.2-1.6 MB for global, and 42-43 MB for glum. Global has `O(p^2)`
storage, however, so table remains the safe choice as `p` grows.

The older tables below primarily describe the table solver and remain useful for
understanding its scaling behavior.

All benchmarks are release builds, and every one of them is **gated on the engines
agreeing about the fitted means** before any timing is reported — a fast wrong answer
fails the benchmark. Times are fit time only and the fastest of repeated runs; the
synthetic table is the best of three independent runs of five repeats each, because a
single run varies by up to 10%. **glum** is the
speed comparison; **statsmodels** is the correctness oracle. Absolute numbers are
machine-dependent; the ratios are the point.

glum is a strong comparison rather than a naive baseline: its `tabmat` backend avoids a
dense dummy-coded design matrix too, and its default `irls-ls` solver is Cholesky-based
IRLS. The dense route is what statsmodels takes.

### Synthetic data

Five tables, 81 parameters, factors drawn independently.

| | Avenue | glum | statsmodels |
|---|-------:|-----:|------------:|
| Poisson, 1M rows | **0.100 s** | 0.416 s | — |
| Gamma, 1M rows | **0.114 s** | 0.408 s | — |
| Tweedie(1.5), 1M rows | **0.182 s** | 0.286 s | — |
| Gaussian, 1M rows | 0.079 s | **0.074 s** | — |
| Poisson, 100k rows | **0.016 s** | 0.034 s | 1.560 s |
| Poisson, 5M rows | **0.695 s** | 2.518 s | — |

Peak memory is measured with one engine per process, so each figure is that whole
process's high-water mark — data, design matrix, solver and interpreter together.

| whole-process peak RSS | Avenue | glum | statsmodels |
|---|-------:|-----:|------------:|
| Poisson, 100k rows | **113 MB** | 191 MB | 932 MB |
| Poisson, 1M rows | **196 MB** | 336 MB | — |
| Poisson, 5M rows | **564 MB** | 1,200 MB | — |
| freMTPL2, tutorial bands | **411 MB** | 464 MB | — |
| house_sales, Gamma | **130 MB** | 202 MB | 538 MB |

A 1.7x advantage on the synthetic cases, widening to 2.1x at five million rows as the
interpreter's fixed footprint stops mattering. Both engines avoid a dense matrix, so this
is a constant factor rather than a different scaling law; the `O(n · parameters)` blowup
Avenue genuinely avoids belongs to the dummy-coded route, and the 932 MB statsmodels
spends on the *smallest* problem in the table is what that costs.

Each factor is stored in the dtype its data calls for — `Int32` for a category code,
`Float64` for a band's upper bound — which is worth roughly half the frame on a
categorical design. The freMTPL2 row is the narrow one at 1.13x because most of that
process is the pandas source frame both engines are built from, not either engine.

### Real data

Five public datasets across four families, all with the correlated factors that synthetic
data does not have.

| | Avenue fit | Avenue peak | glum fit | glum peak |
|---|---:|---:|---:|---:|
| freMTPL2, 678k rows, 79 params, Poisson | **0.26 s** | **87 MB** | 0.49 s | 165 MB |
| freMTPL2, 678k rows, 270 params, Poisson | **0.41 s** | **87 MB** | 1.64 s | 119 MB |
| nyc_taxi, 2.75M rows, 577 params, Gamma | 5.22 s | **272 MB** | **3.82 s** | 479 MB |
| census_income, 45.2k rows, 116 params, Binomial | **0.15 s** | **6 MB** | 0.21 s | 11 MB |
| house_sales, 21.6k rows, 92 params, Gamma | **0.046 s** | **9 MB** | 0.055 s | 81 MB |
| house_sales, 21.6k rows, 92 params, Gaussian | 0.034 s | 1 MB | **0.012 s** | 0 MB |

Peak here is what each engine adds inside one process, which is why the figures are far
below the whole-process numbers above; both are measured, they answer different questions.
Avenue takes four of the six rows on speed and five of six on memory. The two speed losses
are informative:

- **nyc_taxi** is the largest real problem here and the only one with high-cardinality
  geography (252 pickup and 261 dropoff zones). It is the case where a factorisation wins:
  the plan is hard enough that Avenue needs many passes, and narrow enough that `O(n·T²)`
  never bites. It still fits in 57% of glum's memory.
- **Gaussian house_sales** goes to glum by 3x for a structural reason: under an identity
  link a single IRLS step *is* the exact answer for a linear model, and no number of
  cheaper passes beats being finished after one. **If you are fitting an unpenalised
  Gaussian model, use a direct solver.**

The French motor data is deliberately a hard case: `Area` is a six-band rebanding of
`Density`, correlated at 0.972. Avenue detects that pair and solves it as one block —
worth 3.9x here and 10.3x on the census data — and
`scripts/bench_fremtpl.py --drop area` shows how well: dropping the redundant table
outright — what a modeller who noticed it would do — now changes the fit time by 1.00x on
the tutorial bands and 1.01x on the wide ones. The redundancy costs essentially nothing
once the pair is solved together.

### At twenty million rows

Every case above finishes in under two seconds. A single fit at a size where the absolute
numbers stand on their own: 20M rows, 501 parameters, Poisson with an exposure offset, one
engine per process because the two representations do not fit in memory together.

| 20M rows, 501 parameters | Avenue | glum `irls-ls` |
|---|---:|---:|
| 100 tables of 6 levels | **39.3 s**, 10.8 GB | 865.9 s, 21.1 GB |
| 5 tables of 101 levels | **3.1 s**, 1.2 GB | 16.5 s, 3.6 GB |

Fitted means agree to 5.6e-09 and 3.2e-09. Those two rows carry the *same* 501 parameters
over the same data and differ only in how the parameters are laid out, which isolates what
the comparison turns on: moving them from 100 tables into 5 collapses the per-iteration
gap from 18x to 2.7x, exactly as `O(n·T)` against `O(n·T²)` predicts.

Those factors are stored as `Int32` category codes, which is what they are — unordered
draws over a level count. That matters for the memory column: 100 factors over 20M rows is
an 8 GB frame as `Int32` against 16 GB as `Float64` bands, and the fit is identical either
way. It is why Avenue's peak here is 10.8 GB rather than the 18.7 GB this benchmark
reported when it built everything as bands.

The remaining gap is ours. glum is handed 2 GB of `int8` codes, and 4 bytes per factor per
row is as narrow as Avenue's matching path goes — anything narrower falls back to a slow
path. Supporting `Int8` and `Int16` is the largest memory win still available.

### When table descent is the wrong tool

Two situations, both worth knowing before you choose it.

**An unpenalised Gaussian model.** One IRLS step is exact; a sweep is not. Use a direct
solver.

**Many tables sharing one common direction.** A backfit's convergence rate is set by how
much information the tables share, and enough shared structure will make the sweep count
dominate everything else. Loading 100 tables on a common latent driver, at 1M rows:

| pairwise correlation | `table_conditioning` | Avenue | glum | |
|---:|---:|---:|---:|--|
| 0.00 | 1.8 | **4.3 s** | 35.6 s | 8.2x faster |
| 0.10 | 10.1 | 29.6 s | 31.2 s | parity |
| 0.20 | 19.2 | 95.8 s | 31.2 s | 3.2x slower |
| 0.30 | 28.4 | 240.6 s | 30.9 s | 7.8x slower |

Avenue does the same work 8.2x faster in every one of those rows; the entire swing is how
many passes it needs, because a factorisation is indifferent to conditioning and glum's
cost is flat throughout. Hence the rule worth remembering, and the reason the measure is
reported: **this fitter wins while `table_conditioning` stays under about 10**, which is
exactly where the two engines cross above.

What decides that is *not* any pairwise correlation. Holding the pairwise figure at 0.28
and varying only how many tables share the driver, a hundred tables cost eighty times what
five do. What matters is how strongly the tables share one direction across all of them at
once, and Avenue measures it and reports it as `table_conditioning`: 1.0 for orthogonal
tables, rising to the table count when they all carry the same information. **Above
roughly 10 the fit slows sharply; above 25 this is the wrong tool.** It costs nothing to
compute and is available on the diagnostics before you commit to a long fit.

In practice this appears to be a synthetic pathology. Every real dataset in the suite:

| dataset | rows | tables | `table_conditioning` | worst pair |
|---|---:|---:|---:|---:|
| freMTPL2, tutorial bands | 678,013 | 9 | 2.85 | 0.972 |
| freMTPL2, wide bands | 678,013 | 9 | 2.94 | 0.994 |
| house_sales | 21,613 | 10 | 4.11 | 0.766 |
| census_income | 45,222 | 12 | 3.76 | 0.987 |
| nyc_taxi | 2,753,989 | 10 | 4.11 | 1.000 |

All between 2.85 and 4.11, well under the threshold — *including the three that contain a
pair over the alias threshold*. That is not a coincidence. A real alias is local to two
tables, usually one driver banded twice, and the joint pair solve handles it; conditioning
measures a direction shared across many tables at once, which real rating plans do not
tend to have. The weakness is real. It also appears to be rare.

### Reproducing

```bash
python scripts/bench_glm.py       # synthetic, four families
python scripts/bench_fremtpl.py   # French motor third-party liability
python scripts/bench_housing.py   # King County house sales
python scripts/bench_real.py      # NYC taxi, census income
python scripts/bench_large.py     # 20M rows; --correlation for the table above
python scripts/bench_isolated.py  # peak memory, one engine per process
```

Stock `cargo --release`. LTO, `codegen-units = 1` and `target-cpu=native` were all
measured and none moved a number outside run-to-run noise, so none are configured; the
reasoning is in the [module docs](src/glm/README.md#build-settings).

---

## Known gaps

- **Global solver coverage.** Variates, locked rows, non-base normalization, and models
  above 6,000 parameters currently fall back to table descent under `solver="auto"`.
- **Penalized inference.** Penalized fits intentionally omit standard errors; selective
  or debiased inference is not implemented.
- **Wald standard errors only.** No likelihood-ratio tests, profile intervals, or
  robust/sandwich covariance.
- **Step lookup only.** A table cannot interpolate between its rows, so two ages in the
  same band get the same factor. Interpolating tables are designed but not built.
- **Variate degree is not chosen for you.** Only a Wald z on the top degree; refit at
  each degree and compare `aic` yourself.
- **Poorly conditioned table-descent plans are slow.** Use `solver="auto"` or
  `solver="global"` when the global path supports the model.
- **No narrow dtypes.** Feature columns may be `Int32` or `Float64` — both match at the
  same speed — but nothing narrower is recognised, so a wide design still carries 4 bytes
  per factor per row where glum is handed 1.

## Roadmap

- Automatic degree selection for variates (likelihood-ratio or AIC across a sequence)
- Difference penalties on adjacent ordinal levels and monotonicity constraints
- Narrower dtypes on the matching path, to close the memory gap on wide designs

---

## Development

```bash
cargo test --lib                  # 255 tests
cargo test --features benchmarks  # include the benchmark tests
```

**Building the Python bindings.** The extension is `abi3-py312`, so it needs Python
3.12 or newer, and `pyo3-polars 0.21` pairs with a *specific* range of the Python polars
package — not the latest. Newer polars fails at the first DataFrame handed across the
boundary with ``TypeError: argument 'df': `compat_level` has invalid type: 'int'``,
which does not name the real cause.

```bash
uv venv --python python3.12 .venv
uv pip install --python .venv/bin/python maturin "polars==1.31.0" pyarrow
source .venv/bin/activate
maturin develop --release
python -m unittest discover -s tests    # 29 tests
```

Bumping `polars` in `Cargo.toml` means bumping `pyo3-polars` and the pinned Python
`polars` together; the three move as one.

## Documentation

- [GLM module](src/glm/README.md) — update rules, variates, identifiability, anchoring,
  convergence, standard errors, and the full performance methodology
- [Rating model module](src/rating_model/README.md) — table structure, matching rules,
  LightGBM conversion
- `src/plan.rs` — terms, band rules, base levels, dtype normalisation, and what `check`
  looks for
- `src/validation.rs` — the warning set, and what each code means
- `src/report.rs` — how a verdict is reached

## Built with

- [Polars](https://www.pola.rs/) — fast DataFrames
- [PyO3](https://pyo3.rs/) — Python bindings
- [Rayon](https://github.com/rayon-rs/rayon) — parallelism
