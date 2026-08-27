# Avenue Model

**A GLM fitter that works directly on rating tables, plus exact conversion of LightGBM
models into the same representation.**

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

### Fit a GLM on rating tables

```python
from avenue_model import RatingModel, fit_glm_with_diagnostics, GLMOptions
import polars as pl

# Tables define the structure: levels, bands, interactions, matching rules.
model = RatingModel(base_tables, objective="poisson")

# Frequency models normally carry exposure as an offset, not a weight.
training_df = training_df.with_columns(pl.col("exposure").log().alias("log_exposure"))

options = GLMOptions(objective="poisson", max_iterations=100, tolerance=1e-9)
fitted_model, diag = fit_glm_with_diagnostics(
    model, training_df, "claim_count",
    offset_col="log_exposure",
    options=options,
)

print(diag)                  # iterations, converged, deviance, pseudo_r2, dispersion, aic
print(diag.standard_errors)  # per table, per row; aligns with model_tables()

predictions = fitted_model.predict(new_data)
tables = fitted_model.model_tables()   # list of Polars DataFrames
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

### What the fitter does

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
| `table_conditioning` | how hard this plan is for a coordinate method — see [below](#when-avenue-is-the-wrong-tool) |
| `variate_terms` | fitted polynomial, standard errors and top-degree z for each variate |

Standard errors follow R's pivoted-QR convention: an anchoring reference row is exactly
`0`, and a row that is aliased or carries no exposure is `NaN` rather than a large
meaningless number. A level that is perfectly separated, or whose Fisher information
collapses during the fit, is detected and reported as aliased instead of being given a
standard error of `1e7`.

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

---

## Performance

All benchmarks are release builds, and every one of them is **gated on the engines
agreeing about the fitted means** before any timing is reported — a fast wrong answer
fails the benchmark. Times are fit time only, fastest of three runs. **glum** is the
speed comparison; **statsmodels** is the correctness oracle. Absolute numbers are
machine-dependent; the ratios are the point.

glum is a strong comparison rather than a naive baseline: its `tabmat` backend avoids a
dense dummy-coded design matrix too, and its default `irls-ls` solver is Cholesky-based
IRLS. The dense route is what statsmodels takes.

### Synthetic data

Five tables, 81 parameters, factors drawn independently.

| | Avenue | glum | statsmodels |
|---|-------:|-----:|------------:|
| Poisson, 1M rows | **0.096 s** | 0.424 s | — |
| Gamma, 1M rows | **0.114 s** | 0.410 s | — |
| Tweedie(1.5), 1M rows | **0.213 s** | 0.287 s | — |
| Gaussian, 1M rows | 0.088 s | **0.071 s** | — |
| Poisson, 100k rows | **0.016 s** | 0.035 s | 1.523 s |
| Poisson, 5M rows | **0.709 s** | 2.455 s | — |

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

| | Avenue | glum `irls-ls` |
|---|-------:|---------------:|
| freMTPL2, 678k rows, 79 params, Poisson | **0.27 s** (15 sweeps) | 0.46 s (5 iter) |
| freMTPL2, 678k rows, 270 params, Poisson | **0.52 s** (25) | 1.61 s (18) |
| nyc_taxi, 2.75M rows, 577 params, Gamma | 5.36 s (35) | **3.77 s** (9) |
| census_income, 45.2k rows, 116 params, Binomial | **0.16 s** (36) | 0.20 s (21) |
| house_sales, 21.6k rows, 92 params, Gamma | **0.052 s** (50) | 0.057 s (6) |
| house_sales, 21.6k rows, 92 params, Gaussian | 0.039 s (53) | **0.013 s** (1) |

Avenue takes four of the six rows. The two it loses are informative:

- **nyc_taxi** is the largest real problem here and the only one with high-cardinality
  geography (252 pickup and 261 dropoff zones). glum wins it on 9 IRLS iterations against
  35 sweeps — the expected outcome wherever the sweep count climbs but the table count is
  too small for `O(n·T²)` to bite.
- **Gaussian house_sales** goes to glum by 3x for a structural reason: under an identity
  link a single IRLS step *is* the exact answer for a linear model. One iteration against
  53 sweeps is not a contest. **If you are fitting an unpenalised Gaussian model, use a
  direct solver.**

The French motor data is deliberately a hard case: `Area` is a six-band rebanding of
`Density`, correlated at 0.972. Avenue detects that pair and solves it as one block, which
is worth 3.4x here and 9.7x on the census data. It is close to free when there is nothing
to find — on independent synthetic factors, enabling detection costs between −0.3% and
+4.9%, inside run-to-run noise.

### At twenty million rows

Every case above finishes in under two seconds. A single fit at a size where the absolute
numbers stand on their own: 20M rows, 501 parameters, Poisson with an exposure offset, one
engine per process because the two representations do not fit in memory together.

| 20M rows, 501 parameters | Avenue | glum `irls-ls` | per iteration |
|---|-------:|---------------:|--------------:|
| 100 tables of 6 levels | **39.3 s** (4 sweeps, 10.8 GB) | 865.9 s (5 iter, 21.1 GB) | 9.8 s vs 173 s |
| 5 tables of 101 levels | **3.1 s** (4 sweeps, 1.2 GB) | 16.5 s (8 iter, 3.6 GB) | 0.8 s vs 2.1 s |

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

### When Avenue is the wrong tool

Two situations, both worth knowing before you choose it.

**An unpenalised Gaussian model.** One IRLS step is exact; a sweep is not. Use a direct
solver.

**Many tables sharing one common direction.** A backfit's convergence rate is set by how
much information the tables share, and enough shared structure will make the sweep count
dominate everything else. Loading 100 tables on a common latent driver, at 1M rows:

| pairwise correlation | `table_conditioning` | sweeps | Avenue | glum | |
|---:|---:|---:|---:|---:|--|
| 0.00 | 1.8 | 5 | **4.3 s** | 35.6 s | 8.2x faster |
| 0.10 | 10.1 | 121 | 29.6 s | 31.2 s | parity |
| 0.20 | 19.2 | 494 | 95.8 s | 31.2 s | 3.2x slower |
| 0.30 | 28.4 | 1,124 | 240.6 s | 30.9 s | 7.8x slower |

Avenue is 8.2x faster *per iteration* in every one of those rows. The entire swing is the
sweep count, because a factorisation is indifferent to conditioning and glum sits at five
iterations throughout. Hence the rule worth remembering: **this fitter wins while the plan
needs fewer than about forty sweeps.**

What decides that is *not* any pairwise correlation. At a fixed pairwise 0.28, five tables
converge in 14 sweeps and a hundred take 1,124. What matters is how strongly the tables
share one direction across all of them at once, and Avenue measures it and reports it as
`table_conditioning`: 1.0 for orthogonal tables, rising to the table count when they all
carry the same information. **Above roughly 10, expect hundreds of sweeps; above 25,
thousands.** It costs nothing to compute and is available on the diagnostics before you
commit to a long fit.

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

- **Unpenalised only.** No ridge, lasso, elastic net, credibility shrinkage, or
  monotonicity constraints yet.
- **Wald standard errors only.** No likelihood-ratio tests, profile intervals, or
  robust/sandwich covariance.
- **Step lookup only.** A table cannot interpolate between its rows, so two ages in the
  same band get the same factor. Interpolating tables are designed but not built.
- **Variate degree is not chosen for you.** Only a Wald z on the top degree; refit at
  each degree and compare `aic` yourself.
- **Poorly conditioned plans are slow.** See [above](#when-avenue-is-the-wrong-tool). The
  planned fix is conjugate gradient preconditioned by the sweep — `O(√κ)` iterations
  rather than `O(κ)`, each still `O(n·T)`.
- **No narrow dtypes.** Feature columns may be `Int32` or `Float64` — both match at the
  same speed — but nothing narrower is recognised, so a wide design still carries 4 bytes
  per factor per row where glum is handed 1.

## Roadmap

- Automatic degree selection for variates (likelihood-ratio or AIC across a sequence)
- Penalized regression: credibility shrinkage for sparse levels, difference penalties on
  adjacent ordinal levels, monotonicity constraints, elastic net
- Narrower dtypes on the matching path, to close the memory gap on wide designs

---

## Development

```bash
cargo test --lib                  # 150 tests
maturin develop --release         # build the Python bindings
cargo test --features benchmarks  # include the benchmark tests
```

## Documentation

- [GLM module](src/glm/README.md) — update rules, variates, identifiability, anchoring,
  convergence, standard errors, and the full performance methodology
- [Rating model module](src/rating_model/README.md) — table structure, matching rules,
  LightGBM conversion

## Built with

- [Polars](https://www.pola.rs/) — fast DataFrames
- [PyO3](https://pyo3.rs/) — Python bindings
- [Rayon](https://github.com/rayon-rs/rayon) — parallelism
