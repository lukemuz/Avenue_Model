# GLM Fitting Module for Avenue_Model

Generalized Linear Model fitting by coordinate descent directly on rating tables — no design matrix.

## Features

### Supported Distributions

| Family | Link | Variance `V(mu)` | Typical use |
|--------|------|------------------|-------------|
| Gaussian | Identity | `1` | Linear regression |
| Poisson | Log | `mu` | Claim counts |
| Gamma | Log | `mu^2` | Claim severity |
| Tweedie(p) | Log | `mu^p` | Pure premium (mixed zero / positive) |
| Binomial | Logit | `mu(1-mu)` | Binary classification |

### Algorithm

Coordinate descent over the tables: each table's factors are updated in turn while
every other table is held fixed, sweeping until the deviance stops moving.

Because a step table assigns each observation to exactly one row, the weighted
least-squares step for a level collapses to a scalar. Two update rules are used:

**Log-link families** get an exact closed-form coordinate solve. With `mu_i` the
current fitted mean and `a_i` the prior weight, the level's score equation solves to

```
beta_r  <-  beta_r + ln( A / E )

  A = sum over the level of  a * mu^(1-p) * y      "actual"
  E = sum over the level of  a * mu^(2-p)          "expected"
```

For Poisson (`p = 1`) this is literally `ln(actual / expected)` — the classic
actual-over-expected update. Because it minimises the deviance exactly along that
coordinate, the fit is monotone and cannot diverge.

**Other links** take a single IRLS step:

```
beta_r  <-  beta_r + sum(a * w * r) / sum(a * w)

  w = (dmu/deta)^2 / V(mu)        IRLS weight
  r = (y - mu) / (dmu/deta)       residual on the link scale
```

`w` and `r` are never formed separately — `loss.rs` returns their product directly,
because the two contain matching factors of `mu` (or `mu(1-mu)`) that cancel
analytically. Forming them apart would divide by a quantity approaching zero only to
multiply it straight back in.

| Family | `dmu/deta` | `V(mu)` | `w` | increment to `beta_r` |
|--------|-----------|---------|-----|------------------------|
| Gaussian | `1` | `1` | `a` | `sum a(y-mu) / sum a` |
| Poisson | `mu` | `mu` | `a·mu` | `sum a(y-mu) / sum a·mu` |
| Gamma | `mu` | `mu^2` | `a` | `sum a(y-mu)/mu / sum a` |
| Tweedie(p) | `mu` | `mu^p` | `a·mu^(2-p)` | `sum a·mu^(1-p)(y-mu) / sum a·mu^(2-p)` |
| Binomial | `mu(1-mu)` | `mu(1-mu)` | `a·mu(1-mu)` | `sum a(y-mu) / sum a·mu(1-mu)` |

Steps are capped at 10 on the link scale and factors at ±500, which only binds under
separation — a level whose observations are all 0 or all 1 has no finite MLE, so the
fit walks it to the boundary and stops rather than producing `NaN`.

### Identifiability

A model carrying an intercept table *and* a free factor for every level is
over-parameterised: you can add a constant to one table and subtract it from the
intercept without changing any prediction. Left alone, backfitting settles anywhere
along that flat direction, so the tables — the actual deliverable — would depend on
table order and starting values rather than on the data.

After every sweep the fit is re-anchored. `GLMOptions.normalization` selects how:

- `BaseLevel` (default) — each feature table's first row goes to zero and the shift
  moves into the intercept. Every other level then reads directly as a relativity
  against that base level.
- `WeightedMean` — each table is centred on its exposure-weighted mean, so the
  intercept carries the overall average level.
- `None` — leave factors where the fit put them. Predictions are still correct, but
  the split between intercept and tables is arbitrary.

Anchoring never changes a prediction. Tables that are offsets, or that contain locked
rows, are left alone.

### Convergence

The fit stops when the *relative change in deviance* falls below `tolerance`
(default `1e-8`), or when `max_iterations` sweeps have run.

`fit_glm_with_diagnostics` returns a `GLMDiagnostics` alongside the model reporting
iterations, whether it converged, final and null deviance, the full deviance history,
and any table rows that received no exposure and so kept their starting factor.

## Python API

```python
from avenue_model import RatingModel, fit_glm, fit_glm_with_diagnostics, GLMOptions
import polars as pl

mean_table = pl.DataFrame({"Rating_Factor": [0.0]})
age_table = pl.DataFrame({
    "Age": [25.0, 35.0, 50.0, 65.0, float("inf")],   # inclusive upper bounds
    "Rating_Factor": [0.0, 0.0, 0.0, 0.0, 0.0],
})

model = RatingModel([mean_table, age_table], objective="poisson")

options = GLMOptions(
    objective="poisson",
    max_iterations=100,
    tolerance=1e-8,
    verbose=True,
    normalization="base_level",
)

fitted, diag = fit_glm_with_diagnostics(
    model, training_data,
    target_col="claims",
    weight_col="exposure",
    options=options,
)
print(diag)          # iterations, converged, deviance, null_deviance, pseudo_r2
predictions = fitted.predict(test_data)
```

### Exposure as an offset

For frequency models the standard idiom is `log(exposure)` as an offset rather than
as a weight, so the fitted factors are rates:

```python
df = df.with_columns(pl.col("exposure").log().alias("log_exposure"))
fitted = fit_glm(model, df, "claim_count", offset_col="log_exposure", options=options)
```

Offset **columns** are fixed per observation. Offset **tables** (`table.as_offset()`)
and offset **rows** are fixed factors that the fit carries but never updates.

### Distribution-specific examples

```python
# Counts
fit_glm(model, df, "claim_count", "exposure", options=GLMOptions(objective="poisson"))

# Severity, weighted by claim count
fit_glm(model, df, "severity", "claim_count", options=GLMOptions(objective="gamma"))

# Pure premium
fit_glm(model, df, "loss_amount", "exposure",
        options=GLMOptions(objective="tweedie", tweedie_power=1.5))

# Binary; predictions come back as probabilities in [0, 1]
fit_glm(model, df, "is_claim", options=GLMOptions(objective="binary"))
```

## Rust API

```rust
use avenue_model::glm::{fit_glm_with_diagnostics, GLMOptions, Normalization};

let options = GLMOptions {
    objective: "poisson".to_string(),
    max_iterations: 100,
    tolerance: 1e-8,
    verbose: true,
    tweedie_power: 1.5,
    normalization: Normalization::BaseLevel,
};

let (fitted, diag) = fit_glm_with_diagnostics(
    &model, &training_df, "target", Some("weight"), Some("log_exposure"), options
)?;
println!("deviance {:.6} after {} sweeps", diag.deviance, diag.iterations);
```

## Input requirements

Fitting rejects, rather than silently working around:

- a target, weight or offset column that is not `Float64`, or contains nulls or
  non-finite values
- negative weights
- **any observation that fails to match a row of any table.** An unmatched observation
  would contribute nothing to that table's linear predictor and be excluded from its
  update, producing a plausible-looking fit from a model that quietly dropped a term.
  Numeric tables therefore need a final unbounded (`inf`) row, categorical tables need
  every level covered or a `-999` wildcard, and every table's feature columns must be
  present with the expected dtype.
- an intercept table (index 0) with more than one row

## Testing

```bash
cargo test --lib glm                 # all GLM tests
cargo test --lib glm_correctness     # the correctness harness specifically
```

`glm_correctness_tests.rs` asserts on numbers, in two ways:

1. **Closed form.** A saturated model — intercept plus one factor with every level
   free — has an exact ML solution for every exponential family: each level's fitted
   mean equals that level's weighted mean of `y`. This holds for every link, which
   makes it a sharp check that the link scale is handled correctly.

2. **External reference.** Fits pinned from statsmodels in `glm_reference_data.rs`,
   covering all five families plus an offset case. Avenue's parameterisation carries
   an intercept *and* every level, so raw coefficients are not comparable to
   statsmodels' treatment coding; the tests compare the two quantities that are
   invariant to the parameterisation — fitted means per row, and level contrasts
   within a table.

Regenerate the reference fixtures (requires `numpy`, `scipy`, `statsmodels`):

```bash
python scripts/gen_glm_reference.py
```

## Performance

No benchmark against other GLM libraries exists yet, and the numbers previously
published here were taken in debug mode, which understates release builds by roughly
an order of magnitude. Rather than replace them with equally unanchored figures, this
section stays empty until there is a release-mode comparison against **glum** (the
purpose-built insurance GLM library, and the honest speed target) and **statsmodels**
(the correctness oracle), fitting the same data.

The interesting claim is not raw speed but that Avenue never materialises the design
matrix, so its advantage should grow with the number of levels — a benchmark should
hold rows fixed and sweep the level count to show that.

## Module Structure

```
src/glm/
├── mod.rs              # Public API exports
├── fitting.rs          # Coordinate descent, normalization, diagnostics
├── loss.rs             # Families: links, variance functions, IRLS weights, deviance
├── matching.rs         # Observation-to-table-row matching
└── utils.rs            # Helper functions (weighted means, etc.)
```

## Known gaps

- No standard errors or covariance for the fitted factors — the largest remaining gap
  for variable selection. The final `X'WX` is block-sparse in this representation, so
  per-level standard errors are cheap once wired up.
- No dispersion estimate, AIC or BIC.
- Numeric tables are step functions only. Continuous and linear terms need knot
  semantics (interpolating between rows rather than stepping), which is the natural
  next extension.
- No regularisation. For rating tables the high-value forms are credibility shrinkage
  of sparse levels and difference penalties on adjacent ordinal levels, more than L1
  variable selection.
- Matching allocates per observation per table, and is the dominant cost on large
  datasets.

## References

- Nelder, J. A., & Wedderburn, R. W. (1972). *Generalized linear models*
- McCullagh, P., & Nelder, J. A. (1989). *Generalized Linear Models*, 2nd ed.
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning*
- Wood, S. N. (2017). *Generalized Additive Models: An Introduction with R*
