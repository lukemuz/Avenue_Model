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

### Variates: continuous drivers in a table

By default every row of a table carries its own free factor, so a five-row age table
spends four parameters. Often that is not what you want — age drives risk smoothly, and
four free levels spend degrees of freedom estimating wiggle you do not believe in.

Marking a table a **variate** ties its factors to a single slope:

```
factor[r] = slope * values[r]     (+ a constant the intercept absorbs)
```

Five rows, **one** parameter, whatever the row count. Three things follow:

- The fitted curve is smooth by construction, not by penalty, and at degree 1 monotone
  as well.
- Rows with little or no exposure still get a sensible factor, read off the line rather
  than left stranded at their starting value.
- **Lookup does not change.** The table is still an ordinary step table with the same
  bounds and the same `Rating_Factor` column, so any rating engine reads it unchanged.
  Nothing interpolates and nothing is approximated — the table *is* the fitted model.

```rust
use avenue_model::rating_model::RatingTable;

// Bounds as usual: inclusive upper bounds, ascending, last one unbounded.
let age = RatingTable::new(age_df, None)
    .as_variate(vec![20.0, 30.0, 40.0, 50.0, 65.0])?;
```

`values` is what each row is worth on the driver's scale, one per row. It is supplied
rather than derived from the table's own numeric column, because that column holds bin
*upper bounds*: the top bin's bound is normally `inf`, and a bound is the edge of a bin
rather than a point inside it. Supplying the values is also how you say what the
open-ended top band is worth.

After fitting, the table looks like this — five factors, all exactly on one line:

```
 Age (bound)   values    Rating_Factor
       20        20         0.0000        <- anchored base
       30        30         0.0850
       40        40         0.1700
       50        50         0.2550
      inf        65         0.3825
```

`table.variate_slope()` recovers the slope. Standard errors work out as you would
expect: each row's is the slope's, scaled by that row's distance from the base value,
so the base row's is exactly 0 and the whole table contributes one column to `X'WX`.

Rows cannot be locked on a variate table — every factor comes from the fitted curve, so
pinning a single row has no representation. Lock the whole table with `as_offset()`
instead.

#### Polynomials

`as_polynomial_variate(values, degree)` fits a curve instead of a line:

```
factor[r] = sum over m of  beta_m * values[r]^m
```

Degree 1 is a line and costs one parameter, degree 2 bends once and costs two, and so
on. The table always keeps every row and is always read as a step table — the degree
only decides how many parameters the fit spends describing the shape.

The degree cannot reach the number of distinct values. At `distinct - 1` the polynomial
already passes exactly through every row, which is the same fit as free levels; beyond
that the extra terms are not identified. The hard ceiling is `MAX_VARIATE_DEGREE` (8),
well past anything defensible — high-degree polynomials oscillate between the points
they pass through, which is the opposite of what a variate is for.

**Is the curve earning its bend?** `GLMInference.variate_terms` carries each variate's
coefficients and their standard errors, and `top_degree_z()` gives the Wald statistic
for the highest power:

```rust
for terms in &diag.inference.unwrap().variate_terms {
    println!("table {}: degree {}, coefficients {:?}", terms.table_index, terms.degree, terms.coefficients);
    if let Some(z) = terms.top_degree_z() {
        println!("  top degree z = {:.2}", z);   // |z| < 2 -> try one degree lower
    }
}
```

The standard errors are on the rescaled basis the fit uses, where the driver is mapped
onto `[-1, 1]`. That basis is triangular — the `m`th column involves no power above `m`
— so the **top** degree's z statistic is the same whatever scale the lower terms are
expressed on, which makes it a valid test for dropping that degree. The lower degrees'
z statistics depend on how the basis is centred; to judge them, refit a degree lower and
compare deviance.

`coefficients` is on the raw scale, so the fitted curve can be written down directly.
`variate_slope()` is a degree-1 convenience and returns `None` above that — a curve has
no single slope.

#### Conditioning

Two things keep the small solve well behaved, both pure reparameterisations that leave
the fit and its fixed point identical:

- **Rescaling.** The powers are taken of `u = (v - centre) / half_range`, which lies in
  `[-1, 1]`. Age to the fourth is around ten million while age is around forty; without
  rescaling the normal matrix spans orders of magnitude and the solve loses most of its
  significant digits.
- **Centring.** Each column is then centred on its weight-weighted mean, making the step
  orthogonal to the intercept under the current weights — the exact Newton step for the
  shape given the level is free to adjust. Without it, a slope column that never crosses
  zero is nearly collinear with the intercept, and coordinate descent between the two
  crawls: the deviance goes flat long before the coefficients have settled, so the fit
  reports convergence having not arrived.

The powers are also solved **jointly**, as one `d x d` system rather than one coordinate
at a time. `v` and `v^2` are strongly correlated over any range that does not straddle
zero, so cycling between them would converge just as slowly.

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

The fit stops when the **largest absolute score** over every free parameter falls to
`tolerance` (default `1e-9`), or when `max_iterations` sweeps have run.

The score of a parameter is the derivative of the log-likelihood with respect to it,
and it is zero only at the optimum — so this measures how far the *factors* still have
to move. It is the same criterion glum applies as `gradient_tol`. The score is scaled
by the total absolute residual, which makes the threshold independent both of the
number of observations and of the units the response is measured in; a Gaussian fit on
currency and one on log-odds mean the same thing by `1e-9`.

> **This replaced a test on the relative change in deviance, which did not work.**
> Deviance is *quadratic* in the parameter error near the optimum, so a deviance
> tolerance of `1e-t` buys only about `1e-(t/2)` on the factors — and when convergence
> is slow, the deviance goes flat while the parameters are still moving. On the French
> Motor Third-Party Liability data the old rule reported convergence with the fitted
> means `1.1e-04` away from the answer. Under the score rule the same fit reaches
> `2.1e-07`. It costs more sweeps, because the earlier fit was not finished.

A fit that cannot reach the tolerance stops once the score has failed to improve for
twelve consecutive sweeps, and reports `converged = false` with the score it achieved.
Two near-aliased tables — `Area` and `Density` in the French motor data, where one is a
rebanding of the other — are the usual cause. **`converged` is the flag to check; it
now means what it says.**

`fit_glm_with_diagnostics` returns a `GLMDiagnostics` alongside the model reporting
iterations, whether it converged, `max_gradient` and the full `gradient_history`, final
and null deviance, the deviance history, and any table rows that received no exposure
and so kept their starting factor. A gradient history that falls steeply and then
crawls is the signature of near-aliased tables.

### Standard errors

Also on `GLMDiagnostics`, unless `compute_standard_errors` is turned off.

The full design — an intercept plus every level of every table — is rank deficient, so
`X'WX` is singular and the individual factors have no standard error at all. What *is*
estimable is any contrast invariant to shifting a constant between a table and the
intercept. Inference therefore runs in a reduced, full-rank basis (an intercept plus
every level except each table's reference row, i.e. treatment coding), and the standard
error reported against each row is that of the contrast the row actually represents
under the model's anchoring. Under the default `BaseLevel` anchoring the reported
factors *are* the treatment contrasts, so the numbers line up directly with what
statsmodels or R report.

`standard_errors[t][r]` aligns index-for-index with `model_tables()`:

| Row | Standard error |
|-----|----------------|
| The anchoring reference | exactly `0` — fixed by construction, not estimated |
| No exposure | `NaN`; also listed in `unfitted_rows` |
| Aliased | `NaN`; also listed in `aliased_rows` |
| Anything else | the contrast's standard error |

Rank deficiency is treated as information, not failure. Two tables keyed on the same
feature are collinear; a completely separated level has zero IRLS weight and confounds
with the intercept. Those parameters are set aside and listed in `aliased_rows`, and
everything else keeps usable standard errors — the same convention as R marking a
coefficient `NA`. Degrees of freedom spend the model's rank, not its nominal parameter
count.

Also reported: `dispersion` (1 for Poisson and Binomial, Pearson chi-squared over
residual degrees of freedom otherwise), `pearson_chi2`, `df_residual`, `n_parameters`
(the rank), `log_likelihood`, `aic` and `bic`.

The log-likelihood, and so AIC and BIC, are `None` for Tweedie: its density is an
infinite series with no closed form, and reporting a number would mean quietly
substituting an approximation. For Gaussian the likelihood is evaluated at the ML
variance `SSE/n`, not the Pearson estimate `SSE/(n-p)` used for standard errors — the
same distinction statsmodels makes. AIC and BIC count the mean parameters only, again
matching statsmodels.

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
print(diag)          # iterations, converged, deviance, pseudo_r2, dispersion, aic
predictions = fitted.predict(test_data)
```

### A continuous driver

```python
# Age as a single slope rather than five free levels.
age_table = pl.DataFrame({
    "Age": [20.0, 30.0, 40.0, 50.0, float("inf")],   # inclusive upper bounds
    "Rating_Factor": [0.0] * 5,
})
model = RatingModel([mean_table, age_table], objective="poisson")
model = model.as_variate(1, [20.0, 30.0, 40.0, 50.0, 65.0])

fitted, diag = fit_glm_with_diagnostics(model, df, "claims", "exposure", options=options)
print(fitted.variate_slope(1))     # the one estimated parameter
print(fitted.model_tables()[1])    # five factors, all on that line

# A curve instead of a line: same table, two parameters.
model = model.as_variate(1, [20.0, 30.0, 40.0, 50.0, 65.0], degree=2)
fitted, diag = fit_glm_with_diagnostics(model, df, "claims", "exposure", options=options)

print(fitted.variate_coefficients(1))      # [beta_1, beta_2] on the raw age scale
for table_index, degree, coefs, ses, z in diag.variate_terms:
    print(f"table {table_index}: degree {degree}, top-degree z = {z:.2f}")
    # |z| < 2 suggests the bend is not earning its parameter; try degree=1.
```

### Reading the standard errors

`diag.standard_errors` lines up index-for-index with `model_tables()`:

```python
for t, table in enumerate(fitted.model_tables()):
    for r, factor in enumerate(table["Rating_Factor"]):
        se = diag.standard_errors[t][r]
        z = diag.z_value(t, r, factor)          # None for reference/aliased rows
        flag = " (base)"    if se == 0          else \
               " (aliased)" if (t, r) in diag.aliased_rows else \
               " (no data)" if (t, r) in diag.unfitted_rows else ""
        print(f"table {t} row {r}: {factor:+.4f}  se={se:.4f}  z={z}{flag}")
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
   statsmodels' treatment coding; the tests compare quantities that are invariant to
   the parameterisation — fitted means per row and level contrasts within a table —
   along with deviance, dispersion, residual degrees of freedom, standard errors,
   log-likelihood and AIC. A variate case is included, fitted in statsmodels as an
   ordinary continuous covariate taking each record's band value.

Regenerate the reference fixtures (requires `numpy`, `scipy`, `statsmodels`):

```bash
python scripts/gen_glm_reference.py
```

## Performance

Two benchmarks, both release-mode, both gated on the engines agreeing about the fitted
means: `scripts/bench_glm.py` (synthetic) and `scripts/bench_fremtpl.py` (the French
motor data glum builds its own `wide-insurance` benchmark from). Fit seconds only,
fastest of three runs, 5 tables and 81 parameters; **glum** is the speed target and
**statsmodels** the correctness oracle.

| synthetic, 1M rows | Avenue | glum | statsmodels |
|--------------------|-------:|-----:|------------:|
| Poisson            | 0.100  | 0.441 | — |
| Gamma              | 0.119  | 0.414 | — |
| Tweedie(1.5)       | 0.212  | 0.287 | — |
| Gaussian           | 0.081  | 0.075 | — |
| Poisson, 100k rows | 0.012  | 0.035 | 1.520 |
| Poisson, 5M rows   | 0.740  | 2.540 | — |

Peak memory at 1M rows is 54 MB against glum's 112, and at 5M rows 369 against 683 —
a flat ~1.8x, so it is a constant-factor advantage rather than a different scaling law.
Both engines are `O(n · tables)`; glum's tabmat does not materialise a dense design
matrix either. The `O(n · parameters)` blowup Avenue genuinely avoids belongs to the
dummy-coded route, which is what the 1.5 GB statsmodels row above is measuring.

**The real result is the one that does not flatter us.** Those synthetic factors are
drawn independently, which is the best case for coordinate descent. On the French motor
data, where `Area` is essentially a rebanding of `Density` and driver age tracks
bonus-malus:

| freMTPL2, 678k rows, 79 parameters | fit seconds | sweeps |
|------------------------------------|------------:|-------:|
| Avenue                             | 3.61        | 254    |
| glum, `irls-ls`                    | 0.45        | 5      |
| glum, `irls-cd`                    | 0.53        | 6      |

Same per-sweep machinery, 50x the sweeps. Backfitting converges at a rate set by the
canonical correlation between the blocks, and correlated rating factors are the norm
rather than the exception. Per-sweep cost is not the problem and further micro-optimising
it will not close this gap; see *Known gaps*.

What does keep the per-sweep cost competitive: `mu` is carried through a sweep as
`mu *= exp(delta_r)` instead of being re-derived per observation per table, which is the
difference between `n_rows` exponentials and `n` of them; `mu^(1-p)` is specialised away
from `powf` for the exponents the common families produce (Poisson's is `mu^0`); and the
scatter-adds run under rayon above 100k rows.

## Module Structure

```
src/glm/
├── mod.rs              # Public API exports
├── fitting.rs          # Coordinate descent, normalization, diagnostics
├── inference.rs        # Standard errors, dispersion, aliasing, AIC/BIC
├── loss.rs             # Families: links, variance, IRLS weights, deviance, likelihood
├── matching.rs         # Observation-to-table-row matching
└── utils.rs            # Helper functions (weighted means, etc.)
```

## Known gaps

- Only Wald z for the top polynomial degree. Choosing a degree properly wants
  likelihood-ratio tests or AIC across a fitted sequence; for now, refit at each degree
  and compare `deviance` or `aic` yourself.
- Lookup is always a step lookup. A table cannot interpolate between its rows, so a
  variate's continuity lives in the *pattern of factors*, not in the prediction — two
  ages in the same band get the same factor. Interpolating tables are designed but not
  built.
- Only Wald standard errors. No likelihood-ratio tests, profile intervals, or robust
  / sandwich covariance.
- No regularisation. For rating tables the high-value forms are credibility shrinkage
  of sparse levels and difference penalties on adjacent ordinal levels, more than L1
  variable selection.
- **Correlated tables converge slowly.** This is the live limitation, not a micro-
  optimisation: backfitting's rate is set by the canonical correlation between blocks,
  so two tables that carry nearly the same information trade a constant back and forth
  for hundreds of sweeps. The measured cost is 254 sweeps against glum's 5 on the French
  motor data. The fix is structural — assemble `X'WX` from the tables and take a real
  Newton step (its diagonal blocks are diagonal, its off-diagonal blocks are weighted
  contingency tables, and it costs `O(p²)` memory rather than `O(np)`), falling back to
  accelerated coordinate descent when `p` is too large for that.

## References

- Nelder, J. A., & Wedderburn, R. W. (1972). *Generalized linear models*
- McCullagh, P., & Nelder, J. A. (1989). *Generalized Linear Models*, 2nd ed.
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning*
- Wood, S. N. (2017). *Generalized Additive Models: An Introduction with R*
