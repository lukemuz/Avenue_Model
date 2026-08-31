# GLM fitting

Avenue fits generalized linear models either directly on rating tables or through a
global IRLS solver. The table solver avoids materializing a dummy-coded
observation-by-parameter matrix; the global solver handles smaller, well-supported
problems with direct linear algebra.

**Contents:** [Solvers](#solvers) · [Table algorithm](#table-algorithm) ·
[Benchmarks](#benchmarks) · [Variates](#variates) · [Convergence](#convergence) ·
[Inference](#inference) · [API](#python-api) · [Requirements](#input-requirements)

## Solvers

`solver="auto"` selects the global solver when the model supports it and otherwise uses
table descent.

| Setting | Behavior |
|---|---|
| `auto` | Choose global when supported; otherwise use table descent |
| `global` | Require global IRLS and error when unsupported |
| `table` | Require the low-memory table algorithm |

The global path uses a treatment-coded Gram matrix, direct solves for unpenalized and
ridge fits, and coordinate descent for lasso and elastic net. It is limited to
base-level-normalized step tables with an updatable intercept, no locked rows and at most
6,000 parameters. Its memory cost is `O(p²)`.

The table path supports the full model representation and scales with table-row match
indices rather than a design matrix. It is strongest when `n` is large and the model has
ordinary rating-factor structure.

## Table algorithm

One sweep visits each table and updates its rows while holding other tables fixed. Row
matching is resolved once before fitting, so later sweeps read cached indices.

| | Avenue table solver | Design-matrix IRLS |
|---|---|---|
| per iteration | `O(n · T)` | `O(n · T²)` to form `X'WX` |
| iterations | usually tens | usually 5–20 |
| memory | table-row match indices | design representation plus `X'WX` |

For a log-link family, a level has the exact coordinate update

```text
beta_r <- beta_r + log(A / E)

A = sum a * mu^(1-p) * y
E = sum a * mu^(2-p)
```

For Poisson this is the familiar `log(actual / expected)`. Gaussian and Binomial use an
IRLS coordinate step. Ridge, lasso and elastic-net penalties modify the same updates.

Three optimizations address the usual weakness of backfitting:

- SQUAREM extrapolates the dominant error mode and accepts only improving steps.
- Near-aliased table pairs are detected by canonical correlation and solved jointly.
- Sweep order updates the most strongly coupled table first.

`GLMDiagnostics.table_conditioning` summarizes collective dependence. Values near 1 are
well-conditioned; above roughly 10, expect hundreds of sweeps. A direct/global solver is
usually preferable for strongly conditioned plans and unpenalized Gaussian models.

## Benchmarks

Every reported result is gated on comparable fitted means. Times are fit-only release
builds and the fastest of repeated runs. Absolute times depend on hardware; compare
engines within a table. glum is the primary competitor because its `tabmat` backend also
avoids a dense dummy-coded matrix. statsmodels is used as a correctness reference.

```bash
python scripts/bench_glm.py       # synthetic families
python scripts/bench_fremtpl.py   # French motor
python scripts/bench_housing.py   # King County housing
python scripts/bench_real.py      # NYC taxi and census income
python scripts/bench_engines.py   # glum, scikit-learn and H2O
python scripts/bench_large.py     # 20M rows and conditioning sweep
python scripts/bench_isolated.py  # whole-process peak memory
```

### Synthetic data

Five independent tables and 81 parameters favor coordinate descent and represent an
upper bound rather than a typical result.

| | Avenue | glum | statsmodels |
|---|---:|---:|---:|
| Poisson, 1M rows | **0.100 s** | 0.416 s | — |
| Gamma, 1M rows | **0.114 s** | 0.408 s | — |
| Tweedie(1.5), 1M rows | **0.182 s** | 0.286 s | — |
| Gaussian, 1M rows | 0.079 s | **0.074 s** | — |
| Poisson, 5M rows | **0.695 s** | 2.518 s | — |

Whole-process peak RSS, including data and interpreter:

| | Avenue | glum | statsmodels |
|---|---:|---:|---:|
| Poisson, 100k rows | **113 MB** | 191 MB | 932 MB |
| Poisson, 1M rows | **196 MB** | 336 MB | — |
| Poisson, 5M rows | **564 MB** | 1,200 MB | — |

### Real data

| unpenalized | Avenue fit | Avenue peak | glum fit | glum peak |
|---|---:|---:|---:|---:|
| freMTPL2, 678k rows, 79 params, Poisson | **0.26 s** | **87 MB** | 0.49 s | 165 MB |
| freMTPL2, 678k rows, 270 params, Poisson | **0.41 s** | **87 MB** | 1.64 s | 119 MB |
| NYC taxi, 2.75M rows, 577 params, Gamma | 5.22 s | **272 MB** | **3.82 s** | 479 MB |
| census income, 45k rows, 116 params, Binomial | **0.15 s** | **6 MB** | 0.21 s | 11 MB |
| house sales, 21.6k rows, 92 params, Gamma | **0.046 s** | **9 MB** | 0.055 s | 81 MB |
| house sales, 21.6k rows, 92 params, Gaussian | 0.034 s | 1 MB | **0.012 s** | **0 MB** |

Avenue wins four of six fits and five of six memory comparisons. glum wins the
high-cardinality taxi model. Its direct solve also wins the small Gaussian model, where
one factorization is the exact linear-model answer.

### The rest of the field

`bench_engines.py` compares Avenue with glum, scikit-learn and H2O across three families
and three penalty settings on a separate four-core machine. Avenue is fastest in five of
the six scenarios where every engine returns a comparable solution.

| fit time, Avenue = 1.00x | Avenue | glum | scikit-learn | H2O |
|---|---:|---:|---:|---:|
| freMTPL2, Poisson, unpenalized | **1.00x** | 2.5x | 2.9x | 3.2x |
| census income, Binomial, ridge | **1.00x** | 1.8x | 1.3x | 4.3x |
| freMTPL2, Poisson, lasso | **1.00x** | 2.6x | n/a | n/a |

`n/a` means an engine cannot express the model or fails the fitted-mean agreement gate.
scikit-learn's `newton-cholesky` wins the small house-sales Gamma cases; H2O is slower on
every comparable case. The full nine-case output remains reproducible with
`scripts/bench_engines.py`.

### At twenty million rows

| 20M rows, 501 parameters | Avenue | glum |
|---|---:|---:|
| 100 tables of 6 levels | **39.3 s, 10.8 GB** | 865.9 s, 21.1 GB |
| 5 tables of 101 levels | **3.1 s, 1.2 GB** | 16.5 s, 3.6 GB |

The two cases have the same data and parameter count. Their different table layouts
show the `O(n · T)` versus `O(n · T²)` iteration cost directly. Fitted means agree to
`5.6e-09` and `3.2e-09`.

### Conditioning limit

With 1M rows and 100 tables loaded on a shared latent driver:

| pairwise correlation | `table_conditioning` | Avenue | glum |
|---:|---:|---:|---:|
| 0.00 | 1.8 | **4.3 s** | 35.6 s |
| 0.10 | 10.1 | **29.6 s** | 31.2 s |
| 0.20 | 19.2 | 95.8 s | **31.2 s** |
| 0.30 | 28.4 | 240.6 s | **30.9 s** |

Per-sweep cost remains low, but the number of sweeps rises with collective dependence.
Near-alias pair solving handles two redundant tables; it cannot remove a direction shared
across many tables. `solver="auto"` selects the global path when supported.

## Variates

A variate uses the values attached to table rows as a continuous covariate rather than
estimating one free factor per row. Polynomial degrees are solved jointly to avoid slow
coordinate cycling between correlated powers.

```python
model = model.as_variate(
    table_index=1,
    values=[20.0, 30.0, 40.0, 50.0, 65.0],
    degree=2,
)
result = fit_glm_with_diagnostics(model, df, "claims", "exposure")
print(result.model.variate_coefficients(1))
```

Inference reports the highest-degree coefficient and its Wald statistic. Refit at
different degrees and compare deviance or AIC when selecting polynomial complexity.

## Identifiability

An intercept plus every level of every table is over-parameterized. After each sweep,
Avenue re-anchors factors without changing predictions:

- `base_level` fixes each table's first row at zero and is the default.
- `weighted_mean` centers tables by exposure-weighted mean.
- `none` leaves the arbitrary split between intercept and tables unchanged.

Offset tables and locked rows are not re-anchored. Inference uses a reduced,
treatment-coded basis and reports unestimable rows as aliased rather than failing the
whole fit.

## Convergence

The fit stops when the largest scaled absolute score reaches `tolerance` (default
`1e-9`). This tests the parameters directly and is comparable to glum's `gradient_tol`.
Deviance alone is insufficient near the optimum because it is quadratic in parameter
error and reaches its floating-point floor early.

If deviance ceases to improve at machine precision, the fit eventually returns
`converged = false` with its final score. SQUAREM proposals are accepted only when the
score does not worsen. Callers should always check `converged`.

## Inference

For unpenalized fits, `GLMDiagnostics` reports standard errors, dispersion, Pearson
chi-squared, residual degrees of freedom, log-likelihood, AIC and BIC where defined.
Standard errors are treatment contrasts against the anchoring reference; aliased or
uninformative rows receive `NaN`. Penalized fits do not report standard errors because
their coefficients are deliberately biased.

The implementation is validated against statsmodels. On the census-income case, 113
estimable standard errors agree to `3.9e-10`; Avenue marks two separated levels as
unestimable.

## Python API

Most Python users should start with `Plan`; see the [root README](../../README.md). The
lower-level API accepts an existing `RatingModel`:

```python
from avenue_model import GLMOptions, fit_glm_with_diagnostics

result = fit_glm_with_diagnostics(
    model,
    training_data,
    target_col="claims",
    weight_col="exposure",
    options=GLMOptions(solver="auto", tolerance=1e-9),
)

print(result.diagnostics.converged)
predictions = result.predict(test_data)
```

For a frequency model, either fit claims per exposure using exposure as a prior weight,
or fit counts with `log(exposure)` supplied through `offset_col`.

## Rust API

```rust
use avenue_model::glm::{fit_glm_with_diagnostics, GLMOptions};

let options = GLMOptions {
    objective: "poisson".to_string(),
    ..Default::default()
};

let (fitted, diagnostics) = fit_glm_with_diagnostics(
    &model,
    &training_df,
    "target",
    Some("weight"),
    Some("log_exposure"),
    options,
)?;
```

## Input requirements

Fitting rejects:

- null, non-finite or incorrectly typed target, weight and offset columns;
- negative weights;
- observations that fail to match a row in any table;
- an intercept table with more than one row; and
- unsupported feature-column dtypes.

Numeric tables should end in an unbounded row. Categorical tables must cover every level
or provide the `-999` wildcard. `Plan.check()` reports these faults before fitting.

## Testing and source

```bash
cargo test --lib glm
python scripts/gen_glm_reference.py  # regenerate statsmodels fixtures
```

- [`fitting.rs`](fitting.rs) — global and table solvers, acceleration and diagnostics
- [`matching.rs`](matching.rs) — observation-to-row matching
- [`redundancy.rs`](redundancy.rs) — correlations, joint pairs and conditioning
- [`inference.rs`](inference.rs) — standard errors and fit statistics
- [`loss.rs`](loss.rs) — links, variance, deviance and likelihood

## References

- McCullagh, P., & Nelder, J. A. (1989). *Generalized Linear Models*, 2nd ed.
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning*
- Varadhan, R., & Roland, C. (2008). *Simple and globally convergent methods for
  accelerating the convergence of any EM algorithm* (SQUAREM)
