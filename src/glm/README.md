# GLM Fitting Module

Generalized linear models fitted by block coordinate descent directly on rating tables —
no design matrix, no coefficient vector to map back.

**Contents:** [How it works](#how-it-works) · [Benchmarks](#benchmarks) ·
[Variates](#variates-continuous-drivers-in-a-table) · [Identifiability](#identifiability) ·
[Convergence](#convergence) · [Standard errors](#standard-errors) ·
[Python API](#python-api) · [Rust API](#rust-api) ·
[Input requirements](#input-requirements) · [Testing](#testing) ·
[Known gaps](#known-gaps)

---

## How it works

### The sweep

Each **sweep** visits every table in turn and updates all of that table's rows at once,
holding every other table fixed. Sweeps repeat until the score is flat.

A step table assigns each observation to exactly one row, so the weighted least-squares
step for a level collapses to a **scalar** — a weighted sum over the observations that
landed on that row. No matrix is formed and no matrix is factorised.

| | Avenue | IRLS with a design matrix |
|---|---|---|
| per iteration | `O(n · T)` — touch each row once per table | `O(n · T²)` — form `X'WX` from every pair of blocks |
| iterations | tens (linear convergence) | 5–20 (quadratic convergence) |
| memory | table-row match indices | `X`, or `tabmat`'s block representation, plus `X'WX` |

That trade — a much cheaper iteration against more of them — is the whole performance
story, in both directions. The [benchmarks](#benchmarks) below measure where it pays and
where it does not.

### Matching

"Which row does this observation fall in" is answered once per table, before the first
sweep, into a `Vec<u32>` per table. Every sweep then reads indices rather than re-deciding
anything, which is why the fixed cost is 3% of a real fit.

One row-by-row scan in `matching.rs` defines the answer, and everything else is a shortcut
that has to reproduce it exactly — the tests assert that directly against the scan rather
than against expected values:

| plan | when | cost per observation |
|---|---|---|
| `Constant` | no feature columns (the intercept table) | none |
| `SortedNumeric` | one `Float64` column, non-decreasing bounds | binary search, `O(log rows)` |
| `Categorical` | one `Int32` column, no null table values | one hash lookup |
| pre-resolved scan | anything else, including interactions | `O(rows)`, no allocation |
| reference scan | dtypes or nulls the shortcuts cannot express | `O(rows)` plus a `HashMap` per observation |

The distinction that matters to a caller is `Float64` versus `Int32`, because it decides
which shortcut applies: a `Float64` column is a numeric band and an `Int32` one is a
category code. Both reach the same answer at the same speed, and `bench_real.py` fits
every design both ways to keep it that way.

### Supported families

| Family | Link | Variance `V(mu)` | Typical use |
|--------|------|------------------|-------------|
| Gaussian | Identity | `1` | Linear regression |
| Poisson | Log | `mu` | Claim counts |
| Gamma | Log | `mu^2` | Claim severity |
| Tweedie(p) | Log | `mu^p` | Pure premium (mixed zero / positive) |
| Binomial | Logit | `mu(1-mu)` | Binary classification |

### The two update rules

**Log-link families get an exact closed-form coordinate solve.** With `mu_i` the current
fitted mean and `a_i` the prior weight, the level's score equation solves to

```
beta_r  <-  beta_r + ln( A / E )

  A = sum over the level of  a * mu^(1-p) * y      "actual"
  E = sum over the level of  a * mu^(2-p)          "expected"
```

For Poisson (`p = 1`) this is literally `ln(actual / expected)` — the classic
actual-over-expected update. Because it is the exact *minimiser* along that coordinate
rather than a step towards it, the deviance falls every sweep and the fit cannot diverge.

**Other links take a single IRLS step:**

```
beta_r  <-  beta_r + sum(a * w * r) / sum(a * w)

  w = (dmu/deta)^2 / V(mu)        IRLS weight
  r = (y - mu) / (dmu/deta)       residual on the link scale
```

| Family | `dmu/deta` | `V(mu)` | `w` | increment to `beta_r` |
|--------|-----------|---------|-----|------------------------|
| Gaussian | `1` | `1` | `a` | `sum a(y-mu) / sum a` |
| Poisson | `mu` | `mu` | `a·mu` | `sum a(y-mu) / sum a·mu` |
| Gamma | `mu` | `mu^2` | `a` | `sum a(y-mu)/mu / sum a` |
| Tweedie(p) | `mu` | `mu^p` | `a·mu^(2-p)` | `sum a·mu^(1-p)(y-mu) / sum a·mu^(2-p)` |
| Binomial | `mu(1-mu)` | `mu(1-mu)` | `a·mu(1-mu)` | `sum a(y-mu) / sum a·mu(1-mu)` |

`w` and `r` are never formed separately — `loss.rs` returns their product directly,
because the two contain matching factors of `mu` (or `mu(1-mu)`) that cancel
analytically. Forming them apart would divide by a quantity approaching zero only to
multiply it straight back in.

Steps are capped at 10 on the link scale and factors at ±500, which only binds under
separation — a level whose observations are all 0 or all 1 has no finite MLE, so the fit
walks it to the boundary and stops rather than producing `NaN`.

### What makes it fast

Four things, in rough order of how much they are worth:

- **The exact `ln(A/E)` solve.** A coordinate minimiser rather than a gradient step, so
  no line search and no step-size tuning.
- **SQUAREM acceleration** (`squarem_steplength`). Three-point extrapolation reads the
  dominant error mode off the iterates themselves and jumps along it. A clean geometric
  decay is exactly what it annihilates: 254 sweeps to 66 on the French motor data, 79 to
  50 on housing. Every jump is checked and rejected if it does not help, so the worst
  case is a few wasted passes.
- **Joint solving of near-aliased table pairs** (`update_pair`). Two tables describing
  the same driver — a density band and an area code — are the case that brings a backfit
  to a crawl. Solved as one block instead of alternately: 66 sweeps to **15**.
- **Incremental `mu`.** Carried through a sweep as `mu *= exp(delta_r)` rather than
  re-derived per observation per table, which is `n_rows` exponentials instead of `n` of
  them. `mu^(1-p)` is specialised away from `powf` for the exponents the common families
  produce (Poisson's is `mu^0`), and the scatter-adds run under rayon above 100k rows.

The pair detector (`redundancy.rs`) is what makes the third one possible without a
heuristic. The first canonical correlation between two tables is *exactly* the factor by
which their shared direction survives each sweep, and it comes from a weighted contingency
table rather than anything the size of the design. On the French motor data it reads
`Density`/`Area` at **0.9713**, whose square — 0.9434 — is the 0.9431 tail rate the fit
actually exhibits. It predicts the convergence rate before fitting anything, in 13 ms over
36 pairs.

Its cost is `O(k_a · k_b)` per power iteration per pair, so it is negligible on ordinary
plans — 0.8% of the taxi fit and 2.2% of the census one — but it grows with the square of
the level count, and the pairs go out to the thread pool for that reason. The worst case
is a wide plan whose factors are close to independent: a flat spectrum is what power
iteration converges slowest on, so the pairs that turn out not to matter are the ones that
cost the most to dismiss. Ten independent 250-level tables spend 86 ms deciding, against
1.8 ms for the same plan at 5 levels. Real designs are nowhere near that — no pair in
either real dataset needs more than 70 of the 200 available iterations.

The null deviance that `pseudo_r2` is measured against also has a closed form under a log
link. Writing `A` for `sum a·e^((1-p)·o)·y` and `B` for `sum a·e^((2-p)·o)`, the update
from any `beta` is `ln(A/B) - beta`, so `beta + step(beta) = ln(A/B)` wherever it started:
one pass is the answer and every later one re-derives it. That fixed cost is 34 ms on
freMTPL2 and agrees with statsmodels' intercept-only fit to `1.4e-13`. Identity and logit
links have no such collapse and iterate until the step stops shrinking.

---

## Benchmarks

**Methodology.** All release builds. Every benchmark is **gated on the engines agreeing
about the fitted means** before any timing is reported — a fast wrong answer fails.
Times are fit time only, fastest of three runs. Absolute numbers are machine-dependent;
the ratios are the point.

- **glum** is the speed comparison. Its `tabmat` backend avoids a dense dummy-coded
  design matrix too, so this is a strong comparison rather than a naive baseline. Its
  default unpenalised solver is `irls-ls` (Cholesky); `irls-cd` is shown where it was
  affordable to run.
- **statsmodels** is the correctness oracle, and takes the dense route.

```bash
python scripts/bench_glm.py       # synthetic, four families
python scripts/bench_fremtpl.py   # French motor third-party liability
python scripts/bench_housing.py   # King County house sales
python scripts/bench_real.py      # NYC taxi, census income
python scripts/bench_large.py     # 20M rows; --correlation for the conditioning sweep
python scripts/bench_isolated.py  # peak memory, one engine per process
```

### Synthetic data

Five tables, 81 parameters, factors drawn independently — the best case for a coordinate
method, and worth reading as an upper bound rather than a typical result.

| | Avenue | glum | statsmodels |
|---|-------:|-----:|------------:|
| Poisson, 1M rows | **0.096 s** | 0.424 s | — |
| Gamma, 1M rows | **0.114 s** | 0.410 s | — |
| Tweedie(1.5), 1M rows | **0.213 s** | 0.287 s | — |
| Gaussian, 1M rows | 0.088 s | **0.071 s** | — |
| Poisson, 100k rows | **0.016 s** | 0.035 s | 1.523 s |
| Poisson, 5M rows | **0.709 s** | 2.455 s | — |

### Memory

`scripts/bench_isolated.py` gives each engine its own process and reports that process's
high-water RSS — data, design matrix, solver and interpreter together. Sampling several
engines inside one process instead makes the second one look free, because CPython
allocates it inside the pool the first one left behind; that is an artifact of the harness
rather than a property of the engine.

| whole-process peak RSS | Avenue | glum `irls-ls` | statsmodels |
|---|-------:|---------------:|------------:|
| synthetic Poisson, 100k rows | **113 MB** | 191 MB | 932 MB |
| synthetic Poisson, 1M rows | **196 MB** | 336 MB | — |
| synthetic Poisson, 5M rows | **564 MB** | 1,200 MB | — |
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

Five public datasets across four families. These have the correlated factors that
synthetic data does not.

| unpenalised, fit seconds | Avenue | glum `irls-ls` | glum `irls-cd` |
|---|-------:|---------------:|---------------:|
| freMTPL2, 678k rows, 79 params, Poisson | **0.27** (15 sweeps) | 0.46 (5) | 0.53 (6) |
| freMTPL2, 678k rows, 270 params, Poisson | **0.52** (25) | 1.61 (18) | 1262 (500) |
| nyc_taxi, 2.75M rows, 577 params, Gamma | 5.36 (35) | **3.77** (9) | — |
| census_income, 45.2k rows, 116 params, Binomial | **0.16** (36) | 0.20 (21) | 254 (500) |
| house_sales, 21.6k rows, 92 params, Gamma | **0.052** (50) | 0.057 (6) | 0.776 (9) |
| house_sales, 21.6k rows, 92 params, Gaussian | 0.039 (53) | **0.013** (1) | 0.336 (1) |

Avenue takes four of the six. What each dataset is there to cover:

- **freMTPL2** is close to a worst case for a backfit: `Area` is a six-band rebanding of
  `Density`, correlated at 0.972. It is the dataset glum builds its own `wide-insurance`
  benchmark from, and it appears at two band widths.
- **nyc_taxi** is the largest *real* problem here and the only one with credible
  high-cardinality geography — 252 pickup and 261 dropoff zones. **glum wins it
  outright**, 9 IRLS iterations against 35 sweeps: the expected outcome wherever the
  sweep count climbs but the table count stays too small for `O(n·T²)` to bite.
- **census_income** is the only real Binomial, the one link whose coordinate update is a
  Newton step rather than an exact `ln(A/E)` minimiser.
- **house_sales** is the ordinary case: correlated (`sqft_living` against `grade` at
  0.76) with nothing aliased. Its Gaussian fit goes to glum by 3x for a structural
  reason — under an identity link a single IRLS step *is* the exact answer for a linear
  model. One iteration against 53 sweeps is not a contest. **For an unpenalised Gaussian
  model, use a direct solver.**

`glum[irls-cd]`, its own coordinate-descent solver, is 8–15x slower than Avenue on the
housing data and is the only engine that fails the agreement check on any of these
problems. On the 270-parameter design it takes 21 minutes to reach its 500-iteration cap,
still `1.5e-3` from the answer, against Avenue's 0.52 s — so `bench_fremtpl.py` runs it
only under `--solver-sweep`. The comparison worth making is against `irls-ls`.

Both new datasets needed complete cases rather than a missing-value level, for the same
reason: two factors missing on the *same* rows have identical missing indicators and are
therefore exactly aliased. `workclass`/`occupation` in the census data, and three pairs in
the taxi data, read rho = 1.0000 that way, and glum refuses such a design with
`LinAlgError: Matrix is singular`.

#### What the two accelerants are worth

Backfitting's rate is set by the canonical correlation between blocks, and the two
best-understood datasets differ in exactly that:

| | tail `rho` | sweeps per decade | worst single table |
|--|-----------:|------------------:|--------------------|
| freMTPL2 | 0.943 | 39 | drop `Area` **or** `Density`: 254 sweeps → 20 |
| house_sales | 0.765 | 8.6 | drop `sqft_living`: 79 → 54 |

On the French motor data one near-duplicate pair is the *entire* problem for a plain
sweep — every other table is irrelevant to the rate. On the housing data no table
dominates. So the insurance result is a tail case rather than the typical one, and
`bench_housing.py --diagnose` runs the drop-one sweep that tells the two apart.

SQUAREM takes freMTPL2 from 254 sweeps to 66; the joint pair solve takes it from 66 to
**15**. Between them they leave little for a modeller to recover by hand: `Area` is a
table someone reviewing this plan would drop, and `bench_fremtpl.py --drop area` fits it
both ways — dropping it moves 15 sweeps to 14 and 0.27 s to 0.21 s. Most of that 1.2x is
the five parameters no longer being carried, and glum gains the same 1.2x from the same
drop while being indifferent to the redundancy by construction. **The 254-sweep figure is
what the aliasing costs an unaccelerated backfit, not what it costs this one.**

### At twenty million rows

The cases above all finish in under two seconds. `scripts/bench_large.py` runs one fit at
a size where the absolute numbers stand on their own: 20M rows, 501 parameters, Poisson
with an exposure offset, one engine per process because the two representations do not fit
in memory together.

| 20M rows, 501 parameters | Avenue | glum `irls-ls` | per iteration |
|---|-------:|---------------:|--------------:|
| 100 tables of 6 levels | **39.3 s** (4 sweeps, 10.8 GB) | 865.9 s (5, 21.1 GB) | 9.8 s vs 173 s |
| 5 tables of 101 levels | **3.1 s** (4 sweeps, 1.2 GB) | 16.5 s (8, 3.6 GB) | 0.8 s vs 2.1 s |

Fitted means agree to `5.6e-09` and `3.2e-09`.

Those two rows carry the **same 501 parameters over the same data** and differ only in how
the parameters are laid out, which isolates what the comparison turns on. Moving them from
100 tables into 5 collapses the per-iteration gap from 18x to 2.7x — exactly what `O(n·T)`
against `O(n·T²)` predicts.

Fitting `c1·n·T + c2·n·T(T+1)/2` to glum's two per-iteration figures gives `c1 = 16.5 ns`
per row-table and `c2 = 1.39 ns` per row-pair, both physically sensible — the second is
about what a scatter-add into a contingency table should cost. On those coefficients the
quadratic term is 81% of its time at 100 tables and 20% at 5. Two points fitted with two
coefficients is a consistency check rather than a test; the measured collapse from 18x to
2.7x is the evidence. Avenue's own cost per row-table is 4.8 ns and 8.0 ns across the two
rows — within a factor of two while `T` changes by twenty, which is what `O(n·T)` requires.

Those factors are stored as `Int32` category codes, which is what they are — unordered
draws over a level count. That matters for the memory column: 100 factors over 20M rows is
an 8 GB frame as `Int32` against 16 GB as `Float64` bands, and the fit is identical either
way. It is why Avenue's peak here is 10.8 GB rather than the 18.7 GB this benchmark
reported when it built everything as bands.

The remaining gap is ours. glum is handed 2 GB of `int8` codes, and 4 bytes per factor per
row is as narrow as Avenue's matching path goes — anything narrower falls back to a slow
path. Supporting `Int8` and `Int16` is the largest memory win still available.

### Where it loses: correlated tables

The factors above are independent, so the shape that produces the largest advantage is
also the one most exposed to correlation. Loading every factor on a shared latent driver
(`bench_large.py --correlation`), at 1M rows and 100 tables:

| pairwise `rho` | `table_conditioning` | sweeps | Avenue | glum | |
|---:|---:|---:|---:|---:|--|
| 0.00 | 1.8 | 5 | **4.3 s** | 35.6 s | 8.2x faster |
| 0.10 | 10.1 | 121 | 29.6 s | 31.2 s | parity |
| 0.20 | 19.2 | 494 | 95.8 s | 31.2 s | 3.2x slower |
| 0.30 | 28.4 | 1,124 | 240.6 s | 30.9 s | 7.8x slower |

Avenue is 8.2x faster **per iteration** in every one of those rows — 0.9 s against 7.1 s.
The entire swing is the sweep count, because a factorisation is indifferent to
conditioning and glum sits at five iterations throughout. Hence the rule worth
remembering: **this fitter wins while the plan needs fewer than about forty sweeps.**

#### A pairwise measure cannot see it

A pairwise correlation of 0.10 already costs 121 sweeps — two orders of magnitude below
the `NEAR_ALIAS` threshold, so the pair detector sees nothing and the joint solve never
fires. That is not a badly chosen threshold. **The count of correlated tables matters more
than the correlation itself.** Holding the pairwise figure at 0.28 and varying only how
many tables share the driver, at 200k rows:

| tables | worst pair | `table_conditioning` | sweeps to converge |
|-------:|-----------:|---------------------:|-------------------:|
| 5 | 0.280 | 2.11 | 14 |
| 25 | 0.282 | 7.65 | ~101 |
| 50 | 0.284 | 14.57 | 248 |
| 100 | 0.284 | 28.45 | 1,119 |

The pairwise column is constant; the sweep count moves eighty-fold. What governs the rate
is the canonical correlation between one block and the **span of all the others**, and a
hundred tables on a common latent cover that shared direction almost exactly while every
pair stays modest.

`collective_strength` is the measure that does see it: the largest eigenvalue of the
pairwise correlation matrix, which for an equicorrelated set is exactly `1 + (T-1)·rho` —
verified to two decimals in every row above. It costs nothing, since the pairwise
correlations are already computed, and it is reported as
`GLMDiagnostics::table_conditioning` whether or not any pair crosses the joint-solve
threshold. **1.0 for orthogonal tables, up to the table count when they all carry the same
information. Above roughly 10, expect hundreds of sweeps; above 25, thousands.**

#### How conditioned is real data?

Every real dataset in the suite, measured:

| dataset | rows | tables | `table_conditioning` | worst pair |
|---------|-----:|-------:|---------------------:|-----------:|
| freMTPL2, tutorial bands | 678,013 | 9 | 2.85 | 0.972 |
| freMTPL2, wide bands | 678,013 | 9 | 2.94 | 0.994 |
| house_sales | 21,613 | 10 | 4.11 | 0.766 |
| census_income | 45,222 | 12 | 3.76 | 0.987 |
| nyc_taxi | 2,753,989 | 10 | 4.11 | 1.000 |

All between 2.85 and 4.11, far below the threshold — **including the three that contain a
pair over the alias threshold**. That is not a coincidence. A real alias is local to two
tables, usually one driver banded twice, and the joint pair solve handles it; conditioning
measures a direction shared across *many* tables at once. The hundred-tables-on-one-latent
design that stalls the sweep is a synthetic pathology and nothing here resembles it. The
weakness is real; it also appears to be rare.

The fix that would remove it while keeping the `O(n·T)` advantage is conjugate gradient on
the normal equations, preconditioned by the sweep: `O(sqrt(kappa))` iterations rather than
`O(kappa)`, each costing one sweep plus one `X'WX·v` product, both `O(n·T)`. Assembling
`X'WX` outright would also work and costs `O(n·T²)` — precisely glum's cost, and precisely
the advantage being defended. Not built.

### Build settings

Stock `cargo --release`. Three obvious levers were measured on the synthetic suite and
**none moved a single number outside run-to-run noise**, so none are configured:

| | build time | effect |
|--|-----------:|--------|
| `lto = "fat"`, `codegen-units = 1` | 5s → 379s | none |
| `-C target-cpu=native` | 5s → 106s | none |
| eliding the scatter-add bounds check (`get_unchecked`) | — | none |

The reason is the shape of the inner loop rather than anything about the compiler. The
work is a scatter-add at a data-dependent index — `numer[table_matches[i]] += ...` — which
cannot vectorise, because each lane would need a different address and the indices repeat
constantly (that is what a rating table *is*). Wider registers have nothing to fill, and
the bounds check disappears into the latency of the dependent load it guards. LTO has
nothing to inline across either: the hot loops are all inside this module and already
marked `#[inline]`, and the data is copied out of Polars into plain `Vec`s before fitting
precisely so the fit never crosses back.

`target-cpu=native` is also actively wrong to ship — the wheel would fault with an illegal
instruction on any machine older than the one that built it — so it would need to earn its
place, and it does not.

---

## Variates: continuous drivers in a table

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

### Polynomials

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

### Conditioning of the variate solve

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

---

## Identifiability

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

## Convergence

The fit stops when the **largest absolute score** over every free parameter falls to
`tolerance` (default `1e-9`), or when `max_iterations` sweeps have run.

The score of a parameter is the derivative of the log-likelihood with respect to it, and
it is zero only at the optimum — so this measures how far the *factors* still have to
move. It is the same criterion glum applies as `gradient_tol`, on the same scale, so the
two are comparable. The score is scaled by the total absolute residual, which makes the
threshold independent both of the number of observations and of the units the response is
measured in; a Gaussian fit on currency and one on log-odds mean the same thing by `1e-9`.

> **Why not deviance?** Deviance is *quadratic* in the parameter error near the optimum,
> so a deviance tolerance of `1e-t` buys only about `1e-(t/2)` on the factors — and when
> convergence is slow, the deviance goes flat while the parameters are still moving. On
> freMTPL2 a relative-deviance rule reported convergence with the fitted means `1.1e-04`
> from the answer; the score rule reaches `2.1e-07` on the same fit. It costs more sweeps
> because the earlier fit was not finished.

**Giving up.** A fit that cannot reach the tolerance stops once the *deviance* has failed
to improve by more than `DEVIANCE_STALL` (`1e-15`) for twelve consecutive sweeps, and
reports `converged = false` with the score it achieved. Progress has to be judged on the
deviance and not on the score, because the score **oscillates**: the error rotates as it
decays, so on a hundred correlated tables it falls to `7.2e-04` by sweep 34, is back up at
`1.2e-03` by sweep 46, and reaches tolerance on sweep 1,119. Any rule that treats a score
increase as failure abandons a fit that is converging, and a windowed version fails too —
the oscillation period is a property of the data, so no window length is safe. The
deviance cannot oscillate: every coordinate update is an exact minimiser along its own
coordinate, so it falls every sweep and flattens only at a fixed point.

`DEVIANCE_STALL` sits at the deviance's rounding floor and not a decade higher, for the
same quadratic reason as above — a threshold of `1e-12` cut off a fit at
`max|score| = 4.4e-09` against a `1e-9` tolerance. The consequence is accepted
deliberately: past a score of about `1e-8` the deviance can no longer certify that
anything is happening, so a genuinely stalled fit runs to `max_iterations` rather than
stopping early. Wasted sweeps are a far smaller failure than an abandoned fit.

**SQUAREM is accepted on the score, not the deviance**, for exactly that reason: close to
the optimum the deviance cannot distinguish a step that halves the remaining error from
one that doubles it. Requiring the score not to worsen costs nothing, because that score
is already being computed. On 50 correlated tables the unguarded accelerator turned a
248-sweep fit into one still unconverged after 5,000; guarded, it finishes in 253 and
still earns its keep where a single mode dominates (`house_sales`, 79 sweeps to 53).

**`converged` is the flag to check, and it means what it says.** Near-aliased tables are
the usual cause of a false; a `gradient_history` that falls steeply and then crawls is
their signature, and `table_conditioning` says up front whether to expect it.

## Standard errors

On `GLMDiagnostics`, unless `compute_standard_errors` is turned off.

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

Detecting that takes **two** tests, because there are two ways to be unestimable and each
is invisible to the other's:

1. A parameter that is a **linear combination of earlier ones** has a pivot that collapses
   relative to *its own* diagonal.
2. A parameter that **never carried any information** — a separated logistic level, or a
   Poisson band with no claims whose weight underflows once its factor hits the clamp —
   has a diagonal already negligible relative to *the design*, and measured against itself
   it looks perfectly healthy.

Judging only the first way left the census income fit (pivot `1.7e-15`) and the wide
freMTPL2 fit (pivot `1.7e-111`) with no standard errors at all instead of the two and
sixteen `NA`s they should have had.

Validated against statsmodels on the census income fit: **113 standard errors agree to
`3.9e-10`**. The two rows Avenue declines are the two perfectly separated levels, where
statsmodels does not detect the separation and reports standard errors of `6.6e5` and
`1.9e7` — numbers that look like answers and are not.

Also reported: `dispersion` (1 for Poisson and Binomial, Pearson chi-squared over
residual degrees of freedom otherwise), `pearson_chi2`, `df_residual`, `n_parameters`
(the rank), `log_likelihood`, `aic` and `bic`.

The log-likelihood, and so AIC and BIC, are `None` for Tweedie: its density is an
infinite series with no closed form, and reporting a number would mean quietly
substituting an approximation. For Gaussian the likelihood is evaluated at the ML
variance `SSE/n`, not the Pearson estimate `SSE/(n-p)` used for standard errors — the
same distinction statsmodels makes. AIC and BIC count the mean parameters only, again
matching statsmodels.

---

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

`GLMOptions` also carries `accelerate` (SQUAREM, default on) and
`solve_aliased_pairs_jointly` (default on). Both are safeguarded; turn them off only to
reproduce the unaccelerated sequence exactly, or to bisect a fit that misbehaves.

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

### Checking a plan before fitting it

```python
from avenue_model import table_correlations

# (table_i, table_j, rho) for each pair, rho the first canonical correlation.
for i, j, rho in table_correlations(model, df, weight_col="exposure"):
    if rho > 0.9:
        print(f"tables {i} and {j} are near-aliased at {rho:.4f}")

fitted, diag = fit_glm_with_diagnostics(model, df, "claims", options=options)
print(diag.table_conditioning)   # > 10: expect hundreds of sweeps
print(diag.accelerated_steps)    # SQUAREM jumps accepted
```

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
    ..Default::default()
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
- **a table whose feature column is a numeric dtype other than `Float64` or `Int32`.**
  Those two are the only dtypes the matcher reads; anything else was previously dropped
  during table construction, which left the column constraining nothing and matched every
  observation to row 0 — a wrong model with no unmatched observation to give it away.
  `Int64` is the easy way in, being numpy's default integer dtype and what
  `pandas.Categorical(...).codes` widens to.

## Testing

```bash
cargo test --lib                     # 150 tests
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

## Module structure

```
src/glm/
├── mod.rs              # Public API exports
├── fitting.rs          # Coordinate descent, SQUAREM, anchoring, diagnostics
├── redundancy.rs       # Table correlations, near-alias detection, conditioning
├── inference.rs        # Standard errors, dispersion, aliasing, AIC/BIC
├── loss.rs             # Families: links, variance, IRLS weights, deviance, likelihood
├── matching.rs         # Observation-to-table-row matching
└── utils.rs            # Helper functions (weighted means, etc.)
```

## Known gaps

- **No regularisation.** No ridge, lasso, elastic net, credibility shrinkage, or
  monotonicity constraints.
- **Poorly conditioned plans are slow.** `table_conditioning` above ~10 costs hundreds of
  sweeps; see [above](#where-it-loses-correlated-tables) for the measurements and the
  planned fix.
- **Wald standard errors only.** No likelihood-ratio tests, profile intervals, or robust
  / sandwich covariance.
- **Only Wald z for the top polynomial degree.** Choosing a degree properly wants
  likelihood-ratio tests or AIC across a fitted sequence; for now, refit at each degree
  and compare `deviance` or `aic` yourself.
- **Lookup is always a step lookup.** A table cannot interpolate between its rows, so a
  variate's continuity lives in the *pattern of factors*, not in the prediction — two
  ages in the same band get the same factor. Interpolating tables are designed but not
  built.
- **`update_pair` applies its Newton step unconditionally.** Everything else in the
  fitter checks its work; a guard here would make the joint solve safe independently of
  the detector.
- **No dtype narrower than 4 bytes on the matching path.** `Int32` and `Float64` both
  match at full speed, but `Int8`/`Int16` fall back to the reference scan, so a wide
  design carries 4 bytes per factor per row where glum takes 1 — see
  [At twenty million rows](#at-twenty-million-rows).
- **Interaction tables scan every row.** The pre-resolved scan removed the per-observation
  allocation, but the cost is still `O(table rows)` per observation. A hash on the tuple
  of matched codes would make it `O(1)`, as the one-column categorical path already is.

## References

- Nelder, J. A., & Wedderburn, R. W. (1972). *Generalized linear models*
- McCullagh, P., & Nelder, J. A. (1989). *Generalized Linear Models*, 2nd ed.
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning*
- Wood, S. N. (2017). *Generalized Additive Models: An Introduction with R*
- Varadhan, R., & Roland, C. (2008). *Simple and globally convergent methods for
  accelerating the convergence of any EM algorithm* (SQUAREM)
- Greenacre, M. (1984). *Theory and Applications of Correspondence Analysis*
