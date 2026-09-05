# Continuous smooth effects: implementation notes

Status: exact natural-cubic scoring and editable workbook persistence are implemented
and independently tested. Low-level Rust table-sweep fitting now supports unpenalized
continuous splines. `Plan.spline` now resolves explicit, quantile or equal-width knots within training
folds. Unpenalized spline knot covariance and whole-term tests now use continuous
basis loadings. Existing
polynomial variates still score as step tables. The smooth-effect acceptance requirement
remains open until the remaining penalty and workflow acceptance checks pass.

## Canonical curve and numerical representation

`src/spline.rs` represents a natural cubic by function values at finite, strictly
increasing knots. It interpolates those values exactly, is twice continuously
differentiable between intervals, and has zero curvature at both boundary knots.
Outside the knot range it follows the endpoint tangent, rather than extrapolating
the last cubic segment or silently flattening it.

Each interval uses `u = (x-left)/(right-left)` and four polynomial coefficients.
This avoids evaluating powers of large raw predictors. The second-derivative system
uses knot spacing normalized by the full range; a tridiagonal solve produces local
polynomials. Nonfinite predictors, knots, values, coefficients or working equations
produce errors. There is no wildcard/default-rate inference.

The cardinal basis identifies every parameter with one knot value. Its weights sum
to one and reproduce affine functions. Replacing knot values creates a new curve;
the score change equals the corresponding basis-weighted parameter change, including
the tails. The scalar evaluator uses binary search and a single local polynomial.

The kernel also produces a curvature matrix `R` for
`b' R b = integral_0^1 (d2 f / dt2)^2 dt`, where `t` is the normalized whole knot
range. Constants and straight lines lie in its null space. This penalty is invariant
to affine rescaling of the predictor; it is not a ridge penalty toward a base category.
Producing this matrix does not yet implement penalized spline fitting or inference.

## Fitting without an observation-by-knot matrix

For each current IRLS working curvature and score, accumulate a four-by-four Gram
matrix and four-vector in interval-local polynomial coordinates. Tail observations
use two affine coordinates. Exact-knot observations use exact cardinal weights.
Transform these aggregates into the knot-value information matrix and score vector.

This path needs working memory quadratic in knot count, independent of the number
of observations. The current kernel transformation is cubic in knot count; it has
not been benchmarked as a large GLM workload. The current fitter explicitly rejects
singular knot information; broader parameter limits and conditioning exhibits remain open. No performance win is claimed from this design.

## Independent numerical evidence

`studies/reference_natural_spline.py` generates
`tests/fixtures/natural_spline.json` with SciPy `CubicSpline(bc_type='natural')`.
The generator explicitly adds linear endpoint-tangent tails. Three-point Gaussian
quadrature integrates products of second derivatives exactly on each polynomial
interval. It evaluates SciPy's local coefficients for that quadrature, avoiding
rounding quadrature nodes when an input origin is near `1e9`.

Five geometries and 220 probes cover two-knot linear behavior, irregular spacing,
tightly spaced knots, large coordinate offsets, all knots and adjacent floats,
interior points and both tails. Tests compare values, first/second derivatives,
every basis weight, the complete curvature matrix, and information/score aggregates
against the independent dense SciPy basis. Other tests enforce C2 joins, natural
boundary curvature, affine invariance, edit linearity and malformed-input errors.

Value/basis comparisons use absolute tolerance `1e-9` plus relative tolerance
`1e-10`. Derivative reference tolerance additionally accounts for differentiation
amplifying input roundoff by inverse minimum knot spacing, using the recorded
`64 * epsilon * value_scale / min_spacing^order` bound. For the clustered case,
SciPy's nominally zero endpoint curvature has about `3e-9` floating-point residual;
the kernel's natural-boundary condition is also tested directly, separately from
reference agreement. Tolerances are derived from reference inputs, not Avenue errors.
The full Rust suite passes 272 tests, with six ignored tests and one ignored doc test.

Regenerate the fixture with the test environment's SciPy installed:

```sh
python studies/reference_natural_spline.py --output tests/fixtures/natural_spline.json
cargo test --no-default-features --locked spline::tests
```

## Integration gates still open

1. **Plan extensions.** Public spline terms, fold-local knot resolution, declared
   linear tails and malformed-input checks are implemented. Given/offset-model terms
   now retain fixed spline semantics through Plan serialization and refitting.
2. **Fitting extensions.** Unpenalized table sweeps now use continuous updates,
   basis-score convergence and observed-contribution normalization. A roughness penalty
   still needs its own interpretation and covariance/selection rules. Individual locked
   knots, global solving and acceleration remain unsupported for spline fits.
3. **Inference extensions.** Model-based/HC0/CR0 knot covariance, full-design alias
   detection and joint tests now use continuous basis loadings. Pre-fit spline
   conditioning, penalized inference and continuous simultaneous bands remain open.
   A knot without an observation exactly on it is not automatically unestimated.
4. **Acceptance.** Recover known smooth shapes for multiple families; compare means,
   scores and uncertainty with an independent continuous-basis fit. Check raw-quote
   reloads and edits across interior points, knots, adjacent floats and tails. Preserve
   existing step/booster behavior and measure the complete fitting/scoring workflow.

These gates extend Plan/FittedModel/the Rust engine and the rating-table artifact.
They do not call for a second high-level GLM API or an opaque estimator wrapper.

## Continuous scoring and editable workbooks

A Rust scoring table is declared explicitly:

```rust,ignore
let curve = RatingTable::new(
    df!("age" => [20.0, 40.0, 60.0],
        "Rating_Factor" => [0.0, 0.4, 0.0])?,
    None,
).as_natural_cubic()?;
```

The numeric column contains finite, strictly increasing knots, and `Rating_Factor`
contains the curve values **on the link scale**. There must be at least two knots.
The builder rejects categorical columns, null/nonfinite values and combined
monotonic/variate declarations. The low-level GLM fitter accepts unpenalized continuous
tables, including fully fixed curves. Given-term serialization now preserves spline
semantics, enabling the ordinary Plan offset-model update workflow.

Workbooks containing a spline write format **4** and table metadata
`"spline": "natural_cubic_linear_tails"`. Ordinary workbooks still write version 2;
monotonic step workbooks still write version 3. Earlier readers reject version 4.
A spline declaration in a workbook labeled with an older version is rejected.
Knots and knot values are the sole editable authority: coefficients are compiled
once per scoring batch from the current values, never stored as a second set of
editable numbers. Integral-looking CSV knots remain continuous numeric coordinates.
Factor scale preserves knot values; relativity scale writes their exponentials and
reloads their logarithms **before** constructing the cubic.

Rust RatingModel and Python FittedModel scoring use the continuous curve, including
both tails. Strict scoring rejects missing/nonfinite quotes; diagnostic scoring
returns null predictions with row status. Exposure handling is unchanged. Explanations
use `kind="spline"`, the evaluated contribution, and a null `table_row`, because an
interpolated value is not the coefficient of one row. Composition retains separate
spline tables instead of flattening them into bands. Bundle reloads and change review
use these same scoring/explanation paths. Loaded knot values remain `scoring_only`.

Validation A/E tables show explicit support intervals, with open lower bounds and
closed finite upper bounds. For knots `k0,...,kM`, groups are `(-inf,k0]`,
`(k0,k1]`, ..., `(kM-1,inf)`. The last group includes the final knot interval and
right tail. These groups partition finite quotes; they are not piecewise constant
rates or knot-parameter support counts. A/E exhibits therefore omit the knot factor
column and explain this distinction in a finding.

`src/tests/spline_scoring_tests.rs` checks all 220 independent SciPy probes through
scalar/batch scoring and JSON/CSV reloads, plus known continuous values, explanations,
A/E, composition, edits and invalid artifacts/quotes. `tests/test_spline_scoring.py`
checks the public Python bundle/change-review workflow, offset means, zero exposure,
diagnostic nulls and relativity-scale CSV reloads. These are scoring checks, not
independent evidence for spline fitting or inference.

## Unpenalized table-sweep fitting

`src/glm/spline.rs` provides one continuous Fisher-scoring block inside the existing
GLM sweep. It aggregates interval-local working information and scores, solves in knot
coordinates and evaluates the resulting continuous change at each observation.
Backtracking checks the ordinary family deviance, with a floating-point allowance
near an optimum. Positive-weight working predictors must remain within the family's
numerical link range. A rejected step leaves the factors unchanged; the existing
score/stall/iteration rules determine nonconvergence.

The spline's convergence score is `B' W r`, not a sum assigned to support groups.
It uses the same response-scale normalization as the existing ordinary-table scores.
Base-level normalization anchors the first knot value. Weighted-mean normalization
uses the integrated basis loadings `B' weights`, so the **observed continuous
contribution** has weighted mean zero; support-group counts are not used as knot
weights. Constant shifts move into the intercept without changing predictions.

Spline fits currently use the table solver, with SQUAREM acceleration and spline
pair solving disabled. Fully fixed curves are evaluated continuously while other
terms fit. A singular within-spline information matrix raises an actionable error
instead of assigning independent step factors. Knots without exact observations are
not labeled unestimated. Covariance now uses the continuous observation basis, as
described below; discrete-table conditioning is not substituted for spline conditioning.
Ridge/elastic-net options, global solving and individual locked knots remain rejected
pending their integration.

The independent generator `studies/reference_spline_fit.py` constructs a dense SciPy
natural-cubic basis with explicit linear tails and solves the weighted GLM score
using SciPy `root`. It includes a categorical effect, nonuniform weights, zero-weight
rows and offsets. `tests/fixtures/spline_fit.json` records targets, reference
coefficients/means and reference scores for Gaussian, Poisson, Gamma, Tweedie(1.5)
and binomial families. Rust tests compare all three normalization modes to those
means, anchored coefficients where applicable, and JSON reloads. The tolerance is
`2e-7 * (1 + abs(reference))` with fit score tolerance `1e-10`. A separate two-spline
Poisson test recovers a known continuous shape at new interior, knot and tail quotes.
Malformed-design tests check singular information and unsupported options.

Regenerate and run the fitting references:

```sh
python studies/reference_spline_fit.py
cargo test --no-default-features --locked glm_spline
```

The complete Rust suite passes 278 tests (six ignored plus one ignored doc test).
These initial tests establish low-level fitting behavior. The later public Plan and
continuous-basis inference checks are described below.

## Public Plan integration

`Plan.spline(column, knots=..., quantile=..., equal_width=...)` now declares the
continuous term. A dedicated `Knots` enum separates finite spline controls from
band upper bounds. Automatic resolution includes both training boundaries, uses
positive-weight training observations, and collapses ties. Explicit knots are never
repaired silently. Resolved terms expose `knots` separately from `edges`.

Public tests recover SciPy curves in five families, serialize/refit plans and bundles,
exercise fold-local and zero-weight-outlier knot resolution, GLM selection, and
malformed inputs. Plan checks avoid treating spline support bins as independent
coefficients or reporting their discrete-table correlation as spline conditioning.
The runnable `examples/smooth_pricing_study.py` completes development-fold selection,
a reserved final holdout, bundle validation/reload and continuous-curve CSV export.
See [the user guide](SPLINES.md) for API conventions and current statistical limits.

## Fixed spline priors and conditional inference

Given terms now carry optional `SplineKind` metadata, copied by `offset_model`.
Preparation treats their coordinates as numeric, and building checks canonical
spline geometry rather than demanding an unbounded step band. Ordinary Given terms
retain their existing representation. A second fixed constant table is checked as a
one-row carried intercept on workbook reload; malformed multiple-row constants are
still refused.

When every spline is fixed, the ordinary inference layout excludes those tables
and uses the continuously evaluated means. Newly fitted supported discrete/variate
terms can therefore receive model-based, HC0 or CR0 covariance and whole-term tests,
conditional on the prior. Estimated splines now use continuous loadings in that same
covariance engine, as described in the following section.
Tests compare coefficient SEs and joint statistics against independent dense matrices
for weighted-rate and count-offset Poisson updates, including both tails, integer JSON
knots, category encodings, Plan reloads and validated bundle delivery. Prior knot
coefficients remain bit-for-bit unchanged. No prior estimation uncertainty is propagated.

## Continuous-basis covariance and whole-term tests

The existing inference engine now accepts observation-dependent spline loadings.
It retains the first knot as the reduced-basis reference and includes every other
knot parameter, independent of support-bin counts. Both expected information and
HC0/cluster scores use the same continuous basis at actual observation coordinates,
including the linear tails. The design is accumulated without storing an n-by-p
matrix. This inference pass has not been profiled as a large-portfolio workload.

Weighted-mean reporting uses `B' weights` to transform knot contrasts and the
intercept, matching the fitted normalization. Existing full-design null-space checks
identify aliased original coefficients and refuse unsupported whole-term tests.
The joint null is equal knot values (a constant curve), and the statistic uses the
complete within-term covariance. Empty support groups do not silently drop spline
parameters from the test. Parameter counts reflect the identified design rank.

`tests/test_spline_inference.py` compares knot/intercept/category standard errors
and joint statistics with independent dense SciPy basis algebra for five families,
model-based/HC0/CR0 covariance, weighted offsets and both reporting anchors. It also
checks parameterization-invariant unanchored joint tests, Poisson quasi-dispersion,
two fully aliased spline terms, and bundle evidence isolation. Covariance tolerances
are relative `3e-6` plus absolute `1e-9`; joint-statistic tolerance is relative `4e-6`.
The mean references are the independently solved `spline_fit.json` fixtures.

Intervals are marginal knot-coefficient intervals. Continuous simultaneous bands,
roughness-penalized covariance and post-selection coverage are not claimed.
