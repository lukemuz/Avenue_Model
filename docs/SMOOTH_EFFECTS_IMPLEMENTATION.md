# Continuous smooth effects: implementation notes

Status: exact natural-cubic scoring and editable workbook persistence are implemented
and independently tested. There is **no smooth Plan term or spline fitter yet**.
Existing polynomial variates still score as step tables. The smooth-effect acceptance
requirement remains open until fitting, inference and end-to-end recovery checks pass.

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
not been benchmarked as a full GLM workload. A future fitter must handle parameter
limits and conditioning explicitly. No performance win is claimed from this design.

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

1. **Plan and schema.** Add an explicit natural-cubic term with knot resolution inside
   training folds, a declared tail rule and clear duplicate/missing-input errors.
   Knots are control locations, not inclusive band bounds.
2. **Fitting.** Connect the interval aggregates to the existing table sweep, with
   safeguarded IRLS updates, numerical score convergence and identifiable constant
   shifts. Weighted-mean normalization must use observed curve contributions, not
   counts of observations exactly at knots. A roughness penalty needs its own
   interpretation and covariance/selection rules.
3. **Inference and support.** Use observation-dependent basis loadings for covariance
   and joint tests. Scoring validation now shows interval support, but fitting must
   still establish basis identification. A knot without an observation exactly on it
   is not automatically an unestimated factor. Do not reuse free-step-table errors.
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
monotonic/variate declarations. Calling the GLM fitter with this table currently
returns an explicit unsupported error, including when the table is marked fixed.
Carrying it into a Plan offset is likewise refused until the fitter is integrated.

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
