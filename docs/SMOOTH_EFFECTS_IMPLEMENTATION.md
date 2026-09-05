# Continuous smooth effects: implementation notes

Status: the numerical natural-cubic kernel is implemented and independently tested.
There is **no public smooth Plan term yet**. Existing polynomial variates still score
as step tables; this work does not relabel them as continuous predictors. The
smooth-effect acceptance requirement remains open until integration and end-to-end
fit/export checks pass.

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
3. **Shared scoring.** Extend low-level RatingModel scoring, strict FittedModel scoring,
   validation, explanations and composition to evaluate the same continuous curve.
   Integer table-row matches alone do not specify a continuous contribution.
4. **Inference and support.** Use observation-dependent basis loadings for covariance
   and joint tests. Show interval support and basis identification; a knot without an
   observation exactly on it is not automatically an unestimated factor. Do not reuse
   free-step-table row standard errors.
5. **Workbooks.** Persist knots, knot values and the tail rule as the canonical editable
   representation, derive local coefficients on load, and use a new format version
   so older readers reject rather than step-score the curve. Changes to either knot
   geometry or values must invalidate derived scoring coefficients. General workbook
   checks currently expect infinite final band bounds and need subtype-aware handling.
6. **Acceptance.** Recover known smooth shapes for multiple families; compare means,
   scores and uncertainty with an independent continuous-basis fit. Check raw-quote
   reloads and edits across interior points, knots, adjacent floats and tails. Preserve
   existing step/booster behavior and measure the complete fitting/scoring workflow.

These gates extend Plan/FittedModel/the Rust engine and the rating-table artifact.
They do not call for a second high-level GLM API or an opaque estimator wrapper.
