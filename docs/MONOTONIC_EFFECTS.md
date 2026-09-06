# Monotonic rating effects

Declare the direction of a numeric banded effect in the model plan:

```python
from avenue_model import Plan, GLMOptions

plan = (Plan.frequency('exposure')
        .monotone('risk_score', 'increasing', breaks=[20., 40., 60., 80.])
        .categorical('region'))
model = plan.fit(training, 'claim_rate', GLMOptions(tolerance=1e-8))
assert model.converged
model.to_workbook().save_csv_dir('rating_plan')
```

Use `direction='decreasing'` for a decreasing effect. As with `banded`, supply exactly
one of `breaks`, `quantile`, or `equal_width`; edges are inclusive upper bounds and
an unbounded final band is added. Quantile/equal-width boundaries resolve from the
training data, including within training folds. The effect is an exact step function,
not a continuous spline or an approximation to a hidden smooth predictor.

Factors are constrained while minimizing the model loss. Adjacent bands can pool at
the same factor. The direction holds on both the coefficient and relativity scales,
and for predictions when other predictors and any exposure offset are held fixed.
A predictor with a monotonic term cannot also occur in another Plan term: another
effect of that predictor could reverse the declared direction.

## Supported fits and review

- Poisson, Gamma and Tweedie with variance power in `[1, 2]`; unpenalized fits.
- Multiple monotonic terms and ordinary categorical/banded terms. Base-level and
  weighted-mean normalization preserve direction.
- `solver='auto'` selects the table solver. A requested global solve is rejected.
  Extrapolation is disabled and monotonic tables are excluded from joint pair solves.
  Requested fit options remain recorded; `solver_used` records the actual solver.
- Empty bands copy the preceding supported band's factor; leading empty bands copy
  the first supported band's factor. They remain `no_data` in the estimate tables.
  This extension supplies a score for those bands, not evidence of their risk.
- Ordinary covariance, parameter-count-based likelihood statistics and coefficient
  intervals are withheld for the whole constrained fit with an explicit explanation.
  HC0/cluster requests and nonzero regularization are rejected. Plan resolution's
  parameter count describes the unconstrained structural dimension, not an estimated
  effective degrees of freedom after pooling.
- Missing numeric predictors follow existing strict Plan preparation rules. A direct
  Rust monotonic table cannot have missing-only rows or individual row/table locks.

Convergence uses feasible directional scores for pooled bands. Individual band scores
can be nonzero at the constrained optimum, so an ordinary free-coordinate gradient
would incorrectly report nonconvergence. Existing finite numerical coefficient guards
are not treated as statistical constraints that certify a boundary fit.

## Export and edits

The direction survives Plan JSON and workbook JSON/CSV. A workbook
containing a monotonic table uses format version 3 so older readers reject it instead
of silently dropping the declaration. Ordinary workbooks still write version 2; the
current reader accepts versions 1–3.

Editing a workbook still changes its scoring factors. If the factors violate the
declared direction, loading records `monotonicity_violated`, and the model report
carries that finding. It does not silently sort the edited factors. Refitting the
source Plan enforces the declared direction again. Saved source inference never
becomes fitting evidence for an edited scorer.

## Verification and remaining work

Rust tests compare the block algorithm against every contiguous partition for 3,072
loss comparisons. Fitting tests check exact weighted offset solutions, both directions,
both normalizations, empty bands, metadata round trips and edited-factor findings.
Python tests compare multi-term fits with a separate dense SLSQP constrained optimizer
for all three families and both directions, including active pooled constraints. CSV
and workbook reloads preserve quote predictions to relative tolerance `1e-12`.

Continuous splines, smoothing penalties, constrained uncertainty, penalties combined
with ordering, and ordered terms inside interactions remain unimplemented. The full
improvement plan and broader acceptance audit remain open.
