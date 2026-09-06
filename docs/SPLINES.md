# Continuous spline effects

`Plan.spline` fits a natural-cubic effect of a numeric predictor, with linear
endpoint-tangent tails. The fitted knot values are coefficients on the link scale.
Quotes between knots evaluate the continuous curve exactly.

```python
from avenue_model import Plan

plan = Plan.frequency('exposure').spline('driver_age', quantile=5)
model = plan.fit(data, 'claim_frequency')
print(model.resolved[1]['knots'])
quotes = model.predict_rate(new_policies)
```

Choose one knot specification:

| Argument | Meaning |
|---|---|
| `knots=[18., 25., 40., 60., 85.]` | Exact control locations, including both boundaries |
| `quantile=5` | Five nearest-rank locations, including training minimum and maximum |
| `equal_width=5` | Five evenly spaced locations across the training range |
| No knot argument | Defaults to `quantile=5` |

Automatic knots use positive-weight training rows and unweighted ranks among those
rows. Duplicate automatic knots are collapsed; fewer than two distinct locations
are rejected. Explicit knots must already be finite and strictly increasing, with
representable spacing; they are never silently sorted or dropped. Training coordinates
must be finite and non-null, including zero-weight rows. Prediction diagnostics can
report invalid quote rows without dropping them.

The count includes both boundary knots. The final knot is finite; spline tables do
not append an infinite band. Two knots produce a straight line. A natural cubic has
zero second derivative at the boundary knots, then follows the endpoint tangent
outside them. It does not promise monotonicity. Polynomial `variate` terms retain
their existing banded scoring semantics.

## Folds, selection and support

`Plan.fit` resolves automatic knots using only the supplied training rows. Plan JSON retains
the knot specification; `model.resolved` records the resulting knot coordinates in
`knots`, with `edges=None`. A serialized Plan can be refitted and will resolve
its automatic knots on the new training data. The scoring workbook always retains
the already resolved knots.

Evaluate candidate specifications with ordinary loops and a common held-out loss.
Changing knot count changes the allowed shape, so compare alternatives on common
folds. The resolved `parameters` field is the nominal specification dimension after
an intercept constraint, not a verified effective degrees-of-freedom estimate.
The fit reports the identified parameter count from its continuous information
matrix; aliased coefficients are not assigned finite standard errors.

Validation A/E exhibits describe explicit support intervals rather than assigning
an observed rate to a knot value. A knot need not coincide with any observation to
be estimated. The fitter rejects singular within-spline information. Post-fit covariance detects
full-design aliases; pre-fit spline conditioning exhibits are still being developed.

## Export and editing

`model.to_workbook()` retains exact continuous scoring in workbook
format 4. Earlier workbook readers reject that version. Knot locations and knot
values are the editable authority; scoring derives local polynomial coefficients
from them. Relativity-scale workbooks convert knot relativities to log coefficients
before interpolating. Edit either coordinates or factors, reload, and use
prediction differences to review quote effects.

Explanations show `kind='spline'`, the evaluated continuous contribution, and a null
`table_row`. A contribution between knots does not belong to a single table row.
Native factor composition retains separate continuous tables, and workbook reloads use the same
scorer. Loaded/edited values carry no new fitting evidence.

## Current statistical scope

Unpenalized Gaussian, Poisson, Gamma, Tweedie and binomial fits use the existing table
solver with safeguarded Fisher scoring. Base-level normalization anchors the first
knot; weighted-mean normalization centers observed continuous contributions.

Unpenalized spline fits support model-based, HC0 and one-way CR0 covariance, using
the continuous cardinal basis at each observation. `coefficient_intervals(model)`
reports intervals for the **knot coefficients** (and their exponentiated relativities
under a log link). These are not simultaneous confidence bands for the entire curve.
`term_tests(model)` tests whether all knot values are equal, conditional on the other
terms; that is the null of a constant spline contribution. Model-based Poisson fits
also support quasi-Poisson rescaling.

Weighted-mean covariance includes the observed-contribution centering in both the
knot coefficients and intercept. Fully aliased terms receive unavailable joint tests;
a supported contrast's validity is assessed against the full-design null space.
Inference parameter counts reflect the identified design rank. Fix knot specifications
before interpreting conventional intervals/tests; selecting knots on the same data
does not give post-selection coverage. Use reserved data for final evaluation.

Roughness penalties, ridge/elastic-net options, individual locked knots, global solving
and spline acceleration are not implemented. Fully fixed curves can be carried
through `Plan.offset_model`, including Plan/workbook round trips.

Run the complete synthetic pricing example:

```sh
python examples/smooth_pricing_study.py --output /tmp/avenue-smooth-study
```

It reserves a final holdout, fits five training-quantile knots, writes a validation
report and scoring workbook, verifies reload parity, and exports a continuous quote
curve as CSV. This is an API example, not evidence of real-portfolio calibration.

## Update a filed continuous curve

```python
updated_plan = (
    Plan.frequency('exposure')
    .offset_model(prior_model, prefix='prior')
    .categorical('new_factor')
)
updated = updated_plan.fit(update_data, 'claim_frequency')
```

The prior's intercept, continuous knot values and category encodings remain fixed.
The new plan applies its own declared exposure convention once; offset-model tables
do not carry a second observation exposure multiplier. Declare the update target
and exposure convention consistently with the prior's rate/count meaning.

Serialized Given terms retain the spline interpolation/tail declaration, so an
integer-looking knot cannot become a category code. Fixed prior intercepts also
survive workbook reloads. New factors receive ordinary model-based, HC0 or clustered
covariance and term tests when their free design is otherwise supported. This
inference is **conditional on the fixed prior**: it does not propagate uncertainty
from the earlier spline fit. The prior's knot values remain locked and are not
reported as newly estimated parameters. An update that also estimates a new spline uses its continuous basis in the
new fit covariance; uncertainty in the fixed prior still is not propagated.

## Real-portfolio evidence

The [real motor acceptance study](REAL_SPLINE_ACCEPTANCE.md) passes independent
prediction and delivery checks for frequency, severity and pure premium. The tested
five-quantile-knot specifications have worse holdout loss than the banded baseline,
wide uncertainty at a sparse boundary, and slower fitting. That evidence limits the
current performance and modeling-quality claims; it does not justify replacing the
banded specification automatically.
