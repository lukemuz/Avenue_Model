# Model specification

Choose the structure you want to estimate. Avenue keeps those choices in the Plan and resolves data-dependent bands and knots when fitting.

[Interactions](#interactions) · [Monotonic bands](#monotonic-bands) · [Continuous splines](#continuous-splines)

## Existing rating plans: import, refit or hold fixed

An Avenue workbook already contains the tables and their metadata:

```python
from avenue_model import Plan, Workbook

existing = Workbook.load_csv_dir("rating_plan").to_model()
refitted = existing.refit(train)  # Uses its recorded target and exposure convention.
```

For a plan from another system, read its tables into Polars and import them once.
Here the existing frequency is a base rate of 0.10 multiplied by regional relativities:

```python
import polars as pl

book = Workbook.from_tables(
    {"region": pl.DataFrame({
        "region": ["north", "south"],
        "Relativity": [1.0, 0.8],
    })},
    family="poisson",
    base_value=0.10,
    target="frequency",
    exposure="exposure",
)
existing = book.to_model()
book.save_csv_dir("imported_plan")

# Re-estimate the factors, keeping the tables' levels, bands and interactions.
refitted = existing.refit(train)

# Or keep the entire existing plan fixed and estimate adjustments on top.
adjusted = (
    Plan.frequency("exposure")
    .offset_model(existing)
    .banded("age", breaks=[25., 50., 70.])
    .fit(train, "frequency")
)
```

Replace the example dataframe with `pl.read_csv("region.csv")` for your own table.
Supply only predictor columns and the factor column. String and integer predictors
are categories; floating predictors are **inclusive upper band bounds**, with an
infinite final bound. Set CSV schema overrides explicitly when a numeric band could
otherwise be read as an integer category. Multiple predictor columns define an
interaction table. Category labels are encoded consistently across all supplied tables.

`scale="relativity"` is the default for this import: `Relativity` contains positive
multipliers and `base_value` is the base rate. With `scale="factor"`, use
`Rating_Factor` and supply `base_value` on the link scale (log rate for Poisson,
log odds for binomial, response units for Gaussian). Import creates the intercept;
do not include an extra intercept table. Imported tables are checked before use.
This importer handles ordinary step tables. Load saved Avenue workbooks directly
to retain their locks, offsets, splines and other metadata.

`refit` returns a new fitted model; the source is unchanged. It preserves the current
tables, category encodings, locked rows, fixed offsets and smooth/monotonic metadata.
It does not regenerate quantile bands or discover new levels. Supply `target` if it
was not recorded, and pass `GLMOptions(...)` to choose regularization or other fitting
options; omitted options use the normal defaults, not a prior fit's settings.

For rates, `existing.refit(train, "frequency", exposure="exposure")` uses exposure
weights. For counts, specify `exposure_role="offset"` so log exposure enters once.
An existing recorded role is retained unless overridden. Use
`existing.with_response("frequency").refit(train)` to remove a recorded exposure.
Refits carry new diagnostics and inference where supported; workbook reloads retain
scoring metadata, not the original fit's covariance.

`offset_model` holds every prior factor, including its base rate, fixed. The new plan
fits its own intercept for an overall rate-level adjustment, plus any new terms you
add. Omitting `.banded(...)` above fits just that overall adjustment. Prior table
names receive the default `prior.` prefix. The new `Plan` declares the response and
exposure convention for the adjustment fit; its family must interpret the prior
factors on the same link scale.

## Interactions

```python
from avenue_model import Plan

plan = (
    Plan.frequency("exposure")
    .categorical("territory")
    .banded("driver_age", breaks=[25., 50.])
    .interaction(["territory", "driver_age"], [None, [25., 50.]])
)
model = plan.fit(training, "frequency")
print(model.resolved)
print(model.rating_tables_by_name())
```

When both corresponding main effects are present, Avenue recognizes a compatible
two-way categorical/banded interaction and uses identifiable treatment contrasts.
Interaction cells touching either main-effect reference level are fixed at zero.
Only the remaining interior cells are estimated. With `a` and `b` levels the interaction
therefore has `(a-1)*(b-1)` free parameters in a fully supported design. Reference
constraints are visible as locked table rows and in `resolved` as
`kind="interaction_contrast"`. They survive workbook export/reload.

Categorical references follow their main-effect selections, including most-exposed
levels; banded references are the first band. Terms may appear in any order. Numeric
axes must use the same break specification as their main effect so their learned bands
agree within each training fold. Main effects retain their usual interpretation at
the other factor's reference; the interaction is the additional joint effect on the
link scale.

The table solver supports the fixed reference rows. `solver="auto"` selects that
path; explicitly forcing the global solver still rejects partially locked tables.
Inference builds the actual free-parameter design and excludes all fixed reference
cells, rather than treating the resulting full cell table as unidentified. Independent
Poisson tests cover fitted means, estimable interaction contrasts and standard errors.
Sparse/unsupported cells can still be nonestimable and require review.

An interaction without both matching main effects retains the existing full-cell
representation. Arbitrary higher-order hierarchical contrasts and automatic interaction selection
are not implemented. Inference is conditional on the selected structure. Use the validation/complexity evidence to justify additional structure.

## Monotonic bands

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

### Supported fits and review

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

### Export and edits

The direction survives Plan JSON and workbook JSON/CSV. A workbook
containing a monotonic table uses format version 3 so older readers reject it instead
of silently dropping the declaration. Ordinary workbooks still write version 2; the
current reader accepts versions 1–3.

Editing a workbook still changes its scoring factors. If the factors violate the
declared direction, loading records `monotonicity_violated`, and the model report
carries that finding. It does not silently sort the edited factors. Refitting the
source Plan enforces the declared direction again. Saved source inference never
becomes fitting evidence for an edited scorer.

### Verification and scope

Rust tests compare the block algorithm against every contiguous partition for 3,072
loss comparisons. Fitting tests check exact weighted offset solutions, both directions,
both normalizations, empty bands, metadata round trips and edited-factor findings.
Python tests compare multi-term fits with a separate dense SLSQP constrained optimizer
for all three families and both directions, including active pooled constraints. CSV
and workbook reloads preserve quote predictions to relative tolerance `1e-12`.

Monotonic constraints are currently limited to banded effects. Smoothing penalties,
constrained uncertainty, regularization combined with ordering, and ordered terms
inside interactions are not implemented.

## Continuous splines

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

### Folds, selection and support

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
full-design aliases.

### Export and editing

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

### Current statistical scope

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

### Update a filed continuous curve

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

### Real-portfolio checks

The [motor evaluation](../studies/README.md#real-motor-comparison) compares banded and
five-knot spline fits with independent glum predictions. Its retained single-split
results favor bands on holdout loss for this specification. Knot counts should be
chosen for the problem and evaluated on reserved observations. Correct numerical
fitting alone does not establish a better predictive specification.
