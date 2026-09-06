# Statistical inference

An original GLM fit supports coefficient intervals and joint term tests, with model-based, HC0 or clustered covariance. The sections below explain the assumptions and supported combinations.

[Coefficient intervals](#coefficient-intervals) · [HC0 covariance](#hc0-covariance) · [Clustered covariance](#clustered-covariance) · [Whole-term tests](#whole-term-tests)

## Coefficient intervals

Numeric band tables also include [predictor interval bounds](scoring.md#numeric-band-bounds), separate
from the statistical uncertainty bounds on coefficients.

`coefficient_intervals` provides named tables containing coefficients, standard
errors and direct intervals for an original converged fit:

```python
from avenue_model import coefficient_intervals

review = coefficient_intervals(frequency, confidence=.95)
print(review.tables['region'])
quasi = coefficient_intervals(frequency, dispersion='quasi_poisson')
print(quasi.metadata)
```

`Coefficient_Lower` and `Coefficient_Upper` are normal Wald intervals on the fitted
linear-predictor scale. Log-link tables also include exponentiated
`Relativity_Lower` and `Relativity_Upper`. The original `Standard_Error` column is
retained; `Interval_Standard_Error` reports the uncertainty actually used for this
interval. `Interval_Status` is `wald`, `fixed` for a zero-error reference constraint,
or `unavailable` for an unestimable/no-data/locked factor without finite uncertainty.
A fixed reference is a coding constraint, not a precisely estimated effect.
Exponentiation can overflow to infinity for very wide intervals.

For model-based covariance, the default uses the engine's dispersion: one for Poisson/Binomial,
and estimated dispersion for Gaussian/Gamma/Tweedie. The optional quasi-Poisson
route estimates

`phi = sum(weight * (actual - mean)^2 / mean) / residual_degrees_of_freedom`

from the **original fit's** Pearson statistic and residual degrees of freedom.
For count models with exposure offsets, `mean` is the fitted count; for
exposure-weighted rates, `mean` is the fitted rate. These formulations give the
same Pearson statistic and coefficient uncertainty on the same records.
Zero-weight observations do not add residual degrees of freedom. Weights follow
the existing GLM precision/exposure convention, not a claim of independent replicated
observations. An aggregate record is one observation for residual degrees of freedom.

Quasi-Poisson multiplies Poisson coefficient standard errors by `sqrt(phi)`; it
leaves coefficients, predictions, family, likelihood and AIC untouched. It is an
explicit uncertainty review, not a negative-binomial fit or a new likelihood.
Dispersion is not floored at one: a positive value below one narrows intervals.
Nonpositive dispersion or residual degrees of freedom is rejected. Large dispersion
can reflect omitted structure, dependence or heterogeneous risk; scaling alone
does not diagnose or fix those issues.

These intervals are conditional on the specified structure and use a normal
approximation. They do not account for tuning, data-selected bands or uncertainty in the dispersion
estimate. Clustering requires [explicit cluster covariance](inference.md#clustered-covariance)
at fitting time. [HC0 robust covariance](inference.md#hc0-covariance)
can be selected at fitting time; it is labeled separately and is not dispersion-scaled. Penalized fits,
nonconverged fits, and fits without computed inference are rejected. Loaded/edited
workbooks cannot acquire the source fit's intervals. The original numerical evidence
is available as `model.inference_summary`.

The auto tutorial exports ordinary intervals for its fixed GLMs and separate
quasi-Poisson frequency intervals. Independent tests verify OLS intervals and
Poisson information-matrix/Pearson calculations under both exposure conventions;
the predictions are checked unchanged.

Unnormalized (`normalization='none'`) per-row factors are not identified: the engine
withholds their standard errors and the interval API explains the required anchoring.
`model.fit_options` records the effective normalization and other fitting controls.
In ordinary rating-table review, `Status='locked'` identifies fixed priors in a new fit;
`scoring_only` identifies loaded/converted factors without original fitting evidence.
The interval-specific status remains `unavailable` when such a row has no finite
uncertainty. These labels do not turn fixed or edited factors into new estimates.

## HC0 covariance

Request HC0 when fitting an unpenalized plan:

```python
from avenue_model import GLMOptions, coefficient_intervals

model = plan.fit(training, 'frequency', GLMOptions(covariance='hc0'))
print(model.inference_summary)
review = coefficient_intervals(model)
print(review.tables['region'])
```

`covariance='model_based'` remains the default. HC0 changes uncertainty only:
coefficients, score convergence, response predictions, dispersion estimation and
likelihood/AIC calculations keep their existing meanings. The report, low-level
`GLMDiagnostics`, `FittedModel.inference_summary`, and interval metadata identify
the covariance method explicitly.

The implementation uses the same identifiable reduced design as ordinary inference,
including two-way interaction reference constraints and polynomial variate loadings.
Let `a_i` be an observation's precision/exposure weight, `d_i = d mu_i / d eta_i`,
and `V(mu_i)` the family's unit variance. It computes

```
H = sum_i a_i * d_i^2 / V(mu_i) * x_i x_i'
s_i = a_i * (y_i - mu_i) * d_i / V(mu_i) * x_i
Cov_HC0 = H^-1 * sum_i s_i s_i' * H^-1
```

This is an **expected-information sandwich**: variance misspecification is allowed
under a correctly specified conditional mean and independent observations. It is
not an observed-Hessian sandwich under arbitrary mean misspecification. Poisson and
Binomial canonical links and Gaussian identity have the corresponding familiar
score/information forms. Dispersion cancels in the sandwich; the family dispersion
record is not multiplied into HC0 a second time. Weights enter the score squared,
not as a claim that each unit of exposure is an independent replicated record.
Zero-weight observations contribute neither information nor score.

Supported families are Gaussian, Poisson, Gamma, Tweedie and Binomial, with their
existing links. Rate/exposure-weight and count/offset Poisson formulations are
supported. Both global and table solvers use the same inference calculation.
Base-level and weighted-mean normalization are supported; a weighted-mean intercept
includes the covariance of the table averages shifted into it. Existing rank/alias
checks and unavailable-row handling still apply.

HC0 requests reject penalties, disabled inference and unanchored `normalization='none'`.
Inference calculation failures remain visible as fit diagnostics rather than making
an otherwise usable scorer disappear. Intervals require a converged original fit.
`coefficient_intervals(..., dispersion='quasi_poisson')` rejects HC0: quasi-Poisson
scaling is a separate alternative, not another multiplier for robust errors.

Supported joint contrasts can be assessed with [whole-term tests](inference.md#whole-term-tests).
HC0 has no leverage correction, finite-sample correction, clustering or adjustment
for model selection. It can be unreliable in small samples
or high-leverage cells. Repeated policies, geographic dependence or other cluster
structures need [explicit clustered inference](inference.md#clustered-covariance); HC0 alone does
not account for them. Standard errors
on edited/loaded scorers remain unavailable; retain source fit evidence separately.

Independent tests form dense information and score matrices across all five families,
nonuniform weights and both solvers, and also verify offsets, hierarchical constraints,
weighted-mean contrasts and unchanged fitted means. These are numerical-reference
checks, not simulation evidence of interval coverage in every finite-sample setting.
The optional calculation adds a second parameter-square matrix and matrix products;
the default model-based path does not allocate that matrix.

## Clustered covariance

Use a cluster column when repeated observations within a group may be dependent:

```python
from avenue_model import GLMOptions, coefficient_intervals

model = plan.fit(
    training, 'frequency',
    GLMOptions(covariance='cluster', cluster='policy_id'),
)
print(model.inference_summary)  # cluster_cr0, policy_id, positive-weight group count
print(coefficient_intervals(model).tables['territory'])
quotes = model.predict(new_business)  # no policy_id needed unless it is a rating predictor
```

The method is an **uncorrected one-way CR0 expected-information sandwich**. It first
sums weighted observation score vectors within each group, then takes their outer
products:

```
S_g = sum_{i in group g} s_i
Cov_CR0 = H^-1 * sum_g S_g S_g' * H^-1
```

`H` and `s_i` follow the formulas in [robust inference](inference.md#hc0-covariance). Covariance
allows dependence within each group while assuming independence across groups and
a correctly specified conditional mean. It changes uncertainty only; point estimates,
means, likelihood and family dispersion are unchanged. No extra dispersion factor is
applied. One observation per group reproduces HC0.

The cluster column must contain non-null integer or string identifiers on **all fitting
rows**, including zero-weight rows. Floating-point, date or categorical identifiers
should be explicitly converted to a stable integer/string representation before fitting.
At least two groups must have positive observation weight. Groups containing only
zero-weight rows contribute neither scores nor the reported group count. Weights retain
the engine's precision/exposure interpretation, not independent replicated observations.
Choose IDs at the actual dependence level; a row number does not address repeated-policy
or geographic dependence. A multi-column grouping must be formed explicitly by the
caller without ambiguous string concatenation.

Both global and table solvers support all five existing families, including Poisson
exposure offsets. Base-level and weighted-mean normalization use the existing reduced
design and contrast machinery. Penalties, disabled inference and unanchored normalization
are rejected. Invalid cluster definitions fail before fitting. Numerical inference
failures remain visible in diagnostics while retaining the scorer, as with other
inference methods.

Reports, diagnostics and intervals record
`covariance_method='cluster_cr0'`, `cluster_column` and `n_clusters`. Cluster identities
are not added to quote requirements or copied into source evidence as data rows.
The source column/count do not replace a training-data identifier or reproducible split.
Loaded and edited scoring workbooks cannot inherit original standard errors.

Intervals use a normal approximation, **not** a Student-t distribution with group-based
degrees of freedom. The ordinary residual degrees of freedom retained in diagnostics
remain a fit/dispersion quantity. CR0 has no small-sample or leverage correction and
can be unreliable with few or highly unequal clusters. Two groups merely meets the
computational minimum; it is not evidence that inference is reliable. Multi-way
clustering, CR1/CR2 corrections, cluster bootstrap and post-selection
uncertainty remain outside this implementation. Quasi-Poisson interval rescaling is
rejected for clustered fits.

Supported joint contrasts can be assessed with [whole-term tests](inference.md#whole-term-tests).

The [homeowners tutorial](../examples/homeowners_perils.py) clusters uncertainty by
home ID across renewal years, separately from its grouped train/holdout split.
It exports labeled CR0 factor intervals;
it still models attritional water/theft only, not catastrophe aggregation.

Tests independently aggregate dense scores for all families and both solvers, check
count offsets, singleton equivalence to HC0, reversed-row invariance, zero-weight group
counts, invalid IDs, unavailable combinations, and source evidence isolation after
reload. These verify numerical formulas, not universal finite-sample coverage.
Grouping sorts row indices and holds one group score at a time, using O(n + p²) memory
rather than a dense groups-by-parameters matrix.

## Whole-term tests

Use `term_tests` to assess a factor's supported contrasts jointly, conditional on
the other terms in an original converged fit:

```python
from avenue_model import Plan, term_tests

model = (Plan.frequency('exposure')
         .banded('age', breaks=[25., 40., 60.])
         .categorical('region')
         .fit(training, 'claim_rate'))
tests = term_tests(model)
print(tests.table.select('term', 'df', 'statistic', 'p_value', 'status'))
print(tests.metadata)
```

For a categorical/banded term, the null is equality of its supported level factors.
For a polynomial variate, all nonconstant polynomial coefficients are tested together.
For a table with locked rows, the null sets its free row factors to zero while leaving
locked values unchanged. Fixed tables and terms without free supported contrasts have
an explicit unavailable result. The intercept is not included.

The result includes `null_hypothesis`, `excluded_rows` and `note` alongside the test
statistics. `excluded_rows` lists zero-support table rows: the test does not infer
their effects. A main effect in a hierarchical model tests the corresponding table
contrasts at the other interacting predictors' reference levels. It does **not** test
the main effect and every interaction involving that predictor together. Changing
those reference levels can therefore change that conditional main-effect null.

### Method and interpretation

For a term's reduced coefficients `b` and full within-term covariance `V`, the statistic
is `b' V^-1 b`. It is compared with an asymptotic chi-square distribution with one
degree of freedom per tested coefficient. Off-diagonal covariance matters: summing
squared per-row z scores would give the wrong test. Ordinary categorical/banded tests
and polynomial joint tests are invariant to a change between base-level and weighted-
mean normalization. Unnormalized fits also support these identified contrasts, even
though their individual coefficient intervals are unavailable.

The test uses the covariance fitted with the model: model-based, HC0, or one-way CR0.
For Poisson model-based inference, `term_tests(model, dispersion='quasi_poisson')`
uses Pearson dispersion to rescale the covariance. Point estimates are unchanged.
HC0/cluster covariance cannot receive this extra dispersion rescaling.

These are conditional, asymptotic tests of a prespecified structure. There is no
multiple-testing correction, small-sample F reference, cluster finite-sample adjustment
or correction for choosing terms/bands using the same data. They are not valid
post-selection evidence merely because the selected model was refitted. P-values do
not measure business importance or replace holdout loss, calibration and support review.
Very small tails can underflow to zero in double precision; the statistic is retained.

### Unavailable cases and identification

Penalized, monotonic, nonconverged and loaded/edited scorer requests fail explicitly.
Disabled/failed inference has no covariance to test. Other unavailable terms remain
rows in the result rather than disappearing:

- A complete term that is not separately identifiable from other terms.
- Singular or non-positive-definite joint covariance.
- At least as many joint constraints as independent clusters under CR0.
- Fixed terms, no supported contrasts, or unrecoverable polynomial coefficients.

Singular covariance is not silently converted into a test of a smaller hypothesis.
An unrelated identifiable term can still be tested when nuisance terms are aliased.
Identification uses numerical rank and orthogonality to the information matrix's null
directions. For example, duplicate tables cannot receive separate coefficient
uncertainty when effects can move between them without changing predictions.

### Evidence and delivery

Within-term coefficient/covariance blocks are retained during inference. Joint solves
are lazy, on request; no second global fit or full model matrix is required. Additional
retained covariance storage is quadratic in each term's reduced width, rather than
the full model width; the existing inference parameter limit still applies.

Retain the requested `tests.table.to_dicts()` and `tests.metadata` with your fit
records. A scoring workbook does not acquire the source fit's convergence or inference.
The real motor study exports the table and its interpretation metadata.

Python tests compare joint statistics and p-values
with independent dense calculations for Gaussian, Poisson, Gamma, Tweedie and binomial,
under classical/HC0/CR0 covariance and both reporting anchors. They cover quasi-Poisson,
hierarchical interactions, polynomial terms, empty bands, aliased variates, fixed priors,
insufficient clusters and scoring-artifact isolation. Chi-square tails are separately
checked against SciPy across degrees of freedom 1–5000 and extreme statistics.
Rust tests protect singular-test rejection and identifiable terms beside aliased nuisance
tables. This evidence does not supply post-selection or small-sample validity.
