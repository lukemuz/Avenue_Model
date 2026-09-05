# Whole-term Wald tests

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

## Method and interpretation

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

## Unavailable cases and identification

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
directions, not merely whether a column survived the rank solver. This also corrects
an earlier standard-error defect: duplicate tables must not give the first table
confident-looking errors for an effect that can be moved into the second table without
changing any prediction.

## Evidence and delivery

Within-term coefficient/covariance blocks are retained during inference. Joint solves
are lazy, on request; no second global fit or full model matrix is required. Additional
retained covariance storage is quadratic in each term's reduced width, rather than
the full model width; the existing inference parameter limit still applies.

`save_bundle` records the default covariance-based joint table and interpretation
metadata automatically, or an explicit reason when the model cannot supply it.
This is source-fit evidence. Reloaded source and edited scorers never acquire a new
convergence certificate or the right to rerun inferential tests. Older schema-1 bundles
without the additive `term_tests` field simply lack that evidence. User-requested
quasi-Poisson tests can be retained separately using `tests.table.to_dicts()` and
`tests.metadata`.

The auto, homeowners and real motor examples export `*_term_tests.json` with both
the named table and its metadata. Python tests compare joint statistics and p-values
with independent dense calculations for Gaussian, Poisson, Gamma, Tweedie and binomial,
under classical/HC0/CR0 covariance and both reporting anchors. They cover quasi-Poisson,
hierarchical interactions, polynomial terms, empty bands, aliased variates, fixed priors,
insufficient clusters and source-bundle isolation. Chi-square tails are separately
checked against SciPy across degrees of freedom 1–5000 and extreme statistics.
Rust tests protect singular-test rejection and identifiable terms beside aliased nuisance
tables. This evidence does not supply post-selection or small-sample validity.
