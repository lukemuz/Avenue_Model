# Coefficient intervals and quasi-Poisson uncertainty

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
approximation. They do not account for tuning, data-selected bands, clustering,
uncertainty in the dispersion estimate. [HC0 robust covariance](ROBUST_INFERENCE.md)
can be selected at fitting time; it is labeled separately and is not dispersion-scaled. Penalized fits,
nonconverged fits, and fits without computed inference are rejected. Loaded/edited
workbooks cannot acquire the source fit's intervals. The original numerical evidence
is available as `model.inference_summary` and is preserved in analytical bundles.

The auto tutorial exports ordinary intervals for its fixed GLMs and separate
quasi-Poisson frequency intervals. Independent tests verify OLS intervals and
Poisson information-matrix/Pearson calculations under both exposure conventions;
the predictions are checked unchanged.
