# Independent-observation robust covariance

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
`GLMDiagnostics`, `FittedModel.inference_summary`, interval metadata and analytical
bundle source evidence identify the covariance method explicitly.

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

HC0 has no leverage correction, finite-sample correction, clustering, whole-term
Wald tests or adjustment for model selection. It can be unreliable in small samples
or high-leverage cells. Repeated policies, geographic dependence or other cluster
structures need [explicit clustered inference](CLUSTER_INFERENCE.md); HC0 alone does
not account for them. Standard errors
on edited/loaded scorers remain unavailable; source bundle evidence stays separate.

Independent tests form dense information and score matrices across all five families,
nonuniform weights and both solvers, and also verify offsets, hierarchical constraints,
weighted-mean contrasts and unchanged fitted means. These are numerical-reference
checks, not simulation evidence of interval coverage in every finite-sample setting.
The optional calculation adds a second parameter-square matrix and matrix products;
the default model-based path does not allocate that matrix.
