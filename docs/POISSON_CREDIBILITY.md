# Conditional credibility for sparse groups

`poisson_credibility` partially pools one categorical group's frequency relativities
around a fixed Poisson baseline. It returns ordinary Avenue scoring tables and a
separate posterior exhibit. Install `avenue_model[credibility]` for SciPy quantiles.

```python
from avenue_model import poisson_credibility, CredibilityResult

# baseline is a previously prepared Poisson FittedModel declaring its exposure.
# experience contains integer paid_claims, territory strings and baseline inputs.
result = poisson_credibility(
    experience, baseline, group="territory", counts="paid_claims",
    prior_strength=2.0, confidence=0.95,
)
print(result.posterior)
counts = result.model.predict(quotes)       # quote exposure applied once
rates = result.model.predict_rate(quotes)   # exposure not required
result.save("territory_credibility")
loaded = CredibilityResult.load("territory_credibility")
```

For group g, let E_g be the sum of baseline expected claims and K_g the observed
integer claim count. The model is K_g | theta_g ~ Poisson(E_g theta_g), with independent
theta_g ~ Gamma(shape=a, rate=a) priors. The posterior is Gamma(a+K_g, a+E_g).
Its mean is `(a+K_g)/(a+E_g)`, equivalently `Z_g * (K_g/E_g) + (1-Z_g)`, with
`Z_g = E_g/(a+E_g)` when E_g > 0. This follows the
[Poisson-with-exposure conjugacy derivation](https://statproofbook.github.io/P/poissexp-post.html).

`prior_strength` is a prespecified positive a, measured against baseline expected
claims, not raw exposure. Larger values pull group relativities more strongly toward
one. The default prior mean of one retains the baseline risk level. Prior variance
is 1/a. The helper does not estimate a from the supplied groups, select it on a holdout,
or claim generic ridge coefficients represent a random-effects posterior.

The posterior exhibit gives counts, expected counts, rows, unpooled A/E, credibility
weight, posterior shape/rate, mean, standard deviation and equal-tail Gamma credible
intervals. A declared group with zero expectation and zero claims retains its prior
and is labeled `prior_only`; positive claims with zero expectation are rejected.
No rows are silently excluded. Negative, fractional, missing or nonfinite counts are
rejected. The group must contain non-null strings and must not already be a baseline
predictor. Unseen quote groups raise the ordinary unmatched error; they do not silently
receive a fallback rate.

The baseline and prior strength are treated as fixed. Use an independently developed
baseline when that matches the study design; fitting it on the same experience does
not make its estimation uncertainty disappear. These intervals exclude baseline,
hyperparameter, selection and model misspecification uncertainty. They describe group
relativities, not future realized claim counts. Integrating over a group's uncertainty
also introduces predictive overdispersion and dependence within that group; an ordinary
Poisson variance assumption for the delivered mean does not capture that uncertainty.

The model stores log posterior means in a categorical rating table, retains the
baseline's predictor encodings and declares count output. `predict_rate`, quote
explanations, frequency/severity composition and fixed-prior `Plan.offset_model`
updates use the existing engine. The scoring artifact has no GLM convergence or Wald
inference claim. For comparison, explicitly use `training_status="completed"` with
the returned model. The saved posterior stays separate from the workbook; integrity
checks refuse to attach it to edited scoring/evidence files. Edited workbooks remain
loadable through `Workbook` for scoring and fresh review.

Acceptance includes independent integration of prior times likelihood, risk-adjusted
baseline encodings, both exposure conventions, composition, reload, invalid counts
and a fixed-prior update. Run:

```sh
python studies/poisson_credibility_acceptance.py --output /tmp/credibility-study
```

In the prespecified 6,000-group simulation, sparse-group mean squared error was 0.464
for posterior means versus 11.650 for unpooled estimates; overall 95% credible interval
coverage was 94.97%. [Retained results](../studies/results/poisson_credibility/result.json)
also show supported groups and future-count comparison. This is a well-specified,
prior-predictive simulation with a known baseline. It does not establish coverage for
every fixed group rate, empirical-prior fitting, severity pooling or arbitrary mixed models.
