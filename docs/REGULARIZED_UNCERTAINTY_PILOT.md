# Regularized bootstrap coverage pilot

A generic percentile bootstrap around a regularized fit must not be presented as a
confidence interval for the underlying unpenalized response mean. This controlled
Gaussian ridge experiment demonstrates a large coverage failure caused by shrinkage
bias even when every fit is numerically correct.

The study uses 100 independently generated datasets of 300 observations each. A binary
predictor has population probability 1/2; true means are 0.2 and 1.0, with independent
unit-variance Gaussian noise. For each dataset, Avenue fits the same prespecified
categorical Plan at ridge penalties 0, 0.01 and 0.3. Each penalty uses identical sets
of 100 pairs-bootstrap resamples. The intercept is unpenalized; penalty mixing is zero.
All 30,300 original/bootstrap fits converge and their quote predictions agree with an
independent two-by-two ridge normal-equation solution to about 2.4e-15 absolute error.
No interval is computed from a reduced subset if a replicate fails.

The 2.5th and 97.5th percentiles of the refitted quote predictions were assessed against
two different quantities: the true group mean and the population optimum of the
fixed-penalty objective. These are equal only at zero penalty in this experiment.

| Ridge alpha | Group | Coverage of true mean | Coverage of penalized target | Mean bias vs true mean |
|---|---:|---:|---:|---:|
| 0 | 0 | 94% | 94% | 0.002 |
| 0 | 1 | 95% | 95% | 0.002 |
| 0.01 | 0 | 93% | 94% | 0.017 |
| 0.01 | 1 | 95% | 95% | -0.014 |
| 0.3 | 0 | 6% | 91% | 0.221 |
| 0.3 | 1 | 10% | 92% | -0.216 |

At alpha 0.3, the bootstrap distributions follow the shrunken predictor, leaving the
true means well outside most intervals. Correct solver output and plausible interval
widths do not establish the intended coverage. Coverage of the penalized target also
does not establish coverage of the true mean. With only 100 outer datasets, a coverage
estimate near 95% has Monte Carlo standard error about 2.2 percentage points; the
100 inner resamples give coarse tail estimates. This pilot rejects a misleading
interpretation, rather than validating a general interval method.

The public regularized-uncertainty API remains open. Its next design must specify the
estimand, the resampled pipeline, and what bias or model-selection uncertainty is
included. Fixed-penalty fit-stability bands can be useful if labeled as such, but are
not a substitute for valid confidence intervals for true regression effects. Lasso
has further bootstrap-consistency issues; this ridge experiment does not validate a
lasso method. See Chatterjee and Lahiri,
[Bootstrapping Lasso Estimators](https://www.tandfonline.com/doi/abs/10.1198/jasa.2011.tm10159).

Run `studies/regularized_bootstrap_coverage.py --output <new-directory>` to reproduce.
[Retained evidence](../studies/results/regularized_bootstrap/result.json) includes the
seed, script/extension hashes, library versions and failures; the neighboring history
and summary CSVs retain each dataset's intervals, targets and coverage. This is an
editable-source Gaussian ridge pilot, not an insurance-portfolio acceptance result,
a post-selection coverage claim, or a computational benchmark.
