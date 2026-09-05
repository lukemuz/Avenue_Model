# Refit stability for a Plan

`bootstrap_stability` measures how predictions change when the training data are
resampled and the same Plan recipe is fitted again. Its percentile bands are **stability
diagnostics, not confidence intervals with guaranteed true-mean coverage**. They are
also not coefficient intervals or prediction intervals for future observed claims.
The [coverage pilot](REGULARIZED_UNCERTAINTY_PILOT.md) shows why this distinction matters
for regularized models.

```python
from avenue_model import bootstrap_stability

stability = bootstrap_stability(
    training, plan, review_quotes,
    target="claim_rate", options={"alpha": 0.03, "l1_ratio": 0.0},
    resamples=200, seed=42, group="policy_id", mass=0.95,
)
print(stability.summary)
print(stability.history)
stability.save("refit_stability")
```

`summary` contains the original fit's predictions, positional quote row numbers,
`stability_lower`, `stability_upper`, and complete/incomplete status. `mass` is the
central empirical bootstrap mass. `draws` contains one column per requested resample;
failed columns are null. `history` retains replicate seeds, sampled row counts, distinct
resampling-unit counts, convergence and errors. The saved output includes these tables,
the Plan/options, training/quote fingerprints and resampling metadata; it is not a
deployable model bundle.

With `group=None`, individual rows are sampled uniformly with replacement. With a
group column, whole groups are sampled uniformly with replacement, preserving their
rows and original exposure weights. Unequal group sizes can give different row counts
in different replicates. Group labels must be non-null finite scalar values, and there
must be at least two sampling units. This does not establish exchangeability across
groups or make temporal dependence disappear.

The Plan is serialized before fitting and replayed for each resample. Learned bands,
knots and categories are resolved using that resample, not the quote data or the full
training population. Penalties and other fitting options stay fixed. Upstream model
selection is not repeated, and carried priors stay fixed; their uncertainty is not
included. Ordinary coefficient covariance is disabled because this routine uses
refitted means rather than attaching unpenalized standard errors to penalized fits.

The original fit must converge and score every quote. Every bootstrap replicate must
also converge and score every quote before any percentile bands are reported. A missing
resampled category, singular fit, nonconvergence or scoring failure remains in history
and withholds all bands. Successful draws remain inspectable; failed fits are never
silently discarded to make the result look complete.

Reproduce a replicate by rebuilding the sampling units in their original row/group
insertion order, initializing `random.Random(history.sample_seed)`, and drawing
`len(units)` calls to `randrange(len(units))`. Concatenate the selected units' row
indices, then refit the saved Plan and options. The Python version and dataset
fingerprint are retained. Memory for draws grows with quote rows times resamples;
start with a meaningful set of review quotes rather than assuming a full portfolio
times hundreds of refits is cheap.

Tests compare row/group resamples with independent weighted Poisson means and Gaussian
ridge matrix solutions, verify reproducibility and saved draws, and retain missing-level
failures. The coverage pilot's `--public-api` mode verifies every public-API refit against
the independent ridge solution and compares returned bands with independently computed
percentiles. Passing these mechanics does not establish nominal confidence coverage.
