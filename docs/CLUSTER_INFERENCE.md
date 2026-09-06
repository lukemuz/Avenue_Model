# One-way clustered covariance

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

`H` and `s_i` follow the formulas in [robust inference](ROBUST_INFERENCE.md). Covariance
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
clustering, CR1/CR2 corrections, cluster bootstrap, whole-term tests and post-selection
uncertainty remain outside this implementation. Quasi-Poisson interval rescaling is
rejected for clustered fits.

The [homeowners tutorial](../examples/homeowners_perils.py) now clusters uncertainty by
home ID across renewal years, separately from its grouped train/holdout split and its
fixed-prediction cluster bootstrap comparison. It exports labeled CR0 factor intervals;
it still models attritional water/theft only, not catastrophe aggregation.

Tests independently aggregate dense scores for all families and both solvers, check
count offsets, singleton equivalence to HC0, reversed-row invariance, zero-weight group
counts, invalid IDs, unavailable combinations, and source evidence isolation after
reload. These verify numerical formulas, not universal finite-sample coverage.
Grouping sorts row indices and holds one group score at a time, using O(n + p²) memory
rather than a dense groups-by-parameters matrix.
