# GLM selection with a common evaluation loss

`select_glm` evaluates an explicit grid of unresolved plans and fitting options.
It retains each fold's convergence, fit findings, parameter count, iterations,
score gradient, support, loss, A/E and errors. Every trial sees the same validation
rows. Categories and data-dependent bands resolve only on training rows.

```python
from avenue_model import GLMTrial, Plan, SplitSpec, select_glm

trials = {
    f'bands={bands},alpha={alpha},mix={mix}': GLMTrial(
        Plan.frequency('exposure').banded('age', quantile=bands).categorical('region'),
        {'alpha': alpha, 'l1_ratio': mix},
    )
    for bands in (3, 5)
    for alpha in (0.001, 0.01)
    for mix in (0., 0.5)
}
selection = select_glm(
    training, trials, target='frequency', unit='claims/exposure', metric='poisson',
    weight='exposure', split=SplitSpec.grouped('policy_id', n_splits=3, seed=47),
)
print(selection.summary)
print(selection.history)
selection.save('review/selection-v1')  # new directory
model = selection.refit(training)
print(model.report(final_holdout).markdown)
```

Candidate plans can also differ in fixed bands, variate degrees or interaction
structure. Put fitting Tweedie power on each `Plan(..., tweedie_power=...)`.
The `tweedie_power` argument to `select_glm` is the **common evaluation power**,
independent of the fitting powers. Selection rejects power in trial `GLMOptions`
to avoid the plan silently overriding it. Do not select powers using their own
incomparable raw deviances.

The default recommendation minimizes pooled held-out loss: sum of weighted losses
across folds divided by total validation weight. It requires successful scoring and
known convergence in **every** fold. A failed or nonconverged trial remains visible;
its favorable partial scores cannot win. Fit findings remain separate from selection
loss and should be reviewed for conditioning and support. Exact loss ties use trial
insertion order. No automatic complexity preference or hard table limit is imposed.

Pass a `SplitSpec` or existing `Fold` objects tied to the same dataset. Validation
sets across folds must not overlap; repeated CV needs separate runs. Out-of-time
splits can have a single fold, and validation need not cover the entire dataset.
The saved JSON includes all trial specifications, fold memberships, fingerprints,
seeds and the evaluation definition. Row membership is stored, not observation data.
`refit` checks the original selection dataset fingerprint and raises on failed
convergence. It creates a new full-data fit; the final model must be evaluated on a
separate holdout that was not used in selecting candidates.

Penalty semantics are inherited from `GLMOptions`: categorical/banded contrasts
shrink toward the first/reference row, the intercept is unpenalized, and polynomial
variate tables are not penalized. Reference choices affect penalized fits. Penalized
fits omit ordinary standard errors. Unpenalized standard errors after choosing a
structure are conditional and do not account for selection. These CV results provide
no post-selection confidence intervals.

The [auto study](../examples/auto_pricing_study.py) evaluates four penalty/power trials
within the training years, then compares the selected refit on the untouched final
year and exports its analytical bundle. Independent regression tests compare
elastic-net selection with scikit-learn and unequal-weight Tweedie evaluation with
an independent deviance calculation. Warm starts, prepared-data reuse, adaptive
search and bootstrap refit uncertainty remain future work.
