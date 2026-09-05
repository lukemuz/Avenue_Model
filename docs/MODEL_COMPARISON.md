# Compare means on one evaluation population

```python
from avenue_model import Candidate, compare_models

comparison = compare_models(
    holdout,
    {
        "avenue": Candidate(fitted_frequency, unit="rate"),
        "glum": Candidate(glum_predictions, unit="rate", converged=True),
    },
    target="frequency", weight="exposure", unit="rate", metric="poisson",
    segments=["region", "accident_year"],
    bootstrap=500, bootstrap_group="policy_id", seed=42,
)
print(comparison.summary)
print(comparison.segments["region"])
print(comparison.recommended)
```

External prediction vectors must be in the same row order as `holdout`. For external
models, supply evidence-backed `converged=True` only after checking their fit; it is
caller-provided evidence. Avenue models expose their own convergence, and a recorded
failure cannot be overridden. A candidate with unknown convergence is compared but
is not recommended. The recommendation is the lowest common evaluation loss among
successfully scored candidates with known convergence, not a complete actuarial
approval or a claim of statistical superiority.

Every candidate explicitly declares a common unit. These declarations are assertions
by the caller; Avenue does not infer external physical units or perform conversions.
Use rates with exposure weights or counts without those weights, consistently across
all candidates. The target and weights are validated once, and no row is silently
excluded. A candidate that cannot score all rows remains in the summary as failed,
with its error and null predictions, instead of receiving an easier population.

Choose `poisson`, `gamma`, `tweedie`, or `squared_error`. The comparison metric is
independent of model family, including for composed means. Tweedie uses one explicit
`tweedie_power` for all candidates. Comparing fits trained at different powers under
this common loss is valid; comparing their own raw training deviances is not the
selection criterion here. Gamma requires positive responses and means; Poisson and
Tweedie permit zero responses (and a zero mean only when the response is also zero).

Summary and segment tables report row counts, weight, weighted actual/expected totals,
A/E and mean weighted loss. A/E is null for zero expected totals. Each segment exhibit
reconciles to the common portfolio; null segment labels remain an explicit group.
`predictions` preserves row positions, actuals and weights alongside named predictions.
Metadata records the population fingerprint, unit, metric, evaluation power and seed.

Optional 95% paired percentile bootstrap intervals describe loss differences relative
to the first named candidate. Negative differences favor the candidate. Row resampling
is the default; `bootstrap_group` samples whole groups with replacement. Predictions
are fixed: these intervals do not include refitting, model selection or future regime
uncertainty. Zero-support replicates are skipped and effective replicate counts are
reported. If the baseline fails, no difference interval is reported. For temporal
uncertainty choose appropriate blocks explicitly; this is not a time-series bootstrap.
