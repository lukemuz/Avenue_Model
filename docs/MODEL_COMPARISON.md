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
print(comparison.discrimination["avenue"])  # use a candidate name from your comparison
print(comparison.segments["region"])
print(comparison.recommended)
```

External prediction vectors must be in the same row order as `holdout`. For external
models, supply evidence-backed `converged=True` only after checking their fit; it is
caller-provided evidence. Avenue models expose their own convergence, and a recorded
failure cannot be overridden. A candidate with unknown training status is compared but
is not recommended. The recommendation is the lowest common evaluation loss among
successfully scored candidates with known convergence or explicitly completed training, not a complete actuarial
approval or a claim of statistical superiority.

For learners such as finite-schedule boosting, use
`Candidate(booster_model, unit="rate", training_status="completed")` after verifying
that the intended training procedure finished. The booster tutorial does this after
the selected schedule returns and conversion/reload parity checks pass. Completion
is caller-provided training evidence; successful conversion alone establishes no
training or generalization claim. It leaves `converged` null instead of inventing a
GLM convergence certificate. Unknown external predictions remain ineligible by default.

The summary separates scoring `status`, effective `training_status` (`converged`,
`completed`, `failed`, `unknown`), the caller's `declared_training_status`, and
`converged`. Set `training_status="failed"` to record a failed training procedure;
`"unknown"` or `None` supplies no completion evidence. Any model- or caller-reported
nonconvergence, or an explicit failed training status, defeats completion evidence.
A scoring failure remains ineligible even if training completed. Existing callers
using `converged=True` remain supported. Contradictory failure evidence is resolved
conservatively and retained in the separate columns.

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

The summary also reports `gini`, `normalized_gini` and `discrimination_status`.
`discrimination[candidate]` contains a concentration curve ordered by ascending
predicted response mean. Its `score`, `rows`, `weight` and `actual` columns describe
each tied-score group; `weight_share` and `actual_share` are cumulative portfolio
shares. The first row is the origin with a null score and zero support. Zero-weight
observations remain in the common prediction population but contribute no rank support.

Gini is one minus twice the area under this curve, integrating linearly between
tied-score groups. Computation uses an equivalent centered rank-covariance formula
to avoid cancellation when observed targets are nearly constant. Positive values
indicate that higher predicted means concentrate
more observed loss; reversed rankings can produce negative values. Normalized Gini
divides by the same statistic obtained by ranking on the observed target. This oracle
is descriptive and uses the evaluation outcomes; it is never a fitted candidate.
The weights are exactly the comparison weights, so loss-cost targets with exposure
weights rank loss cost against cumulative exposure and cumulative observed loss.
Ties are aggregated before integration, preventing arbitrary row order from affecting
the result. Constant predictions have zero discrimination. Multiplying predictions
by a positive constant leaves ranking unchanged but can worsen calibration and loss.

Negative targets on positive-weight rows and zero total actuals make these measures
unavailable (`negative_target` or `zero_actual`); their curves are omitted. Constant
supported targets yield Gini zero and normalized Gini null (`constant_target`). Failed
candidates have null measures and `scoring_failed` status. These cases do not disable
otherwise valid loss comparisons. Discrimination is an empirical diagnostic with no
intervals here; it does not change convergence eligibility or loss-based recommendation.

Optional 95% paired percentile bootstrap intervals describe loss differences relative
to the first named candidate. Negative differences favor the candidate. Row resampling
is the default; `bootstrap_group` samples whole groups with replacement. Predictions
are fixed: these intervals do not include refitting, model selection or future regime
uncertainty. Zero-support replicates are skipped and effective replicate counts are
reported. If the baseline fails, no difference interval is reported. For temporal
uncertainty choose appropriate blocks explicitly; this is not a time-series bootstrap.
