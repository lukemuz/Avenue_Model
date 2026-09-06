# Real-data fork challenger acceptance, September 5, 2026

Historical evidence for the builds and scripts recorded below. The current library
has removed study orchestration and analytical bundle APIs; see the
[scope review](COMPLEXITY_REVIEW.md). Saved results are retained, but commands and
artifact layouts below describe the historical run.

The fork challenger now completes the real motor study from a fresh wheel environment,
using the same paid-claim definition, training rows and untouched holdout as the GLMs.
Both conversion modes preserve held-out predictions, restored category labels and
raw-quote workbook scoring. The study also produces a named booster-frequency × GLM-
severity composition, quote explanations, source evidence and an audited rate edit.

The [real motor study](REAL_MOTOR_ACCEPTANCE.md) explains the paid-record definition,
orphan-record exclusion, uncapped losses/exposure and grouped policy split. No claim is
made that these observed losses are developed prospective ultimate losses. Historical
evaluation results are unchanged. New compact evidence is in
[studies/results/real_fork](../studies/results/real_fork/result.json).

## What was run

A fresh Python 3.12 environment installed the release Avenue wheel and the existing
fork wheel, LightGBM 4.6.0.99. The full 103-test Python suite passed in that environment.
The GLM reference checks also reran successfully against glum's native categorical path.
The fork trained on 508,509 policies and evaluated 169,504 held-out policies. Three
inner folds were grouped by policy ID, with all category mappings learned on training
data. The inner LightGBM folds share Dataset bins; the final holdout does not participate
in Dataset construction.

Four trials searched learning rate and both fork interaction penalties. Trees had at
most four leaves and depth two, with a minimum leaf size of 2,000. The selected trial
used 80 rounds, the search cap, so more rounds could alter the result. This is a bounded
acceptance search, not evidence of the optimal structure or penalty. Trial specifications,
fold counts and selected iteration are retained in
[trials.json](../studies/results/real_fork/trials.json).

The final model's CV mean table count was six, with six in each fold. Actual conversion
complexity differs by consolidation mode:

| Conversion | Tables | Total rows | Largest table | Maximum interaction order | Warm scoring, 169,504 quotes |
|---|---:|---:|---:|---:|---:|
| analysis | 10 | 153 | 42 | 2 | 0.0208 s |
| max | 6 | 329 | 210 | 2 | 0.0259 s |

Fewer tables did not mean fewer cells or lower observed scoring cost. These timings
are medians of three local warm batches, not isolated comparative benchmarks. Single-
quote median observations were approximately 43 and 42 microseconds respectively.
Tuning took 3.30 seconds and final booster fitting 0.223 seconds. The complete study,
including GLMs, independent references, conversion, parity and review artifacts, took
26.10 seconds. Whole-process peak RSS was about 3.06 GiB, not isolated model memory.

## Scoring and delivery evidence

Both modes passed all 169,504 holdout comparisons and 810 additional probes at every
numerical split threshold, adjacent floating-point values, numerical missing routes,
and categorical missing/unseen codes. Maximum holdout absolute error was 5.0e-16;
maximum relative error was 2.73e-15, within the declared `atol=rtol=1e-12` contract.
Every workbook was reloaded with external category identity restored and scored raw
quote labels without training weights. These are tested-input parity results, not
proof of arbitrary unsupported booster semantics.

The max-mode model retains build/conversion lineage in an analytical bundle. Its
original fit diagnostics remain unknown, as appropriate for a converted booster.
A 5% intercept edit was checked on every holdout quote. Detailed change attribution
was retained on a 1,000-row sample, where the weighted change is exactly 5% within
floating-point precision. Fresh edited-model validation uses the entire holdout.
[Parity/complexity evidence](../studies/results/real_fork/result.json) and
[sample edit totals](../studies/results/real_fork/sample_edit_totals.csv) record the checks.

## Predictive result and interpretation

On identical exposure-weighted holdout rows:

| Candidate | Common evaluation loss | A/E |
|---|---:|---:|
| GLM frequency | Poisson deviance 0.453825 | 0.9902 |
| Fork booster frequency | Poisson deviance 0.451044 | 0.9904 |
| GLM frequency × GLM severity | Power-1.5 Tweedie deviance 87.6173 | 1.4279 |
| Fork frequency × GLM severity | Power-1.5 Tweedie deviance 85.7881 | 1.4330 |
| Direct Tweedie GLM | Power-1.5 Tweedie deviance 87.7905 | 1.4188 |

The frequency challenger improves the tested losses, but it does not solve the
uncapped loss-cost calibration problem. Twenty paired bootstrap replicates exercise
the workflow; they are insufficient for stable uncertainty claims. No holdout rebasing
or additional holdout-driven search was used to conceal the remaining mismatch.
[Frequency comparison](../studies/results/real_fork/frequency_comparison.csv) and
[loss-cost comparison](../studies/results/real_fork/comparison.csv) retain all candidates.

The finite boosting schedule has no GLM score-convergence certificate. The common
comparison therefore displays unknown convergence and does not automatically recommend
the booster or its product. Its predictive evidence is available for explicit review;
this remains a limitation of using one convergence eligibility rule across estimators.
No classical GLM standard errors are assigned to booster-derived factors.

## Reproduction and remaining scope

Install a newly built Avenue wheel plus glum and the tuning/test dependencies in a
fresh Python 3.12 environment. Install the intended fork wheel in that environment,
then use the same public policy/severity Parquets as the real motor exercise:

```sh
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 POLARS_MAX_THREADS=4 RAYON_NUM_THREADS=4 \
python studies/real_motor_acceptance.py --frequency frequency.parquet --severity severity.parquet --output /tmp/real-fork-new --booster
python -m unittest discover -s tests
```

`--booster` selects the installed stock/fork module through the public resolver; inspect
the resulting build metadata. Input, source and installed-wheel hashes are retained in
[provenance](../studies/results/real_fork/provenance.json). Exact observed versions and
workflow timings are in [run.json](../studies/results/real_fork/run.json). Full local
artifacts reside at `/tmp/avenue-real-fork-acceptance`.

This completes the real-data fork arm of the acceptance exercise. Remaining gates
include smooth/monotonic effects, controlled performance/memory studies, broader wheel
and dependency support, and a final requirement-by-requirement audit. HC0 and one-way
cluster inference now exist; small-sample and post-selection inference remain limited.
The broader first-choice goal remains active.
