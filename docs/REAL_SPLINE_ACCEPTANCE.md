# Real motor study: continuous spline acceptance

Historical evidence for the builds and scripts recorded below. The current library
has removed study orchestration and analytical bundle APIs; see the
[scope review](COMPLEXITY_REVIEW.md). Saved results are retained, but commands and
artifact layouts below describe the historical run.

The continuous workflow passes fitting, independent-prediction and delivery checks
on the French motor portfolio. The tested spline specifications **do not improve
holdout loss over the banded baseline**. Their fitting time and boundary uncertainty
leave material work before claiming that this is the preferred modeling path.

This is editable-source evidence for engine commit `7f2579d`, with the rebuilt
release extension. It is not a refreshed clean-wheel or cross-platform acceptance run.
The [provenance manifest](../studies/results/real_splines/provenance.json) records
extension/script hashes, versions, thread settings, the common split hash and hashes
of all generated artifacts. Historical evaluation outputs remain unchanged.

## Population and specifications

The study uses the same 678,013 policy rows and positive-payment definition as the
[earlier motor study](READINESS_AUDIT.md). Paid claim records are joined by policy ID;
195 orphan claim records totaling 788,714.18 are excluded and reported. There are no
loss, exposure or predictor caps, development factors, trends or monetary adjustments.
Reported claim counts remain separate from paid-record frequency. The retained
[source audit](../studies/results/real_splines/smooth/source_audit.json) gives totals.

The grouped policy split has 508,509 development policies and 169,504 holdout policies;
the severity subsets have 18,764 and 6,180 policies. Both runs have identical input
hashes and byte-identical split specifications. The split SHA-256 is
`1154267c719539752b27e70e22217d5c11582ca0613d1a9dbda691cd69850791`.
There is no time variable, so this is a grouped random holdout rather than temporal
validation.

Poisson frequency, Gamma severity and Tweedie(1.5) pure premium each include driver
age, vehicle age, bonus/malus, region and fuel. The baseline keeps the existing
prespecified bands. The continuous alternative requests five training-quantile knots
for each numeric effect, with linear endpoint-tangent tails. Ties reduce bonus/malus
to three knots for frequency/pure premium and four for severity. Resulting identified
parameter counts are 33, 34 and 33, versus 37 for each banded model. Knot locations
are retained in [model records](../studies/results/real_splines/smooth/models.json).
These specifications were not selected using this holdout.

## Independent agreement and numerical sensitivity

The glum reference receives an independently evaluated SciPy natural-cubic cardinal
basis using Avenue's resolved training knots, plus native categorical columns. Its
first spline column is dropped so the intercept represents the constant direction.
This checks continuous evaluation and fitting rather than comparing a band approximation.

At Avenue relative-score tolerance `1e-9`, the Tweedie comparison failed the existing
`rtol=2e-6, atol=1e-6` threshold on 14 of 169,504 holdout rows. The maximum relative
difference printed for those violations was `3.25043563e-6`. The
[failed comparison log](../studies/results/real_splines/initial_parity_failure.log)
is retained. Frequency and severity passed in that run.

Tightening only Avenue's stopping tolerance to `1e-11` passed all three comparisons.
Glum's gradient tolerance stayed `1e-9`, and the acceptance threshold stayed unchanged.
The two engines use different score normalizations, so equal numerical tolerance
values do not imply equal coefficient precision.

| Model | Holdout rows | Maximum relative difference from glum |
|---|---:|---:|
| Frequency | 169,504 | 6.60e-9 |
| Severity | 6,180 | 1.97e-9 |
| Tweedie pure premium | 169,504 | 3.92e-8 |

The [per-model parity records](../studies/results/real_splines/smooth/premium_parity.json)
retain thresholds and failed-row counts. All models converge, produce knot intervals
and whole-term tests, survive bundle reloads to `1e-12` relative/absolute tolerance,
and support explanations, frequency/severity composition and a reviewed 5% intercept
edit. The successful [run record](../studies/results/real_splines/smooth/run.json)
was written only after those assertions passed.

## Holdout quality

Both specifications are evaluated on the same pure-premium target with exposure
weights and Tweedie deviance at power 1.5; lower loss is better.

| Structure | Model | Mean loss | Actual / expected |
|---|---|---:|---:|
| Banded | Frequency × severity | 87.61734 | 1.42788 |
| Continuous spline | Frequency × severity | 89.50891 | 1.42353 |
| Banded | Tweedie | 87.79046 | 1.41876 |
| Continuous spline | Tweedie | 89.87306 | 1.42282 |

Sources: [banded comparison](../studies/results/real_splines/banded/comparison.csv)
and [continuous comparison](../studies/results/real_splines/smooth/comparison.csv).
The spline loss costs total about 29.7% below actual held-out losses. This is poor
calibration, despite close agreement between implementations. No production readiness
or statistically significant improvement is inferred from these results.

The Tweedie vehicle-age coefficient at the upper boundary of 100 has standard error
about 6.64 on the link scale. Its
[knot interval exhibit](../studies/results/real_splines/smooth/premium_vehicle_age_intervals.csv)
makes that weak precision visible. Exact continuous scoring does not solve sparse-tail
estimation. Knot selection and stabilization still need evaluation inside training folds.

## Runtime and practical limits

These are single-run workflow observations on the recorded machine, with five thread
settings held at four. They are not controlled speed benchmarks.

| Model | Avenue spline fit | glum spline fit | Reference basis/category preparation | Avenue banded fit |
|---|---:|---:|---:|---:|
| Frequency | 3.719 s | 0.367 s | 0.276 s | 0.268 s |
| Severity | 0.097 s | 0.016 s | 0.009 s | 0.024 s |
| Tweedie | 16.339 s | 0.311 s | 0.232 s | 0.601 s |

Avenue's fit includes Plan checks and covariance; glum's timed fit uses its default
`store_covariance_matrix=False`, with independent design preparation timed separately.
These scopes differ, so the ratios are not isolated solver speedups. The observed gap
still warrants profiling. Tweedie needed 103 table sweeps, versus four glum iterations.

The spline workflow took 30.42 s total, versus 11.31 s for bands. Raw continuous scoring
took about 18–20 ms for 169,504 frequency/pure-premium quotes. Whole-process peak RSS
was 1,843,012 KiB for the spline workflow; that includes data preparation, pandas,
SciPy/glum, validation and delivery, and is not incremental engine memory. Detailed
[timings](../studies/results/real_splines/smooth/models.json) are retained.

Follow-up [profiling and acceptance](../studies/results/spline_profile/README.md)
identified repeated Tweedie deviance evaluation as a substantial cost. Reusing the
existing square-root specialization at power 1.5 and removing unused information-matrix
work from convergence checks reduced the observed full-inference Tweedie fit to
10.758 seconds and 97 sweeps. All three independent prediction comparisons and delivery
checks passed again; Tweedie's maximum relative difference from glum was 1.00e-9.
These remain single-run observations. The model specification, stopping tolerance and
comparison thresholds were unchanged, and the holdout quality limitation remains.

Further performance work should address repeated sweeps and reusable training geometry.
Roughness penalties, rare-tail stabilization, broader model selection and refreshed
wheel acceptance remain open. The historical banded-model performance claims do not
establish spline fitting performance.

## Reproduce

Use an output path that does not already exist:

```sh
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
POLARS_MAX_THREADS=4 RAYON_NUM_THREADS=4 \
python studies/real_motor_acceptance.py \
  --frequency /path/to/freMTPL2freq.parquet \
  --severity /path/to/freMTPL2sev.parquet \
  --output /tmp/avenue-real-spline-study --smooth --fit-tolerance 1e-11
```

Omit `--smooth` for the banded baseline. SciPy is needed for the independent spline
reference; it is not a core Avenue fitting dependency. Full generated artifacts for
this run remain under the `/tmp` paths recorded in the provenance manifest; compact
results and review exhibits are retained in this repository.
