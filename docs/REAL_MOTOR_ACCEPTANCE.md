# Real motor acceptance exercise, September 5, 2026

The real frequency/severity/pure-premium arm now completes from a freshly installed
release wheel. All three GLMs converge, agree with glum's native categorical path,
and deliver raw-quote scoring artifacts with retained inference and review evidence.
This proves a substantial part of the ordinary workflow. It does **not** establish
that Avenue is yet the first choice for the complete improvement-plan scope.

The [executable study](../studies/real_motor_acceptance.py) and
[recorded results](../studies/results/real_motor/run.json) are separate from the unchanged
historical evaluation. It uses the public freMTPL2 frequency and severity tables,
joined by policy identifier as described in the
[scikit-learn insurance example](https://scikit-learn.org/stable/auto_examples/linear_model/plot_tweedie_regression_insurance_claims.html).
Its own modeling decisions are explicit: no exposure, claim-count or monetary caps;
no development or trend assumptions; grouped random policy holdout because these
inputs have no suitable temporal field. This is observed paid-loss modeling, not a
prospective ultimate-loss rate recommendation.

## Data reconciliation

There are 678,013 policies and 26,639 source severity records. The join excludes 195
severity records without policy predictors, totaling 788,714.18 source loss units;
the full run retains those records as an exclusion audit. The matched data contains
26,444 paid claims and 59,909,216.50 loss units. Reported `ClaimNb` totals 36,102 and
has 9,117 policy-level disagreements with available paid-record counts. The modeled
frequency explicitly uses **matched positive-payment records per exposure**; it does
not silently relabel reported frequency. These definitions make frequency × severity
and direct pure premium commensurate.

The largest individual claim is 4,075,400.56. The 1,224 exposure values above one are
retained. All policy rows pass the declared preparation rules; positive-loss severity
uses 24,944 policies. Original reported claims remain available in the study frame.
[Source audit](../studies/results/real_motor/source_audit.json) and
[population totals](../studies/results/real_motor/populations.csv) preserve these choices.

## Numerical and delivery evidence

Frequency and Tweedie use 508,509 training and 169,504 holdout policies. Severity uses
18,764 training and 6,180 holdout policies. Each prespecified design has 37 parameters:
three banded predictors, region and fuel, with an intercept. Independent glum fits
use pandas categorical inputs and its native categorical processing with treatment
coding, rather than a manually assembled sparse dummy matrix.

| Model | Maximum relative prediction difference vs glum | Avenue fit seconds | glum fit seconds |
|---|---:|---:|---:|
| Poisson frequency | 8.66e-14 | 0.247 | 0.161 |
| Gamma severity | 8.92e-9 | 0.020 | 0.012 |
| Tweedie pure premium | 5.65e-8 | 0.497 | 0.122 |

These are single workflow observations, not controlled solver benchmarks: Avenue
computed inference; glum inference was not requested. They do not support a speed-win
claim. Native-reference input preparation took about 0.48 seconds for frequency and
0.45 seconds for premium. Avenue raw scoring took about 0.013 seconds for 169,504
quotes. First converged/validated Avenue model: 2.86 seconds after data loading began,
excluding imports, installation and download. Full study: 14.11 seconds. Whole-process
peak RSS was about 2.91 GiB, including data, both engines, review and change exhibits;
it is not isolated engine memory and warrants workflow profiling.

All three models export direct intervals, quote explanations and analytical bundles.
Reloaded bundles agree with original raw-quote means at `atol=rtol=1e-12`; no training
response or weight columns are supplied for quote means. Named frequency × severity
composition is saved separately. A 5% premium intercept edit produces a 5% portfolio
change within floating-point precision and receives a fresh validation report.
[Timing/reference records](../studies/results/real_motor/models.json) and
[edit totals](../studies/results/real_motor/edit_totals.csv) provide the numerical evidence.

## Predictive assessment and remaining gates

Holdout A/E is **1.428** for frequency × severity and **1.419** for Tweedie. Both materially
underpredict this uncapped loss sample. Common power-1.5 Tweedie loss is 87.6173 versus
87.7905 respectively; glum's matching Tweedie predictions produce the same loss.
The 20-replicate paired bootstrap is a workflow check, too small for a stable interval
recommendation. No holdout rebasing is performed to conceal the calibration problem.
The [comparison](../studies/results/real_motor/comparison.csv) and
[model report](../studies/results/real_motor/premium_report.md) retain this unfavorable evidence.

The exercise found and fixed two review defects: an incorrect percentage interpretation
of A/E and a bucket warning that asserted aggregate calibration even when it failed.
Reports now give totals and A/E, flag the need to investigate population differences
and loss volatility, and avoid diagnosing missing interactions from bucket thresholds
alone. A regression test protects the corrected wording and actual bucket count.

Package-native preparation, splits, named reviews, composition, comparisons, explanations,
bundles and edit analysis handled the modeling workflow. Study-specific glue remains for
joining policy/claim sources, defining paid-record counts, auditing orphan records and
building the independent reference's categorical frame. It requires no custom Avenue
encoding, fabricated quote weights or bespoke factor-review infrastructure.

Still unverified or incomplete: a real-data fork challenger in this same fresh environment,
controlled latency/memory profiling, broad platform-wheel installation, smooth/monotonic
models, robust/clustered inference, and the full requirement-by-requirement acceptance
audit. Synthetic homeowners and stock/fork studies are separately verified. The remaining
statistical capabilities include ordinary pricing needs; this report does not mark the
broader first-choice goal achieved.

## Reproduction

Build a release wheel with `maturin build --release --out dist`, install it into a new
Python 3.12 environment together with `glum`, `scikit-learn` and the wheel's `test,tuning`
extras, and obtain the public datasets:

```python
from sklearn.datasets import fetch_openml
for dataset_id, name in [(41214, 'frequency'), (41215, 'severity')]:
    fetch_openml(data_id=dataset_id, as_frame=True, parser='auto').frame.to_parquet(name + '.parquet')
```

```sh
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 POLARS_MAX_THREADS=4 RAYON_NUM_THREADS=4 \
python studies/real_motor_acceptance.py --frequency frequency.parquet --severity severity.parquet --output /tmp/real-motor-new
python -m unittest discover -s tests
```

Use a new output directory. Exact package versions and input hashes are in `run.json`;
Parquet hashes can differ across writer versions even for equivalent source records.
The retained provenance records the script, validation source and installed wheel hashes.
The clean wheel passed all 95 Python tests; Rust passed 258 tests with six ignored tests
and one ignored doc test. The study is deliberately outside ordinary CI because its
public data and independent glum dependency are larger than the synthetic smoke studies.

## Follow-up

The [real-data fork challenger arm](REAL_FORK_ACCEPTANCE.md) has since completed in a
fresh fork/wheel environment, on this same paid-record definition and holdout.
Its numerical agreement, predictive improvements and remaining calibration weakness
are reported separately; the original measurements above remain unchanged.
