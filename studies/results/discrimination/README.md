# Shared discrimination acceptance

Numerical follow-up: the initial `1 - 2 * area` calculation lost relative precision
for nearly constant targets, yielding normalized Gini -2 for a reversed ranking
whose correct value is -1. The implementation now computes the equivalent weighted
rank covariance after centering targets. Regression cases with target differences
from 1e-12 through 1e-15 agree with independent pairwise calculations. All 130 Python
tests and the auto example pass. Recomputing the statistic from the retained real
tied-score group totals differs from the original area calculation by less than
2e-13 (`stability_check.json`). This follow-up does not rerun model fitting or replace
the historical acceptance source hashes below.

The real motor smooth study was rerun with common concentration curves and Gini
exhibits for the composed frequency/severity model, Avenue Tweedie and independent
glum predictions. The population, specification and interpretation remain those in
`docs/REAL_SPLINE_ACCEPTANCE.md`: 169,504 held-out policies, exposure-weighted observed
paid loss cost, no temporal validation or prospective loss adjustments.

| Candidate | Gini | Normalized Gini | A/E | Mean Tweedie loss |
|---|---:|---:|---:|---:|
| Frequency × severity | 0.481715 | 0.486882 | 1.423526 | 89.508906 |
| Avenue Tweedie | 0.488283 | 0.493521 | 1.422819 | 89.873058 |
| glum Tweedie | 0.488283 | 0.493521 | 1.422819 | 89.873058 |

The richer ranking exhibit does not overturn the observed calibration limitation or
establish predictive superiority. Recommendation still uses the common prespecified
loss and recorded convergence. No uncertainty intervals for discrimination are claimed.

All 129 Python tests passed, including an independent pairwise weighted-rank calculation,
ties and row permutations, zero support, inverse/constant rankings, scale changes and
undefined target cases. The auto example exports concentration curves. The real study
passed all independent mean, inference, bundle and delivery gates. Exported curves were
also checked for row/weight/actual reconciliation and their trapezoidal Gini recomputed
from CSV coordinates with `math.fsum`.

`comparison.csv` retains full precision. `provenance.json` identifies the parent source,
hashes of modified source and all generated real-study files, and curve checks. The
complete curves and models remain in `/tmp/avenue-real-discrimination`; they are not
duplicated here. Run `studies/real_motor_acceptance.py --smooth --fit-tolerance 1e-11`
with the documented frequency/severity inputs and an unused output directory to reproduce.
