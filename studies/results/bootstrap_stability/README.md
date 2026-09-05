# Public bootstrap stability acceptance

The public `bootstrap_stability` API passes independent weighted-mean checks for row
and unequal-size group resampling, ridge matrix-reference checks, deterministic replay,
saved-draw roundtrips, and failure retention when a sampled training population omits
a quoted category. All 141 Python tests pass. Missing levels withhold all percentile
bands rather than silently reducing the bootstrap population.

The coverage pilot was rerun through the public API on 100 datasets, three penalties
and 100 resamples each: 30,300 fits including original fits. Every prediction agrees
with the independent ridge solution within 2.4e-15 absolute error. Returned percentile
bands also agree with independently calculated NumPy quantiles. No refits failed.

This verifies mechanics, not confidence coverage. At the strong ridge penalty, only
10% of the true group means fall in the nominal central-95% resampling bands, for both
groups. The API therefore labels its output as refit stability and makes no true-mean,
coefficient or future-observation coverage guarantee. Upstream selection and fixed-prior
uncertainty are not resampled; Plan preprocessing is resolved within each replicate.

`result.json` identifies the script and loaded extension, plus the exact stability-module
hash. `provenance.json` retains the remaining modified source hashes and artifact hashes.
Full pilot outputs remain in `/tmp/avenue-bootstrap-stability-reviewed`. Reproduce with
`studies/regularized_bootstrap_coverage.py --public-api --output <new-directory>`.
Historical coverage and wheel acceptance records remain unchanged; this is an
editable-source Python increment.
