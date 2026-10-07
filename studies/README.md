# Evaluation and development checks

Use the regression suite for correctness, examples for integration, and benchmarks
for performance. Study scripts write to a **new** output directory and never install
or publish packages. Benchmark inputs can be downloaded explicitly with `--download`.
Run from the repository root.

## Installed-wheel acceptance

Build a wheel, then install it into a separate Python environment with the test and
tuning extras. Replace the placeholder below with the actual wheel filename:

```sh
maturin build --release --out dist
python -m pip install 'dist/ACTUAL_WHEEL_FILENAME.whl[test,tuning]'
python studies/wheel_acceptance.py --wheel dist/ACTUAL_WHEEL_FILENAME.whl --output /tmp/avenue-acceptance
```

The runner verifies the installed Python files and native extension against the
supplied wheel, rejects editable/source imports, and runs all Python tests with no
skips allowed. It then runs the auto, homeowners, booster/refit and continuous-spline
examples with four-thread limits. Outputs include `acceptance.json`, logs and the
example artifacts. CI and release jobs use this same runner.

For native changes, also run:

```sh
cargo fmt -- --check
cargo test --no-default-features --locked
```

The release workflow configures tests on Python 3.12/3.13 and Linux, macOS and Windows
wheels. A local run covers only its recorded environment; inspect the actual CI jobs
for other platforms. [API documentation checks](../docs/API_REFERENCE.md) separately
verify generated public anchors from an installed wheel.

## Real motor comparison

Install `glum` in addition to the test/tuning dependencies. Supply local OpenML
41214 (freMTPL2freq) and 41215 (freMTPL2sev) parquet files:

```sh
python studies/real_motor_acceptance.py --frequency /path/freMTPL2freq.parquet --severity /path/freMTPL2sev.parquet --output /tmp/avenue-motor --booster
python studies/real_motor_acceptance.py --frequency /path/freMTPL2freq.parquet --severity /path/freMTPL2sev.parquet --output /tmp/avenue-motor-spline --smooth --fit-tolerance 1e-11
```

This evaluates frequency, positive-payment severity and Tweedie loss cost. Frequency
counts observed positive-payment records, which can differ from reported `ClaimNb`.
Orphan claim records are excluded and audited. The script applies no claim/exposure
caps, development or trend. It records invalid numeric rows and uses a deterministic
25% grouped policy holdout. This random split is not an out-of-time validation.

The glum reference uses native categories and an independently evaluated SciPy basis
for splines. Every model must converge; each holdout prediction must agree within
`atol=1e-6, rtol=2e-6`. Workbook reloads must agree within `atol=rtol=1e-12`; a 5%
intercept edit must produce the expected change. Reports, interval/term-test results,
source audit and input hashes are written alongside comparison metrics.

`--booster` adds a short training-only CV search with the installed stock/fork LightGBM.
The challenger checks both consolidation modes on holdout quotes and threshold,
adjacent-float and missing/category probes. CSV reloads must preserve raw-label means.
It does not treat successful finite-round booster training as GLM convergence.

[Compact motor results](results/real_motor/README.md) retain numerical evidence for a
specific local build. Single-run fit times are diagnostic observations, not controlled
engine benchmarks or a claim that one package is universally faster. The two
specifications use the same held-out policies and common evaluation loss.

## Performance benchmarks

The [published-wheel benchmarks](results/release_0_1_2/README.md) support the
[performance claims](../README.md#performance-at-a-glance). They compare Avenue and
glum on six real-data specifications, Tweedie loss cost with and without penalties,
and two portfolios of 20 million observations. The runner verifies the installed wheel
payload, isolates engines in fresh processes, records convergence and checks fitted
means before a comparison is used in the documentation. Its results include repeated
timings, thread settings, package versions, input hashes and whole-process peak memory.

See the linked results for exact reproduction commands. Run engines sequentially
without competing CPU work. The older `scripts/bench_*.py` runners remain available
for targeted experiments; their historical timings are superseded by the release
benchmark tables.

[Large-table scoring](../docs/LARGE_TABLE_SCORING.md) documents a narrower matching
benchmark and its historical before/after records. Its runner accepts explicit policy,
booster and optional workbook paths; it does not depend on an old evaluation folder.
Run the same saved workbook on two builds when measuring an implementation change.

## Independent reference fixtures

`tests/fixtures` contains the small deterministic inputs used by regression tests.
The natural-cubic reference generators use NumPy/SciPy independently of Avenue:

```sh
python studies/reference_natural_spline.py --output /tmp/natural_spline.json
python studies/reference_spline_fit.py --output /tmp/spline_fit.json
```

Compare generated values and recorded producer versions before intentionally updating
fixtures. Ordinary acceptance runs do not rewrite reference data.

## Results policy

Retain compact measurements that substantiate a documented claim, with source/input
hashes and methodology. Write exploratory reports, fitted workbooks, logs, prediction
arrays and downloaded datasets outside the source tree. Regression tests should
assert correct behavior; obsolete defect-expecting evaluation scripts are not a
second test suite. The earlier expansion plans and run-by-run reports are archived,
not maintained as current documentation.
