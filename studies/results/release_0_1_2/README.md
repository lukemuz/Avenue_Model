# Published-wheel performance: Avenue 0.1.2

These measurements compare the published Linux wheel of Avenue 0.1.2 with glum 3.4.1
on an AMD Ryzen 9 9950X desktop CPU: 16 physical cores, 32 logical threads and 60 GiB
of usable RAM. The environment uses Linux x86-64 and Python 3.13.15. Results were
collected on October 7, 2026.

## Results at 32 threads

Fit times are seconds; memory is whole-process peak RSS in GiB. All nine real-data
comparisons use Avenue's global solver. The largest tables separately show both
Avenue solvers. Raw records are in [real.json](real.json), [tweedie.json](tweedie.json)
and [large.json](large.json).

| Model | Rows | Parameters | Avenue fit (s) | glum fit (s) | Avenue peak (GiB) | glum peak (GiB) |
|---|---:|---:|---:|---:|---:|---:|
| French motor, Poisson | 678,013 | 79 | **0.163** | 2.01 | 0.76 | 0.81 |
| French motor, wide Poisson | 678,013 | 270 | **0.503** | 5.59 | 0.81 | 0.82 |
| NYC taxi, Gamma | 2,753,989 | 577 | 10.3 | **5.16** | 2.50 | 2.39 |
| Census income, Binomial | 45,222 | 116 | **0.13** | 7 | 0.48 | 0.54 |
| House sales, Gamma | 21,613 | 92 | **0.0239** | 0.0429 | 0.45 | 0.45 |
| House sales, Gaussian | 21,613 | 92 | **0.00403** | 0.0248 | 0.45 | 0.44 |
| Motor loss cost, Tweedie | 678,013 | 79 | **0.734** | 1.6 | 0.77 | 0.79 |
| Tweedie, ridge | 678,013 | 79 | **1.41** | 1.83 | 0.77 | 0.79 |
| Tweedie, elastic net | 678,013 | 79 | **1.03** | 1.93 | 0.78 | 0.78 |

| 20M rows, 501 parameters | Avenue global: s / GiB | Avenue table: s / GiB | glum: s / GiB |
|---|---:|---:|---:|
| 5 tables, 101 levels each | 2.97 / 2.73 | 3.11 / 2.43 | 12.6 / 3.43 |
| 100 tables, 6 levels each | 70.5 / 18.85 | 36.6 / 18.53 | 763 / 19.75 |

## Thread sensitivity

Median fit times in seconds at fixed thread limits. These are supplemental tuning
results; the headline comparison above uses the same 32-thread setting throughout.
A dash means no verified selected comparison at that setting. The initial one-thread
Tweedie attempts remain in the supporting evidence.

| Model | Avenue 1 | 4 | 16 | 32 | glum 1 | 4 | 16 | 32 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| French motor, Poisson | 0.357 | 0.195 | 0.161 | 0.163 | 0.507 | 0.265 | 0.252 | 2.01 |
| French motor, wide Poisson | 1.1 | 0.571 | 0.493 | 0.503 | 1.66 | 1.05 | 2 | 5.59 |
| NYC taxi, Gamma | 15.9 | 9.7 | 9.36 | 10.3 | 3.43 | 2.61 | 2.94 | 5.16 |
| Census income, Binomial | 0.134 | 0.131 | 0.129 | 0.13 | 0.22 | 0.14 | 0.195 | 7 |
| House sales, Gamma | 0.025 | 0.0235 | 0.0238 | 0.0239 | 0.0345 | 0.0284 | 0.0455 | 0.0429 |
| House sales, Gaussian | 0.00515 | 0.00439 | 0.00393 | 0.00403 | 0.00735 | 0.00677 | 0.00705 | 0.0248 |
| Motor loss cost, Tweedie | — | 0.89 | 0.762 | 0.734 | — | 0.283 | 0.224 | 1.6 |
| Tweedie, ridge | — | 1.73 | 1.46 | 1.41 | — | 0.587 | 0.272 | 1.83 |
| Tweedie, elastic net | — | 1.22 | 1.06 | 1.03 | — | 0.37 | 0.355 | 1.93 |

Among the verified settings shown, Avenue leads five of the six original real-data
specifications. glum leads taxi and the three additional Tweedie specifications.
Smaller thread pools can substantially improve glum's times on these workloads.

## Reproduce

Create an isolated Python 3.13 environment and install the
[v0.1.2 Linux wheel](https://github.com/lukemuz/Avenue_Model/releases/tag/v0.1.2).
The benchmark runner checks every installed package file against the supplied wheel,
including the native extension; editable imports cannot supply release evidence.

```sh
python -m pip install /path/avenue_model-0.1.2-cp312-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
python -m pip install glum==3.4.1 numpy==2.5.3 scipy==1.18.1 pandas==3.0.6 polars==1.31.0 tabmat==4.2.1 scikit-learn==1.9.1 pyarrow psutil
python studies/release_benchmarks.py --download --data-dir /tmp/avenue-data
```

Set `WHEEL` to the downloaded wheel and run each command to completion before starting
the next. Every output directory must be new. Source loading and common feature
construction are shared specifications, outside the engine timers.

```sh
WHEEL=/path/avenue_model-0.1.2-cp312-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
python studies/release_benchmarks.py --data-dir /tmp/avenue-data --output /tmp/avenue-real --wheel "$WHEEL" --cases motor motor_wide taxi census housing_gamma housing_gaussian tweedie tweedie_ridge tweedie_elastic --threads 1 4 16 32 --repeats 3
python studies/release_benchmarks.py --data-dir /tmp/avenue-data --output /tmp/avenue-real-strict --wheel "$WHEEL" --cases motor_wide census housing_gamma --threads 32 1 4 16 --repeats 3 --tolerance 1e-12
python studies/release_benchmarks.py --data-dir /tmp/avenue-data --output /tmp/avenue-tweedie-strict --wheel "$WHEEL" --cases tweedie tweedie_ridge tweedie_elastic --threads 32 4 16 --repeats 3 --tolerance 1e-11
python studies/release_benchmarks.py --data-dir /tmp/avenue-data --output /tmp/avenue-large --wheel "$WHEEL" --cases large_few large_many --engines avenue avenue_table glum --threads 32 --repeats 3
```

The tighter runs establish the accuracy needed for the comparisons. Combine Avenue's
tighter results with glum's initial results, which already supply a sufficiently
accurate reference, and validate every selected pair:

```sh
python studies/release_benchmarks.py --data-dir /tmp/avenue-data --output /tmp/avenue-verified-real --avenue-results /tmp/avenue-real /tmp/avenue-real-strict --glum-results /tmp/avenue-real --threads 1 4 16 32
python studies/release_benchmarks.py --data-dir /tmp/avenue-data --output /tmp/avenue-verified-tweedie --avenue-results /tmp/avenue-tweedie-strict --glum-results /tmp/avenue-real --cases tweedie tweedie_ridge tweedie_elastic --threads 4 16 32
```

Combining creates links to existing worker outputs and records their source paths.
It does not rerun a fit or select the fastest replicate. Later Avenue input directories
take precedence for cases they contain. The two validated grids contain 48 and 18
engine/thread configurations, respectively, with 33 prediction comparisons in total.
The large grid adds six configurations and four comparisons. All 37 selected
comparisons pass; each configuration includes one warmup and three timed fits.

The data sources are OpenML 41214 (motor frequency), 41215 (motor severity), 1590
(census income), `house_sales` version 1, and NYC TLC yellow taxi trips for January
2024. Downloaded parquet bytes can vary with the conversion-library version; each
record retains the exact local input hashes. The two synthetic cases use the same
generator and fixed seed, 20260827, with independent categorical factors.

## Measurement and agreement

Each case, engine and thread setting runs sequentially in a fresh process. Numerical
thread limits are set before importing libraries: OpenMP, OpenBLAS, MKL, NumExpr,
Rayon and Polars. Each worker performs one full-size warmup followed by three timed
fits. Tables report the median fit time; JSON also records individual runs, engine
preparation, preparation-plus-fit and prediction times. The real-data headline uses
the same 32-thread limit for both engines, matching the desktop's logical CPU count.
The thread sweep separately shows the effect of tuning. Both large cases also use
32 threads. A fresh-process [default-thread probe](default_threads.json), with no
thread environment variables set, confirms 32 threads for OpenBLAS, OpenMP and Polars
on this machine. The fit tolerances and solver choices remain explicit benchmark
settings. These are in-sample fitting comparisons; prediction timers cover the full
training data and do not measure held-out accuracy.

Both engines use identical response, exposure, weight, category and reference-level
definitions, an unpenalized intercept, and no standard-error calculation. glum uses
native pandas categoricals through tabmat, `drop_first=True`, `scale_predictors=False`
and `solver="auto"`. Avenue's `avenue` records explicitly request `solver="global"`;
`avenue_table` explicitly requests table descent. Reported solver names make this
choice visible. The default iteration limit is 5,000 for both engines.

The initial gradient tolerances are 1e-10 for the standard specifications, 1e-11 for
taxi and 1e-9 for Tweedie. Wide motor, census and housing Gamma are additionally rerun
at 1e-12, and Tweedie at 1e-11, because the original fits passed per-row agreement but
missed the stricter aggregate check. Both engines
must report convergence. Every fitted mean must agree within
`atol=1e-6, rtol=2e-6`, and the maximum absolute difference divided by the reference
predictions' RMS must be below 1e-7. Regularized Tweedie also checks coefficient
differences below 1e-5 and relative penalized-objective agreement within 1e-8.
Failed or nonconverged comparisons cannot substantiate a speed claim.

The reported comparisons keep glum's initial tolerances and use Avenue's tighter
settings for those six cases. All selected pairs pass the same prediction-accuracy
requirements. The additional tight glum runs are retained as supporting experiments;
wide motor becomes singular at 1e-12 at 4, 16 and 32 threads. These failures do not
enter the timing tables: the initial glum fits converge and match the tighter Avenue
fits. The one-thread glum elastic-net attempt was manually stopped after 290 seconds
during its first fit to prioritize the primary 32-thread comparison. It has no
reported timing or convergence conclusion; the verified Tweedie sweep covers
4, 16 and 32 threads. Worker timeouts are now 600 seconds for real cases and
3,600 seconds for large cases, including warmup and repetitions.

Memory is the Linux whole-process peak RSS through the first fit. It includes the
interpreter, imported libraries, source loading, common feature construction and
engine preparation; it excludes later predictions and timing repetitions. Both
workers import the same libraries. This metric describes peak process requirements
and replaces the older incremental-memory figures.

Motor frequency preserves the historical count cap of four and exposure clipping to
[0.001, 1]. Tweedie instead uses uncapped matched claim amounts divided by raw policy
exposure, with exposure as the sample weight, power 1.5 and alpha 0.1 for the penalized
fits. Ridge uses `l1_ratio=0`; elastic net uses 0.5. The audit records excluded orphan
claim rows and total matched loss. No predictor scaling changes the penalty between
packages.

The penalized comparisons cover ridge and elastic net. An
[earlier published-wheel lasso check](earlier_lasso_diagnostic.json) at alpha 0.1 and
tolerance 1e-9 returned `converged=False` after 572 iterations. It is retained as a
convergence diagnostic and supplies no performance claim. This documentation refresh
does not change the fitting engine.

## Evidence

Compact JSON files retain timings, convergence diagnostics, package versions, wheel
and input hashes, numerical agreement and the source hashes of the runners. Raw data,
logs and full prediction arrays remain outside the repository. The initial attempts
are retained so the tighter reruns are visible. The runner exits with an error for
an incomplete grid, a nonconverged fit or a failed agreement check; the initial grid
therefore has expected agreement failures. Rebuild and validate existing output with
`--aggregate --check`, passing the original case, engine and thread grid.

This refresh supersedes the historical headline timing and memory tables. The old
four-library ranking against scikit-learn and H2O is retired from the current claims;
those libraries were not included in this release comparison.

- [Environment and installed package versions](environment.json)
- [Initial grid, including rejected comparisons and the stopped trial](initial_grid.json)
- [Tighter real-data attempts, including singular glum fits](tight_real_attempts.json)
- [Tighter Tweedie attempts](tight_tweedie_attempts.json)
