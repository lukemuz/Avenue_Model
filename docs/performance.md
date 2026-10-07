# Performance

Avenue offers state-of-the-art GLM fitting speed: competitive with and sometimes
faster than glum, with efficient memory use. These results compare the published
**Avenue 0.1.2 Linux wheel** with **glum 3.4.1** on a regular desktop: AMD Ryzen 9
9950X, 16 cores / 32 threads, 60 GiB RAM, Python 3.13.15. Measured October 7, 2026.

## Fitting speed

Seconds per fit, median of three runs after one warmup. Avenue uses the global
solver. The default-thread comparison uses all 32 threads; the best-tested columns
show how each package performs with thread tuning (thread count in parentheses).

| Model | Rows / parameters | Avenue, 32 threads | glum, 32 threads | Avenue, best tested | glum, best tested |
|---|---:|---:|---:|---:|---:|
| Motor, Poisson | 678,013 / 79 | 0.163 | 2.01 | 0.161 (16) | 0.252 (16) |
| Motor, wide Poisson | 678,013 / 270 | 0.503 | 5.59 | 0.493 (16) | 1.05 (4) |
| Taxi, Gamma | 2,753,989 / 577 | 10.3 | 5.16 | 9.36 (16) | 2.61 (4) |
| Census, Binomial | 45,222 / 116 | 0.13 | 7 | 0.129 (16) | 0.14 (4) |
| Housing, Gamma | 21,613 / 92 | 0.0239 | 0.0429 | 0.0235 (4) | 0.0284 (4) |
| Housing, Gaussian | 21,613 / 92 | 0.00403 | 0.0248 | 0.00393 (16) | 0.00677 (4) |
| Motor, Tweedie | 678,013 / 79 | 0.734 | 1.6 | 0.734 (32) | 0.224 (16) |
| Tweedie, ridge | 678,013 / 79 | 1.41 | 1.83 | 1.41 (32) | 0.272 (16) |
| Tweedie, elastic net | 678,013 / 79 | 1.03 | 1.93 | 1.03 (32) | 0.355 (16) |

Thread settings tested: 1, 4, 16 and 32 for the first six cases; 4, 16 and 32 for
Tweedie. Avenue leads five of the six core cases at the best tested settings;
glum leads taxi and the three Tweedie cases. More threads do not always help.

## Large portfolios and memory

Both synthetic portfolios have **20 million rows and 501 parameters**, with
independent categorical factors. All engines use 32 threads. Memory is the OS-recorded
whole-process peak through the first fit, including inputs and libraries.

| Portfolio | Avenue global: seconds / GiB | Avenue table: seconds / GiB | glum: seconds / GiB |
|---|---:|---:|---:|
| 5 tables, 101 levels | 2.97 / 2.73 | 3.11 / 2.43 | 12.6 / 3.43 |
| 100 tables, 6 levels | 70.5 / 18.85 | 36.6 / 18.53 | 763 / 19.75 |

**Historical memory correction:** the old Python background sampler stalled while
Avenue held the GIL and missed temporary allocations. Rebuilding commit `08b1b67`
and comparing it with the published wheel on the 100-table portfolio gave **18.56 GiB
for both**, while the sampler saw only **10.22 GiB**. The older roughly-twofold memory
advantage was overstated; switching solvers did not cause that discrepancy.

## Reproduce

Install the release wheel into an isolated Python 3.13 environment, then install
`glum==3.4.1`, `numpy`, `scipy`, `pandas`, `scikit-learn` and `threadpoolctl`.
The [single runner](../studies/benchmark.py) checks the installed wheel payload,
executes each engine in a fresh process, and rejects failed convergence or prediction
agreement. Run without competing CPU work. Large cases can take about an hour.

```sh
python studies/benchmark.py --download --data-dir /tmp/avenue-data
python studies/benchmark.py --data-dir /tmp/avenue-data --output /tmp/avenue-real --wheel /path/to/avenue.whl --threads 32 --cases motor motor_wide taxi census housing_gamma housing_gaussian tweedie tweedie_ridge tweedie_elastic
python studies/benchmark.py --data-dir /tmp/avenue-data --output /tmp/avenue-large --wheel /path/to/avenue.whl --threads 32 --cases large_few large_many --engines avenue avenue_table glum
```

Omit `--threads` to use the machine's logical CPU count; pass multiple counts for
an optional sweep. Each output directory must be new. Logs, predictions, package
versions, hashes, timings and agreement checks are generated there, outside the repo.
Use `--aggregate --check` with the same cases, engines and threads to recheck a run.

Input sources are OpenML 41214 (motor frequency), 41215 (severity), 1590 (census),
`house_sales`, and January 2024 NYC yellow taxi data. The runner contains all cleaning,
banding and synthetic generation. Frequency caps claim count at four and exposure
at [0.001, 1]; Tweedie uses uncapped matched loss divided by exposure, with exposure
weights, power 1.5, and alpha 0.1 for ridge/elastic net (`l1_ratio=0.5`).

Timing excludes source loading and common preprocessing; engine preparation is recorded
separately. Every prediction must meet `atol=1e-6, rtol=2e-6`, with maximum error divided
by reference RMS below `1e-7`. Penalized fits also check coefficients and objective.
The runner uses the verified engine-specific stopping tolerances; equal numerical
answers, rather than identical tolerance labels, are the comparison requirement.
Pure lasso is not included: an earlier strict-tolerance wheel check did not converge.
