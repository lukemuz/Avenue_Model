# Large numeric table scoring

Avenue recognizes complete numeric Cartesian grids and searches each axis by binary
search. A dense coordinate index identifies the original factor row. The index is
rebuilt for each matching call, so workbook edits cannot leave stale factors or geometry
in a persistent cache. Public prediction still prepares inputs and checks unmatched and
nonfinite results.

Recognition requires unique complete coordinates, non-null/non-NaN thresholds, and
row order in which each coordinate's immediate predecessor occurs earlier. Transitivity
then proves that the coordinate found by binary search precedes every other matching
row. This supports valid orders beyond a particular lexicographic layout. Incomplete,
duplicate, reordered, missing-route and mixed categorical/numeric tables retain the
general matcher. This optimization does not approximate a table or change its factors.

The independent row matcher checks a three-dimensional grid against all 2,197
combinations of boundary, tail, signed-zero, infinite, null and NaN probes, plus invalid
grid variants. Existing conversion, scoring, fitting and workbook tests exercise the
shared matcher. Verification passed 283 active Rust tests (six ignored) and 143 Python
tests; the doctest remains ignored.

## Reproduced historical workload

The saved fork booster and editable workbook from the original evaluation contain four
tables, 19,181 total rows and an 18,810-row largest table. The study reconstructs exactly
the original 508,509 training / 169,504 quote split and training-only numeric category
maps. It scores the saved workbook without refitting or reconverting it. Historical
exposure/claim caps do not enter this quote-only scoring benchmark.

On the same AMD Ryzen 9 9950X Linux x86-64 machine, with four-thread limits:

| Engine checkpoint | Median public scoring time, five calls |
|---|---:|
| Current engine before grid index (`fbc192d`) | 1.1753 seconds |
| Grid index, first process | 0.0086 seconds |
| Grid index, second process | 0.0090 seconds |
| Grid index, final source verification | 0.0092 seconds |

This is approximately 128–136 times faster than the immediately preceding engine on
this specific large-grid workload. The original historical median was 3.0966 seconds;
the plan's fivefold engineering target is met locally on its original workload.
All four runs produce byte-identical predictions, with maximum relative error
`4.996e-15` against the saved booster. Every timed call must pass `atol=rtol=1e-12`
before the runner records a successful result.

These are complete repeated batch-scoring times, including preparation, index construction
and result checks, excluding file loading and reference prediction. First calls are
retained. They are not single-quote latency or prepared-cache measurements. Process peak
RSS includes data, reference booster and workbook; it is not isolated scoring memory.
Small workloads, irregular tables and other hardware need their own measurements.

The [retained results](../studies/results/large_table_scoring/) record every timing,
input and prediction fingerprint, source/matcher/script/native-extension hashes, thread
settings and dependency versions. They use an editable local release extension, not a
fresh installed wheel. Run from the repository with the historical evaluation artifacts:

```sh
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
POLARS_MAX_THREADS=4 RAYON_NUM_THREADS=4 python studies/large_table_scoring.py \
  --evaluation evaluation --output /tmp/avenue-large-table-scoring
```

The output directory must be new. NumPy, pandas, Polars and LightGBM are required for
this benchmark. The recorded run uses the fork environment; it loads the saved model
without training. A reusable prepared-scoring API and full stage/memory profiling remain
separate work.
