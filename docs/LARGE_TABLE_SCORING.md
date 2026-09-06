# Large numeric table scoring

Avenue recognizes complete numeric Cartesian grids and searches each axis by binary
search. A dense coordinate index identifies the original factor row. The index is
rebuilt for each matching call, so workbook edits cannot leave stale factors or geometry
in a persistent cache. Public prediction still prepares inputs and checks unmatched and
nonfinite results.

For fewer than 32 quote rows, multi-column numeric tables use the direct matcher to
avoid index construction overhead. That scan stops at the first match using zero
categorical wildcards: no later row can improve its specificity, and ties preserve
the first row. Matches using wildcards continue searching for a more specific row.
Tests compare both sides of the batch-size cutoff with the independent row matcher,
including empty and chunked frames. The cutoff is a conservative local heuristic,
not a guarantee of optimal dispatch for every table geometry and machine.

Recognition requires unique complete coordinates, non-null thresholds, and
row order in which each numeric coordinate's immediate predecessor occurs earlier. Transitivity
then proves that the coordinate found by binary search precedes every other matching
row. Explicit missing-only `NaN` bounds use a separate coordinate on each axis,
outside the ordered numeric search. Null quote values and NaNs select that coordinate
when it exists; finite quotes cannot select it. No predecessor relation is imposed
between missing and numeric coordinates. A null *table threshold* still means an
unconstrained bound and prevents this index from being used.

This supports valid orders beyond a particular lexicographic layout. Incomplete,
duplicate, incorrectly ordered and mixed categorical/numeric tables retain the
general matcher. This optimization does not approximate a table or change its factors.

The independent row matcher checks a three-dimensional grid against all 2,197
combinations of boundary, tail, signed-zero, infinite, null and NaN probes, plus invalid
grid variants. Two additional three-dimensional grids check explicit missing routes,
including a missing-only axis, 2,662 probe combinations, NaN payloads and chunked data.
Conversion tests exercise tiny and indexed-size quote batches through both
consolidation modes and workbook reloads.

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
these observations apply to the recorded workload and builds.
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
fresh installed wheel.

## Reproduce a scoring measurement

The runner requires a policy parquet (OpenML 41214) and a saved numerical LightGBM
booster with features `age`, `vehicle_age`, `bonus`, `region`, `fuel`. It reconstructs
the historical seeded 75% training split and sorted training-only category maps.
Only use a booster trained with those exact feature conventions.

```sh
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
POLARS_MAX_THREADS=4 RAYON_NUM_THREADS=4 python studies/large_table_scoring.py \
  --frequency /path/freMTPL2freq.parquet --booster /path/selected_booster.txt \
  --output /tmp/avenue-scoring
```

By default it converts the supplied booster and saves the resulting `model.json`.
Pass `--workbook /path/model.json` to score that exact artifact on another build.
This keeps factors and row order fixed. File loading and reference predictions occur
outside timed calls; every timed prediction must pass booster parity. Inputs, native
extension, matcher and script hashes are retained with every timing sample.

The output directory must be new. NumPy, pandas, Polars and LightGBM are required.
The RSS measurement uses Python's Unix `resource` module and covers the whole process.
The script does not fetch historical models or data. The archived original evaluation
contains those inputs for exact replay of the records below; other boosters define
new workloads and cannot reproduce the old table counts or timings.

## Current conversion with explicit missing routes

Converting the same saved booster with the missing-route implementation produces
four tables and 21,723 rows, including explicit missing routes, rather than the
historical workbook's 19,181 rows. Before missing-coordinate indexing, the current
conversion scored the same 169,504 finite quotes in 0.7566 seconds (median of five).
With indexing it takes about 9 ms, with byte-identical predictions and maximum
relative booster error `4.996e-15`. This extends the speed improvement to the current
conversion path; it is not a claim that the two workbook snapshots are identical.

The baseline saves its exact `model.json`. Subsequent runs use
`--workbook /path/to/baseline/model.json`, preserving row order as well as table geometry
and factors. This matters because fresh conversions can enumerate equivalent grid
coordinates in different valid orders. The [retained follow-up](../studies/results/missing_grid_scoring/)
checks exact workbook equality excluding only the creation timestamp, prediction
bytes, input fingerprints and booster parity. Missing-valued quotes are covered by
the independent tests, not by this finite-quote timing workload. No persistent cache,
irregular-grid or categorical-grid speedup is claimed.

For small-batch measurements, add `--quote-rows 1 --repeats 21` (or another positive
batch size). This scores the leading holdout rows with the same training-only category
maps. It measures repeated public calls on a loaded model, not cold process startup
or a distribution of randomly sampled individual quotes. The separate
[small-batch results](../studies/results/small_batch_scoring/) retain these checks.
