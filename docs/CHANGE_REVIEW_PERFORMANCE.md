# Change-review memory and timing, September 5, 2026

`compare_changes` now combines long factor exhibits with a columnar Polars full join
instead of converting them into millions of Python dictionaries and tuple keys.
The public result still contains policy changes, matched old/new table rows,
added/removed factors, portfolio totals, segment totals and stable row ordering.
Identical negative-infinity offsets for zero exposure still have zero coefficient
change, rather than NaN.

The real acceptance study had a whole-process peak near 3 GiB. A separate measurement
identified change analysis as a large contributor: 169,504 policies × six tables
produced 1,017,024 factor-contribution rows. Inputs were raw predictors from the real
motor data and the study's saved Tweedie workbook before/after a 5% intercept edit.

## Controlled before/after measurement

Each implementation ran in a fresh process with the same artifacts, rows, Polars 1.31.0
and four-thread environment. There were three runs per implementation; repetitions
were sequential, alternating baseline and new code. The first baseline/new pair
preceded the alternating repetitions. The baseline function was loaded from commit
`b158b2c`; the rest of the environment and engine were shared.

| Median measurement | Before | After |
|---|---:|---:|
| Change-review time | 3.944 s | 0.516 s |
| Process peak RSS after review | 2.44 GiB | 0.99 GiB |
| Increase in process high-water RSS during review | 2.24 GiB | 0.78 GiB |

This is **7.65× faster** with a **65.0% smaller high-water RSS increase** on this workload.
RSS is the Linux process high-water mark, not an allocator-isolated measurement of
Avenue memory. The result includes a million-row exhibit, so its remaining allocation
is not negligible. These numbers do not establish performance on every table shape,
and are not measurements of the plan's separate large-booster scoring target.

Every output table from every repetition—policies, all 1,017,024 contributions,
portfolio totals, region totals and fuel totals—was compared against the saved baseline
with exact values, order and dtype checks. Metadata matched too. Additional regression
tests cover added/removed terms, absent table-row indices, zero-exposure negative-infinity
offsets and undefined relative changes for zero original means. All 104 Python tests pass.
No Rust code changed in this increment.

[Recorded measurements and source hashes](../studies/results/change_review/summary.json)
include all six individual runs. Full Arrow exhibits used for exact comparison remain
in `/tmp/avenue-change-{before,after}` and the `-2`/`-3` repetition directories.

## Whole-study follow-up

The complete real GLM/glum acceptance study passed again, including independent means,
raw-quote bundle reload and the 5% edit. This run took 9.88 seconds with a 1.67 GiB
whole-process peak, compared with the earlier 14.11-second/2.91-GiB observation.
These are separate whole-study observations, not the controlled three-run comparison
above. Loss comparisons and edit totals agreed at `atol=rtol=1e-12`.
[Run record](../studies/results/change_review/full_study_run.json) retains the versions,
input hashes and interpretation limits. Full output is `/tmp/avenue-real-memory-acceptance`.

## Reproduction

First reproduce the real motor study to obtain its original JSON source workbook and
edited CSV workbook. Then save the old function from Git and profile each implementation:

```sh
git show b158b2c:python/avenue_model/changes.py > /tmp/changes-baseline.py
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 POLARS_MAX_THREADS=4 RAYON_NUM_THREADS=4 \
python studies/profile_change_review.py --data frequency.parquet \
  --old /tmp/real-motor/premium_bundle/source/workbook.json \
  --new /tmp/real-motor/edited_premium --output /tmp/change-before \
  --implementation /tmp/changes-baseline.py
```

Repeat without `--implementation` and with a new output path to measure current code.
The optional implementation argument executes the supplied local `changes.py` as the
baseline; use the exact saved Git source for this comparison. Run processes sequentially
and compare every saved `.arrow` file with `polars.testing.assert_frame_equal(...,
check_exact=True)` before interpreting a speed result.
