# Small-batch scoring follow-up

Same machine, inputs, four-thread environment and saved four-table/19,181-row model
as the [large-table study](../../../docs/LARGE_TABLE_SCORING.md). Each process retains
21 public scoring calls, including the first call, and checks all against the saved
fork booster with `atol=rtol=1e-12`.

| Leading holdout rows | Before small-batch dispatch | Final dispatch |
|---|---:|---:|
| 1 | 0.691 ms | 0.136 ms; separate process 0.134 ms |
| 31 | Unmeasured | 0.306 ms |
| 32 | Unmeasured | 0.710 ms |
| 169,504 | Previous study 8.6–9.2 ms | 8.669 ms |

Single-quote scoring is approximately five times faster on this quote. Predictions
are byte-identical to the preceding engine; the full batch also retains its original
prediction fingerprint. The dispatch cutoff of 32 is conservative, not an optimal
crossover estimate: the table shows its discontinuity explicitly. Small-batch costs
depend on matching row positions, geometry and machine. These measurements do not
establish a randomly sampled quote-latency distribution or cold process startup cost.

The earlier engine is commit `39cdb29`; final records identify the changed matcher
by its SHA-256 before the evidence commit. No persistent cache is introduced.
Verification: 283 active Rust tests (six ignored), 143 Python tests, and one ignored
doctest. Batch-boundary tests include empty and chunked frames and compare directly
with the independent row matcher. JSON records retain inputs, script/native-extension
and matcher hashes, thread settings, dependency versions and all timings.
