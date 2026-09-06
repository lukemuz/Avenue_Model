# Missing-route grid performance

The current conversion of the original saved fork booster contains four tables and
21,723 rows, including explicit missing routes. On 169,504 finite quotes its median
public scoring time falls from 0.7566 seconds to 0.0092 and 0.0089 seconds in separate
processes (five calls each), approximately 82–85 times faster locally.

The baseline uses the runner's `--reconvert` mode and saves the exact resulting
workbook. After/repeat use `--model /tmp/avenue-missing-grid-before/model.json` to
preserve that artifact, including row order. Identity checks compare all workbook
contents except the creation timestamp and require identical prediction bytes.
All timed predictions pass original-booster parity at atol=rtol=1e-12.

See [method, limitations and reproduction](../../../docs/LARGE_TABLE_SCORING.md).
Before and after records retain source/extension hashes and all timings. The runner
was extended with --model after the baseline; the final script hash is in after/repeat.
This is editable local release-extension evidence, not fresh-wheel acceptance.
All 284 active Rust and 147 Python tests pass; six Rust tests and one doctest remain
ignored. Independent missing-coordinate tests cover actual null/NaN quote inputs;
the real-data timing population contains finite quotes.

The flag names above describe the recorded script. The current runner converts by
default and uses `--workbook` for a saved artifact; see the linked reproduction guide.
