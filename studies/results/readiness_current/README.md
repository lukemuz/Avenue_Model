# Current wheel acceptance

Engine/package revision: `7d6d0d8a654d4a0e5cfb90c6109bc472b2b3d496`.
Wheel: `avenue_model-0.1.0-cp312-abi3-manylinux_2_39_x86_64.whl`.
SHA-256: `b7d5a4101ea7e25aeaa4fdc4a34fc5195c408143129909c132a9796ab75e53f6`.

The release wheel was installed into two newly created CPython 3.12.14 environments
using cached, explicitly pinned dependencies. One uses stock LightGBM 4.7.0; the other
uses the locally built fork 4.6.0.99. Both runs verify all 13 installed package payload
files byte-for-byte against the wheel and reject editable/source imports.

| Gate | Stock | Fork |
|---|---|---|
| Public Python suite | 133 passed, no skips | 133 passed, no skips |
| Auto pricing tutorial | Passed | Passed |
| Homeowners/peril composition tutorial | Passed | Passed |
| Booster tuning/conversion/edit tutorial | Passed | Passed |
| Real motor study with booster challenger | Passed | Passed |
| Real continuous-spline study with inference/delivery | Passed | Passed |

The source Rust suite passed 282 tests, with six ignored tests and one ignored doctest
(`rust.log`). The runner now includes the real spline arm at the previously established
1e-11 stopping tolerance and unchanged independent-prediction comparison thresholds.
Its exact modified source hash is retained in each acceptance manifest.

The manifests record source hashes, dependency versions, input hashes, step results
and hashes of all generated artifacts. Retained compact artifacts were verified against
those hashes before copying here. Complete artifacts, including curves and workbooks,
remain in `/tmp/avenue-current-{stock,fork}-acceptance`; the wheel remains under
`/tmp/avenue-readiness-current-wheels`. Historical readiness evidence is unchanged.

This verifies the current package on local Linux x86-64 / CPython 3.12 only. It is not
evidence for other Python/platform/dependency combinations, publishing, controlled speed
benchmarks, or adequate prospective calibration. The real spline specification still
does not beat the banded baseline in held-out loss. Credibility, regularized uncertainty,
broader performance work and the remaining release matrix remain open.

Reproduce using the updated `studies/readiness_acceptance.py` command documented in
`docs/READINESS_AUDIT.md`, with an installed wheel and unused output directory.
