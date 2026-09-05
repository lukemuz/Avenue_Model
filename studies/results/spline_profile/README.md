# Initial real spline profile

## Follow-up validation

The optimized deviance passed a fresh independent SciPy/glum acceptance run with
inference and delivery checks (`fast_power_acceptance/`). The next change specializes
the shared spline accumulation routine for score-only loadings, avoiding the unused
information matrix and observation-length zero-curvature allocation during normalization
and convergence. It preserves the accumulation order. Independent kernel fixtures cover
interior coordinates, exact knots and both tails; their score-only results also equal
the full normal-equation scores exactly.

The fresh no-inference profile (`loadings.json`) took 10.716 seconds and 97 sweeps,
with byte-identical predictions to the preceding fast-power profile. This small
single-run difference does not establish an additional speedup. The new implementation
passed 279 active Rust tests and 126 Python tests; five kernel tests were rerun after
adding invalid-input checks.

Full inference and delivery acceptance also passed on this implementation
(`loadings_acceptance/`, source patch `loadings.patch`). Fits took 3.718 seconds for
frequency, 0.097 for severity and 10.758 for Tweedie. Maximum relative prediction
differences from glum were 6.60e-9, 1.97e-9 and 1.00e-9 respectively, passing the
original tolerance. These records supersede the initial lack of independent acceptance
below, without changing the historical observations. The full artifact hashes are in
`loadings_acceptance/provenance.json`.

## Original measurements

The saved real motor split and specification from `docs/REAL_SPLINE_ACCEPTANCE.md`
are reproduced by `studies/profile_real_spline.py`. All runs use 508,509 training
rows, 169,504 holdout rows, tolerance 1e-11 and four threads per library. Inference
is disabled in these three fresh-process measurements.

The original release took 16.243 seconds and 103 sweeps. Temporary stage timing
took 16.419 seconds with byte-identical predictions. Summed stage times were
2.245 seconds for previous deviance and 5.514 for line search, versus 1.706 for
normal equations and 1.727 for loadings. The instrumentation patch and log are
retained here; instrumentation was removed before the optimized release build.

Reusing the existing special-power helper in Tweedie deviance replaces general
powers with square roots at power 1.5. The first optimized run took 10.893 seconds
and 97 sweeps. Maximum relative prediction change versus the original Avenue run
was 3.841e-8; all holdout predictions pass the unchanged 2e-6 relative / 1e-6
absolute tolerance. The stopping tolerance is unchanged. Floating-point differences
affect line search and iteration count, so the observed reduction includes both
cheaper sweeps and fewer sweeps.

These are initial single-run measurements, not a repeated benchmark or refreshed
independent glum acceptance. The Python suite briefly overlapped the optimized
process (0.573 seconds). All 279 active Rust and 126 Python tests passed, including
independent spline references and a closed-form deviance check across numerical
scales. Historical full-inference acceptance evidence remains separate.

JSON records contain extension hashes and parent source commit. `optimization.patch`
identifies the source changes present in the optimized build; `instrumentation.patch`
identifies the timed build. Prediction arrays and complete models remain in the
corresponding `/tmp/avenue-spline-profile-{full,timed,fast-power}` directories.

Reproduce from the repository root, substituting an unused output directory:

```sh
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 POLARS_MAX_THREADS=4 RAYON_NUM_THREADS=4 \
  /tmp/avenue-eval-venv/bin/python studies/profile_real_spline.py \
  --frequency evaluation/data/freMTPL2freq.parquet \
  --severity /tmp/avenue-freMTPL2sev.parquet \
  --split /tmp/avenue-real-spline-tight/split.json \
  --output /tmp/avenue-spline-profile-reproduction --no-inference
```
