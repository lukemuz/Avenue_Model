# Initial real spline profile

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
