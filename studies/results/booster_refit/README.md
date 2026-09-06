# Booster structure refit acceptance

The installed local Linux CPython 3.12 wheel passes all 151 Python tests without
skips and all three tutorials, including the new optional booster GLM refit branch.
The source Rust suite passes 284 tests (six ignored; one ignored doctest), and
`cargo fmt -- --check` passes. No remote platform run is claimed.

Stock 4.7.0 and fork 4.6.0.99 both complete the ordinary synthetic study. The ridge
refit converges, preserves raw-quote predictions through analytical bundle reload,
and enters the common held-out comparison. Its aggregate calibration improves while
its Poisson loss worsens: this is a workflow demonstration, not a predictive win.
The penalty is prespecified, not CV-selected; no post-selection intervals are claimed.

Additional categorical runs use the same generated CSV, replacing claims by 1 for
north and 6 for south, and loss by 1500 times claims. Both builds select region in
their structure and preserve north/south identity in the refitted bundle. These
artificial signal probes establish label handling, not actuarial performance.

The branch exposed a false nonconvergence report: the global treatment-coded solver
was testing the fixed reference row as if it were a free table-sweep coordinate.
With a nonzero unsupported row, that score includes a frozen penalty contribution
that need not vanish. The solver now checks its actual free coordinates. The
regression fails in all eight cases on the preceding installed wheel, then passes
against independent SciPy Poisson/ridge score equations for two penalty strengths,
two unused starting factors, and both automatic/global solver selection. One-step
fits still correctly report nonconvergence. Table-sweep convergence is unchanged.

The compact comparison, support and bundle records are retained here. Full local
outputs are under `/tmp/avenue-refit-*`; wheel identity and source hashes are in
`wheel_acceptance.json`. This replaces a wheel in an existing test environment.
