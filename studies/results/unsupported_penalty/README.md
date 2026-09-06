# Unsupported penalized row scoring

The final installed Linux CPython 3.12 wheel passes 153 Python tests without skips
and all three tutorials, including the optional booster GLM refit. Source Rust tests
pass 284 cases (six ignored and one ignored doctest); formatting also passes. The
fork booster refit tutorial separately passes in the editable fork environment.
This is local evidence, not remote platform certification.

The regression extends the previous free-coordinate convergence fixture to score
multiple unused numeric/missing routes, including null quotes, and to inspect their
no-data status and workbook reload. Independent Poisson/ridge score equations verify
supported predictions. Tests also cover correlated table pairs, elastic-net/lasso,
and rows present with zero weights across all five supported families. Automatic,
global and table solvers agree within the stated numerical tolerances.

An initial extended regression failed against the preceding installed wheel because
unsupported predictions retained arbitrary starting factors or depended on the
solver. The final rule sets each unlocked, unsupported penalized step contrast to
zero, its penalty-only optimum. It remains marked no_data. This changes predictions
on unused routes after a new penalized fit, not existing saved scoring workbooks.
The shared scoring contract describes the migration and scope.

The before log records the initial extended Poisson regression; the after log and
wheel manifest cover the final expanded three-method regression. This acceptance
replaces a wheel in the existing local release-test environment. Compact stock/fork
comparison records show that the normal tutorial's supported holdout performance
is unchanged; the new behavior concerns unsupported scoring routes.
