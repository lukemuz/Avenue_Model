# Fundamentals scope review

The maintenance criterion is a strong modeling/scoring primitive that is difficult
to reproduce correctly outside Avenue. A workflow being useful does not by itself
justify adding a library method for it.

## Removed

Nine Python modules were removed, rather than hidden behind specialist imports:
preparation, splitting, selection, comparison, changes, composition, bundle,
credibility and stability. This removes generic study/result wrappers, named model
graphs, analytical bundle schemas, bootstrap orchestration and the narrow conditional
Poisson–Gamma helper. Associated feature tests and obsolete guides were removed.
Historical benchmark and acceptance outputs remain as records of their tested builds.

Final-conversion resource-policy arguments and tuning scoring-data/timing machinery
were removed. Actual table sizes and conversion parity remain available for callers
to inspect and apply their own limits.

## Retained

Correctness and speed fixes; native model fitting, penalties and fixed priors;
hierarchical interactions; monotonic bands; continuous spline scoring; diagnostics
and explanations; model-based/HC0/CR0 inference and whole-term tests; safe workbook
round trips; verified booster conversion; explicit pandas conversion; and the
existing LightGBM tuning interface with structural counts.

The native `Plan`, `FittedModel` and `Workbook` remain the model API. Examples now
prepare data, split observations, compare predictions, combine perils and calculate
rate changes with ordinary Polars, NumPy and scikit-learn operations. They do not
introduce replacement frameworks elsewhere in the repository.

Independent numerical, inference, missing-route and export regression checks are
retained. Real-data studies still compare predictions with glum and probe converted
boosters at split boundaries. Benchmark evidence is not a reason to add more APIs;
Avenue's existing favorable results remain relevant on their measured workloads.

## Verification of the reduced package

The root export list falls from 50 to 26 entries; all 19 pre-expansion exports remain.
Python implementation falls from 14 modules / 2,517 lines to 5 modules / 827 lines.
The native Rust implementation is unchanged by this scope reduction.

The installed Linux CPython 3.12 wheel passes 109 Python tests with no skips and
the auto, homeowners and booster/refit acceptance examples. The stock and fork
booster examples, continuous-effect example, and real motor banded/spline studies
also pass locally. The real study retains glum prediction checks and booster
threshold/missing-route parity. Generated API documentation verifies 193 anchors.
Removed modules are absent from the installed wheel, not merely unexported.

These runs validate this local build; they do not claim results for unexecuted
platforms or replace the separate, controlled performance benchmarks.
