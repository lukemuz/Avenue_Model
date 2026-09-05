# Improvement plan implementation record

Branch: `improvement/pricing-workflow`.

The full scope and acceptance criteria remain in [IMPROVEMENT_PLAN.md](IMPROVEMENT_PLAN.md).
This record is a progress log, not a declaration that the plan is complete.
Existing evaluation reports, scripts and recorded runs have been preserved.

## Completed increment: scoring exposure and intercept-only fitting

- Prediction prepares rating predictors independently of training weights.
- Offset prediction applies log exposure before the inverse link, agreeing with validation.
- Zero scoring exposure is supported; negative/null/nonfinite scoring exposure raises.
- Intercept-only plans build and fit through the ordinary GLM engine.
- Fitted/workbook-derived models are covered by `tests/test_scoring_contract.py`.
- Migration and target definitions: [SCORING_CONTRACT.md](SCORING_CONTRACT.md).

Verification on September 5, 2026: 34 Python tests passed; 257 Rust tests passed,
6 ignored (plus one ignored doc test). Python extension built in release mode with
`maturin develop --release --uv`. Temporary toolchain/environment locations are
`/tmp/avenue-cargo`, `/tmp/avenue-rustup`, `/tmp/avenue-eval-venv`, and `/tmp/avenue-tools`.
Build/test output is in `/tmp/avenue-improvement-{build,python,rust}.log`.

## Completed increment: strict scoring and row diagnostics

- `predict()` rejects unmatched rows and nonfinite factors/response means with row context.
- `predict_diagnostics()` preserves row order, returns null failed means, and names
  unmatched tables. Schema and invalid exposure failures still raise.
- Scoring uses the fitting/validation batch matcher, including explicit wildcards.
- Python coverage checks fitted, workbook-derived and composed models. Rust coverage
  checks wildcard routing and nonfinite/overflow behavior.
- Verification: 35 Python tests and 258 Rust tests passed; 6 Rust tests and one doc
  test remain ignored. Release extension rebuilt successfully.

## Completed increment: finite-input conversion parity

Minimized the captured categorical failure and adjacent-threshold failures before
fixing their causes:

- Matchers now prefer fewer categorical wildcards, including partial wildcard rows.
- JSON float parsing retains round-trip precision; path construction deduplicates
  only exactly equal thresholds.
- Analysis-mode node differences include every tree's root contribution.
- Consolidation of two intercepts retains one row.
- Workbook duplicate-row checks use exact floating-point identity instead of rounded
  display text (and separate field keys instead of a delimiter-joined string).

Verification: all 40 Python tests and 258 Rust tests pass. The five conversion tests
also pass with fork LightGBM 4.6.0.99; stock tests use LightGBM 4.7.0. Both modes and
CSV reload are checked against actual boosters, a captured tree, adjacent float
boundaries and eight deterministic randomized trees. Six Rust tests and a doc test
remain ignored.

Additional read-only checks of original captured fork boosters passed on finite
boundary probes: 2,396 numeric rows (max absolute errors 3.89e-16 analysis / 3.33e-16
max), and 1,901 categorical rows (1.06e-15 analysis / 5.00e-16 max), using absolute
plus relative tolerances of 1e-12. Historical evaluation outputs were not overwritten.
The temporary reproduction is `/tmp/avenue-check-captured.py`; output is
`/tmp/avenue-captured-parity.log`.

These findings do not establish missing/default-route correctness or support for
all booster objectives. Those remain correctness gates before conversion can be
called dependable for arbitrary quote inputs.

## Next required work

Complete scoring semantics (explicit units/conversions, schema, consistent exposure checks); complete LightGBM missing/default routing and unsupported-semantics checks, and add
a Booster entry point with conversion metadata and optional parity evidence.
Then complete the ordinary study workflow, reproducible splits/comparison, named
review/explanations, composition and analytical preservation, and selected modeling
extensions from the plan. Run the plan's fresh-user acceptance exercise before making
a first-choice-tool claim. Historical evaluation verifiers still check historical
failure artifacts; new regression tests establish current behavior without overwriting
that evidence.
