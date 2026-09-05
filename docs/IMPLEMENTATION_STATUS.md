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

## Next required work

Complete scoring semantics (explicit units/conversions, schema, strict/permissive
unmatched diagnostics, consistent exposure checks); minimize and fix categorical and
threshold LightGBM conversion failures across consolidation modes and workbook reload.
Then complete the ordinary study workflow, reproducible splits/comparison, named
review/explanations, composition and analytical preservation, and selected modeling
extensions from the plan. Run the plan's fresh-user acceptance exercise before making
a first-choice-tool claim. Historical evaluation verifiers still check historical
failure artifacts; new regression tests establish current behavior without overwriting
that evidence.
