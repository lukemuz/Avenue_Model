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

## Completed increment: conversion entry point and semantic guards

- Python `from_booster()` returns a model, build/objective metadata, source dump
  fingerprint and an optional parity report. No-data conversion states not verified.
- Reports retain failed rows and error/unmatched/nonfinite summaries; saving writes
  an editable workbook plus separate evidence JSON without training data.
- Core JSON conversion rejects unsupported objectives, multiclass, averaged ensembles,
  linear leaves, split operators and non-unit binary sigmoid. Binary objective options
  no longer accidentally select an identity link.
- Added failure-report, metadata persistence and binary-link regression checks.
- Verification: 42 Python tests pass, including the entry point and workbook evidence;
  the seven conversion tests also run against the installed fork. Usage and remaining
  missing/default limitations are documented in [CONVERSION.md](CONVERSION.md).

## Completed increment: explicit Poisson prediction conveniences and schema

- `predict_rate` and `predict_count` use the recorded Poisson convention and reject
  ambiguous/severity/composed means. Offset rates can score without exposure.
- `prediction_kind` exposes the recorded Poisson convention; it is not a general
  physical-unit system. `input_schema` separates prediction and validation inputs
  and exposes category mappings from the current tables/artifact.
- Invalid exposure values now fail consistently in plan preparation, fitted/loaded
  validation and count scoring. Zero weight versus positive offset training exposure
  is documented separately from zero scoring exposure.
- Regression coverage exercises both Poisson conventions and workbook reload,
  schema mappings, ambiguous conversion rejection and invalid exposure in fitting
  and validation. Verification: 44 Python tests and 258 Rust tests passed.

## Completed increment: reproducible split specifications

- Added `SplitSpec.random`, `grouped` and `out_of_time`, retained row membership,
  seeds/specifications, content/order/schema fingerprint and stable split IDs.
- `Fold.fit` resolves an ordinary Plan on training rows only; `FoldFit.validate`
  uses exactly the retained holdout. Externally prepared datasets remain supported.
- Versioned membership serialization rejects overlaps, invalid indices, future
  schema versions and reuse against changed/reordered data.
- Six tests cover complete K-fold coverage, reproducibility, group isolation, time
  boundaries, dates, membership round trips and holdout-only category/extreme-value
  leakage. Verification: all 50 Python tests passed. No Rust changes in this increment.
- Usage and limitations: [VALIDATION_SPLITS.md](VALIDATION_SPLITS.md).

## Completed increment: common candidate comparison

- Added `Candidate`, `compare_models` and `Comparison` for Avenue models and external
  prediction vectors on one explicitly declared response unit/population/loss.
- Named summary, row predictions and segment/period A/E exhibits reconcile weighted
  portfolio totals. Failed scoring is retained without reducing the population.
- Recommendations require known convergence; recorded nonconvergence cannot be
  overridden by a lower holdout loss or caller-supplied flag.
- Paired row/cluster bootstrap percentile intervals report fixed-prediction loss
  differences against a named baseline, with seed and effective replicate counts.
- Independent scikit-learn reference tests cover weighted Poisson/Gamma/Tweedie losses;
  tests also verify portfolio reconciliation, failure retention, paired uncertainty
  and shared review of Avenue/external means. All 56 Python tests passed.
- Declared the independent test extra and installed it in Python CI. Usage and
  statistical limits: [MODEL_COMPARISON.md](MODEL_COMPARISON.md).

## Completed increment: preparation, experience and executable auto study

- `prepare_pricing` supplies frequency, positive-loss severity and pure-premium
  populations with preserved source rows, explicit invalid-row exclusion policy,
  row-level reasons/flags and reconciled population totals. It never adjusts loss
  or exposure. Zero-payment severity limitations are documented explicitly.
- Common factor experience includes exposure, claims, loss and derived rates.
  `rating_tables_by_name` provides named estimated-factor access.
- `examples/auto_pricing_study.py` generates/loads a synthetic CSV, prepares/audits,
  resolves an out-of-time split, fits/checks/reviews all three model types, compares
  loss-cost means and exports/reloads workbooks to score raw quote predictors.
- The example completed with 3,000 rate records, 371 positive-loss severity records
  and a common 1,000-row holdout. The paired loss-difference interval includes zero;
  this synthetic run is workflow evidence, not a predictive-superiority claim.
- CI now runs the complete example. All 60 Python and 258 Rust tests pass. Installed
  rustfmt and formatted the accumulated Rust changes; `cargo fmt -- --check` passes.
- Documentation: [PRICING_PREPARATION.md](PRICING_PREPARATION.md). Local example outputs
  are in `/tmp/avenue-auto-study`; log is `/tmp/avenue-auto-study.log`.

## Completed increment: quote explanations and model-change exhibits

- `FittedModel.explain` returns named summary/contribution tables with matched table
  rows, link-scale effects, log-link multipliers, exposure application and final means.
- `compare_changes` reconciles old/new means at policy, segment and portfolio levels,
  retains matched factor changes and explicitly identifies added/removed terms.
- Tests reconstruct predictions for log/identity links, fitted/loaded models and zero
  exposure; unmatched rows remain strict. A known 10% manual workbook edit is correctly
  attributed to its factor and reconciles to portfolio/segment changes.
- The auto study now demonstrates a 5% manual factor edit, quote explanations, largest
  policy movements and separate validation of the edited artifact. Its observed
  portfolio change is 5.000000000000062%, within floating-point precision of 5%.
- All 64 Python tests, 258 Rust tests, formatting and the extended auto study pass.
  Usage and attribution limits: [EXPLANATIONS.md](EXPLANATIONS.md).

## Completed increment: pandas boundary and constant boosters

- Optional `from_pandas` preserves categorical labels instead of positional codes,
  nullable numerical/boolean values, and an explicitly requested source index.
  Ambiguous mixed/object/float-category inputs fail with actionable encoding guidance.
- Tests demonstrate pandas/Polars prediction and label agreement, unused-level
  isolation, missing/unseen diagnostics and nonmutation of category order.
- Constant-only boosters convert to an intercept artifact; later constant tree
  contributions also consolidate correctly and survive CSV reload in both modes.
- All 68 Python tests, eight stock/fork conversion tests, 258 Rust tests and
  formatting pass. Constant-booster parity includes freshly trained stock and fork
  boosters, not only handcrafted JSON.
- Declared pandas/test extras. Adapter usage: [PANDAS.md](PANDAS.md).

## Completed increment: explicit missing/default routing

- Numeric null/NaN values no longer silently select the first ordinary GLM band.
  Tables without a missing-only row report them as unmatched.
- Converted numeric tables carry explicit NaN bounds for missing-only rows, preserving
  LightGBM NaN defaults and None-type missing-to-zero decisions through path parsing,
  consolidation and CSV/JSON reload. Categorical integer nulls use complement routes.
- `zero_as_missing` is explicitly rejected pending its separate near-zero semantics.
- Workbook format 2 identifies the new matching contract and blocks older readers;
  version-1 normal artifacts still load, and future versions fail explicitly.
- Tests cover both default directions, repeated splits, null/NaN inputs, numerical
  and categorical routes, fresh stock/fork boosters, and workbook version migration.
- Verification: 72 Python tests, 10 fork conversion tests, 258 Rust tests and formatting
  pass. Original captured fork boosters also pass 2,401 numeric and 1,904 categorical
  boundary/missing probes in both modes at atol=rtol=1e-12 (max absolute error 1.06e-15).
  Read-only reproduction: `/tmp/avenue-check-captured-missing.py`; output is in
  `/tmp/avenue-captured-missing-parity.log`. Historical evaluation evidence is intact.

## Completed increment: named composition and default preservation

- Fixed a reproduced legacy composition defect: wildcard category sentinels were
  renumbered as ordinary levels when merging named category encodings. Defaults now
  survive composition and workbook reload even with different component code maps.
- Added `frequency_severity`, `sum_loss_costs` and `ComposedModel` with explicit unit,
  retained names/components, response-scale operations and no inherited likelihood.
  Validation requires a common explicit metric. Count frequencies are converted to
  rates without requiring exposure on quote data.
- Nested composition manifests plus independently editable workbooks preserve scoring
  lineage/encodings. They do not claim to preserve full analytical fit evidence.
- The auto study now uses and saves the named frequency-severity model. All 75 Python
  tests and the extended study pass; 258 Rust tests and formatting pass.
- The exact-cell composition test also exposes nonconvergence on perfectly fitted
  synthetic models. Composition preserves the failure status. Investigate the solver's
  stopping rule on these fixtures; do not treat successful scoring as convergence.
- Usage and limits: [COMPOSITION.md](COMPOSITION.md).

## Completed increment: exact-fit convergence

- Diagnosed the exact-cell failure as residual-only score normalization: numerator
  and denominator both became rounding noise, leaving a nonzero ratio indefinitely.
- The denominator now sums per-row reference score magnitudes, taking the larger of
  residual magnitude and mean-score magnitude. The free-parameter/KKT score remains
  the stopping criterion; deviance-only convergence was not introduced.
- Exact weighted cells converge across Gaussian/Poisson/Gamma/Tweedie and response
  scales 1e-6, 1 and 1e6. Noisy weighted cells match independent closed-form cell
  means; exact offset counts reconcile; a one-iteration unfinished fit still fails.
- Updated the existing Rust regression that deliberately asserted the historical
  false nonconvergence and its downstream false validation alarm.
- All 79 Python tests, 258 Rust tests, formatting and the auto study pass. Existing
  independent family/offset references and difficult convergence tests remain green.
  Documentation now states the actual normalization and no longer equates its
  tolerance numerically with glum's gradient_tol.

## Completed increment: homeowners workflow and support matrix

- Added an executable synthetic attritional water/theft study with CSV loading,
  explicit development/trend adjustment totals, grouped home/renewal splits,
  separately fitted peril models, factor reviews and common loss-cost comparison.
- Named peril composition saves/reloads raw-quote scoring and reconciles component
  predictions. A known 5% water factor edit changes the total by precisely 5% of the
  water contribution; the edited composition receives fresh validation.
- The study completed on 4,500 policy-year records with a 1,500-row grouped holdout.
  Holdout A/E is about 0.878 and the paired loss-difference interval includes zero.
  These findings are retained; no predictive-superiority or production-calibration
  claim is made for the synthetic exercise.
- CI now exercises both complete public studies. Added [WORKFLOW_SUPPORT.md](WORKFLOW_SUPPORT.md)
  and linked it from README, including explicit remaining statistical/delivery limits.
- All 79 Python tests pass. Example outputs: `/tmp/avenue-homeowners-study`; log:
  `/tmp/avenue-homeowners-study.log`. No Rust changes in this increment.

## Completed increment: identifiable two-way hierarchical interactions

- Compatible categorical/banded two-way interactions with both main effects now
  fix either reference margin at zero, preserving estimable interior contrasts.
  References follow main-effect choices independently of term order; matched numeric
  break specifications resolve from the training population.
- Constraints are stored as fixed table rows, visible in resolved/review output and
  portable through workbook reload. Interaction-only representation is unchanged.
- Independent treatment-coded Poisson tests verify means, interaction contrasts,
  parameter counts and covariance/standard errors; additional tests cover banded
  axes, most-exposed references and reordered terms.
- Fixed-row inference was previously unaware of row locks. It now constructs the
  actual free design; the existing public unconstrained inference entry point is
  preserved as a wrapper. Pairwise full-table redundancy checks no longer mislabel
  constrained tables as unconstrained duplicate main effects.
- All 82 Python tests, 258 Rust tests and formatting pass. Supported scope and global
  solver/higher-order limitations: [HIERARCHICAL_INTERACTIONS.md](HIERARCHICAL_INTERACTIONS.md).

## Next required work

Complete general response-unit semantics and complete preprocessing persistence and evaluate the remaining explicit conversion limitations.
Then complete the ordinary study workflow, reproducible splits/comparison, named
review/explanations, composition and analytical preservation, and selected modeling
extensions from the plan. Run the plan's fresh-user acceptance exercise before making
a first-choice-tool claim. Historical evaluation verifiers still check historical
failure artifacts; new regression tests establish current behavior without overwriting
that evidence.

## Completed increment: individual-model analytical bundles

- `save_bundle`/`load_bundle` retain a source workbook, source plan, schema/category
  mappings, resolved terms, fitting summary, coefficient/inference tables, findings,
  and report alongside an editable exact-factor CSV workbook.
- Optional validation is run at export and retains aggregate metrics, calibration
  and factor exhibits. Optional fold records preserve membership and split ID.
- Caller context explicitly distinguishes supplied fit options, dataset identifiers,
  preprocessing, units and lineage from automatically captured source evidence.
- Source file hashes reject accidental evidence edits; scoring file changes are
  conservatively flagged, including formatting-only edits. Loaded scorers do not
  inherit source fitting diagnostics. Unknown bundle versions fail explicitly.
- Tests cover original/edited predictions, evidence isolation, source integrity,
  version rejection, no overwrite, non-JSON metadata rejection, fold persistence,
  and refitting from the retained plan. All 85 Python tests pass. No Rust changes
  were required; the previous 258-test Rust verification remains applicable.
- [Format and scope](ANALYTICAL_BUNDLES.md): composition-graph evidence, automatic
  fitting-option capture and executable preprocessing remain open.

## Completed increment: explicit GLM selection

- `GLMTrial` and `select_glm` evaluate candidate plans and fitting-option grids on
  identical reproducible fold populations, resolving all learned terms locally.
- History retains convergence, fit findings, gradient, iterations, parameter count,
  validation support/loss/A-E and failures. Eligibility requires every fold to
  converge and score; pooled loss is weighted by validation support.
- Tweedie fitting powers live on trial plans and share one explicit evaluation
  power. Ambiguous power overrides in trial options are rejected.
- Selection artifacts retain declarative trial/fold records. The selected refit
  checks the original dataset fingerprint and refuses nonconverged results.
- Tests agree with scikit-learn elastic-net losses and the selected penalty;
  independent Tweedie calculations verify common evaluation power and unequal
  support aggregation. Failure/nonconvergence, unseen holdout categories, overlap
  rejection and refit/save behavior are covered. All 89 Python tests pass.
- The auto study now selects four penalty/power candidates only within training
  years, preserves a final-year holdout, compares the selected model, and exports
  its analytical bundle. Executed successfully at `/tmp/avenue-auto-selection-study`.
  All four trials converged in all three folds. Final selected loss was 130.966056
  versus 130.976567 for the fixed Tweedie and 130.970579 for frequency × severity;
  paired intervals include zero, so this synthetic evidence does not establish
  superiority. Historical evaluation outputs remain untouched.
- [Semantics and limitations](GLM_SELECTION.md): no adaptive search, warm-start reuse,
  repeated overlapping CV, or post-selection confidence intervals in this increment.

## Completed increment: tuning complexity and executable booster study

- Tuning now measures the same boosting prefix used for the selected CV loss, retains
  per-fold counts, calls its aggregate a mean, and propagates the supplied CV seed.
  One-shot fold iterators are materialized for reuse across trials.
- Constant ensembles correctly estimate one intercept table. Removed the obsolete
  arbitrary failure-complexity substitution and broad BaseException suppression.
- Real-booster tests use a full ensemble with more tables than its selected prefix,
  verify generator-fold reuse, and check constant-ensemble tuning. Both tests pass
  against stock 4.7.0 and fork 4.6.0.99. All 91 Python tests and 258 Rust tests pass;
  six Rust tests and one doc test remain ignored. Release extension rebuilt.
- `examples/booster_pricing_study.py` completes synthetic CSV preparation, time holdout,
  grouped inner tuning, build metadata, conversion parity, category identity, common
  GLM comparison, quote explanations, workbook reload and a 5% intercept edit with
  fresh validation. The stock study is now exercised in CI.
- Both build runs completed: `/tmp/avenue-stock-booster-study-v2` and
  `/tmp/avenue-fork-booster-study`. Each selected one round, produced two tables/five
  rows, and passed parity and raw-label reload on 1,000 holdout quotes at absolute
  plus relative tolerance 1e-12. This validates mechanics, not predictive superiority.
- [Study limitations](BOOSTER_STUDY.md) explicitly cover shared LightGBM CV binning,
  unknown booster GLM-style convergence, post-selection inference, timing scope and
  CV table screening versus final-artifact resource limits.
