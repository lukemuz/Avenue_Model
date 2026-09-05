# Improvement plan implementation record

Branch: `improvement/pricing-workflow`.

The full scope and acceptance criteria remain in [IMPROVEMENT_PLAN.md](IMPROVEMENT_PLAN.md).
This record is a progress log, not a declaration that the plan is complete.
Existing evaluation reports, scripts and recorded runs have been preserved.

## In progress: exact continuous spline effects

- Added a private natural-cubic kernel parameterized by knot values, with local
  polynomial evaluation, linear endpoint tails, cardinal basis weights and a
  normalized-range curvature penalty. It rejects invalid/nonfinite geometry and inputs.
- Added interval-local IRLS sufficient-statistic accumulation, transformed to knot
  coordinates without an observation-by-knot matrix. This is numerical groundwork;
  no public smooth Plan term or fitting/scoring integration is claimed yet.
- Independent SciPy fixtures cover five geometries and 220 probes, including irregular
  and clustered knots, large offsets, adjacent floats and tails. Values, derivatives,
  basis weights, curvature and dense information/score references agree. C2 joins,
  affine invariance, exact knot interpolation and edit linearity are also tested.
- All 272 Rust tests pass, with six ignored tests and one ignored doc test. Existing
  public Python behavior is unchanged. [Implementation notes](SMOOTH_EFFECTS_IMPLEMENTATION.md)
  identify the remaining Plan, fitting, inference, shared-scoring and workbook gates.

## Completed increment: whole-term Wald inference and estimability correction

- Added `term_tests(model, dispersion=...)` with named joint statistics, chi-square
  tails, tested dimensions, explicit null hypotheses, unsupported row indexes and
  unavailable-term reasons. Covers ordinary supported term contrasts, hierarchical
  interaction free cells, polynomial degrees and model-based/HC0/CR0 covariance;
  model-based Poisson additionally supports quasi-Poisson rescaling.
- Within-term reduced covariance is retained during inference; joint solves are lazy.
  Penalized, constrained, nonconverged or loaded scorers cannot acquire tests. Singular
  covariance and insufficient cluster rank do not silently yield a smaller hypothesis.
  Bundles preserve default source tests and interpretation metadata automatically.
- Corrected a rank-deficiency defect: a column retained by the rank solver can still
  represent an effect confounded with a dropped column. Null-direction checks now
  withhold separate errors/tests for both duplicate terms while preserving unrelated
  estimable terms. Predictions and fitting paths are unchanged.
- Independent dense calculations cover five families, ordinary/HC0/cluster covariance,
  both normalizations, interactions, polynomials and Pearson scaling. Tests also cover
  empty/leading bands, locked priors, aliases, bundle isolation and chi-square tails
  against SciPy through 5,000 degrees of freedom. All 116 Python and 267 Rust tests
  pass; six Rust tests and one doc test remain ignored. Release extension rebuilt.
- Auto, homeowners and real motor workflows pass with joint-test JSON exports.
  [Usage and limits](TERM_TESTS.md) explicitly exclude post-selection, multiplicity and
  small-sample corrections. Continuous smooth effects and the broader goal remain open.

## Completed increment: current-wheel workflow acceptance and full-scope audit

- Added `studies/readiness_acceptance.py`: verifies installed Python/native payloads
  against the supplied wheel, rejects editable/source imports, and runs all tests,
  three public tutorials and the real motor study with its booster arm. Failures,
  skipped suites, source/dependency/input identities and artifact hashes are retained.
- Fresh stock 4.7.0 and fork 4.6.0.99 environments each passed all 110 Python tests,
  all tutorials and the real study from the wheel built at `4e5d796`. Both conversion
  modes passed on 169,504 holdout quotes plus 698 stock / 810 fork boundary probes.
  A deliberately mismatched wheel failed its payload check before running tests.
- First valid model took about 2.77 seconds in each run; the complete real study with
  challenger took 20.31 / 19.88 seconds and peaked at about 1.76 GiB. These are local
  workflow observations, not isolated comparative benchmarks. Poor holdout calibration
  remains visible; predictions are approximately 30% below actual loss.
- [The requirement-by-requirement audit](READINESS_AUDIT.md) links current evidence and
  explicitly retains missing smooth effects, joint-term tests, pooling, regularized
  uncertainty, review/comparison gaps, broader performance and release-platform gates.
  No first-choice completion claim is made. Compact manifests, logs and results live
  under `studies/results/readiness`; original evaluation artifacts remain unchanged.

## Completed increment: monotonic banded rating effects

The private ordered log-link block solver is implemented and tested. It pools
Poisson/Gamma/Tweedie sufficient statistics exactly in log space, supports either
direction and common finite coefficient bounds, and handles zero actuals. An
independent exhaustive contiguous-partition oracle checks 3,072 loss comparisons;
additional checks cover unequal weights, extreme statistics and invalid inputs.
`Plan.monotone(column, direction, ...)` now connects it to ordinary fitting for
unpenalized Poisson/Gamma/Tweedie. The table solver uses ordered-cone optimality
residuals, excludes constrained tables from joint updates, and disables extrapolation.
Empty bands extend supported factors and retain no-data flags. Unsupported families,
global solves, penalties, robust covariance, locks and repeated predictor terms fail
explicitly. Ordinary inference and likelihood parameter counts are withheld with
an explanation that reaches intervals, reports and analytical bundles.

Plan JSON and version-3 workbooks preserve direction; ordinary workbooks still write
version 2. Edited violations reach the model report. Tests compare multi-term fits
with an independent dense constrained optimizer for three families and both directions;
Rust tests additionally verify exact weighted offset fits under both normalizations.
Full CSV/bundle quote reloads pass at 1e-12 relative tolerance. Export acceptance also
found and fixed numeric-looking category labels being inferred as numeric values;
JSON/CSV now respect recorded encodings, including labels with leading zeros.

All 110 Python tests and 265 Rust tests pass (six Rust tests and one doc test ignored).
Release extension rebuilt. [Usage and limits](MONOTONIC_EFFECTS.md) distinguish this
exact banded constraint from the still-unimplemented smooth/spline requirement;
[implementation notes](MONOTONIC_IMPLEMENTATION.md) retain the derivation.

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

## Completed increment: direct intervals and quasi-Poisson review

- Original fitted models expose `inference_summary` with Pearson statistic, residual
  degrees of freedom, dispersion, parameter counts and unavailable-inference notes.
  Analytical bundles retain that source evidence.
- `coefficient_intervals` returns named coefficient and log-link relativity intervals,
  adjusted standard errors, reference/unavailable status and explicit conditional-Wald
  metadata. It refuses loaded, nonconverged, penalized or inference-disabled models.
- The explicit quasi-Poisson route rescales original Poisson uncertainty using Pearson
  dispersion. It does not change means, family, likelihood or AIC. It supports the
  existing rate/exposure-weight and count/offset conventions, not replicate weights.
- Independent OLS and information-matrix/Pearson calculations agree, including both
  exposure conventions; prediction arrays remain unchanged. Rejection paths are tested.
  All 94 Python tests and 258 Rust tests pass; six Rust tests and one doc test remain
  ignored. Release extension rebuilt successfully.
- The complete auto study ran at `/tmp/avenue-auto-interval-study`, now exporting direct
  intervals for fixed GLMs and separate quasi-Poisson frequency intervals.
- [Interpretation and limitations](COEFFICIENT_INTERVALS.md) distinguish reference
  constraints, model-based/estimated dispersion, aggregate-record degrees of freedom,
  underdispersion, data-selected structures, and clustering. Robust covariance,
  whole-term tests and post-selection uncertainty remain open.

## Completed increment: fresh-wheel real motor acceptance arm

- Built and installed a release wheel into a new Python 3.12 environment, then ran
  `studies/real_motor_acceptance.py` on public freMTPL2 frequency and severity data.
  Historical evaluation scripts and recorded outputs remain unchanged.
- The join audit found 195 orphan severity records (788,714.18 source loss units),
  retained separately, and 9,117 disagreements between reported and observed paid
  claim counts. The study explicitly models matched positive-payment frequency,
  claim-count-weighted severity and uncapped paid pure premium; no exposure caps,
  monetary caps, development or trend adjustments are applied.
- All three 37-parameter GLMs converged. Independent glum native-categorical means
  agree within maximum relative errors 8.66e-14 Poisson, 8.92e-9 Gamma and 5.65e-8
  Tweedie. Full raw-quote bundle reload and an exact 5% rate edit passed.
- Real holdout calibration is unfavorable: A/E 1.428 product and 1.419 Tweedie.
  This remains in the comparison/report; it is not hidden through holdout rebasing.
- The study exposed and fixed misleading validation text: incorrect A/E percentage
  interpretation, an unconditional claim of aggregate calibration, and an overly
  definite missing-interaction explanation. Bucket messages now use actual bucket
  count and describe support/volatility checks. Regression coverage added.
- Fresh-wheel tests: 95 Python tests passed. Rust: 258 passed, six ignored, one doc
  test ignored. Study completed in 14.11 seconds after loading began; first model
  2.86 seconds. Timings are single observations; whole-process RSS around 2.91 GiB
  motivates profiling and is not isolated solver memory.
- [Acceptance report](REAL_MOTOR_ACCEPTANCE.md) and compact `studies/results/real_motor`
  evidence include versions, input/source/wheel hashes, numerical comparisons, model
  reports and explicit remaining gates. Full output: `/tmp/avenue-real-acceptance-v3`.
  Real fork comparison, broader installation/performance checks, statistical extensions
  and full-scope audit remain outstanding; the overall goal is not marked complete.

## Completed increment: independent-observation HC0 covariance

- `GLMOptions(covariance='hc0')` requests an expected-information sandwich on the
  existing reduced design. Both solvers retain their point estimates/convergence;
  the default model-based path avoids the extra covariance matrix allocation.
- HC0 uses squared weighted observation scores, with no second dispersion factor,
  leverage correction, finite-sample correction or clustering. Penalties, disabled
  inference and unanchored normalization are rejected for this option.
- Method identity is exposed in low-level diagnostics, fit/inference summaries,
  Markdown reports, interval metadata and source bundles. Quasi-Poisson rescaling
  of HC0 intervals is rejected to prevent double adjustment.
- Independent dense matrix tests cover all five families, heterogeneous precision
  weights, both solvers, count offsets, hierarchical contrasts and unchanged means.
  Weighted-mean tests exposed and fixed an existing intercept-inference defect: its
  contrast must include table averages shifted into it. The correction applies to
  classical and HC0 covariance and agrees with an independent contrast calculation.
- All 99 Python tests and 258 Rust tests pass; six Rust tests and one doc test remain
  ignored. Release extension rebuilt. [Semantics and examples](ROBUST_INFERENCE.md)
  state the correctly specified conditional-mean/independent-observation assumptions.
  Cluster covariance, whole-term tests, finite-sample/coverage studies and
  post-selection uncertainty remain open.

## Completed increment: one-way clustered covariance

- `GLMOptions(covariance='cluster', cluster='column')` computes CR0 by summing weighted
  observation scores within independent groups before forming their outer products.
  Both solvers share the reduced design/contrast machinery; predictions are unchanged.
- Cluster identity is explicit, requires non-null string/integer IDs and at least two
  positive-weight groups, and is not added to quote input requirements. Metadata and
  reports record `cluster_cr0`, the source column and positive-weight group count.
- Grouping sorts observation indices and holds one score vector, avoiding dense
  groups-by-parameters storage. Penalties, disabled inference, unanchored normalization
  and quasi-Poisson rescaling are rejected. Source bundle evidence remains separate
  from loaded/edited scoring artifacts.
- Independent dense-score checks cover all five families and both solvers. Additional
  tests verify count offsets, singleton equivalence to HC0, row-order invariance,
  zero-weight group counts, invalid definitions, metadata and reload isolation.
- All 103 Python tests and 258 Rust tests pass; six Rust tests and one doc test remain
  ignored. Release extension rebuilt. The homeowners study completed at
  `/tmp/avenue-homeowners-cluster-study`, with 1,000 training-home clusters per peril,
  CR0 intervals, raw-quote reload and a validated component edit.
- [Assumptions and limitations](CLUSTER_INFERENCE.md) distinguish one-way uncorrected
  CR0, independent clusters, precision/exposure weights and normal intervals from
  small-sample corrections, multi-way clustering, group-based t intervals, coverage
  guarantees, whole-term tests and post-selection uncertainty. Those remain open.

## Completed increment: real-data fork challenger acceptance arm

- `studies/real_motor_acceptance.py --booster` now invokes a reusable challenger arm
  on exactly the GLM study's paid-record definition and policy train/holdout split.
  It tunes the installed stock/fork build, restores external category identity,
  converts both modes, records complexity/timing, and compares the same holdout means.
- Built the current release wheel and installed it with fork LightGBM 4.6.0.99 into
  a new environment. All 103 Python tests passed there; the real GLM/glum reference
  checks reran successfully. Historical evaluation outputs remain untouched.
- Four fork trials/three inner folds selected 80 boosting rounds (the cap). Both
  modes passed all 169,504 holdout quotes and 810 threshold/adjacent-float/missing/
  unseen probes at atol=rtol=1e-12; max holdout absolute error 5.0e-16 and relative
  error 2.73e-15. Raw-label workbook reload and every-quote 5% edit checks passed.
- Actual complexity illustrates the limits of table count: analysis 10 tables/153
  rows/largest 42; max 6 tables/329 rows/largest 210. Local warm-batch medians were
  0.0208/0.0259 seconds for 169,504 quotes; these are observations, not benchmark wins.
- Fork frequency deviance 0.451044 versus GLM 0.453825. Fork-frequency × GLM-severity
  common Tweedie loss 85.7881 versus all-GLM product 87.6173. Loss-cost A/E remains
  unfavorable at 1.433, and no holdout rebasing conceals it. Bootstrap count 20 is a
  mechanics check, not stable uncertainty evidence. Booster convergence remains
  explicitly unknown under the current GLM-oriented recommendation rule.
- [Acceptance report](REAL_FORK_ACCEPTANCE.md) and `studies/results/real_fork` preserve
  comparisons, parity, trial history, build identity and input/source/wheel hashes.
  Full output: `/tmp/avenue-real-fork-acceptance`; complete run 26.10 seconds and
  whole-process peak around 3.06 GiB (not isolated model memory). Remaining statistical,
  performance, release and final-audit gates are explicitly retained.

## Completed increment: lower-memory change review

- Profiling isolated a large allocation in `compare_changes`: converting both long
  contribution frames into Python dictionaries, unioned tuple keys and output dictionaries.
  Replaced that materialization with a Polars full join, column expressions and stable
  sorting. Policy/portfolio/segment calculations retain their existing arithmetic.
- On 169,504 real raw-quote rows and six tables (1,017,024 contributions), three fresh
  process runs per implementation gave median 3.944→0.516 seconds (7.65×) and peak RSS
  increase 2.24→0.78 GiB (65.0% lower). Total process peak 2.44→0.99 GiB. This is one
  controlled change-review workload, not a universal performance or raw-scoring claim.
- Every Arrow exhibit from all six runs exactly matches the baseline's values, dtypes
  and ordering; metadata matches. Tests additionally cover added/removed factors and
  zero-exposure offsets. All 104 Python tests pass; no Rust changes were needed.
- The complete real GLM/glum study passed again at `/tmp/avenue-real-memory-acceptance`,
  including raw-quote bundles and the edited plan. Whole-run time 9.88 seconds and peak
  1.67 GiB, compared with earlier single observations 14.11 seconds/2.91 GiB. These
  whole-study observations are distinct from the controlled process benchmark.
- [Method and evidence](CHANGE_REVIEW_PERFORMANCE.md) includes the reusable profiler,
  exact baseline source reference, all measurements, source hashes and interpretation
  limits. Broader model shapes, preparation/inference profiling and the original
  large-booster scoring target remain separate work.

## Completed increment: effective fit configuration and honest factor provenance

- Fitting diagnostics retain effective GLM options and the actual global/table solver
  path. `FittedModel.fit_options` exposes a complete replayable GLMOptions dictionary;
  `solver_used` distinguishes automatic selection from the path actually run.
- Analytical bundles automatically retain this source configuration. The older optional
  `fit_options=` annotation remains separately under caller context and cannot override
  captured evidence. Additive fields are absent/unknown in older version-1 bundles;
  loaded and converted scorers do not acquire fitting options retrospectively.
- Per-row review now labels loaded/converted factors `scoring_only` and fixed priors in
  a new fit `locked`, preserving identifiable interaction-reference labels. Unanchored
  factors already had withheld standard errors; their inference note now explains why,
  and the interval API rejects them with that explanation instead of returning an
  unexplained all-unavailable interval table.
- Tests replay a clustered Tweedie fit exactly from captured options, verify Plan-owned
  power and automatic solver resolution, preserve source evidence despite conflicting
  caller notes, and check locked/loaded labels and unchanged predictions.
- All 107 Python tests and 258 Rust tests pass; six Rust tests and one doc test remain
  ignored. Release extension rebuilt. Bundle/interval guides and support matrix updated.
