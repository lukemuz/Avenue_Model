# Improvement-plan readiness audit — September 5, 2026

The broader first-choice goal is **not complete**. The current branch passes the
ordinary workflow acceptance exercise from fresh installed wheels, including a real
motor study and stock/fork challengers. Statistical and review gaps still include
ordinary pricing tasks. This audit follows [IMPROVEMENT_PLAN.md](IMPROVEMENT_PLAN.md),
including work not implemented; it does not redefine the goal around passing tests.

Latest installed-wheel checkpoint: `7d6d0d8a654d4a0e5cfb90c6109bc472b2b3d496` passes
133 Python tests without skips, all three public tutorials, the real motor/booster
study, and the real continuous-spline study in each of two newly created environments.
The Rust suite passes 282 active tests (six ignored tests and one ignored doctest).
The [current manifests](../studies/results/readiness_current/README.md) verify the
installed package bytes against the same wheel for stock LightGBM 4.7.0 and fork
4.6.0.99. This supersedes the earlier lack of wheel evidence for spline inference,
shared discrimination, completed-training status and numeric band exhibits. It remains
local Linux x86-64 / CPython 3.12 evidence; other release platforms are unverified.

The historical checkpoints below record what was known at each revision; completed
follow-ups are reflected in the requirement tables and remaining priorities.

Original audited implementation: `4e5d79643e2befb4c693bf1ed142718bb5c5e08c`. The acceptance runner
added with this audit records its own source hash separately. Historical evaluation
reports and their snapshot verifiers remain intact. Those verifiers describe old
results, including known failures; the current regression suite and acceptance runner
are the evidence for current behavior.

Subsequent implementation: [whole-term Wald tests](TERM_TESTS.md) now cover supported
fixed unpenalized designs, with independent covariance checks, explicit unavailable
cases and source-bundle retention. The follow-up passes 116 Python and 267 Rust tests;
the clean-wheel manifests below remain evidence for the earlier audited revision.

The subsequent [continuous-spline increment](SMOOTH_EFFECTS_IMPLEMENTATION.md) adds
exact scoring, explanations, validation support intervals, composition and editable
version-4 workbook persistence. Independent SciPy probes cover scalar/batch scoring
and both workbook formats. All 275 Rust and 118 Python tests pass from editable source.
Smooth Plan terms, spline fitting, inference and shape recovery remain open; this
does not supersede the historical clean-wheel evidence below.

Low-level unpenalized spline fitting now uses the existing table sweeps and agrees
with independent dense SciPy GLM references for five families, weights, offsets and
three normalization modes. A two-spline Poisson case recovers new-quote curves and
tails. The follow-up passes 278 Rust and 118 Python tests from editable source.
Public smooth Plan/fold integration, spline inference and roughness penalties remain
open; the earlier scoring increment alone did not establish fitting behavior.

The public [Plan spline API](SPLINES.md) now resolves knots within folds and survives
Plan/bundle serialization and refitting. Five-family Python recovery checks and a
synthetic selection/final-holdout delivery example pass. Current editable-source
verification is 278 Rust and 121 Python tests; spline inference, penalties and broader
real-portfolio acceptance remain open.

Fixed spline priors now survive the ordinary offset-model update and delivery path.
New ordinary factors support covariance and term tests conditional on the fixed prior,
verified against dense model-based/HC0/CR0 Poisson calculations. The corresponding
editable-source suites pass 278 Rust and 123 Python tests; inference for newly
estimated splines remains unavailable.

Newly estimated unpenalized splines now support continuous-basis model-based/HC0/CR0
covariance, marginal knot intervals and constant-effect whole-term tests. Independent
dense checks cover five families, both reporting anchors, weighted offsets, aliases,
empty support groups and quasi-Poisson scaling. The rebuilt editable-source suites
pass 278 Rust and 126 Python tests. This does not establish penalized inference,
simultaneous curve bands, post-selection coverage or large-portfolio performance.

The [real continuous-spline study](REAL_SPLINE_ACCEPTANCE.md) now passes the same
motor-portfolio prediction/delivery checks after tightening Avenue convergence to
1e-11. A fresh identical-split banded baseline has lower holdout loss; the spline
Tweedie fit also shows an important runtime gap and sparse-tail uncertainty. These
findings keep model-selection, stabilization and performance work open. They do not
replace the historical installed-wheel evidence below.

## Fresh installed-wheel evidence

[The runner](../studies/readiness_acceptance.py) verifies every installed package payload
against the supplied wheel, rejects editable/source imports, executes the complete
Python suite without skips, runs all three public tutorials, and runs the real motor
study including its booster arm. Each step retains its exact command, exit status,
elapsed time and log hash. Source, dependency, input and output hashes are retained.
An intentional [mismatched-wheel run](../studies/results/readiness/mismatched_wheel.json)
failed before running tests, proving that identical version labels alone cannot pass
the package identity check.

| Evidence | Stock environment | Fork environment |
|---|---|---|
| Full manifest and artifact hashes | [stock](../studies/results/readiness/stock/acceptance.json) | [fork](../studies/results/readiness/fork/acceptance.json) |
| LightGBM | 4.7.0 | 4.6.0.99 |
| Python regression suite | 110 passed, no skips | 110 passed, no skips |
| Synthetic auto, homeowners, booster studies | All passed | All passed |
| Real study: first converged/validated model | 2.77 s | 2.77 s |
| Real study including challenger | 20.31 s | 19.88 s |
| Whole-process peak RSS | 1,846,432 KiB | 1,841,136 KiB |
| Conversion holdout, both modes | 169,504 rows passed | 169,504 rows passed |
| Additional boundary/default probes, both modes | 698 rows passed | 810 rows passed |

Both environments use CPython 3.12.14 on Linux x86-64, with Polars 1.31.0 and the exact
dependency versions in their manifests. The wheel SHA-256 is
`16871dc2b819e5bbc8b26f14ce48e1cd9cfa6bd882acda0ae273d08c8b910031`.
These are sequential single-run observations, not a controlled speed comparison or
a platform/dependency-range certification. The installed payload includes the native
extension and Python modules. The source Rust suite also passed 265 tests, with six
ignored tests and one ignored doc test, before this wheel was built.

The real study preserves the paid-record population, orphan-claim audit and uncapped
losses described in [REAL_MOTOR_ACCEPTANCE.md](REAL_MOTOR_ACCEPTANCE.md). Its
[stock](../studies/results/readiness/stock/real_motor/comparison.csv) and
[fork](../studies/results/readiness/fork/real_motor/comparison.csv) comparisons retain
poor loss calibration: product/Tweedie A/E is approximately 1.428/1.419. Predictions
are about 30% below observed loss. Neither workflow success nor booster loss improvement
is a production rate recommendation. Twenty bootstrap replicates test mechanics,
not stable uncertainty estimates. The homeowners study is synthetic attritional
water/theft with explicit adjustment assumptions; it provides no catastrophe evidence.

## Requirement-by-requirement assessment

“Verified” means the cited evidence covers the stated bounded requirement. “Partial”
identifies an implemented subset and its missing portion. “Open” means no adequate
implementation or acceptance evidence exists. A checked-in workflow is configuration,
not evidence that its remote jobs ran successfully.

### 1. Reliable scoring and conversion

| Requirement / acceptance gate | Status and authoritative evidence |
|---|---|
| Preserve Plan/FittedModel/Rust/table architecture | Verified: existing architecture extended in `src/plan.rs`, `src/glm/`, `src/workbook.rs`; no replacement high-level GLM API. |
| Separate fit, prediction and validation inputs; six-record count fixture | Verified: [scoring tests](../tests/test_scoring_contract.py) assert counts `[1,2,4,1,2,4]`, constant rates, validation reconciliation and quote-only scoring. |
| Explicit rate/count units; zero/partial/invalid exposure | Verified for declared Poisson conventions: same tests and [contract](SCORING_CONTRACT.md); arbitrary physical-unit inference is deliberately absent. |
| Strict unmatched/nonfinite behavior; diagnostic alternative; wildcard/default routes | Verified on fitted, loaded and composed fixtures: scoring, [conversion](../tests/test_conversion_contract.py) and [explanation](../tests/test_explanations.py) tests. |
| Captured category/threshold defects; membership; both consolidation modes/builds; reload | Verified for retained fixtures and generated probes: conversion tests pass against both installed builds; real parity records above use `atol=rtol=1e-12`. This is not arbitrary-booster proof. |
| Booster entry point, identity, no-data verification status, unsupported semantics | Verified: `python/avenue_model/conversion.py`, conversion tests and [support contract](CONVERSION.md). |
| Independent Poisson/Gamma/Tweedie, weights/offsets, randomized differentials | Verified for covered designs: Rust reference tests, covariance tests, randomized small-tree tests, monotonic SLSQP tests and real [glum comparisons](../studies/results/readiness/stock/real_motor/models.json). Broader correlated/high-cardinality acceptance remains below. |

### 2. Cohesive everyday workflow

| Requirement / acceptance gate | Status and authoritative evidence |
|---|---|
| Explicit pandas adapter; labels/nulls/dtypes; Polars parity | Verified: [adapter tests](../tests/test_adapters.py) and [guide](PANDAS.md). Numeric-looking label JSON/CSV regression is in [monotonic tests](../tests/test_monotonic.py). |
| Intercept-only plans; named terms, estimates and diagnostics | Verified: scoring/API/provenance tests; tutorials use named access. |
| Frequency/severity/premium preparation; exclusions, large losses and inconsistent totals | Verified for stated positive-paid-loss population: [preparation tests](../tests/test_preparation.py), real source audits and [preparation guide](PRICING_PREPARATION.md). No implicit caps or development assumptions. |
| Runnable complete auto study without manual Avenue encoding/index joins | Verified from both clean environments: `examples/auto_pricing_study.py` and real motor arm. Source joins and independent-reference preparation remain explicit study code. |
| Shared exposure/claims/loss/A-E/factor/support exhibits; exact totals/exclusions | Verified for current exhibits: preparation, comparison and explanation tests; raw/current real artifact hashes retained. |
| Numeric interval bounds/inclusion rules and optional plotting | Follow-up: estimates, coefficient intervals and A/E tables share explicit bounds for provably ordered grids, including interactions; irregular/missing-route tables explicitly require matching review. [Guide](BAND_REVIEW.md), Rust lookup checks and Python reload/reconciliation tests verify the semantics. No unified plotting API. |
| Component contributions, exposure and final mean; reconciled edits and segments | Verified: explanation/composition tests, manual 5% edit assertions in all tutorials and real study. |

### 3. Validation and selection

| Requirement / acceptance gate | Status and authoritative evidence |
|---|---|
| Reusable random/group/time specifications, seeds/IDs, external holdouts | Verified: [split tests](../tests/test_splitting.py), real grouped holdout and synthetic temporal holdout. |
| Fold-local learned boundaries/encodings; no holdout leakage | Verified for Avenue Plan decisions: split/selection tests. LightGBM inner CV shares training Dataset bins; untouched final holdout is separately verified. |
| Common GLM/GBM/composed/external populations, units, weights and deviance | Verified: [comparison tests](../tests/test_comparison.py) and real glum/booster comparisons. Composed means require an explicit metric. |
| Calibration, discrimination, segment/period A-E, resampling uncertainty in comparison | Follow-up: shared weighted Gini, normalized Gini and tied-score concentration curves now join common losses, aggregate/segment A-E and paired/group loss bootstrap; [comparison tests](../tests/test_comparison.py) check independent pairwise rank calculations and undefined cases. Discrimination intervals and temporal-block/refit uncertainty remain absent. |
| Penalty/mixing/power/structure selection and retained failures/history | Verified for explicit grids: [selection tests](../tests/test_glm_selection.py) include independent ElasticNet selection and common-loss power selection; synthetic auto retains trial artifacts. |
| Warm starts and prepared-data reuse | Open: not supplied by the selection API. |
| Separate convergence, conditioning, support and generalization; no false recommendation | Follow-up: comparison accepts explicit completed-training evidence for finite-schedule learners while retaining null GLM convergence. Any reported training/convergence failure or scoring failure prevents recommendation. Independent warnings remain separate. Comparison tests and both stock/fork booster tutorials verify the contract; completion does not establish generalization. |
| Bucket warnings supplemented by support/uncertainty | Partial: warning interpretation corrected and support shown; no full support-aware calibration uncertainty model. |

### 4. Statistical modeling

| Capability / acceptance gate | Status and authoritative evidence |
|---|---|
| Identifiable main effects plus interactions | Verified for two-way treatment contrasts: [tests](../tests/test_hierarchical_interactions.py) compare independent means and contrasts, term order and numeric edges. Higher-order hierarchical contrasts remain open. |
| Smooth/piecewise smooth plus monotonic effects; recover shapes and export | Verified for unpenalized natural cubic terms and monotonic bands: independent shape recovery, five-family fitting/inference, fold-local knots and exact export checks now pass. The real spline specification has worse holdout loss than bands and uncertain sparse tails; regularized smoothing and broader selection remain open. |
| Direct intervals and robust/cluster covariance | Verified for supported fixed unpenalized designs: interval, HC0 and CR0 tests compare independent calculations and cluster definitions; homeowners exports cluster intervals. Small-sample/multiway corrections are not implemented. |
| Whole-term tests | Open at the audited wheel revision; implemented in the follow-up described above, with [independent tests](../tests/test_term_tests.py). Small-sample and post-selection joint tests remain open. |
| Regularized uncertainty, simulation checks and post-selection interpretation | Partial: naive unpenalized errors remain withheld. A [ridge bootstrap coverage pilot](REGULARIZED_UNCERTAINTY_PILOT.md) shows why generic percentile bands cannot be labeled true-mean confidence intervals. [Bootstrap stability](BOOTSTRAP_STABILITY.md) now supplies reproducible row/group refits, retained failures and explicitly descriptive bands; its public-API pilot independently verifies all refits. A coverage-valid regularized interval API remains open. |
| Narrow credibility/partial pooling with sparse-group simulation | Follow-up: [Poisson–Gamma group relativity](POISSON_CREDIBILITY.md) now supplies conditional posterior means/intervals, ordinary scoring workbooks, composition and fixed-prior updates. Independent likelihood quadrature and a 6,000-group prior-predictive simulation verify recovery, sparse-group stability and conditional coverage. Prior strength is prespecified; empirical-prior, baseline-uncertainty and severity-pooling extensions remain open. |
| Quasi-Poisson and separate negative-binomial evaluation | Partial: quasi-Poisson reference agreement and point-estimate/uncertainty explanation are verified. Negative-binomial estimation remains unimplemented and has no comparative acceptance study. |
| Booster-derived structure support, simplification proposals and selected penalties | Partial: structure checks/support and general GLM selection exist; no native cell-merging proposal workflow or dedicated post-structure-selection uncertainty. Exact conversion remains distinguished from refitting. |

### 5. Analytical record and composition

| Requirement / acceptance gate | Status and authoritative evidence |
|---|---|
| Named frequency×severity and response-scale peril sums; units/components | Verified: [composition tests](../tests/test_named_composition.py), offset-to-rate behavior, nested graph reload and homeowners reconciliation. Legacy `+` remains documented separately. |
| Versioned Plan/options/schema/maps/preprocessing/IDs/fit/validation/lineage bundle | Verified for individual fits: [bundle](../tests/test_bundle.py) and [provenance](../tests/test_fit_provenance.py) tests; current real bundles reload raw quotes. Caller JSON context never replaces captured fit evidence. |
| Complete analytical evidence for a composed graph | Partial: component scoring graph persists; caller-supplied lineage exists, but a native analytical graph bundle is not implemented. |
| Edited artifact detection, source evidence isolation, semantic impact and fresh validation | Verified: bundle integrity/edit tests and known-factor edit exhibits; loaded factors are `scoring_only`, fixed priors `locked`. Monotonic edit violations reach reports. |
| Migration, future-version failures, incremental/locked prior updates | Verified on supported fixtures: workbook v1/v2 reading, v3 monotonic export, unknown-version rejection, Rust composition/refit tests and Plan replay. |

### 6. Tuning and performance

| Requirement / acceptance gate | Status and authoritative evidence |
|---|---|
| Table count as proxy, mean/CV distribution, selected round/final artifact | Verified: [tuning tests](../tests/test_tuning_contract.py), fold counts and selected-round calculations; stock/fork study complexity exhibits show final rows/count differences. |
| Rows/largest table/order/parameters/support/scoring cost in every trial summary | Follow-up: every `Trial.fold_complexity` records converted selected-prefix fold row counts, largest table, interaction order, stored coefficient cells and conversion cost. Text summaries distinguish equal-count artifacts with 40 versus 19,181 rows. Statistical rank, support and scoring cost remain explicitly unmeasured; final refit complexity remains separate. |
| Explicit hard resource constraint on final artifact | Partial: CV `max_tables` is documented as a selection proxy, not a guarantee. No general final-artifact hard-resource enforcement API. |
| Stock/fork identity, controls, CPU/GPU distinctions and installation guidance | Verified for CPU identities and supported controls in current tests/guides. GPU behavior has not been revalidated in this acceptance. Fork source/penalties remain intact. |
| Stage profiling, prepared predictors, index/cache reuse and invalidation | Partial: change-review allocation bottleneck fixed; [numeric grid indexes](LARGE_TABLE_SCORING.md) now avoid repeated table scans and are rebuilt per call. No reusable prepared-scoring cache or stage-complete profile. |
| Equivalent native categorical and sparse reference workloads; cold/warm/inference/memory | Partial: real glum native categorical agreement and local scoring observations retained; prior Rust/sparse references exist. No current full controlled matrix of correlated/high-cardinality designs and all requested timing modes. |
| Correctness before speed claims; investigate repeatable >10% regressions | Partial: current numerical gates pass and the [change-review benchmark](CHANGE_REVIEW_PERFORMANCE.md) has repeated exact-output checks. No automated controlled regression series spanning the full workflow. |
| Fivefold target on original ~3-second large-table scoring case | Locally verified: [original saved four-table/19,181-row workload](LARGE_TABLE_SCORING.md), same 169,504 quotes and hardware, now scores in 8.6–9.2 ms versus 1.175 seconds immediately before indexing and 3.097 seconds historically. Predictions are byte-identical and pass original-booster parity. This does not establish single-quote, irregular-table or cross-platform performance. |

### 7. Installation, learning and maintenance

| Requirement / acceptance gate | Status and authoritative evidence |
|---|---|
| Tested published wheel/platform/Python matrix and source prerequisites | Partial: source guide and release matrix exist; this audit verifies local CPython 3.12 Linux x86-64 wheels only. Release jobs build artifacts but do not install/test every platform wheel. Publishing and other platforms are unverified. |
| Accurate extras and dependency ranges | Partial: test/tuning/pandas extras installed successfully at recorded versions; Polars remains pinned. Broad lower/upper dependency-range evidence is absent. |
| Hosted Python API reference | Open: no built/hosted reference in the inspected source/workflows. |
| Three data-loading tutorials with definitions, splits, convergence, review, quotes, export/edit | Verified for the current synthetic auto/homeowners/booster paths and real motor/challenger. Optional booster-structure GLM refit is not a complete branch of the public booster tutorial. |
| Explicit development/trend/coverage/tail scope | Verified within the tutorials' declared scope: homeowners adjustment audit and attritional-only statement; real study uncapped paid-loss caveats retained. |
| Support matrix and corrected contracts/signatures/tolerance/budget language | Partial: focused guides/matrix updated as features land; the old user-edited roadmap remains historical worktree content, not proof of implementation. Complete documentation/API consistency needs a release audit. |
| CI public examples plus scheduled/release independent comparisons | Partial: ordinary CI config runs tests and all three tutorials, reproduced locally here. No scheduled large-study job or verified remote run for this branch. The new runner supplies a repeatable local/release acceptance command. |

## Remaining priorities

1. Establish defensible regularized uncertainty and broaden the initial conditional
   credibility workflow where baseline/prior estimation affects actuarial decisions.
2. Improve structure selection and stabilization for sparse spline tails. The completed
   unpenalized spline mechanics and inference do not establish superior held-out models.
3. Complete final-artifact trial complexity, original large-table scoring profiling,
   wheel installation tests across the supported matrix, and an executable API reference.
4. Extend irregular-table review, prepared scoring and conditional/post-selection
   uncertainty. Ordered-grid interval labels, shared discrimination and explicit
   completed-training status now close the earlier ordinary comparison/review gaps.

The user allows some plan items to remain unfinished, but that is not evidence that
these current ordinary-task gaps are acceptable for a first-choice claim. The goal
remains active. No publishing or remote communication was performed during this audit.

## Reproduction

Build and install the wheel into a new environment with its `test,tuning` extras and
`glum`; install the local fork wheel into a separate environment for the fork run.
Then run with the documented public parquet sources and a new output directory:

```sh
python studies/readiness_acceptance.py \
  --wheel /absolute/path/avenue_model.whl \
  --frequency /absolute/path/frequency.parquet \
  --severity /absolute/path/severity.parquet \
  --output /tmp/avenue-readiness-new
```

The runner is independent of editable source installation, but runs source-checkout
tests/examples whose hashes it captures. It retains failure evidence and stops on
the first failed step or skipped suite. Full outputs from this audit remain at
`/tmp/avenue-readiness-stock` and `/tmp/avenue-readiness-fork`; compact manifests,
logs and numerical results are committed under `studies/results/readiness/`.
