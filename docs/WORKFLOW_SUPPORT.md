# Pricing workflow and current support

Avenue's current workflow starts with an explicit `Plan`, fits or converts rating
tables, and carries them into review and editable scoring artifacts. This page describes
implemented behavior, not the full scope of the improvement plan.

## Executable studies

Run these from a source checkout after installation:

```sh
python examples/auto_pricing_study.py --output /tmp/auto-study
python examples/homeowners_perils.py --output /tmp/homeowners-study
python examples/booster_pricing_study.py --output /tmp/booster-study
```

The auto study also selects penalty and Tweedie power within its training years,
retains the trials, and exports the selected model as an analytical bundle. Use a
fresh output directory for each study run.

The auto and homeowners studies generate clearly labeled synthetic CSVs and load them, or accept `--data` with
the same column schema. Auto uses an out-of-time split; homeowners uses grouped
home identifiers so renewals stay together. Both check convergence, produce factor
reviews, compare models on a common population, save/reload quote-scoring artifacts,
and inspect an explicit factor edit with fresh validation evidence. Homeowners factor
uncertainty uses one-way CR0 covariance by home across renewal years.

The homeowners example models **attritional water and theft** only. It does not model
catastrophe loss, tail aggregation or reinsurance. Input losses are assumed at a common
coverage/limit/deductible basis. Supplied positive development and trend factors are
applied explicitly, and their before/after loss totals are exported. Synthetic factors
are one. Selecting those factors and coverage assumptions remains the analyst's work.

The synthetic runs verify mechanics; they do not establish predictive superiority on
real insurance data. Their bootstrap loss-difference intervals include zero. Review
held-out calibration as well as loss and fit diagnostics before selecting an artifact.

The [booster study](BOOSTER_STUDY.md) exercises stock/fork CPU tuning, selected-round
complexity, conversion parity, raw-category reload and a rate edit.

## Capability matrix

| Area | Implemented | Limits / remaining work |
|---|---|---|
| Inputs | Native Polars; explicit optional pandas adapter retaining labels/nulls | Polars 1.31.0 pinned; no arbitrary object/preprocessing inference |
| Pricing populations | Frequency, positive-loss severity, pure premium; row audits; factor experience | Development/trend/coverage assumptions supplied by user; zero-payment severity needs explicit treatment |
| GLM structures | Intercept, categorical/banded factors, polynomial variates, [natural-cubic splines](SMOOTH_EFFECTS_IMPLEMENTATION.md), [monotonic banded effects](MONOTONIC_EFFECTS.md), two-way main-effect/interaction contrasts, interaction tables, supplied/locked tables | Higher-order hierarchical contrasts and penalized spline fitting remain open; monotonic fits currently unpenalized with constrained inference withheld; polynomial variates retain step-table scoring |
| Families | Gaussian, Poisson, Gamma, Tweedie, binary | No negative-binomial or general mixed-model estimator |
| Penalties/inference | GLM penalties, classical/HC0/one-way CR0 inference, direct conditional intervals, [whole-term Wald tests](TERM_TESTS.md) and quasi-Poisson uncertainty; named estimates | [Explicit GLM grid selection](GLM_SELECTION.md) supported; small-sample cluster corrections and post-selection uncertainty remain open |
| Regularized stability | [Row/group bootstrap refits](BOOTSTRAP_STABILITY.md), retained failures, reproducible draws and descriptive prediction bands | Bands do not guarantee true-mean confidence coverage; upstream selection and carried-prior uncertainty are excluded |
| Credibility | [Poisson–Gamma group relativity pooling](POISSON_CREDIBILITY.md), conditional posterior intervals, scoring workbooks and fixed-prior updates | Fixed baseline and prespecified prior strength; no estimated hyperparameters, severity pooling or general mixed models |
| Validation | Random/grouped/out-of-time splits; common candidate comparison; deviance, calibration, weighted Gini/concentration curves, segment/period A/E; paired fixed-prediction bootstrap | No automatic temporal-block uncertainty; explicit common units/metric required; discrimination does not establish calibration |
| Scoring | Strict unmatched/nonfinite checks; row diagnostics; Poisson rate/count conveniences; input schema; quote explanations; [indexed numeric grids and small-batch dispatch](LARGE_TABLE_SCORING.md) | General physical-unit inference is not automatic; no reusable prepared-scoring cache; irregular grids retain the general matcher |
| Review | Named factor estimates, covariance/intervals where supported, A/E support; [numeric interval bounds](BAND_REVIEW.md); category/reference labels; prediction-change exhibits | Irregular tables retain explicit matching-review status without invented bounds; conflicting generated column names raise errors |
| Composition | Named frequency-severity product and peril sum; nested component persistence; no inherited likelihood | Full analytical bundle and broader composition algebra remain open |
| Delivery | CSV/JSON workbooks; source category mappings; versions 1–4 reading, version 2 writing for ordinary tables, version 3 for monotonic constraints, version 4 for splines; edit/change exhibits | [Individual-model analytical bundles](ANALYTICAL_BUNDLES.md) preserve source evidence and effective fit options; edited artifacts require new validation |
| Booster conversion | Stock/fork finite and missing-route parity fixtures; categorical complements; exact thresholds; constant boosters; optional parity evidence | Multiclass, linear leaves, averaged ensembles, non-unit sigmoid and zero_as_missing explicitly unsupported |
| Tuning/devices | Optional stock/fork LightGBM tuning; selected-prefix fold table counts, total/largest rows, interaction order, stored coefficient cells and conversion cost | Statistical rank, support and scoring cost remain unmeasured per trial; final artifact complexity differs from CV estimates; fork penalties require a supporting build; GPU parity not verified here |
| Installation | Source builds; fresh Python 3.12 Linux wheel verified locally; [release wheel-test matrix](WHEEL_ACCEPTANCE.md) gates publication on tests/tutorials for Python 3.12/3.13 across five platforms | Remote matrix execution, publication and broader dependency-range validation remain unverified |

## API guides

- [Preparation and experience](PRICING_PREPARATION.md), [pandas adapter](PANDAS.md)
- [Scoring contract](SCORING_CONTRACT.md), [quote explanations and model changes](EXPLANATIONS.md)
- [Reproducible splits](VALIDATION_SPLITS.md), [common model comparison](MODEL_COMPARISON.md)
- [Coefficient intervals and quasi-Poisson](COEFFICIENT_INTERVALS.md), [HC0 covariance](ROBUST_INFERENCE.md), [cluster covariance](CLUSTER_INFERENCE.md)
- [Hierarchical interactions](HIERARCHICAL_INTERACTIONS.md), [GLM selection](GLM_SELECTION.md)
- [Continuous splines](SMOOTH_EFFECTS_IMPLEMENTATION.md), [numeric band review](BAND_REVIEW.md)
- [Bootstrap stability](BOOTSTRAP_STABILITY.md), [conditional credibility](POISSON_CREDIBILITY.md)
- [Named composition](COMPOSITION.md), [booster conversion and parity](CONVERSION.md)
- [Implementation evidence and outstanding work](IMPLEMENTATION_STATUS.md)

## Choosing an uncertainty exhibit

The methods answer different questions. Preserve the stated conditioning when sharing
their results; a nominal interval percentage alone does not identify its meaning.

| Question | Avenue exhibit | What it does not establish |
|---|---|---|
| How uncertain is an unpenalized fitted coefficient or estimable whole term? | `coefficient_intervals` and `term_tests`, using the supported model-based, HC0 or one-way CR0 covariance | Post-selection validity, small-sample cluster corrections, or uncertainty in fixed prior factors |
| How uncertain are unpenalized spline knot values? | Continuous-basis knot covariance/intervals and whole-term tests | Simultaneous confidence bands for the entire curve, or penalized-spline inference |
| How does a held-out loss difference vary under resampling of evaluation rows/groups? | Paired bootstrap in the common model comparison | Variation from refitting or selecting the candidate models |
| How sensitive are fitted predictions to row/group resampling of training data? | `bootstrap_stability` with the same Plan recipe and fixed fitting options | Guaranteed true-mean confidence coverage, future-observation prediction intervals, or repeated upstream selection |
| What are sparse-group frequency relativities under a specified pooling model? | `poisson_credibility` posterior means and credible intervals | Baseline/prior-strength estimation uncertainty, severity pooling, or future realized claim-count intervals |

For regularized fits, the [controlled ridge pilot](REGULARIZED_UNCERTAINTY_PILOT.md)
demonstrates that apparently stable bootstrap predictions can still miss the true
unpenalized mean because of shrinkage bias. Avenue therefore labels refit bands as
stability diagnostics and withholds naive unpenalized coefficient errors.

## Real-data acceptance evidence

The [fresh-wheel motor exercise](REAL_MOTOR_ACCEPTANCE.md) verifies real frequency,
severity and Tweedie fitting against glum, bundles, quote scoring and rate edits.
It also retains the unfavorable held-out calibration and remaining acceptance gates.

The [real fork challenger](REAL_FORK_ACCEPTANCE.md) also passes full holdout, threshold
and default-route parity in both modes, with raw-label reload and explicit complexity
measurements. Its improved loss does not resolve the real study's loss-cost calibration.

[Change-review profiling](CHANGE_REVIEW_PERFORMANCE.md) identified and removed a large
Python-object allocation, with exact before/after exhibit checks and three-process
measurements on the real study's million-row contribution table.

The [real spline study](REAL_SPLINE_ACCEPTANCE.md) verifies continuous scoring,
inference and export against independent references. Its tested spline specifications
have worse held-out loss than the banded comparison; implemented smooth effects do
not by themselves establish a better insurance model.

The [large-table scoring study](LARGE_TABLE_SCORING.md) reproduces the original
four-table/19,181-row workload with unchanged predictions and locally meets its
fivefold performance target. [Release-wheel acceptance](WHEEL_ACCEPTANCE.md) verifies
the latest local installed package while retaining the unverified remote-platform scope.
