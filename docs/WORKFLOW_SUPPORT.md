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

Both generate clearly labeled synthetic CSVs and load them, or accept `--data` with
the same column schema. Auto uses an out-of-time split; homeowners uses grouped
home identifiers so renewals stay together. Both check convergence, produce factor
reviews, compare models on a common population, save/reload quote-scoring artifacts,
and inspect an explicit factor edit with fresh validation evidence.

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
| GLM structures | Intercept, categorical/banded factors, polynomial variates, two-way main-effect/interaction contrasts, interaction tables, supplied/locked tables | Higher-order hierarchical contrasts, splines, monotonic constraints and partial pooling remain planned |
| Families | Gaussian, Poisson, Gamma, Tweedie, binary | No negative-binomial or general mixed-model estimator |
| Penalties/inference | Existing GLM penalty controls and classical supported inference; named factor estimates | [Explicit GLM grid selection](GLM_SELECTION.md) supported; robust/cluster covariance and post-selection uncertainty remain open |
| Validation | Random/grouped/out-of-time splits; common candidate comparison; deviance, segment/period A/E; paired fixed-prediction bootstrap | No automatic temporal-block or refit uncertainty; explicit common units/metric required |
| Scoring | Strict unmatched/nonfinite checks; row diagnostics; Poisson rate/count conveniences; input schema; quote explanations | General physical-unit inference is not automatic |
| Composition | Named frequency-severity product and peril sum; nested component persistence; no inherited likelihood | Full analytical bundle and broader composition algebra remain open |
| Delivery | CSV/JSON workbooks; source category mappings; version-1 reading/version-2 writing; edit/change exhibits | [Individual-model analytical bundles](ANALYTICAL_BUNDLES.md) preserve source evidence; fit options are caller-supplied; edited artifacts require new validation |
| Booster conversion | Stock/fork finite and missing-route parity fixtures; categorical complements; exact thresholds; constant boosters; optional parity evidence | Multiclass, linear leaves, averaged ensembles, non-unit sigmoid and zero_as_missing explicitly unsupported |
| Tuning/devices | Existing optional stock/fork LightGBM tuning | Fork penalties require a supporting build; device availability is inherited from the installed LightGBM build; GPU parity not verified here |
| Installation | Source builds; Python 3.12 exercised locally; wheel build in CI | Published platform-wheel matrix and broader dependency-range validation remain release work |

## API guides

- [Preparation and experience](PRICING_PREPARATION.md), [pandas adapter](PANDAS.md)
- [Scoring contract](SCORING_CONTRACT.md), [quote explanations and model changes](EXPLANATIONS.md)
- [Reproducible splits](VALIDATION_SPLITS.md), [common model comparison](MODEL_COMPARISON.md)
- [Hierarchical interactions](HIERARCHICAL_INTERACTIONS.md), [GLM selection](GLM_SELECTION.md)
- [Named composition](COMPOSITION.md), [booster conversion and parity](CONVERSION.md)
- [Implementation evidence and outstanding work](IMPLEMENTATION_STATUS.md)
