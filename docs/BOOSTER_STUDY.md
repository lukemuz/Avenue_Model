# Booster structure and editable pricing tables

```sh
python examples/booster_pricing_study.py --output /tmp/avenue-booster-study
python examples/booster_pricing_study.py --output /tmp/avenue-booster-refit --refit-glm
```

The example reserves the latest year, encodes predictor categories explicitly,
tunes the installed stock/fork LightGBM on development folds, and converts the
selected booster into exact editable tables. It checks raw-label quote parity after
workbook reload and verifies a 5% intercept edit. Fold construction and held-out
Poisson comparison use scikit-learn directly.

`--refit-glm` uses the converted table structure through `Plan.given`, with a
prespecified ridge penalty, then validates and exports the fitted workbook. This is
a demonstration of the existing primitives, not a new selection API. Refit inference
does not account for choosing the structure with the booster.

Outputs include tuning evidence, conversion metadata/parity, workbooks, model
reports, prediction comparisons and quote explanations. The short synthetic search
is a runnable example, not a claim of optimal model quality or comparative speed.
See [conversion](CONVERSION.md) and [tuning](lightgbm.md) for their contracts.
