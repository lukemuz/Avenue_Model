# Rating models

A rating model is a collection of lookup tables combined under an identity, log or
logit link. The same representation is used for models fitted as GLMs, loaded from
editable workbooks and converted from LightGBM.

## Tables and matching

Each `RatingTable` contains feature columns and one factor column. A `RatingModel`
combines those tables and applies the inverse link to their summed factors.

- `Int32` feature columns are categorical. Values match exactly; `-999` is a wildcard.
- `Float64` feature columns are numeric upper bounds. Rows must be ordered and the last
  bound should normally be infinity.
- Multiple feature columns form an interaction table.
- Tables or individual rows can be fixed as offsets during GLM fitting.

Prediction uses the same matching rules as fitting, so the deployed model cannot drift
from its estimation structure. Batch prediction is parallelized automatically.

## Construct a model

```rust
use avenue_model::rating_model::RatingModel;

let model = RatingModel::from_dataframes(
    vec![intercept_table, age_table, region_table],
    "poisson",
    None,
    None,
)?;

let predictions = model.predict(&data)?;
```

Python users will usually construct the same model through `Plan` or load it through
`Workbook`; see the [root README](../../README.md).

## Convert LightGBM exactly

`RatingModel::from_lgbm_json` converts a dumped LightGBM model into tables without
changing its predictions:

```rust
let model = RatingModel::from_lgbm_json(model_json, "max")?;
```

`"max"` consolidates overlapping tree paths into a compact set of tables. `"analysis"`
retains tables closer to the tree structure for inspection. See the
[LightGBM guide](../../docs/lightgbm.md) for tuning, category handling and refitting.

## Combine and constrain models

Models using the same link can be composed. Under a log link, adding linear predictors
multiplies fitted means, so frequency plus severity produces pure premium:

```rust
let pure_premium = (frequency_model + severity_model)?;
```

`combine_many` performs the same operation for several models. `as_offset()` fixes a
whole table; row metadata can lock only selected factors.

## Consolidation

Consolidation combines compatible tables without changing predictions. It is used by
LightGBM conversion and is also available directly when a model has accumulated
overlapping tables. The cost can grow quickly with interacting feature levels, so the
smallest table count is not always the best representation for analysis.

## Source and tests

- [`mod.rs`](mod.rs) — model types, matching, prediction and composition
- [`lgbm_parser.rs`](lgbm_parser.rs) — LightGBM parsing
- [`consolidation.rs`](consolidation.rs) — exact table consolidation

```bash
cargo test --lib rating_model
```
