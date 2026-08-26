# Avenue Model

**Convert LightGBM models to rating tables. Fit GLMs directly on rating tables. Skip design matrix conversion altogether.**

Avenue Model lets you work with factor tables as a first-class data structure. Convert trained LightGBM models into transparent rating tables, or fit GLMs directly on table representations without flattening to design matrices.

## Why Avenue Model?

**The problem:**
- LightGBM models are accurate but hard to explain and deploy in traditional rating systems
- GLMs require constant conversion between factor tables and design matrices
- Regulators want transparent, auditable rating structures
- Production rating engines expect factor tables, not tree ensembles

**What Avenue Model does:**

### Convert LightGBM to Rating Tables
Extract a trained LightGBM model as interpretable factor tables:
- Predictions exactly match the original LightGBM model
- Every prediction becomes explainable as a sum of table lookups
- Tables integrate directly with existing rating engines
- Choose "max" consolidation for minimal tables or "analysis" for full detail
- Works best with shallow trees (max depth ≤ 4)

### Fit GLMs on Factor Tables
Estimate generalized linear models without design matrix conversion:
- Fit directly on rating table structures
- No need to flatten multidimensional tables
- Combine, adjust, and refine tables as native objects
- Gaussian, Poisson, Gamma, Tweedie and Binomial families
- Prior weights, offset columns, offset tables and locked rows
- Tables are anchored to a base level, so the same data always gives the same tables
- Standard errors per level, plus dispersion, deviance, AIC and BIC
- Continuous drivers as *variates*: a five-row table costs one parameter, not four,
  and still deploys as an ordinary step table
- Validated against statsmodels to 1e-7 on fitted means, contrasts and standard errors

### Native Table Operations
Built-in support for insurance pricing workflows:
- Categorical variables without dummy encoding
- Multi-dimensional tables for interactions
- Wildcard matching (`-999`) for sparse representations
- Additive model structure (or multiplicative if log-link)

**Note:** For best results, train LightGBM with the [Avenue fork](https://github.com/avenue-model/LightGBM) which includes penalties for sparsity and shallow trees.

## Installation

**Rust:**
```toml
[dependencies]
avenue_model = "0.1.0"
```

**Python:**
```bash
maturin develop --release
```

## Quick Start

### Path 1: LightGBM → Rating Tables (Recommended)

Train with LightGBM's accuracy, deploy with actuarial transparency:

```python
from avenue_model import RatingModel
import lightgbm as lgb

# 1. Train LightGBM (use Avenue fork for better table conversion)
lgbm_model = lgb.train(params, train_data)
lgbm_json = lgbm_model.dump_model()

# 2. Convert to interpretable rating tables
model = RatingModel.from_lgbm_json(lgbm_json, "max")  # "max" = minimal tables

# 3. Make predictions (exactly matches lgbm_model.predict)
predictions = model.predict(new_data)

# 4. Inspect the factor tables
tables = model.model_tables()  # List of Polars DataFrames
for table_df in tables:
    print(table_df)

# 5. Combine with other table-based models
combined = lgbm_converted_model + manual_adjustments + territory_factors
```

### Path 2: GLM Directly on Factor Tables

Fit generalized linear models without design matrices:

```python
from avenue_model import RatingModel, fit_glm_with_diagnostics, GLMOptions
import polars as pl

# Start with base rating tables (or converted LightGBM)
model = RatingModel(base_tables, objective="poisson")

# Frequency models normally carry exposure as an offset, not a weight
training_df = training_df.with_columns(pl.col("exposure").log().alias("log_exposure"))

# Fit GLM directly on tables (no flattening!)
options = GLMOptions(objective="poisson", max_iterations=100, tolerance=1e-8)
fitted_model, diag = fit_glm_with_diagnostics(
    model, training_df, "claim_count",
    offset_col="log_exposure",
    options=options,
)
print(diag)   # iterations, converged, deviance, pseudo_r2, dispersion, aic
print(diag.standard_errors)   # per table, per row; aligns with model_tables()

# Predictions and table inspection work the same
predictions = fitted_model.predict(new_data)
```

See [the GLM module docs](src/glm/README.md) for the update rules, the anchoring
options, and what the fitter rejects rather than silently working around.

### Rust Example

```rust
use avenue_model::rating_model::RatingModel;
use avenue_model::glm::{fit_glm, GLMOptions};

// Convert LightGBM to rating tables
let model = RatingModel::from_lgbm_json(&lgbm_json, "max")?;

// Inspect the factor tables
for table in &model.tables {
    println!("{:?}", table.data);  // Polars DataFrames
}

// Optionally refine with GLM on tables
let options = GLMOptions {
    objective: "poisson".to_string(),
    max_iterations: 100,
    ..Default::default()
};
let fitted = fit_glm(&model, &data, "target", Some("weight"), None, options)?;
```

## How It Works

**Rating Tables** store multidimensional factor lookups:
- Numeric features use thresholds (e.g., `age <= 25`)
- Categorical features use exact matching (with `-999` wildcard support)
- Tables are combined additively: `final_prediction = link_function⁻¹(Σ table_factors)`

**Link Functions** supported:
- Identity (Gaussian regression)
- Logit (Binary classification)
- Log (Poisson, Gamma, Tweedie)

**GLM Fitting** uses coordinate descent directly on factor tables — no flattening required. Log-link families get an exact closed-form `ln(actual / expected)` update per level; other links take an IRLS step.

**LightGBM Conversion** options:
- `"max"` consolidation → Minimal tables (production-ready)
- `"analysis"` consolidation → One table per tree node (for interpretability)

**Note:** LightGBM conversion works best with shallow trees (max depth ≤ 4). Deeper trees create exponentially more rating table rows. The Avenue LightGBM fork includes regularization to encourage sparsity and shallow trees.


## Roadmap

- Polynomial variates (degree > 1)
- Release-mode benchmarks against glum, statsmodels and R `glm`
- Penalized regression: credibility shrinkage for sparse levels, difference penalties
  on adjacent ordinal levels, monotonicity constraints, Elastic Net
- Faster matching: column indices and binary search instead of per-row hash maps

## Development

```bash
# Run tests
cargo test --lib

# Build Python bindings
maturin develop --release

# Run with benchmarks
cargo test --features benchmarks
```

**Documentation:**
- [Rating Model Module](src/rating_model/README.md) - Detailed module docs

## Built With

- [Polars](https://www.pola.rs/) - Fast DataFrames
- [PyO3](https://pyo3.rs/) - Python bindings
- [Rayon](https://github.com/rayon-rs/rayon) - Parallelism
