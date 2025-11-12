# Avenue Model

**Unlock LightGBM's power with regulatory-friendly rating tables. Skip design matrix hell.**

Avenue Model bridges the gap between machine learning accuracy and actuarial interpretability. Convert LightGBM models to transparent factor tables that regulators accept, or fit GLMs directly on rating tables without flattening to design matrices. Work in the natural language of insurance pricing.

## Why Avenue Model?

**Traditional insurance modeling forces a false choice:**
- **Use LightGBM** → Great accuracy, but black box models regulators reject and actuaries can't explain
- **Use GLMs** → Interpretable, but you're stuck converting factor tables ↔ design matrices forever
- **Manual conversion** → Error-prone, time-consuming, and loses the intuitive table structure
- **Production deployment** → Rating tables are easy to inspect and deploy.  Avenue gievs ultra-fast predictions from optimized Rust engine.

**Avenue Model gives you the best of both worlds:**

### 🎯 Convert LightGBM → Rating Tables
Turn black box gradient boosting into interpretable, production-ready factor tables:
- ✅ **Preserve accuracy** — Extract the full predictive power of your LightGBM model
- ✅ **Gain interpretability** — Every prediction is now explainable as additive table lookups
- ✅ **Pass regulatory review** — Tables are transparent, auditable, and match traditional actuarial formats
- ✅ **Deploy instantly** — Drop into existing rating engines that run on factor tables
- ✅ **Consolidate intelligently** — Choose "max" mode for minimal production tables or "analysis" mode for detailed inspection

### 📊 Fit GLMs Directly on Factor Tables
Skip design matrices entirely and work with the natural table representation:
- ✅ **No conversion overhead** — Fit directly on rating tables without flattening
- ✅ **Natural workflow** — Work with tables the way actuaries think about them
- ✅ **Fast iterations** — Avoid error-prone matrix conversions in every experiment
- ✅ **Native operations** — Combine, adjust, and analyze tables as first-class objects

### 🏗️ Built for Insurance Workflows
- ✅ **Categorical variables** — Native support without dummy encoding
- ✅ **Interactions** — Multi-dimensional tables capture complex relationships
- ✅ **Sparsity** — Wildcard matching (`-999`) for efficient representation
- ✅ **Additive structure** — Tables combine naturally for modular pricing

**Recommended workflow:** Train LightGBM with the [Avenue fork](https://github.com/avenue-model/LightGBM) that penalizes for sparsity and shallow trees, then convert to tables for interpretability and deployment.

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

# 3. Make predictions (exactly match lightgbm.predict!)
predictions = model.predict(new_data)

# 4. Inspect the factor tables (what regulators see)
tables = model.model_tables()  # List of Polars DataFrames
for table_df in tables:
    print(table_df)  # Factor tables, not black boxes

# 5. Combine with other table-based models
combined = lgbm_converted_model + manual_adjustments + territory_factors
```

### Path 2: GLM Directly on Factor Tables

Fit generalized linear models without design matrices:

```python
from avenue_model import RatingModel, fit_glm, GLMOptions

# Start with base rating tables (or converted LightGBM)
model = RatingModel(base_tables, objective="poisson")

# Fit GLM directly on tables (no flattening!)
options = GLMOptions(objective="poisson", max_iterations=100)
fitted_model = fit_glm(model, training_df, "target", "weight", options)

# Predictions and table inspection work the same
predictions = fitted_model.predict(new_data)
```

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

**GLM Fitting** uses IRLS coordinate descent directly on factor tables — no flattening required.

**LightGBM Conversion** options:
- `"max"` consolidation → Minimal tables (production-ready)
- `"analysis"` consolidation → One table per tree node (for interpretability)

**Note:** LightGBM conversion works best with shallow trees (max depth ≤ 4). Deeper trees create exponentially more rating table rows and become impractical. Use the Avenue LightGBM fork to tune for sparsity and shallow trees.


## Roadmap

**Coming soon:**
- Penalized regression (Elastic Net, Ridge, Lasso)
- Easier specification of polynomials and splines
- More flexible variate handling (offsets, exposure, categorical embeddings)

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
