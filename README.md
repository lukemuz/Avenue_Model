# Avenue Model

**Insurance rating models without the hassle of design matrices.**

Avenue Model lets you work directly with factor tables — the natural representation for insurance pricing. Fit GLMs on rating tables, convert LightGBM models to interpretable factors, and skip the tedious design matrix conversions that plague traditional actuarial workflows.

## Why Avenue Model?

**Traditional insurance modeling is painful:**
- Converting between factor tables and design matrices is error-prone
- GLMs require you to flatten your natural table structure
- LightGBM models are black boxes that regulators don't trust
- Sparsity and interpretability take a back seat to accuracy

**Avenue Model fixes this:**
- ✅ **Fit GLMs directly on factor tables** — No design matrix conversion required
- ✅ **Convert LightGBM → factor tables** — Turn black boxes into interpretable rating structures
- ✅ **Native table operations** — Combine, consolidate, and analyze tables naturally
- ✅ **Built for insurance** — Handles categorical variables, interactions, and sparsity properly

**Recommended:** Use the Avenue fork of Lightgbm to penalize and tune for sparse, interpretable models.

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

### Python Example

```python
from avenue_model import RatingModel, fit_glm, GLMOptions

# 1. Convert LightGBM model to factor tables
model = RatingModel.from_lgbm_json(lgbm_json, "max")

# 2. Fit GLM directly on factor tables (no design matrix!)
options = GLMOptions(objective="poisson", max_iterations=100)
fitted_model = fit_glm(model, training_df, "target", "weight", options)

# 3. Make predictions
predictions = fitted_model.predict(new_data)

# 4. Combine models additively
combined = base_model + territory_model + driver_model
```

### Rust Example

```rust
use avenue_model::rating_model::RatingModel;
use avenue_model::glm::{fit_glm, GLMOptions};

// Convert LightGBM to rating tables
let model = RatingModel::from_lgbm_json(&lgbm_json, "max")?;

// Fit GLM on factor tables
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
