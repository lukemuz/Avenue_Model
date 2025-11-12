# GLM Fitting Module for Avenue_Model

High-performance Generalized Linear Model (GLM) fitting using coordinate descent on rating tables.

## Features

### Supported Distributions
- ✅ **Gaussian** (Identity link) - Linear regression
- ✅ **Poisson** (Log link) - Count data
- ✅ **Gamma** (Log link) - Continuous positive data with right skew
- ✅ **Tweedie** (Log link) - Mixed zero and continuous positive data
- ✅ **Binary/Logistic** (Logit link) - Binary classification

### Algorithm
- **IRLS (Iteratively Reweighted Least Squares)** coordinate descent over rating tables
- Iteratively updates each table's factors while holding others fixed
- Computes optimal factors as weighted means of working residuals
- Working residuals are scaled by inverse variance (proper GLM theory)
- Fast convergence (typically 2-10 iterations)

## Performance Benchmarks

Benchmarks run on development machine (debug mode):

| Distribution | Dataset Size | Bins | Time per Iteration | Time per Row/Iter |
|--------------|--------------|------|-------------------|-------------------|
| Gaussian     | 10k rows     | 4    | 5.32 ms           | 0.53 µs           |
| Poisson      | 10k rows     | 4    | 5.20 ms           | 0.52 µs           |
| Gamma        | 10k rows     | 4    | 5.18 ms           | 0.52 µs           |
| Tweedie      | 10k rows     | 4    | 5.19 ms           | 0.52 µs           |
| Binary       | 10k rows     | 4    | 5.15 ms           | 0.52 µs           |

**Large Dataset Performance:**
- 100k rows, 21 bins: ~30-50ms total fit time
- Prediction: ~1-2 µs per row

## Python API

### Basic Usage

```python
from avenue_model import RatingModel, fit_glm, GLMOptions
import polars as pl

# Create or load a RatingModel with table structure
mean_table = pl.DataFrame({"Rating_Factor": [0.0]})
age_table = pl.DataFrame({
    "Age": [25.0, 35.0, 50.0, 65.0, float('inf')],
    "Rating_Factor": [0.0, 0.0, 0.0, 0.0, 0.0]
})

model = RatingModel([mean_table, age_table], objective="poisson")

# Fit GLM
options = GLMOptions(
    objective="poisson",  # Required: distribution family
    max_iterations=100,
    tolerance=1e-6,
    verbose=True,
    tweedie_power=1.5     # Only used for Tweedie objective
)

fitted_model = fit_glm(
    model,
    training_data,
    target_col="claims",
    weight_col="exposure",
    options=options
)

# Make predictions
predictions = fitted_model.predict(test_data)
```

### Distribution-Specific Examples

#### Poisson (Count Data)
```python
options = GLMOptions(objective="poisson", verbose=True)
model = fit_glm(model, df, "claim_count", "exposure", options)
```

#### Gamma (Continuous Positive)
```python
options = GLMOptions(objective="gamma", verbose=True)
model = fit_glm(model, df, "claim_severity", "claim_count", options)
```

#### Tweedie (Mixed Zero/Continuous)
```python
options = GLMOptions(
    objective="tweedie",
    tweedie_power=1.5,  # 1=Poisson, 2=Gamma, between for Tweedie
    verbose=True
)
model = fit_glm(model, df, "loss_amount", "exposure", options)
```

#### Logistic Regression
```python
options = GLMOptions(objective="binary", verbose=True)
model = fit_glm(model, df, "is_claim", weight_col=None, options=options)
# Predictions are probabilities in [0,1]
probs = model.predict(df)
```

## Rust API

### Direct Usage

```rust
use avenue_model::glm::{fit_glm, GLMOptions};
use avenue_model::rating_model::RatingModel;

let options = GLMOptions {
    objective: "poisson".to_string(),
    max_iterations: 100,
    tolerance: 1e-6,
    verbose: true,
    tweedie_power: 1.5,
};

let fitted_model = fit_glm(
    &model,
    &training_df,
    "target",
    Some("weight"),
    options
)?;
```

## Technical Details

### Working Residuals

For each distribution, we compute working residuals differently:

- **Gaussian**: `r = y - η`
- **Poisson**: `r = y - exp(η)`
- **Gamma**: `r = (y - μ) / μ` where `μ = exp(η)`
- **Tweedie**: `r = (y - μ) / μ^(p-1)` where `p` is the power parameter
- **Binary**: `r = y - logistic(η)`

### Convergence

The algorithm converges when:
- Maximum absolute change in rating factors < tolerance, OR
- Maximum iterations reached

Typical convergence: 2-10 iterations depending on data complexity.

### Memory Efficiency

- Reuses DataFrame structures where possible
- Lazy evaluation via Polars
- Parallel processing for large datasets (via Rayon)

## Testing

Run all GLM tests:
```bash
cargo test glm
```

Run benchmarks:
```bash
cargo test --lib glm_benchmarks -- --ignored --nocapture
```

Run specific distribution test:
```bash
cargo test test_poisson_glm -- --nocapture
```

## Module Structure

```
src/glm/
├── mod.rs              # Public API exports
├── fitting.rs          # Main IRLS coordinate descent algorithm
├── loss.rs             # Loss/deviance functions and working residuals
├── matching.rs         # Observation-to-table matching optimization
└── utils.rs            # Helper functions (weighted means, etc.)
```

## Future Enhancements

Potential enhancements:
- [ ] SIMD-accelerated residual computations
- [ ] Parallel table updates for independent tables
- [ ] GPU acceleration for very large datasets
- [ ] Adaptive convergence criteria
- [ ] Warm start from previous fits
- [ ] Regularization (L1/L2) via proximal gradient descent

## References

- Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning*
- Wood, S. N. (2017). *Generalized Additive Models: An Introduction with R*
- Nelder, J. A., & Wedderburn, R. W. (1972). *Generalized linear models*
