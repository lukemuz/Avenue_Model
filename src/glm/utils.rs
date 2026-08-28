/// Utility functions for GLM fitting
use polars::prelude::*;

/// Computes the mean of a series with optional weights
pub fn weighted_mean(
    values: &ChunkedArray<Float64Type>,
    weights: Option<&ChunkedArray<Float64Type>>,
) -> f64 {
    match weights {
        Some(w) => {
            let mut sum_weighted = 0.0;
            let mut sum_weights = 0.0;

            for i in 0..values.len() {
                if let (Some(val), Some(weight)) = (values.get(i), w.get(i)) {
                    sum_weighted += val * weight;
                    sum_weights += weight;
                }
            }

            if sum_weights > 0.0 {
                sum_weighted / sum_weights
            } else {
                0.0
            }
        }
        None => values.mean().unwrap_or(0.0),
    }
}

/// Initializes the mean table (intercept) using the target mean
pub fn initialize_mean_table(
    target: &ChunkedArray<Float64Type>,
    weights: Option<&ChunkedArray<Float64Type>>,
) -> f64 {
    weighted_mean(target, weights)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_weighted_mean() {
        let values = ChunkedArray::from_vec("values".into(), vec![1.0, 2.0, 3.0, 4.0]);
        let weights = ChunkedArray::from_vec("weights".into(), vec![1.0, 1.0, 1.0, 1.0]);

        assert_eq!(weighted_mean(&values, Some(&weights)), 2.5);
    }

    #[test]
    fn test_weighted_mean_unequal() {
        let values = ChunkedArray::from_vec("values".into(), vec![1.0, 2.0]);
        let weights = ChunkedArray::from_vec("weights".into(), vec![3.0, 1.0]);

        // (1*3 + 2*1) / (3 + 1) = 5/4 = 1.25
        assert_eq!(weighted_mean(&values, Some(&weights)), 1.25);
    }
}
