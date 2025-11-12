use polars::prelude::*;
use crate::rating_model::LinkFunction;

/// Loss functions for different GLM families
#[derive(Debug, Clone)]
pub enum LossFunction {
    Gaussian,    // Identity link, squared error
    Poisson,     // Log link, Poisson deviance
    Binary,      // Logit link, binomial deviance
    Gamma,       // Log link, Gamma deviance
    Tweedie(f64), // Log link, Tweedie deviance with power parameter
}

impl LossFunction {
    /// Creates a loss function from a link function and objective string
    pub fn from_link_function(link: &LinkFunction) -> Self {
        match link {
            LinkFunction::Identity => LossFunction::Gaussian,
            LinkFunction::Log => LossFunction::Poisson,  // Default to Poisson for log link
            LinkFunction::Logit => LossFunction::Binary,
        }
    }

    /// Creates a loss function from an objective string
    pub fn from_objective(objective: &str) -> Self {
        match objective.to_lowercase().as_str() {
            "regression" | "gaussian" => LossFunction::Gaussian,
            "poisson" => LossFunction::Poisson,
            "binary" | "binomial" | "logistic" => LossFunction::Binary,
            "gamma" => LossFunction::Gamma,
            "tweedie" => LossFunction::Tweedie(1.5), // Default Tweedie power
            _ => LossFunction::Gaussian,
        }
    }

    /// Computes working residuals for IRLS (Iteratively Reweighted Least Squares)
    ///
    /// For coordinate descent, the working residual is what we want to fit
    /// when updating a single table's factors while holding others fixed.
    pub fn compute_working_residuals(
        &self,
        target: &ChunkedArray<Float64Type>,
        linear_pred: &[f64],  // η = predictions without current table
    ) -> Result<Vec<f64>, PolarsError> {
        let n = target.len();
        let mut residuals = Vec::with_capacity(n);

        match self {
            LossFunction::Gaussian => {
                // For Gaussian (identity link): residual = y - η
                for i in 0..n {
                    let y = target.get(i).unwrap_or(0.0);
                    let eta = linear_pred[i];
                    residuals.push(y - eta);
                }
            }
            LossFunction::Poisson => {
                // For Poisson (log link): residual = y - exp(η)
                for i in 0..n {
                    let y = target.get(i).unwrap_or(0.0);
                    let eta = linear_pred[i];
                    let mu = eta.exp();
                    residuals.push(y - mu);
                }
            }
            LossFunction::Binary => {
                // For Binary (logit link): residual = y - logistic(η)
                for i in 0..n {
                    let y = target.get(i).unwrap_or(0.0);
                    let eta = linear_pred[i];
                    let mu = 1.0 / (1.0 + (-eta).exp());
                    residuals.push(y - mu);
                }
            }
            LossFunction::Gamma => {
                // For Gamma (log link): residual = (y - exp(η)) / exp(η)
                // Weighted by variance function
                for i in 0..n {
                    let y = target.get(i).unwrap_or(0.0);
                    let eta = linear_pred[i];
                    let mu = eta.exp().max(1e-10);
                    // For Gamma, variance is μ², so working residual is (y-μ)/μ
                    residuals.push((y - mu) / mu);
                }
            }
            LossFunction::Tweedie(p) => {
                // For Tweedie (log link): residual = (y - exp(η)) / exp(η)^(p-1)
                // Variance function is μ^p
                for i in 0..n {
                    let y = target.get(i).unwrap_or(0.0);
                    let eta = linear_pred[i];
                    let mu = eta.exp().max(1e-10);
                    // Working residual: (y - μ) / μ^(p-1)
                    let variance_weight = mu.powf(p - 1.0);
                    residuals.push((y - mu) / variance_weight);
                }
            }
        }

        Ok(residuals)
    }

    /// Computes the deviance (loss) for the current predictions
    /// Used for monitoring convergence and model quality
    pub fn compute_deviance(
        &self,
        target: &ChunkedArray<Float64Type>,
        predictions: &[f64],  // Full predictions (all tables)
        weights: &ChunkedArray<Float64Type>,
    ) -> f64 {
        let n = target.len();
        let mut deviance = 0.0;

        match self {
            LossFunction::Gaussian => {
                // Gaussian deviance: sum of weighted squared residuals
                for i in 0..n {
                    let y = target.get(i).unwrap_or(0.0);
                    let mu = predictions[i];
                    let w = weights.get(i).unwrap_or(1.0);
                    deviance += w * (y - mu).powi(2);
                }
            }
            LossFunction::Poisson => {
                // Poisson deviance: 2 * sum(w * [y * log(y/μ) - (y - μ)])
                for i in 0..n {
                    let y = target.get(i).unwrap_or(0.0);
                    let mu = predictions[i].max(1e-10); // Avoid log(0)
                    let w = weights.get(i).unwrap_or(1.0);

                    if y > 0.0 {
                        deviance += 2.0 * w * (y * (y / mu).ln() - (y - mu));
                    } else {
                        deviance += 2.0 * w * mu;
                    }
                }
            }
            LossFunction::Binary => {
                // Binomial deviance: -2 * sum(w * [y * log(μ) + (1-y) * log(1-μ)])
                for i in 0..n {
                    let y = target.get(i).unwrap_or(0.0);
                    let mu = predictions[i].max(1e-10).min(1.0 - 1e-10); // Clip to (0,1)
                    let w = weights.get(i).unwrap_or(1.0);

                    deviance += -2.0 * w * (y * mu.ln() + (1.0 - y) * (1.0 - mu).ln());
                }
            }
            LossFunction::Gamma => {
                // Gamma deviance: 2 * sum(w * [-log(y/μ) + (y - μ)/μ])
                for i in 0..n {
                    let y = target.get(i).unwrap_or(0.0);
                    let mu = predictions[i].max(1e-10);
                    let w = weights.get(i).unwrap_or(1.0);

                    if y > 0.0 {
                        deviance += 2.0 * w * (-(y / mu).ln() + (y - mu) / mu);
                    }
                }
            }
            LossFunction::Tweedie(p) => {
                // Tweedie deviance (simplified for p != 1, 2)
                for i in 0..n {
                    let y = target.get(i).unwrap_or(0.0);
                    let mu = predictions[i].max(1e-10);
                    let w = weights.get(i).unwrap_or(1.0);

                    if *p == 1.0 {
                        // Poisson case
                        if y > 0.0 {
                            deviance += 2.0 * w * (y * (y / mu).ln() - (y - mu));
                        } else {
                            deviance += 2.0 * w * mu;
                        }
                    } else if *p == 2.0 {
                        // Gamma case
                        if y > 0.0 {
                            deviance += 2.0 * w * (-(y / mu).ln() + (y - mu) / mu);
                        }
                    } else {
                        // General Tweedie
                        if y > 0.0 {
                            let term1 = y.powf(2.0 - p) / ((1.0 - p) * (2.0 - p));
                            let term2 = y * mu.powf(1.0 - p) / (1.0 - p);
                            let term3 = mu.powf(2.0 - p) / (2.0 - p);
                            deviance += 2.0 * w * (term1 - term2 + term3);
                        } else {
                            deviance += 2.0 * w * mu.powf(2.0 - p) / (2.0 - p);
                        }
                    }
                }
            }
        }

        deviance
    }
}
