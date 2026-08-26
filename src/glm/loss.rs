use polars::prelude::*;
use crate::rating_model::LinkFunction;

/// The largest magnitude a linear predictor is allowed to reach.
///
/// Guards `exp` against overflow and keeps logistic probabilities strictly inside
/// (0, 1) so IRLS weights never collapse to exactly zero. `exp(500)` is ~1e217, far
/// beyond any plausible rating factor, so this never binds on a well-posed problem —
/// it only catches separation and other degenerate fits.
pub const ETA_CLAMP: f64 = 500.0;

/// Floor for means under the log link. Below this, `(y - mu)/mu` loses all precision.
const MU_FLOOR: f64 = 1e-300;

/// Natural log of the gamma function, by the Lanczos approximation (g = 7, n = 9).
///
/// Needed for the Poisson and Gamma log-likelihoods, which feed AIC and BIC. Accurate
/// to roughly 15 significant figures across the range these families produce, and
/// avoids taking on a dependency for one function.
fn ln_gamma(x: f64) -> f64 {
    const COEFFS: [f64; 9] = [
        0.999_999_999_999_809_93,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_13,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];

    if x < 0.5 {
        // Reflection: gamma(x) gamma(1-x) = pi / sin(pi x)
        return (std::f64::consts::PI / (std::f64::consts::PI * x).sin()).ln() - ln_gamma(1.0 - x);
    }

    let x = x - 1.0;
    let mut a = COEFFS[0];
    let t = x + 7.5;
    for (i, c) in COEFFS.iter().enumerate().skip(1) {
        a += c / (x + i as f64);
    }
    0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + a.ln()
}

/// Loss functions for different GLM families.
///
/// Each family pairs a variance function `V(mu)` with a link. Together these determine
/// the two quantities IRLS needs: the weight `w = (dmu/deta)^2 / V(mu)` and the
/// residual on the link scale `r = (y - mu) / (dmu/deta)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LossFunction {
    Gaussian,     // Identity link, V(mu) = 1
    Poisson,      // Log link,      V(mu) = mu
    Binary,       // Logit link,    V(mu) = mu(1-mu)
    Gamma,        // Log link,      V(mu) = mu^2
    Tweedie(f64), // Log link,      V(mu) = mu^p
}

impl LossFunction {
    /// Creates a loss function from a link function and objective string
    pub fn from_link_function(link: &LinkFunction) -> Self {
        match link {
            LinkFunction::Identity => LossFunction::Gaussian,
            LinkFunction::Log => LossFunction::Poisson, // Default to Poisson for log link
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

    /// True when this family uses the log link, which admits an exact closed-form
    /// solution for a single step-table level. See [`log_link_variance_power`].
    pub fn is_log_link(&self) -> bool {
        matches!(
            self,
            LossFunction::Poisson | LossFunction::Gamma | LossFunction::Tweedie(_)
        )
    }

    /// The variance power `p` in `V(mu) = mu^p`, for log-link families only.
    pub fn log_link_variance_power(&self) -> Option<f64> {
        match self {
            LossFunction::Poisson => Some(1.0),
            LossFunction::Gamma => Some(2.0),
            LossFunction::Tweedie(p) => Some(*p),
            _ => None,
        }
    }

    /// Mean as a function of the linear predictor, `mu = h(eta)`.
    #[inline]
    pub fn inverse_link(&self, eta: f64) -> f64 {
        let eta = eta.clamp(-ETA_CLAMP, ETA_CLAMP);
        match self {
            LossFunction::Gaussian => eta,
            LossFunction::Binary => 1.0 / (1.0 + (-eta).exp()),
            LossFunction::Poisson | LossFunction::Gamma | LossFunction::Tweedie(_) => {
                eta.exp().max(MU_FLOOR)
            }
        }
    }

    /// The IRLS weight `w = (dmu/deta)^2 / V(mu)`, excluding the prior weight.
    ///
    /// | family        | dmu/deta  | V(mu)      | w          |
    /// |---------------|-----------|------------|------------|
    /// | Gaussian      | 1         | 1          | 1          |
    /// | Poisson       | mu        | mu         | mu         |
    /// | Gamma         | mu        | mu^2       | 1          |
    /// | Tweedie(p)    | mu        | mu^p       | mu^(2-p)   |
    /// | Binomial      | mu(1-mu)  | mu(1-mu)   | mu(1-mu)   |
    #[inline]
    pub fn irls_weight(&self, mu: f64) -> f64 {
        match self {
            LossFunction::Gaussian => 1.0,
            LossFunction::Poisson => mu,
            LossFunction::Gamma => 1.0,
            LossFunction::Tweedie(p) => mu.powf(2.0 - p),
            LossFunction::Binary => mu * (1.0 - mu),
        }
    }

    /// The product `w * r` where `r = (y - mu) / (dmu/deta)` is the residual on the
    /// link scale.
    ///
    /// Returned as a single quantity rather than as `w` and `r` separately because
    /// the two contain matching factors that cancel analytically. Computing them
    /// apart and multiplying would divide by `mu` (or by `mu(1-mu)`) only to multiply
    /// it straight back in — catastrophic when `mu` approaches zero or one.
    #[inline]
    pub fn weighted_link_residual(&self, y: f64, mu: f64) -> f64 {
        match self {
            // w = 1, r = y - mu
            LossFunction::Gaussian => y - mu,
            // w = mu, r = (y - mu)/mu
            LossFunction::Poisson => y - mu,
            // w = mu(1-mu), r = (y - mu)/(mu(1-mu))
            LossFunction::Binary => y - mu,
            // w = 1, r = (y - mu)/mu
            LossFunction::Gamma => (y - mu) / mu,
            // w = mu^(2-p), r = (y - mu)/mu
            LossFunction::Tweedie(p) => mu.powf(1.0 - p) * (y - mu),
        }
    }

    /// The variance function `V(mu)`.
    #[inline]
    pub fn variance(&self, mu: f64) -> f64 {
        match self {
            LossFunction::Gaussian => 1.0,
            LossFunction::Poisson => mu,
            LossFunction::Gamma => mu * mu,
            LossFunction::Tweedie(p) => mu.powf(*p),
            LossFunction::Binary => mu * (1.0 - mu),
        }
    }

    /// Whether the dispersion is fixed at 1 by the family, rather than estimated.
    ///
    /// Poisson and Binomial have no free scale parameter. Gaussian, Gamma and Tweedie
    /// do, and it is estimated from the Pearson statistic.
    pub fn has_fixed_dispersion(&self) -> bool {
        matches!(self, LossFunction::Poisson | LossFunction::Binary)
    }

    /// Log-likelihood of the fit, where the family has a tractable one.
    ///
    /// `None` for Tweedie: its density is an infinite series with no closed form, so
    /// reporting an AIC for it would mean quietly substituting an approximation.
    pub fn log_likelihood(
        &self,
        target: &[f64],
        means: &[f64],
        weights: &[f64],
        dispersion: f64,
    ) -> Option<f64> {
        let n: f64 = weights.iter().filter(|w| **w > 0.0).count() as f64;
        let mut llf = 0.0;

        match self {
            LossFunction::Gaussian => {
                // The likelihood is profiled at the ML estimate of the variance,
                // SSE/n, not at the Pearson estimate SSE/(n-p) used for standard
                // errors. Those differ by the degrees-of-freedom correction, and
                // using the unbiased one here would give an AIC that is off by a
                // constant. statsmodels makes the same distinction.
                let mut sse = 0.0;
                let mut log_w = 0.0;
                for i in 0..target.len() {
                    let w = weights[i];
                    if w <= 0.0 {
                        continue;
                    }
                    sse += w * (target[i] - means[i]).powi(2);
                    log_w += w.ln();
                }
                if n <= 0.0 || !(sse > 0.0) {
                    return None;
                }
                let phi_ml = sse / n;
                llf = -0.5
                    * (sse / phi_ml + n * (2.0 * std::f64::consts::PI * phi_ml).ln() - log_w);
            }
            LossFunction::Poisson => {
                for i in 0..target.len() {
                    let w = weights[i];
                    if w <= 0.0 {
                        continue;
                    }
                    let mu = means[i].max(MU_FLOOR);
                    llf += w * (target[i] * mu.ln() - mu - ln_gamma(target[i] + 1.0));
                }
            }
            LossFunction::Binary => {
                for i in 0..target.len() {
                    let w = weights[i];
                    if w <= 0.0 {
                        continue;
                    }
                    let mu = means[i].clamp(1e-15, 1.0 - 1e-15);
                    llf += w * (target[i] * mu.ln() + (1.0 - target[i]) * (1.0 - mu).ln());
                }
            }
            LossFunction::Gamma => {
                if !(dispersion > 0.0) {
                    return None;
                }
                // Shape parameter of the Gamma is 1/dispersion.
                let shape = 1.0 / dispersion;
                for i in 0..target.len() {
                    let w = weights[i];
                    if w <= 0.0 || target[i] <= 0.0 {
                        continue;
                    }
                    let mu = means[i].max(MU_FLOOR);
                    let s = shape * w;
                    let y_over_mu = target[i] / mu;
                    llf += s * s.ln() + s * y_over_mu.ln() - s * y_over_mu
                        - ln_gamma(s)
                        - target[i].ln();
                }
            }
            LossFunction::Tweedie(_) => return None,
        }

        llf.is_finite().then_some(llf)
    }

    /// Unit deviance contribution for one observation, excluding the prior weight.
    pub fn unit_deviance(&self, y: f64, mu: f64) -> f64 {
        match self {
            LossFunction::Gaussian => (y - mu).powi(2),
            LossFunction::Poisson => {
                let mu = mu.max(MU_FLOOR);
                if y > 0.0 {
                    2.0 * (y * (y / mu).ln() - (y - mu))
                } else {
                    2.0 * mu
                }
            }
            LossFunction::Binary => {
                // Clamp only inside the logs; the boundary case y == mu contributes 0.
                let mu = mu.clamp(1e-15, 1.0 - 1e-15);
                let a = if y > 0.0 { y * (y / mu).ln() } else { 0.0 };
                let b = if y < 1.0 {
                    (1.0 - y) * ((1.0 - y) / (1.0 - mu)).ln()
                } else {
                    0.0
                };
                2.0 * (a + b)
            }
            LossFunction::Gamma => {
                let mu = mu.max(MU_FLOOR);
                if y > 0.0 {
                    2.0 * (-(y / mu).ln() + (y - mu) / mu)
                } else {
                    0.0
                }
            }
            LossFunction::Tweedie(p) => {
                let p = *p;
                let mu = mu.max(MU_FLOOR);
                if (p - 1.0).abs() < 1e-12 {
                    return LossFunction::Poisson.unit_deviance(y, mu);
                }
                if (p - 2.0).abs() < 1e-12 {
                    return LossFunction::Gamma.unit_deviance(y, mu);
                }
                let term1 = if y > 0.0 {
                    y.powf(2.0 - p) / ((1.0 - p) * (2.0 - p))
                } else {
                    0.0
                };
                let term2 = y * mu.powf(1.0 - p) / (1.0 - p);
                let term3 = mu.powf(2.0 - p) / (2.0 - p);
                2.0 * (term1 - term2 + term3)
            }
        }
    }

    /// Total weighted deviance for the current fit.
    ///
    /// `means` are on the response scale (post inverse link), one per observation.
    pub fn total_deviance(&self, target: &[f64], means: &[f64], weights: &[f64]) -> f64 {
        target
            .iter()
            .zip(means.iter())
            .zip(weights.iter())
            .map(|((&y, &mu), &w)| w * self.unit_deviance(y, mu))
            .sum()
    }

    /// Total weighted deviance, taking Polars inputs.
    ///
    /// Retained for callers outside the fitting loop.
    pub fn compute_deviance(
        &self,
        target: &ChunkedArray<Float64Type>,
        predictions: &[f64],
        weights: &ChunkedArray<Float64Type>,
    ) -> f64 {
        let mut deviance = 0.0;
        for i in 0..target.len() {
            let y = target.get(i).unwrap_or(0.0);
            let mu = predictions[i];
            let w = weights.get(i).unwrap_or(1.0);
            deviance += w * self.unit_deviance(y, mu);
        }
        deviance
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `weighted_link_residual` must equal `irls_weight * (y - mu)/(dmu/deta)` wherever
    /// that product is numerically well behaved. This pins the analytic cancellation.
    #[test]
    fn weighted_residual_matches_naive_product() {
        let cases: &[(LossFunction, f64, f64, f64)] = &[
            // (family, y, mu, dmu/deta)
            (LossFunction::Gaussian, 3.0, 2.0, 1.0),
            (LossFunction::Poisson, 7.0, 4.0, 4.0),
            (LossFunction::Gamma, 9.0, 5.0, 5.0),
            (LossFunction::Tweedie(1.5), 6.0, 3.0, 3.0),
            (LossFunction::Binary, 1.0, 0.3, 0.3 * 0.7),
        ];
        for &(f, y, mu, dmu_deta) in cases {
            let naive = f.irls_weight(mu) * (y - mu) / dmu_deta;
            let actual = f.weighted_link_residual(y, mu);
            assert!(
                (naive - actual).abs() < 1e-12 * naive.abs().max(1.0),
                "{:?}: naive {} vs actual {}",
                f, naive, actual
            );
        }
    }

    /// Deviance is minimised at mu == y, where it must be exactly zero.
    #[test]
    fn unit_deviance_is_zero_at_perfect_fit() {
        for f in [
            LossFunction::Gaussian,
            LossFunction::Poisson,
            LossFunction::Gamma,
            LossFunction::Tweedie(1.5),
            LossFunction::Binary,
        ] {
            let y = match f {
                LossFunction::Binary => 0.4,
                _ => 2.5,
            };
            let d = f.unit_deviance(y, y);
            assert!(d.abs() < 1e-10, "{:?}: deviance at perfect fit was {}", f, d);
            // And strictly positive away from it.
            assert!(f.unit_deviance(y, y * 0.5) > 0.0, "{:?}: deviance not positive", f);
            assert!(f.unit_deviance(y, y * 1.5) > 0.0, "{:?}: deviance not positive", f);
        }
    }

    #[test]
    fn inverse_link_never_overflows() {
        for f in [
            LossFunction::Gaussian,
            LossFunction::Poisson,
            LossFunction::Gamma,
            LossFunction::Tweedie(1.5),
            LossFunction::Binary,
        ] {
            for eta in [-1e300, -1e6, 0.0, 1e6, 1e300] {
                let mu = f.inverse_link(eta);
                assert!(mu.is_finite(), "{:?}: inverse_link({}) = {}", f, eta, mu);
            }
        }
    }
}
