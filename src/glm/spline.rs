//! A continuous block inside the ordinary table-sweep solver.
use super::{inference::solve_spd, loss::LossFunction};
use crate::{rating_model::RatingTable, spline::NaturalCubicBasis};
use polars::prelude::*;

pub(super) struct SplineBlock {
    basis: NaturalCubicBasis,
    predictors: Vec<f64>,
}

fn error(message: &str) -> PolarsError {
    PolarsError::ComputeError(format!("Continuous spline: {message}").into())
}

impl SplineBlock {
    pub(super) fn new(table: &RatingTable, df: &DataFrame) -> Result<Self, PolarsError> {
        let (name, _) = table
            .spline_curve()?
            .ok_or_else(|| error("missing spline declaration"))?;
        let knots = table
            .data
            .column(&name)?
            .f64()?
            .into_no_null_iter()
            .collect();
        let xs = df.column(&name)?.f64()?;
        if xs.null_count() > 0 || xs.into_no_null_iter().any(|x| !x.is_finite()) {
            return Err(error("training coordinates must be finite and non-null"));
        }
        Ok(Self {
            basis: NaturalCubicBasis::new(knots).map_err(error)?,
            predictors: xs.into_no_null_iter().collect(),
        })
    }

    pub(super) fn weights_at(&self, row: usize) -> Result<Vec<f64>, PolarsError> {
        self.basis.weights(self.predictors[row]).map_err(error)
    }

    pub(super) fn evaluate(&self, factors: &[f64]) -> Result<Vec<f64>, PolarsError> {
        let curve = self.basis.curve(factors).map_err(error)?;
        self.predictors
            .iter()
            .map(|x| curve.evaluate(*x).map_err(error))
            .collect()
    }

    pub(super) fn loadings(&self, weights: &[f64]) -> Result<Vec<f64>, PolarsError> {
        self.basis
            .loadings(&self.predictors, weights)
            .map_err(error)
    }

    pub(super) fn relative_score(
        &self,
        loss: &LossFunction,
        target: &[f64],
        weights: &[f64],
        means: &[f64],
    ) -> Result<f64, PolarsError> {
        let scores: Vec<f64> = (0..target.len())
            .map(|i| {
                if weights[i] == 0. {
                    0.
                } else {
                    weights[i] * loss.weighted_link_residual(target[i], means[i])
                }
            })
            .collect();
        let reference: f64 = (0..target.len())
            .filter(|&i| weights[i] > 0.)
            .map(|i| {
                scores[i]
                    .abs()
                    .max((weights[i] * loss.weighted_link_residual(0., means[i])).abs())
            })
            .sum();
        let score = self
            .loadings(&scores)?
            .into_iter()
            .map(f64::abs)
            .fold(0., f64::max);
        Ok(if reference > 0. {
            score / reference
        } else {
            score
        })
    }

    /// One safeguarded Fisher-scoring step, with exact continuous changes to eta.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn update(
        &self,
        factors: &mut Vec<f64>,
        eta: &mut [f64],
        means: &mut [f64],
        target: &[f64],
        weights: &[f64],
        offset: &[f64],
        loss: &LossFunction,
    ) -> Result<(), PolarsError> {
        let n = target.len();
        let curvature: Vec<f64> = (0..n)
            .map(|i| {
                if weights[i] == 0. {
                    0.
                } else {
                    weights[i] * loss.irls_weight(means[i])
                }
            })
            .collect();
        let scores: Vec<f64> = (0..n)
            .map(|i| {
                if weights[i] == 0. {
                    0.
                } else {
                    weights[i] * loss.weighted_link_residual(target[i], means[i])
                }
            })
            .collect();
        let (information, score) = self
            .basis
            .normal_equations(&self.predictors, &curvature, &scores)
            .map_err(error)?;
        let delta = solve_spd(&information, &score, factors.len())
            .ok_or_else(|| error("knot-value information is singular; reduce knots or provide more distinct positive-weight coordinates"))?;
        let changes = self.evaluate(&delta)?;
        let largest = changes.iter().map(|v| v.abs()).fold(0., f64::max);
        let mut step = if largest > loss.step_limit() {
            loss.step_limit() / largest
        } else {
            1.
        };
        let previous = loss.total_deviance(target, means, weights);
        let mut trial_means = vec![0.; n];
        // Tiny objective increases at an exact optimum can be rounding error. The
        // independent basis score remains the convergence criterion.
        let slack = 32. * f64::EPSILON * (1. + previous.abs());
        for _ in 0..40 {
            let trial: Vec<f64> = factors
                .iter()
                .zip(&delta)
                .map(|(b, d)| b + step * d)
                .collect();
            if trial.iter().all(|v| v.is_finite())
                && (0..n).all(|i| {
                    weights[i] == 0. || {
                        let predictor = eta[i] + step * changes[i] + offset[i];
                        predictor.is_finite() && predictor.abs() <= loss.eta_limit()
                    }
                })
            {
                for i in 0..n {
                    trial_means[i] = loss.inverse_link(eta[i] + step * changes[i] + offset[i]);
                }
                let objective = loss.total_deviance(target, &trial_means, weights);
                if objective.is_finite() && objective <= previous + slack {
                    *factors = trial;
                    for i in 0..n {
                        eta[i] += step * changes[i];
                        means[i] = trial_means[i];
                    }
                    return Ok(());
                }
            }
            step *= 0.5;
        }
        // Preserve the current fit. The sweep's nonzero score will prevent false
        // convergence and the existing stall/iteration rules report nonconvergence.
        Ok(())
    }
}
