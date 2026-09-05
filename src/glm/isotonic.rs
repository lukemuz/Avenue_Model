//! Ordered block minimization for log-link Poisson, Gamma and Tweedie tables.
//!
//! Holding other tables fixed, write `mu_i = q_i exp(beta_r)`. For variance
//! power `p` in [1, 2], the negative score of row r is
//! `B_r exp((2-p) beta_r) - A_r exp((1-p) beta_r)`, where
//! `A_r = sum(w y q^(1-p))` and `B_r = sum(w q^(2-p))`.
//! These objectives are convex. A pooled block's optimum is log(sum A / sum B),
//! so adjacent violating blocks can be merged exactly, without an IRLS approximation.
//! Store sufficient statistics on the log scale to avoid exponentiating large offsets.
//!
//! This kernel is not yet connected to Plan or the fitting loop. Integration must
//! also handle constrained convergence, solver selection and inference eligibility.

#[derive(Clone, Copy)]
struct Block {
    start: usize,
    end: usize,
    log_actual: f64,
    log_expected: f64,
    value: f64,
}

fn log_add(a: f64, b: f64) -> f64 {
    if a == f64::NEG_INFINITY {
        return b;
    }
    if b == f64::NEG_INFINITY {
        return a;
    }
    let high = a.max(b);
    high + (-(a - b).abs()).exp().ln_1p()
}

/// Exact ordered minimizer with common finite coefficient bounds.
/// Zero actuals are represented by -infinity. Every supplied row must have
/// positive expected weight; unsupported/empty bands must be handled by the caller.
/// `increasing=false` imposes a decreasing order in the original row order.
#[allow(dead_code)] // Deliberately staged before fitting/Plan integration.
pub(crate) fn fit_ordered_log_blocks(
    log_actual: &[f64],
    log_expected: &[f64],
    increasing: bool,
    lower: f64,
    upper: f64,
) -> Result<Vec<f64>, &'static str> {
    if log_actual.len() != log_expected.len() || log_actual.is_empty() {
        return Err("ordered statistics must have matching nonempty lengths");
    }
    if !lower.is_finite() || !upper.is_finite() || lower > upper {
        return Err("ordered coefficient bounds must be finite and ascending");
    }
    if log_actual
        .iter()
        .any(|a| !a.is_finite() && *a != f64::NEG_INFINITY)
        || log_expected.iter().any(|b| !b.is_finite())
    {
        return Err("ordered statistics require nonnegative actuals and positive expected weights");
    }
    let mut blocks: Vec<Block> = Vec::with_capacity(log_actual.len());
    for r in 0..log_actual.len() {
        blocks.push(Block {
            start: r,
            end: r + 1,
            log_actual: log_actual[r],
            log_expected: log_expected[r],
            value: (log_actual[r] - log_expected[r]).clamp(lower, upper),
        });
        while blocks.len() >= 2 {
            let right = blocks[blocks.len() - 1];
            let left = blocks[blocks.len() - 2];
            let violates = if increasing {
                left.value > right.value
            } else {
                left.value < right.value
            };
            if !violates {
                break;
            }
            blocks.pop();
            blocks.pop();
            let a = log_add(left.log_actual, right.log_actual);
            let b = log_add(left.log_expected, right.log_expected);
            blocks.push(Block {
                start: left.start,
                end: right.end,
                log_actual: a,
                log_expected: b,
                value: (a - b).clamp(lower, upper),
            });
        }
    }
    let mut result = vec![0.0; log_actual.len()];
    for block in blocks {
        result[block.start..block.end].fill(block.value);
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fit(a: &[f64], b: &[f64], increasing: bool, lower: f64, upper: f64) -> Vec<f64> {
        fit_ordered_log_blocks(
            &a.iter().map(|v| v.ln()).collect::<Vec<_>>(),
            &b.iter().map(|v| v.ln()).collect::<Vec<_>>(),
            increasing,
            lower,
            upper,
        )
        .unwrap()
    }

    fn objective(beta: &[f64], a: &[f64], b: &[f64], p: f64) -> f64 {
        beta.iter()
            .zip(a)
            .zip(b)
            .map(|((&x, &a), &b)| {
                if p == 1.0 {
                    b * x.exp() - a * x
                } else if p == 2.0 {
                    b * x + a * (-x).exp()
                } else {
                    b * ((2.0 - p) * x).exp() / (2.0 - p) + a * ((1.0 - p) * x).exp() / (p - 1.0)
                }
            })
            .sum()
    }

    #[test]
    fn pooled_solution_beats_every_contiguous_partition() {
        // Exhaustive active-set oracle: every ordered optimum is a partition into
        // constant contiguous blocks. Enumerate all partitions independently of PAVA.
        let b = [1.0, 3.0, 0.5, 2.0];
        for code in 0..256 {
            let a: Vec<f64> = (0..4).map(|r| ((code >> (2 * r)) & 3) as f64).collect();
            for increasing in [true, false] {
                for (lower, upper) in [(-5.0, 5.0), (-0.2, 0.3)] {
                    let actual = fit(&a, &b, increasing, lower, upper);
                    for p in [1.0, 1.5, 2.0] {
                        let mut best = f64::INFINITY;
                        for cuts in 0..8 {
                            let mut candidate = vec![0.0; 4];
                            let mut start = 0;
                            for end in 1..=4 {
                                if end == 4 || cuts & (1 << (end - 1)) != 0 {
                                    let x = (a[start..end].iter().sum::<f64>()
                                        / b[start..end].iter().sum::<f64>())
                                    .ln()
                                    .clamp(lower, upper);
                                    candidate[start..end].fill(x);
                                    start = end;
                                }
                            }
                            if candidate.windows(2).all(|v| {
                                if increasing {
                                    v[0] <= v[1]
                                } else {
                                    v[0] >= v[1]
                                }
                            }) {
                                best = best.min(objective(&candidate, &a, &b, p));
                            }
                        }
                        let loss = objective(&actual, &a, &b, p);
                        assert!((loss - best).abs() < 1e-10, "{a:?} {p}: {loss} vs {best}");
                    }
                }
            }
        }
    }

    #[test]
    fn pooled_rates_are_weighted_not_sorted_or_averaged_logs() {
        let result = fit(&[8.0, 2.0, 20.0], &[1.0, 3.0, 2.0], true, -30.0, 30.0);
        assert!((result[0] - 2.5_f64.ln()).abs() < 1e-14);
        assert_eq!(result[0], result[1]);
        assert!((result[2] - 10.0_f64.ln()).abs() < 1e-14);
    }

    #[test]
    fn stable_with_extreme_offsets_and_zero_actuals() {
        let a = [1001.0, 999.0, 1003.0];
        let b = [1000.0; 3];
        let result = fit_ordered_log_blocks(&a, &b, true, -30.0, 30.0).unwrap();
        let shifted =
            fit_ordered_log_blocks(&[1.0, -1.0, 3.0], &[0.0; 3], true, -30.0, 30.0).unwrap();
        for (x, y) in result.iter().zip(shifted) {
            assert!((x - y).abs() < 1e-12);
        }
        assert_eq!(
            fit(&[0.0, 0.0], &[1.0, 2.0], true, -30.0, 30.0),
            vec![-30.0; 2]
        );
        assert!(fit_ordered_log_blocks(&[0.0], &[f64::NEG_INFINITY], true, -30.0, 30.0).is_err());
        assert!(fit_ordered_log_blocks(&[f64::NAN], &[0.0], true, -30.0, 30.0).is_err());
        assert!(fit_ordered_log_blocks(&[], &[], true, -30.0, 30.0).is_err());
        assert!(fit_ordered_log_blocks(&[0.0], &[0.0], true, 1.0, -1.0).is_err());
    }
}
