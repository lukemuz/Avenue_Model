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
//! The table fitter uses this update and the ordered-cone score residual below.

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
/// Zero statistics are represented by -infinity. Empty bands copy the preceding
/// supported band's fitted value (leading empty bands copy the first supported band).
/// This is a deterministic extension, not an estimate from observations in that band.
/// `increasing=false` imposes a decreasing order in the original row order.
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
        || log_expected
            .iter()
            .any(|b| !b.is_finite() && *b != f64::NEG_INFINITY)
        || log_actual
            .iter()
            .zip(log_expected)
            .any(|(a, b)| *b == f64::NEG_INFINITY && *a != f64::NEG_INFINITY)
    {
        return Err("ordered statistics require nonnegative actuals and positive expected weights");
    }
    let mut blocks: Vec<Block> = Vec::with_capacity(log_actual.len());
    for r in 0..log_actual.len() {
        if log_expected[r] == f64::NEG_INFINITY {
            continue;
        }
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
    if blocks.is_empty() {
        return Err("ordered table requires at least one supported band");
    }
    let mut result = vec![blocks[0].value; log_actual.len()];
    let mut previous_end = 0;
    let mut previous_value = blocks[0].value;
    for block in blocks {
        result[previous_end..block.start].fill(previous_value);
        result[block.start..block.end].fill(block.value);
        previous_end = block.end;
        previous_value = block.value;
    }
    result[previous_end..].fill(previous_value);
    Ok(result)
}

/// Largest first-order improvement along the ordered cone's generators.
/// Scores are negative loss derivatives. In each equal-valued block, feasible
/// directions decrease a prefix or increase a suffix. All their directional
/// scores must be nonpositive at the optimum. Unlike free-coordinate scores,
/// this residual vanishes when opposing row scores balance in a pooled block.
/// Numerical coefficient guards are not statistical bounds: a boundary-limited
/// fit with no finite unconstrained optimum is deliberately not certified here.
pub(crate) fn ordered_score_residual(values: &[f64], scores: &[f64], increasing: bool) -> f64 {
    if values.len() != scores.len()
        || values.is_empty()
        || values.iter().chain(scores).any(|x| !x.is_finite())
        || values
            .windows(2)
            .any(|v| if increasing { v[0] > v[1] } else { v[0] < v[1] })
    {
        return f64::INFINITY;
    }
    let mut worst = 0.0_f64;
    let mut start = 0;
    for end in 1..=values.len() {
        if end != values.len() && values[end] == values[start] {
            continue;
        }
        let mut prefix = 0.0;
        for score in &scores[start..end] {
            prefix += score;
            worst = worst.max(if increasing { -prefix } else { prefix });
        }
        let mut suffix = 0.0;
        for score in scores[start..end].iter().rev() {
            suffix += score;
            worst = worst.max(if increasing { suffix } else { -suffix });
        }
        start = end;
    }
    worst
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_bands_extend_supported_values_without_changing_the_fit() {
        for increasing in [true, false] {
            let supported = fit(&[8.0, 2.0, 20.0], &[1.0, 3.0, 2.0], increasing, -30.0, 30.0);
            let extended = fit(
                &[0.0, 8.0, 0.0, 2.0, 0.0, 20.0, 0.0],
                &[0.0, 1.0, 0.0, 3.0, 0.0, 2.0, 0.0],
                increasing,
                -30.0,
                30.0,
            );
            assert_eq!(
                extended,
                vec![
                    supported[0],
                    supported[0],
                    supported[0],
                    supported[1],
                    supported[1],
                    supported[2],
                    supported[2]
                ]
            );
        }
        assert!(fit_ordered_log_blocks(
            &[f64::NEG_INFINITY],
            &[f64::NEG_INFINITY],
            true,
            -30.0,
            30.0
        )
        .is_err());
    }

    #[test]
    fn cone_score_distinguishes_pooling_from_false_convergence() {
        assert_eq!(ordered_score_residual(&[1.0, 1.0], &[5.0, -5.0], true), 0.0);
        assert_eq!(ordered_score_residual(&[1.0, 1.0], &[-5.0, 5.0], true), 5.0);
        assert_eq!(ordered_score_residual(&[1.0, 2.0], &[5.0, -5.0], true), 5.0);
        assert_eq!(
            ordered_score_residual(&[1.0, 1.0], &[-5.0, 5.0], false),
            0.0
        );
        assert_eq!(ordered_score_residual(&[1.0, 1.0], &[5.0, -4.0], true), 1.0);
        assert!(ordered_score_residual(&[2.0, 1.0], &[0.0, 0.0], true).is_infinite());
        assert!(ordered_score_residual(&[f64::NAN], &[0.0], true).is_infinite());
    }

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
