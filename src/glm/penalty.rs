//! L1 and L2 penalties on the rating factors.
//!
//! # Why this is cheap here
//!
//! A table update is three phases: an `O(n)` pass over the data that scatter-adds into
//! two vectors the width of the table, an `O(n_rows)` pass that turns those into a step,
//! and an `O(n)` pass that folds the step back into the linear predictor. A penalty on
//! one table's factors is a function of that table's parameters alone, so it enters
//! **only the middle phase**. The two passes over the data are untouched.
//!
//! That is the whole efficiency argument. On the French motor data the penalty costs
//! seventy-nine extra arithmetic operations a sweep against roughly six million for the
//! data passes. A direct solver has no equivalent shortcut for L1: it has to abandon its
//! factorisation and switch to coordinate descent over the whole design. This fitter
//! *is* coordinate descent, so it pays nothing.
//!
//! # What is penalised, and against what
//!
//! **Contrasts against the table's first row, not the levels themselves.** This is not a
//! stylistic choice; penalising levels directly would be wrong.
//!
//! The model is over-parameterised on purpose — adding a constant to every row of a
//! table and subtracting it from the intercept changes no prediction — and
//! [`crate::glm::fitting::Normalization`] pins that freedom down by shifting each table
//! back onto its base level after every sweep. Under an unpenalised objective that shift
//! is a pure change of gauge. Under a penalty on levels it is not, because
//! `sum (beta_r - c)^2` is not `sum beta_r^2`: the normaliser would move the objective
//! every sweep and the fit would converge to the minimiser of nothing in particular.
//!
//! Penalising `beta_r - beta_0` removes the problem, because that quantity is exactly
//! what the shift leaves alone. It also happens to be the convention every other GLM
//! library uses without saying so: one-hot dummy coding with a dropped reference level
//! penalises coefficients toward zero, and zero there *means* the reference level. So a
//! penalised fit here and a penalised fit in glum are solving the same problem, which is
//! what makes the two comparable at all.
//!
//! Three consequences worth stating plainly:
//!
//! * The intercept is never penalised. Shrinking the overall level of the model toward
//!   zero is not something anyone wants, and it is not what other libraries do either.
//! * Variate tables are not penalised. Their free parameters are polynomial
//!   coefficients rather than rows, and a polynomial of low degree is already a
//!   constraint doing the job a penalty would do. See [`TableSemantics::Variate`].
//! * **The base level now changes the fit.** Unpenalised, the choice of base level is
//!   pure presentation. Penalised, every other level is shrunk toward it, so a base
//!   level chosen carelessly — a thin level, or an extreme one — pulls the whole table
//!   toward a bad reference.
//!
//! [`TableSemantics::Variate`]: crate::rating_model::TableSemantics::Variate

/// How a penalty is scaled against the deviance.
///
/// glum minimises `D / (2 * sum(w)) + alpha * (l1_ratio * |b|_1 + (1 - l1_ratio)/2 *
/// |b|^2)`. This fitter works with the score on the raw weight scale — `numer` and
/// `denom` accumulate `sum(a * ...)` with no division — which is that objective
/// multiplied through by `sum(w)`. So the per-coefficient multipliers are
/// `sum(w) * alpha * l1_ratio` and `sum(w) * alpha * (1 - l1_ratio)`, and `alpha` means
/// the same number in both engines.
///
/// Everything in this module is on the *half-deviance* scale, matching the score: the
/// score `g_r` satisfies `g_r = -d(D/2)/d(beta_r)`, so the penalty that pairs with it is
/// `l2/2 * z^2 + l1 * |z|`. Converting to the deviance scale for reporting means
/// doubling — see [`PenaltyPlan::total`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TablePenalty {
    /// `sum(w) * alpha * l1_ratio`.
    pub l1: f64,
    /// `sum(w) * alpha * (1 - l1_ratio)`.
    pub l2: f64,
}

impl TablePenalty {
    /// The contrast that minimises the local quadratic model plus the penalty.
    ///
    /// The fitter already builds a one-dimensional model of the objective in each row:
    /// `g` is the score and `h` the curvature, so the unpenalised step is `g / h`.
    /// Writing `z` for the current contrast against the base level and `theta` for the
    /// new one, the penalised subproblem is
    ///
    /// ```text
    /// minimise  h/2 * (theta - z)^2 - g * (theta - z) + l2/2 * theta^2 + l1 * |theta|
    /// ```
    ///
    /// whose minimiser is a soft threshold — the standard coordinate-descent step, and
    /// the reason L1 costs nothing here:
    ///
    /// ```text
    /// theta = S(h * z + g, l1) / (h + l2)
    /// ```
    ///
    /// At `l1 = l2 = 0` this is `z + g / h`, which is exactly the step the unpenalised
    /// fitter takes — for the log-link families after the `ln(1 + .)` damping that turns
    /// `A/E - 1` back into `ln(A/E)`. The penalised path is therefore a strict
    /// generalisation rather than a different algorithm.
    #[inline]
    pub fn solve(&self, g: f64, h: f64, z: f64) -> f64 {
        let curvature = h + self.l2;
        if !(curvature > 0.0) {
            return z;
        }
        soft_threshold(h * z + g, self.l1) / curvature
    }

    /// The smallest subgradient of the penalised objective at this coordinate — the
    /// quantity the convergence test has to drive to zero.
    ///
    /// **A penalised fit that tests the raw score never converges.** At the optimum of a
    /// penalised problem the raw score is not zero; it is equal to the penalty gradient,
    /// which is the whole point. Away from the kink that is a plain subtraction. At
    /// `z = 0` the L1 term has no derivative, and optimality is the *inclusion*
    /// `|g| <= l1` rather than an equation, so the right measure is how far outside that
    /// interval the score sits — which is the same soft threshold again.
    #[inline]
    pub fn subgradient(&self, g: f64, z: f64) -> f64 {
        if z == 0.0 {
            soft_threshold(g, self.l1)
        } else {
            g - self.l2 * z - self.l1 * z.signum()
        }
    }

    /// Where a penalised table's **base level** should move to.
    ///
    /// The base level is not a spectator. Every other level is measured against it, so
    /// moving it by `d` moves every contrast by `-d` at once - which is exactly the
    /// direction a backfit needs to be able to take cheaply, and exactly the direction
    /// the intercept would otherwise have to supply one table at a time.
    ///
    /// **Pinning it instead is a disaster, and a quiet one.** It looks reasonable: the
    /// base level is the reference, other libraries drop that column, so hold it still.
    /// But the model carries an intercept *and* a factor per level, and pinning the base
    /// removes the redundancy that lets a table shift its own level in one step. On the
    /// French motor data that took a fit from 12 sweeps to 457 **at `alpha = 1e-12`** -
    /// a penalty far too small to move a coefficient. The penalty was free; the
    /// parameterisation was not.
    ///
    /// So the base level stays free and carries the penalty's gradient with respect to
    /// itself, which is minus the sum of every other level's. That makes the
    /// over-parameterised stationarity conditions consistent again - they sum to the
    /// intercept's condition, exactly as they do unpenalised - and leaves the gauge
    /// freedom that [`Normalization::BaseLevel`] exists to remove.
    ///
    /// Writing `z_r` for the contrasts, `m` for how many there are, and `S` for their
    /// sum, the objective in the step `d` is
    ///
    /// ```text
    /// h/2 * d^2 - g*d + sum_r [ l2/2 * (z_r - d)^2 + l1 * |z_r - d| ]
    /// ```
    ///
    /// which is convex, piecewise quadratic, and has kinks at every `z_r`. Its
    /// derivative is `d*(h + m*l2) - (g + l2*S) - l1 * sum_r sign(z_r - d)`, and is
    /// non-decreasing, so the root is found by walking the sorted contrasts once. With
    /// no L1 term there are no kinks and the walk collapses to a single division.
    ///
    /// `contrasts` is sorted in place. That is `O(m log m)` per table per sweep against
    /// the `O(n)` passes either side of it, so it does not show up.
    ///
    /// [`Normalization::BaseLevel`]: crate::glm::fitting::Normalization::BaseLevel
    pub fn solve_anchor(&self, g: f64, h: f64, contrasts: &mut [f64]) -> f64 {
        let m = contrasts.len();
        let a = h + m as f64 * self.l2;
        if !(a > 0.0) {
            return 0.0;
        }
        let b = g + self.l2 * contrasts.iter().sum::<f64>();
        if !(self.l1 > 0.0) {
            return b / a;
        }

        contrasts.sort_by(|x, y| x.total_cmp(y));

        // `below` counts the contrasts strictly under the interval being examined, so
        // the sign sum over `z_r - d` is `m - 2*below`.
        let mut below = 0usize;
        let mut idx = 0usize;
        loop {
            let lo = if idx == 0 {
                f64::NEG_INFINITY
            } else {
                contrasts[idx - 1]
            };
            let hi = if idx == m {
                f64::INFINITY
            } else {
                contrasts[idx]
            };
            let sign_sum = m as f64 - 2.0 * below as f64;
            let candidate = (b + self.l1 * sign_sum) / a;
            if candidate > lo && candidate < hi {
                return candidate;
            }
            if idx == m {
                return b / a;
            }

            // Step over every contrast equal to this one at once; the derivative jumps
            // by twice their number.
            let v = contrasts[idx];
            let mut equal = 0usize;
            while idx + equal < m && contrasts[idx + equal] == v {
                equal += 1;
            }
            let left = a * v - b - self.l1 * (m as f64 - 2.0 * below as f64);
            let right = a * v - b - self.l1 * (m as f64 - 2.0 * (below + equal) as f64);
            if left <= 0.0 && right >= 0.0 {
                // The derivative changes sign across the kink itself.
                return v;
            }
            below += equal;
            idx += equal;
        }
    }

    /// The smallest subgradient at a penalised table's base level.
    ///
    /// Its score is `g` plus the sum of every other level's penalty gradient, because
    /// the base level appears in all of them with the opposite sign. Contrasts sitting
    /// exactly on zero contribute an interval rather than a value, and the best choice
    /// within it is the usual soft threshold.
    ///
    /// Summed over a table this cancels against the other rows exactly, leaving the
    /// intercept's own condition - which is the check that the conditions are
    /// consistent, and is what pinning the base level broke.
    #[inline]
    pub fn anchor_subgradient(&self, g: f64, contrasts: &[f64]) -> f64 {
        let mut base = g;
        let mut slack = 0.0;
        for z in contrasts {
            base += self.l2 * z;
            if *z > 0.0 {
                base += self.l1;
            } else if *z < 0.0 {
                base -= self.l1;
            } else {
                slack += self.l1;
            }
        }
        soft_threshold(base, slack)
    }

    /// The penalty's contribution to the objective, on the half-deviance scale.
    #[inline]
    pub fn value(&self, z: f64) -> f64 {
        0.5 * self.l2 * z * z + self.l1 * z.abs()
    }

    /// Whether this penalty can drive a coefficient to exactly zero.
    #[inline]
    pub fn selects(&self) -> bool {
        self.l1 > 0.0
    }
}

/// `S(x, t)`: `x` pulled toward zero by `t`, and held there once it arrives.
#[inline]
pub fn soft_threshold(x: f64, t: f64) -> f64 {
    if x > t {
        x - t
    } else if x < -t {
        x + t
    } else {
        0.0
    }
}

/// The row every other row of a table is shrunk toward, and which a penalised table
/// holds still. See [`PenaltyPlan::is_gauge`] for why it is held still.
///
/// Row 0, matching [`Normalization::BaseLevel`]. The anchor's *value* is still read from
/// the factors rather than assumed to be zero, because two things reach this code with a
/// table un-normalised: a table containing a locked row, which `normalize` declines to
/// shift at all, and the extrapolated point SQUAREM builds, which is written directly
/// into the factors without a sweep. Reading it makes the penalty genuinely invariant to
/// the gauge instead of merely invariant to the gauge the sweep usually leaves behind.
///
/// [`Normalization::BaseLevel`]: crate::glm::fitting::Normalization::BaseLevel
pub const ANCHOR_ROW: usize = 0;

/// Which tables carry a penalty and how strong it is.
#[derive(Debug, Clone)]
pub struct PenaltyPlan {
    penalty: TablePenalty,
    /// Indexed by table. False for the intercept, for variate tables, and for
    /// single-row tables, which have no contrast to penalise.
    penalised: Vec<bool>,
}

impl PenaltyPlan {
    /// Builds the plan, or `None` when no penalty was asked for.
    ///
    /// `total_weight` is `sum(w)` over the observations actually being fitted; see
    /// [`TablePenalty`] for why it appears.
    pub fn new(
        alpha: f64,
        l1_ratio: f64,
        total_weight: f64,
        shapes: &[usize],
        is_variate: &[bool],
    ) -> Option<PenaltyPlan> {
        if !(alpha > 0.0) || !total_weight.is_finite() || !(total_weight > 0.0) {
            return None;
        }
        let ratio = l1_ratio.clamp(0.0, 1.0);
        let penalty = TablePenalty {
            l1: total_weight * alpha * ratio,
            l2: total_weight * alpha * (1.0 - ratio),
        };
        if !(penalty.l1 > 0.0) && !(penalty.l2 > 0.0) {
            return None;
        }

        let penalised = shapes
            .iter()
            .enumerate()
            .map(|(t, k)| {
                // Table 0 is the intercept. A table with one row is an intercept in all
                // but name, and has no contrast against itself to shrink.
                t != 0 && *k > 1 && !is_variate[t]
            })
            .collect();

        Some(PenaltyPlan { penalty, penalised })
    }

    /// The penalty on one row of one table, or `None` where nothing is penalised.
    ///
    /// The anchor row returns `None` because it is not a shrunk parameter. It is not a
    /// free parameter at all - see [`PenaltyPlan::is_gauge`].
    #[inline]
    pub fn row(&self, table: usize, row: usize) -> Option<TablePenalty> {
        if row == ANCHOR_ROW || !self.penalised.get(table).copied().unwrap_or(false) {
            None
        } else {
            Some(self.penalty)
        }
    }

    /// Whether any table is penalised at all.
    pub fn is_active(&self) -> bool {
        self.penalised.iter().any(|p| *p)
    }

    /// Whether the penalty can zero a coefficient, which changes what the joint pair
    /// solve and the standard errors are allowed to do.
    pub fn selects(&self) -> bool {
        self.penalty.selects() && self.is_active()
    }

    /// Whether a table is penalised, for callers that need the answer once rather than
    /// per row.
    pub fn covers(&self, table: usize) -> bool {
        self.penalised.get(table).copied().unwrap_or(false)
    }

    /// The penalty's total contribution, **on the deviance scale** — twice
    /// [`TablePenalty::value`], so that it adds to a deviance rather than to half of
    /// one.
    ///
    /// This is what the stall test and the SQUAREM acceptance test have to use. Both ask
    /// whether the thing being minimised went down, and once a penalty is on, the thing
    /// being minimised is not the deviance. The deviance reported to the caller stays
    /// unpenalised, because it is a goodness-of-fit statistic measured against
    /// `null_deviance` and would stop meaning that if the penalty were folded in.
    pub fn total(&self, factors: &[Vec<f64>]) -> f64 {
        let mut sum = 0.0;
        for (t, rows) in factors.iter().enumerate() {
            if !self.covers(t) || rows.len() <= ANCHOR_ROW {
                continue;
            }
            let anchor = rows[ANCHOR_ROW];
            for (r, f) in rows.iter().enumerate() {
                if r == ANCHOR_ROW {
                    continue;
                }
                sum += self.penalty.value(f - anchor);
            }
        }
        2.0 * sum
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plan(alpha: f64, l1_ratio: f64) -> PenaltyPlan {
        PenaltyPlan::new(alpha, l1_ratio, 1.0, &[1, 4, 3], &[false, false, false]).unwrap()
    }

    #[test]
    fn an_unpenalised_solve_is_the_ordinary_newton_step() {
        let p = TablePenalty { l1: 0.0, l2: 0.0 };
        // theta - z should be g / h for any starting contrast.
        for z in [-2.0, 0.0, 0.75] {
            let theta = p.solve(3.0, 6.0, z);
            assert!((theta - z - 0.5).abs() < 1e-15, "z = {z}, theta = {theta}");
        }
    }

    #[test]
    fn ridge_shrinks_the_step_toward_the_anchor() {
        let p = TablePenalty { l1: 0.0, l2: 4.0 };
        // h*z + g = 6*1 + 0 = 6, over h + l2 = 10.
        let theta = p.solve(0.0, 6.0, 1.0);
        assert!((theta - 0.6).abs() < 1e-15);
        // A coefficient already at the anchor with no score stays there.
        assert_eq!(p.solve(0.0, 6.0, 0.0), 0.0);
    }

    #[test]
    fn lasso_zeroes_a_coefficient_the_data_barely_supports() {
        let p = TablePenalty { l1: 5.0, l2: 0.0 };
        // h*z + g = 2*0.5 + 1 = 2, which is inside the threshold.
        assert_eq!(p.solve(1.0, 2.0, 0.5), 0.0);
        // Outside it, the coefficient survives but is pulled in by exactly l1 / h.
        let theta = p.solve(10.0, 2.0, 0.5);
        assert!((theta - 3.0).abs() < 1e-15, "theta = {theta}");
    }

    #[test]
    fn a_zeroed_coefficient_is_a_fixed_point() {
        let p = TablePenalty { l1: 5.0, l2: 1.0 };
        // Optimality at the kink is |g| <= l1, so a score inside that leaves it alone.
        for g in [-5.0, -4.9, 0.0, 4.9, 5.0] {
            assert_eq!(p.solve(g, 3.0, 0.0), 0.0, "g = {g}");
            assert_eq!(p.subgradient(g, 0.0), 0.0, "g = {g}");
        }
        // And a score outside it moves off zero, in the direction of the score.
        assert!(p.solve(6.0, 3.0, 0.0) > 0.0);
        assert!(p.solve(-6.0, 3.0, 0.0) < 0.0);
    }

    #[test]
    fn the_subgradient_vanishes_exactly_where_the_step_does() {
        // The convergence test and the update have to agree about the optimum, or the
        // fit stops somewhere the step would still move away from - or never stops.
        let p = TablePenalty { l1: 2.0, l2: 3.0 };
        for z in [-1.5, -0.4, 0.0, 0.4, 1.5] {
            for g in [-8.0, -2.0, -0.1, 0.0, 0.1, 2.0, 8.0] {
                let h = 4.0;
                let moved = (p.solve(g, h, z) - z).abs() > 1e-14;
                let scored = p.subgradient(g, z).abs() > 1e-14;
                assert_eq!(moved, scored, "z = {z}, g = {g}");
            }
        }
    }

    #[test]
    fn the_subgradient_is_the_plain_score_when_nothing_is_penalised() {
        let p = TablePenalty { l1: 0.0, l2: 0.0 };
        assert_eq!(p.subgradient(1.25, 0.0), 1.25);
        assert_eq!(p.subgradient(1.25, -3.0), 1.25);
    }

    #[test]
    fn the_intercept_and_the_anchor_row_are_never_penalised() {
        let p = plan(0.1, 0.5);
        assert!(p.row(0, 0).is_none());
        assert!(p.row(0, 1).is_none());
        assert!(p.row(1, ANCHOR_ROW).is_none());
        assert!(p.row(1, 1).is_some());
    }

    /// Brute force against the definition: the anchor step has to minimise the
    /// piecewise-quadratic objective it claims to, kinks and all.
    #[test]
    fn the_anchor_step_minimises_its_objective() {
        let cases: [(f64, f64, Vec<f64>); 6] = [
            (3.0, 4.0, vec![0.5, -0.25, 1.0]),
            (-2.0, 1.0, vec![0.0, 0.0, 0.8]),
            (0.0, 2.0, vec![-1.0, -1.0, -1.0]),
            (12.0, 0.5, vec![0.3]),
            (-7.0, 3.0, vec![2.0, -2.0, 0.0, 0.4, 0.4]),
            (0.1, 9.0, vec![]),
        ];
        for (l1, l2) in [(0.0, 1.5), (2.0, 0.0), (1.0, 0.75), (0.0, 0.0)] {
            let p = TablePenalty { l1, l2 };
            for (g, h, z) in cases.iter() {
                let f = |d: f64| {
                    0.5 * h * d * d - g * d + z.iter().map(|zr| p.value(zr - d)).sum::<f64>()
                };
                let mut work = z.clone();
                let d = p.solve_anchor(*g, *h, &mut work);
                let best = f(d);
                // Nothing nearby may be better, at a range of scales.
                for step in [1e-6, 1e-4, 1e-2, 0.1, 1.0] {
                    for probe in [d - step, d + step] {
                        assert!(
                            f(probe) >= best - 1e-9 * best.abs().max(1.0),
                            "l1={l1} l2={l2} g={g} h={h} z={z:?}: f({probe})={}                              beats f({d})={best}",
                            f(probe)
                        );
                    }
                }
            }
        }
    }

    /// The step and the subgradient must agree about where the optimum is, or the fit
    /// stops somewhere the sweep would still move away from.
    #[test]
    fn the_anchor_subgradient_vanishes_where_the_anchor_step_does() {
        let p = TablePenalty { l1: 1.5, l2: 2.0 };
        for g in [-9.0, -3.0, -0.5, 0.0, 0.5, 3.0, 9.0] {
            for z in [
                vec![0.0, 0.0],
                vec![0.6, -0.6],
                vec![1.0, 1.0, 1.0],
                vec![0.0, 2.0, -0.3],
            ] {
                let mut work = z.clone();
                let moved = p.solve_anchor(g, 4.0, &mut work).abs() > 1e-12;
                let scored = p.anchor_subgradient(g, &z).abs() > 1e-12;
                assert_eq!(moved, scored, "g = {g}, z = {z:?}");
            }
        }
    }

    /// Summed over a whole table the penalty gradients cancel, leaving the intercept's
    /// own condition. This is the consistency that pinning the base level destroyed, and
    /// with it the whole argument for letting the base level move.
    #[test]
    fn a_tables_penalty_gradients_sum_to_zero() {
        // Differentiable: every contrast away from the kink.
        let p = TablePenalty { l1: 0.9, l2: 1.7 };
        let z = [0.4, -1.2, 2.5];
        let rows: f64 = z.iter().map(|zr| p.subgradient(0.0, *zr)).sum();
        let anchor = p.anchor_subgradient(0.0, &z);
        assert!(
            (rows + anchor).abs() < 1e-12,
            "rows contribute {rows}, anchor {anchor}"
        );

        // A pure ridge has no kink at all, so a zero contrast cancels too.
        let ridge = TablePenalty { l1: 0.0, l2: 1.7 };
        let z = [0.4, -1.2, 0.0, 2.5];
        let rows: f64 = z.iter().map(|zr| ridge.subgradient(0.0, *zr)).sum();
        let anchor = ridge.anchor_subgradient(0.0, &z);
        assert!(
            (rows + anchor).abs() < 1e-12,
            "ridge: rows contribute {rows}, anchor {anchor}"
        );

        // On a kink each side independently picks its own smallest subgradient, so the
        // two agree only to within `l1` per zeroed contrast. That gap is slack the base
        // level is entitled to, not a disagreement about where the optimum is.
        let rows: f64 = z.iter().map(|zr| p.subgradient(0.0, *zr)).sum();
        let anchor = p.anchor_subgradient(0.0, &z);
        assert!(
            (rows + anchor).abs() <= p.l1 + 1e-12,
            "gap {}",
            rows + anchor
        );
    }

    #[test]
    fn variate_and_single_row_tables_are_skipped() {
        let p =
            PenaltyPlan::new(0.1, 0.0, 1.0, &[1, 5, 1, 6], &[false, true, false, false]).unwrap();
        assert!(!p.covers(1), "a variate is constrained, not penalised");
        assert!(!p.covers(2), "a one-row table has no contrast to shrink");
        assert!(p.covers(3));
    }

    #[test]
    fn no_alpha_means_no_plan() {
        assert!(PenaltyPlan::new(0.0, 0.5, 1.0, &[1, 4], &[false, false]).is_none());
        assert!(PenaltyPlan::new(-1.0, 0.5, 1.0, &[1, 4], &[false, false]).is_none());
        assert!(PenaltyPlan::new(0.1, 0.5, 0.0, &[1, 4], &[false, false]).is_none());
    }

    #[test]
    fn alpha_is_scaled_by_the_total_weight_to_match_glum() {
        // glum minimises D / (2*sum(w)) + alpha * penalty; this fitter works with that
        // multiplied through by sum(w), so the multipliers carry the weight.
        let p = PenaltyPlan::new(0.25, 0.4, 800.0, &[1, 4], &[false, false]).unwrap();
        let row = p.row(1, 2).unwrap();
        assert!((row.l1 - 800.0 * 0.25 * 0.4).abs() < 1e-12);
        assert!((row.l2 - 800.0 * 0.25 * 0.6).abs() < 1e-12);
    }

    #[test]
    fn the_total_is_measured_against_the_anchor_not_against_zero() {
        // The whole reason the penalty is defined on contrasts: shifting a table
        // wholesale is what `normalize` does every sweep, and it must not move the
        // objective.
        let p = plan(1.0, 0.0);
        let before = vec![vec![0.3], vec![0.0, 1.0, -2.0, 0.5]];
        let after: Vec<Vec<f64>> = before
            .iter()
            .enumerate()
            .map(|(t, rows)| {
                if t == 0 {
                    rows.clone()
                } else {
                    rows.iter().map(|f| f + 7.25).collect()
                }
            })
            .collect();
        let (a, b) = (p.total(&before), p.total(&after));
        assert!(a > 0.0);
        assert!((a - b).abs() < 1e-12, "{a} vs {b}");
    }

    #[test]
    fn the_total_is_on_the_deviance_scale() {
        // Twice the half-deviance value, so it adds to a deviance.
        let p = plan(1.0, 1.0);
        let factors = vec![vec![0.0], vec![0.0, 2.0]];
        assert!((p.total(&factors) - 2.0 * 2.0).abs() < 1e-12);
    }

    #[test]
    fn soft_threshold_holds_at_zero_inside_the_interval() {
        assert_eq!(soft_threshold(0.5, 1.0), 0.0);
        assert_eq!(soft_threshold(-1.0, 1.0), 0.0);
        assert_eq!(soft_threshold(1.5, 1.0), 0.5);
        assert_eq!(soft_threshold(-1.5, 1.0), -0.5);
    }
}
