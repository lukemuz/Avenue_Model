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

        Some(PenaltyPlan {
            penalty,
            penalised,
        })
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

    /// Whether this row is a penalised table's gauge, and so must be held still.
    ///
    /// **A penalised table's base level is pinned, not fitted.** This is the one place
    /// where switching a penalty on changes the shape of the problem rather than just
    /// its arithmetic, and getting it wrong makes the fit unsolvable rather than merely
    /// inaccurate.
    ///
    /// The model carries an intercept *and* a free factor for every level, one more
    /// parameter than the model has directions. Unpenalised that is harmless: the
    /// stationarity conditions are `g_r = 0` for every row, the intercept's condition is
    /// their sum, and a redundant condition among consistent ones costs nothing.
    ///
    /// Add a penalty on the contrasts and the redundancy turns into a contradiction.
    /// Every non-base level now wants `g_r = p_r`, the base level would still want
    /// `g_0 = 0`, and the intercept still wants the residuals to sum to zero - which
    /// forces `sum p_r = 0`, something no real penalty satisfies. The two conditions
    /// fight, and the fit converges to whichever the sweep happened to apply last. On a
    /// lasso strong enough to drop every level, the intercept came back holding the base
    /// level's own mean instead of the overall weighted mean, and tighter fits stopped
    /// converging at all because no point satisfied every condition at once.
    ///
    /// Holding the base level still removes the extra parameter and the extra condition
    /// together, leaving exactly the identified problem: an intercept and one contrast
    /// per remaining level. That is also, precisely, what one-hot dummy coding with a
    /// dropped reference gives every other library, which is why the two agree.
    #[inline]
    pub fn is_gauge(&self, table: usize, row: usize) -> bool {
        row == ANCHOR_ROW && self.covers(table)
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

    #[test]
    fn only_a_penalised_table_has_a_pinned_gauge() {
        let p = PenaltyPlan::new(0.1, 0.5, 1.0, &[1, 4, 5], &[false, false, true]).unwrap();
        assert!(p.is_gauge(1, 0), "a penalised table pins its base level");
        assert!(!p.is_gauge(1, 1));
        assert!(!p.is_gauge(0, 0), "the intercept is the one free constant");
        assert!(!p.is_gauge(2, 0), "an unpenalised table keeps every row free");
    }

    #[test]
    fn variate_and_single_row_tables_are_skipped() {
        let p = PenaltyPlan::new(0.1, 0.0, 1.0, &[1, 5, 1, 6], &[false, true, false, false])
            .unwrap();
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
