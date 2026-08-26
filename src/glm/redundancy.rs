//! How much of one rating table's information another already carries.
//!
//! Backfitting updates one table at a time, so its convergence rate is set by how much
//! the tables overlap. For two blocks of a design that rate has an exact name: the
//! **first canonical correlation** between their column spaces. At `rho = 0` the tables
//! are orthogonal and one sweep suffices; as `rho` approaches 1 the sweep spends its
//! passes trading a constant between them and converges arbitrarily slowly.
//!
//! For two step tables both blocks are indicator matrices, which makes that correlation
//! cheap: it is the largest singular value of the standardised weighted contingency
//! table between them — correspondence analysis, on a matrix with one row per level of
//! one table and one column per level of the other. No design matrix is involved and
//! nothing scales with the number of parameters squared.
//!
//! This measures the same thing twice over, which is why it is worth computing:
//!
//! * **As a diagnostic.** `Area` and `Density` in the French motor data are the same
//!   geography banded twice. An actuary wants to be told that, whatever the solver does
//!   about it, because a plan carrying both is a plan whose two geography tables cannot
//!   be interpreted separately.
//! * **As a solver decision.** A pair above the threshold is the pair worth updating
//!   jointly instead of one at a time.

use super::matching::NO_MATCH;
use rayon::prelude::*;

/// Above this first canonical correlation, two tables carry enough of the same
/// information that backfitting between them crawls.
///
/// The rate at which the shared direction decays is `rho^2` per sweep, so `0.9` already
/// means roughly 20 sweeps per decade of accuracy from that pair alone. Below it the
/// pair costs little enough that the joint solve would not repay its assembly.
pub const NEAR_ALIAS: f64 = 0.9;

/// Two tables and how much information they share.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TablePair {
    /// Index of the first table, always the lower of the two.
    pub first: usize,
    /// Index of the second table.
    pub second: usize,
    /// First canonical correlation between the two tables' indicator columns, in
    /// `[0, 1]`. This is the factor by which the shared direction survives each sweep.
    pub correlation: f64,
}

impl TablePair {
    /// Whether these two tables carry enough of the same information to be worth
    /// updating together. See [`NEAR_ALIAS`].
    pub fn is_near_aliased(&self) -> bool {
        self.correlation >= NEAR_ALIAS
    }
}

/// Rows sampled to estimate the correlations, when the data has more than this.
///
/// The first canonical correlation is a property of the joint distribution of two
/// tables' levels, not of the sample size, and the decision it feeds is a comparison
/// against a threshold of 0.9. Two hundred thousand rows settle that far more precisely
/// than it needs to be settled, and on a five-million-row fit this turns a pass that cost
/// most of a sweep into one that costs a twentieth of one.
///
/// Sampling can only ever mislead the *heuristic*: a pair wrongly grouped is solved
/// jointly and a pair wrongly missed is solved singly, and both produce the same fit.
/// Only the number of sweeps it takes can change.
const SAMPLE_ROWS: usize = 200_000;

/// The first canonical correlation between every pair of updatable step tables.
///
/// One pass over the observations builds every pair's contingency table at once —
/// `T(T+1)/2` scatter-adds per row — after which the correlations come from matrices
/// whose size is a table's level count, not the data's row count.
///
/// Tables that are locked, that hold a single row, or that are variates are skipped.
/// A one-row table is confounded with the intercept rather than with any other table,
/// and a variate's parameters are its polynomial coefficients rather than its rows, so
/// the indicator-matrix argument does not apply to it.
pub fn table_correlations(
    matches: &[Vec<u32>],
    weights: &[f64],
    shapes: &[usize],
    eligible: &[bool],
) -> Vec<TablePair> {
    let n_tables = shapes.len();
    let considered: Vec<usize> = (0..n_tables)
        .filter(|t| eligible[*t] && shapes[*t] > 1)
        .collect();
    if considered.len() < 2 {
        return Vec::new();
    }

    let pairs: Vec<(usize, usize)> = considered
        .iter()
        .enumerate()
        .flat_map(|(i, a)| considered[i + 1..].iter().map(move |b| (*a, *b)))
        .collect();

    let tables = contingency_tables(matches, weights, shapes, &pairs);

    let mut out: Vec<TablePair> = pairs
        .iter()
        .zip(tables.iter())
        .map(|((a, b), counts)| TablePair {
            first: *a,
            second: *b,
            correlation: first_canonical_correlation(counts, shapes[*a], shapes[*b]),
        })
        .collect();

    // Worst first: this is read as "which pair is the problem", and the answer is the
    // top row.
    out.sort_by(|x, y| y.correlation.total_cmp(&x.correlation));
    out
}

/// Weighted co-occurrence counts for every requested pair, in one pass over the data.
fn contingency_tables(
    matches: &[Vec<u32>],
    weights: &[f64],
    shapes: &[usize],
    pairs: &[(usize, usize)],
) -> Vec<Vec<f64>> {
    let sizes: Vec<usize> = pairs.iter().map(|(a, b)| shapes[*a] * shapes[*b]).collect();
    let fresh = || sizes.iter().map(|s| vec![0.0f64; *s]).collect::<Vec<_>>();

    let n = weights.len();
    let total: usize = sizes.iter().sum();
    let workers = rayon::current_num_threads().max(1);

    // Visit every `stride`th row rather than all of them. Regular spacing could in
    // principle line up with an ordering in the data; if it did, the cost is a
    // misjudged heuristic rather than a wrong answer - see `SAMPLE_ROWS`.
    let stride = (n / SAMPLE_ROWS).max(1);
    let sampled = n.div_ceil(stride);

    // The same trade as the fitting sweep: replicate the small per-pair tables across
    // workers only while that is cheap against the scan they save.
    if sampled < 100_000 || workers < 2 || total.saturating_mul(workers).saturating_mul(4) > sampled
    {
        let mut counts = fresh();
        accumulate(matches, weights, shapes, pairs, 0..n, stride, &mut counts);
        return counts;
    }

    let chunk = (n / workers).max(1);
    (0..n)
        .into_par_iter()
        .step_by(stride)
        .chunks(chunk.div_ceil(stride).max(1))
        .fold(fresh, |mut counts, idx| {
            for i in idx {
                accumulate_row(matches, weights, shapes, pairs, i, &mut counts);
            }
            counts
        })
        .reduce(fresh, |mut acc, other| {
            for (a, b) in acc.iter_mut().zip(other.iter()) {
                for (x, y) in a.iter_mut().zip(b.iter()) {
                    *x += y;
                }
            }
            acc
        })
}

fn accumulate(
    matches: &[Vec<u32>],
    weights: &[f64],
    shapes: &[usize],
    pairs: &[(usize, usize)],
    rows: std::ops::Range<usize>,
    stride: usize,
    counts: &mut [Vec<f64>],
) {
    for i in rows.step_by(stride) {
        accumulate_row(matches, weights, shapes, pairs, i, counts);
    }
}

#[inline]
fn accumulate_row(
    matches: &[Vec<u32>],
    weights: &[f64],
    shapes: &[usize],
    pairs: &[(usize, usize)],
    i: usize,
    counts: &mut [Vec<f64>],
) {
    let w = weights[i];
    if w == 0.0 {
        return;
    }
    for (p, (a, b)) in pairs.iter().enumerate() {
        let (ra, rb) = (matches[*a][i], matches[*b][i]);
        if ra == NO_MATCH || rb == NO_MATCH {
            continue;
        }
        counts[p][ra as usize * shapes[*b] + rb as usize] += w;
    }
}

/// Largest singular value of the standardised contingency table.
///
/// With `P` the table normalised to sum 1, and `r`, `c` its row and column masses, the
/// matrix `S = D_r^{-1/2} (P - r c') D_c^{-1/2}` has the canonical correlations as its
/// singular values. Subtracting `r c'` removes the trivial correlation of 1 that every
/// contingency table carries — the one saying both tables agree about which rows exist —
/// leaving the largest *informative* one on top.
///
/// Taken by power iteration on `S' S`, which needs only the dominant value and reaches
/// it in a few dozen small matrix-vector products.
fn first_canonical_correlation(counts: &[f64], k_a: usize, k_b: usize) -> f64 {
    let total: f64 = counts.iter().sum();
    if !(total > 0.0) {
        return 0.0;
    }

    let mut row_mass = vec![0.0f64; k_a];
    let mut col_mass = vec![0.0f64; k_b];
    for i in 0..k_a {
        for j in 0..k_b {
            let p = counts[i * k_b + j] / total;
            row_mass[i] += p;
            col_mass[j] += p;
        }
    }

    // S[i][j] = (p_ij - r_i c_j) / sqrt(r_i c_j). Empty levels contribute nothing and
    // would divide by zero, so they are left at zero.
    let mut s = vec![0.0f64; k_a * k_b];
    for i in 0..k_a {
        if row_mass[i] <= 0.0 {
            continue;
        }
        for j in 0..k_b {
            if col_mass[j] <= 0.0 {
                continue;
            }
            let expected = row_mass[i] * col_mass[j];
            s[i * k_b + j] = (counts[i * k_b + j] / total - expected) / expected.sqrt();
        }
    }

    power_iteration_top_singular_value(&s, k_a, k_b).clamp(0.0, 1.0)
}

/// Dominant singular value of a small dense matrix, by power iteration on `S' S`.
fn power_iteration_top_singular_value(s: &[f64], rows: usize, cols: usize) -> f64 {
    const ITERATIONS: usize = 200;
    const TOLERANCE: f64 = 1e-12;

    // A deterministic start that is not orthogonal to the dominant direction for any
    // realistic table; alternating signs avoid the symmetric-cancellation case that a
    // constant vector can hit.
    let mut v: Vec<f64> = (0..cols)
        .map(|j| if j % 2 == 0 { 1.0 } else { -1.0 })
        .collect();
    let mut sv = vec![0.0f64; rows];
    let mut next = vec![0.0f64; cols];
    let mut sigma = 0.0f64;

    normalise(&mut v);
    for _ in 0..ITERATIONS {
        // sv = S v
        for i in 0..rows {
            sv[i] = (0..cols).map(|j| s[i * cols + j] * v[j]).sum();
        }
        // next = S' (S v)
        for j in 0..cols {
            next[j] = (0..rows).map(|i| s[i * cols + j] * sv[i]).sum();
        }

        let norm = next.iter().map(|x| x * x).sum::<f64>().sqrt();
        if !(norm > 0.0) || !norm.is_finite() {
            return 0.0;
        }
        // The eigenvalue of S'S is sigma^2.
        let next_sigma = norm.sqrt();
        v.copy_from_slice(&next);
        normalise(&mut v);

        if (next_sigma - sigma).abs() <= TOLERANCE * next_sigma.max(1.0) {
            return next_sigma;
        }
        sigma = next_sigma;
    }
    sigma
}

fn normalise(v: &mut [f64]) {
    let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm > 0.0 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two tables that are the same partition relabelled carry identical information, so
    /// knowing one determines the other exactly and the correlation is 1.
    #[test]
    fn a_relabelling_is_perfectly_correlated() {
        let a: Vec<u32> = vec![0, 0, 1, 1, 2, 2, 0, 1, 2, 0];
        let b: Vec<u32> = a.iter().map(|v| 2 - *v).collect();
        let weights = vec![1.0; a.len()];

        let pairs = table_correlations(
            &[a, b],
            &weights,
            &[3, 3],
            &[true, true],
        );
        assert_eq!(pairs.len(), 1);
        assert!(
            (pairs[0].correlation - 1.0).abs() < 1e-9,
            "a relabelled partition should be perfectly correlated, got {}",
            pairs[0].correlation
        );
        assert!(pairs[0].is_near_aliased());
    }

    /// A coarsening is still perfectly predictable from the finer table - which is
    /// exactly what `Area` is to `Density` - even though the reverse is not true. The
    /// first canonical correlation is 1 whenever *either* direction is deterministic.
    #[test]
    fn a_coarsening_is_perfectly_correlated() {
        // Four fine levels collapsing onto two coarse ones.
        let fine: Vec<u32> = vec![0, 1, 2, 3, 0, 1, 2, 3, 1, 3];
        let coarse: Vec<u32> = fine.iter().map(|v| *v / 2).collect();
        let weights = vec![1.0; fine.len()];

        let pairs = table_correlations(&[fine, coarse], &weights, &[4, 2], &[true, true]);
        assert!(
            (pairs[0].correlation - 1.0).abs() < 1e-9,
            "a coarsening should be perfectly correlated, got {}",
            pairs[0].correlation
        );
    }

    /// A fully crossed design carries no shared information at all.
    #[test]
    fn a_crossed_design_is_uncorrelated() {
        // Every combination of 3 x 4 appears exactly once.
        let mut a = Vec::new();
        let mut b = Vec::new();
        for i in 0..3u32 {
            for j in 0..4u32 {
                a.push(i);
                b.push(j);
            }
        }
        let weights = vec![1.0; a.len()];

        let pairs = table_correlations(&[a, b], &weights, &[3, 4], &[true, true]);
        assert!(
            pairs[0].correlation < 1e-9,
            "a fully crossed design should be uncorrelated, got {}",
            pairs[0].correlation
        );
        assert!(!pairs[0].is_near_aliased());
    }

    /// Partial overlap has to land strictly between the two extremes, or the measure is
    /// only detecting the degenerate cases.
    #[test]
    fn partial_overlap_lands_in_between() {
        // b follows a on most rows and departs on the rest.
        let a: Vec<u32> = vec![0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1];
        let b: Vec<u32> = vec![0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0];
        let weights = vec![1.0; a.len()];

        let pairs = table_correlations(&[a, b], &weights, &[2, 2], &[true, true]);
        let rho = pairs[0].correlation;
        assert!(
            rho > 0.1 && rho < 0.9,
            "partial overlap should be strictly between 0 and 1, got {}",
            rho
        );
    }

    /// Weights are exposure: a pair that looks correlated only because of a handful of
    /// unweighted rows must not be reported as such.
    #[test]
    fn weights_drive_the_measure() {
        let a: Vec<u32> = vec![0, 1, 0, 1, 0, 1, 0, 1];
        // Agrees with `a` on the first four rows, disagrees on the rest.
        let b: Vec<u32> = vec![0, 1, 0, 1, 1, 0, 1, 0];

        let agree_heavy = vec![100.0, 100.0, 100.0, 100.0, 1.0, 1.0, 1.0, 1.0];
        let disagree_heavy = vec![1.0, 1.0, 1.0, 1.0, 100.0, 100.0, 100.0, 100.0];

        let hi = table_correlations(&[a.clone(), b.clone()], &agree_heavy, &[2, 2], &[true, true])
            [0].correlation;
        let lo = table_correlations(&[a, b], &disagree_heavy, &[2, 2], &[true, true])[0].correlation;

        // Both are lopsided the same way, just in opposite directions, so the magnitudes
        // match; what matters is that reweighting moves the measure at all.
        assert!(hi > 0.9, "heavily agreeing rows should read as correlated, got {}", hi);
        assert!(lo > 0.9, "heavily disagreeing rows are also predictable, got {}", lo);

        let balanced = vec![1.0; 8];
        let mid = table_correlations(&[vec![0, 1, 0, 1, 0, 1, 0, 1], vec![0, 1, 0, 1, 1, 0, 1, 0]],
                                     &balanced, &[2, 2], &[true, true])[0].correlation;
        assert!(mid < 1e-9, "balanced agreement and disagreement cancel, got {}", mid);
    }

    /// Locked tables, single-row tables and one-table models have no pair to report.
    #[test]
    fn ineligible_tables_are_skipped() {
        let a: Vec<u32> = vec![0, 1, 0, 1];
        let b: Vec<u32> = vec![0, 0, 1, 1];
        let intercept: Vec<u32> = vec![0, 0, 0, 0];
        let weights = vec![1.0; 4];

        // The intercept is a single row, so it pairs with nothing.
        let pairs = table_correlations(
            &[intercept.clone(), a.clone(), b.clone()],
            &weights,
            &[1, 2, 2],
            &[true, true, true],
        );
        assert_eq!(pairs.len(), 1, "only the two real tables should pair");
        assert_eq!((pairs[0].first, pairs[0].second), (1, 2));

        // Locking one of them leaves nothing to compare.
        let pairs = table_correlations(
            &[intercept, a, b],
            &weights,
            &[1, 2, 2],
            &[true, true, false],
        );
        assert!(pairs.is_empty(), "a locked table should not be paired");
    }
}
