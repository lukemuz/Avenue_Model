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
/// **Sampling is not harmless, and a pair that crosses [`NEAR_ALIAS`] is re-measured on
/// every row before it is acted on.** This comment used to claim the opposite - that a
/// pair wrongly grouped and a pair wrongly missed both produce the same fit, and only the
/// sweep count changes. The second half is true; the first is not. A wrongly grouped pair
/// is handed to [`update_pair`], which on the NYC taxi data diverged to fitted means
/// 3.2e21 out.
///
/// The error is one-directional, which is why a plain sample cannot be trusted here: a
/// contingency table with more cells than sampled rows is mostly empty, and
/// correspondence analysis on a table that sparse saturates at 1.0 regardless of the
/// data. The taxi `pickup_zone`/`dropoff_zone` pair - 252 by 261 levels, 65,772 cells -
/// reads 1.0000 from a 200k sample and 0.5788 from all 2.75M rows.
const SAMPLE_ROWS: usize = 200_000;

/// Power iterations for [`collective_strength`]. The matrix is tiny and the dominant
/// eigenvalue is well separated whenever the answer matters, so this is never the
/// binding cost; it exists only to bound a pathological input.
const POWER_ITERATIONS: usize = 500;

/// How strongly the tables share **one common direction**, across all of them at once.
///
/// The largest eigenvalue of the matrix of pairwise first canonical correlations, ones on
/// the diagonal. It runs from `1.0`, when no table says anything about any other, up to
/// `T`, when they are all the same table; a set that shares a single driver equally sits
/// at `1 + (T - 1) * rho`.
///
/// **This, rather than any pairwise figure, is what governs how many sweeps a correlated
/// plan needs.** A hundred tables at a pairwise 0.28 score 28.5 here and take about a
/// thousand sweeps to converge; five tables at that same pairwise 0.28 score 2.1 and take
/// fourteen. Nothing in the pairwise list tells those two apart — [`NEAR_ALIAS`] is a
/// long way above 0.28 either way — which is why a plan can be slow to fit with no pair
/// that looks the least bit suspicious.
///
/// Costs nothing worth measuring: the correlations are already in hand and this is a
/// power iteration on a matrix whose size is the number of tables, not the data.
pub fn collective_strength(pairs: &[TablePair]) -> f64 {
    if pairs.is_empty() {
        return 1.0;
    }

    // Compact the table indices these pairs refer to. Tables skipped as ineligible - the
    // intercept, offsets, variates - never appear in a pair and must not take up a row.
    let mut indices: Vec<usize> = pairs.iter().flat_map(|p| [p.first, p.second]).collect();
    indices.sort_unstable();
    indices.dedup();
    let t = indices.len();

    let mut m = vec![0.0f64; t * t];
    for i in 0..t {
        m[i * t + i] = 1.0;
    }
    for pair in pairs {
        let (Ok(i), Ok(j)) = (
            indices.binary_search(&pair.first),
            indices.binary_search(&pair.second),
        ) else {
            continue;
        };
        m[i * t + j] = pair.correlation;
        m[j * t + i] = pair.correlation;
    }

    // Power iteration. Every entry is a correlation and so non-negative, which by
    // Perron-Frobenius puts the dominant eigenvector in the positive orthant - a
    // positive start converges to it without any of the care a general matrix needs.
    let mut v = vec![1.0 / (t as f64).sqrt(); t];
    let mut next = vec![0.0f64; t];
    let mut lambda = 1.0;
    for _ in 0..POWER_ITERATIONS {
        for i in 0..t {
            next[i] = (0..t).map(|j| m[i * t + j] * v[j]).sum();
        }
        let norm = next.iter().map(|x| x * x).sum::<f64>().sqrt();
        if !(norm > 0.0) || !norm.is_finite() {
            break;
        }
        let previous = lambda;
        lambda = norm;
        for (slot, x) in v.iter_mut().zip(next.iter()) {
            *slot = x / norm;
        }
        if (lambda - previous).abs() <= 1e-12 * lambda {
            break;
        }
    }
    lambda
}

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

    // Visit every `stride`th row rather than all of them. Regular spacing could in
    // principle line up with an ordering in the data; if it did, the cost is a misjudged
    // heuristic rather than a wrong answer - see `SAMPLE_ROWS`.
    let stride = (matches[considered[0]].len() / SAMPLE_ROWS).max(1);
    let tables = contingency_tables(matches, weights, shapes, &pairs, stride);

    // Each pair's correspondence analysis is independent and reads only its own
    // contingency table, so the pairs go out to the pool. This is where the survey's
    // time goes once the tables are wide: the work per pair is `O(k_a * k_b)` per power
    // iteration, so a plan of 250-level tables spends longer deciding whether to use the
    // joint solve than it does fitting.
    let mut out: Vec<TablePair> = pairs
        .par_iter()
        .zip(tables.par_iter())
        .map(|((a, b), counts)| TablePair {
            first: *a,
            second: *b,
            correlation: first_canonical_correlation(counts, shapes[*a], shapes[*b]),
        })
        .collect();

    // **A pair that reads as near-aliased gets checked against every row before anyone
    // acts on it.** Sampling is fine for a survey but not for this decision, because it
    // fails in one direction only and that direction is the dangerous one: a contingency
    // table with more cells than sampled rows is mostly empty, and correspondence
    // analysis on a table that sparse saturates at 1.0 whatever the data says. On 2.75M
    // NYC taxi trips the `pickup_zone`/`dropoff_zone` pair - 252 by 261 levels, 65,772
    // cells against a 200k sample - reads **1.0000** sampled and **0.5788** from every
    // row. Two tables sharing nothing much were grouped on the strength of that, and the
    // joint solve diverged.
    //
    // Only the candidates are re-measured, and there are rarely more than a couple, so
    // this costs one pass over the data in the case where a pass is about to be worth
    // hundreds of sweeps either way.
    if stride > 1 {
        let suspects: Vec<(usize, usize)> = out
            .iter()
            .filter(|p| p.is_near_aliased())
            .map(|p| (p.first, p.second))
            .collect();
        if !suspects.is_empty() {
            let exact = contingency_tables(matches, weights, shapes, &suspects, 1);
            for ((a, b), counts) in suspects.iter().zip(exact.iter()) {
                let correlation = first_canonical_correlation(counts, shapes[*a], shapes[*b]);
                if let Some(pair) = out.iter_mut().find(|p| p.first == *a && p.second == *b) {
                    pair.correlation = correlation;
                }
            }
        }
    }

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
    stride: usize,
) -> Vec<Vec<f64>> {
    let sizes: Vec<usize> = pairs.iter().map(|(a, b)| shapes[*a] * shapes[*b]).collect();
    let fresh = || sizes.iter().map(|s| vec![0.0f64; *s]).collect::<Vec<_>>();

    let n = weights.len();
    let total: usize = sizes.iter().sum();
    let workers = rayon::current_num_threads().max(1);
    let stride = stride.max(1);
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

    /// Independent-looking codes, from a hash rather than modular arithmetic.
    ///
    /// `(c1 * i) % levels` and `(c2 * i) % levels` are *perfectly* aliased whenever `c1`
    /// is invertible mod `levels`, because `i` is then recoverable from the first - an
    /// easy way to write a test that proves nothing.
    fn spread(i: usize, salt: u64, levels: u32) -> u32 {
        let mut z = (i as u64)
            .wrapping_add(salt)
            .wrapping_add(0x9E37_79B9_7F4A_7C15);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        ((z ^ (z >> 31)) % levels as u64) as u32
    }

    /// The failure that made the verification pass necessary.
    ///
    /// One row pairs a level of each table that appears nowhere else, and a second row
    /// gives the first of those levels another partner. Seen whole, that is a correlation
    /// of about 0.71 - not aliased. Seen through a sample that keeps every other row, the
    /// second row disappears, the two rare levels occur only with each other, and
    /// correspondence analysis calls it 1.0.
    ///
    /// That is how the NYC taxi `pickup_zone`/`dropoff_zone` pair read 1.0000 against a
    /// true 0.5788 over all 2.75M rows. A pair read as aliased is handed to
    /// `update_pair`, which diverged on it - so this reading has to be the honest one.
    #[test]
    fn a_rare_pairing_lost_to_sampling_is_not_reported_as_aliased() {
        // Over SAMPLE_ROWS, so the survey samples and the verification pass has to undo
        // it: it keeps rows 0, 2, 4, ...
        let rows = 2 * SAMPLE_ROWS;
        let common = 200u32;
        let levels = common + 1; // the last level of each table is the rare one.

        let mut a: Vec<u32> = (0..rows).map(|i| spread(i, 0, common)).collect();
        let mut b: Vec<u32> = (0..rows).map(|i| spread(i, 0xABCD_EF01, common)).collect();

        a[0] = common;
        b[0] = common; // seen only with each other...
        a[1] = common;
        b[1] = 0; // ...until this row, which the sample drops.

        let weights = vec![1.0; rows];
        let shapes = vec![1, levels as usize, levels as usize];
        let matches = vec![vec![0u32; rows], a, b];
        let eligible = vec![false, true, true];

        let pairs = table_correlations(&matches, &weights, &shapes, &eligible);
        assert_eq!(pairs.len(), 1);
        assert!(
            !pairs[0].is_near_aliased(),
            "a pair that is not aliased over all rows was reported at rho = {:.4}",
            pairs[0].correlation
        );
    }

    fn equicorrelated(n_tables: usize, rho: f64) -> Vec<TablePair> {
        let mut pairs = Vec::new();
        for first in 1..=n_tables {
            for second in (first + 1)..=n_tables {
                pairs.push(TablePair {
                    first,
                    second,
                    correlation: rho,
                });
            }
        }
        pairs
    }

    /// The identity the diagnostic rests on: tables sharing one driver equally sit at
    /// `1 + (T - 1) * rho`, so the figure grows with the *number* of correlated tables
    /// while every pairwise correlation stays put. Measured on real fits, 5 tables at
    /// 0.28 converge in 14 sweeps and 100 tables at that same 0.28 take about 1100.
    #[test]
    fn a_shared_driver_scales_with_the_number_of_tables() {
        for n_tables in [2usize, 5, 25, 100] {
            let strength = collective_strength(&equicorrelated(n_tables, 0.28));
            let expected = 1.0 + (n_tables as f64 - 1.0) * 0.28;
            assert!(
                (strength - expected).abs() < 1e-9,
                "{} tables: got {}, expected {}",
                n_tables,
                strength,
                expected
            );
        }
    }

    /// Orthogonal tables share nothing and cost the sweep nothing.
    #[test]
    fn orthogonal_tables_sit_at_one() {
        assert!((collective_strength(&equicorrelated(50, 0.0)) - 1.0).abs() < 1e-9);
        assert_eq!(collective_strength(&[]), 1.0);
    }

    /// Perfectly aliased tables are one table repeated, and the figure says so.
    #[test]
    fn identical_tables_sit_at_the_table_count() {
        let strength = collective_strength(&equicorrelated(7, 1.0));
        assert!((strength - 7.0).abs() < 1e-6, "got {}", strength);
    }

    /// A single bad pair among otherwise unrelated tables must not read as a plan-wide
    /// problem - that is the case `update_pair` already handles.
    #[test]
    fn one_aliased_pair_does_not_look_like_a_shared_driver() {
        let mut pairs = equicorrelated(20, 0.0);
        for pair in pairs.iter_mut() {
            if pair.first == 1 && pair.second == 2 {
                pair.correlation = 0.99;
            }
        }
        let strength = collective_strength(&pairs);
        assert!(
            strength < 2.0,
            "one pair should stay near 1+rho, got {}",
            strength
        );
    }

    /// Two tables that are the same partition relabelled carry identical information, so
    /// knowing one determines the other exactly and the correlation is 1.
    #[test]
    fn a_relabelling_is_perfectly_correlated() {
        let a: Vec<u32> = vec![0, 0, 1, 1, 2, 2, 0, 1, 2, 0];
        let b: Vec<u32> = a.iter().map(|v| 2 - *v).collect();
        let weights = vec![1.0; a.len()];

        let pairs = table_correlations(&[a, b], &weights, &[3, 3], &[true, true]);
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

        let hi = table_correlations(
            &[a.clone(), b.clone()],
            &agree_heavy,
            &[2, 2],
            &[true, true],
        )[0]
        .correlation;
        let lo =
            table_correlations(&[a, b], &disagree_heavy, &[2, 2], &[true, true])[0].correlation;

        // Both are lopsided the same way, just in opposite directions, so the magnitudes
        // match; what matters is that reweighting moves the measure at all.
        assert!(
            hi > 0.9,
            "heavily agreeing rows should read as correlated, got {}",
            hi
        );
        assert!(
            lo > 0.9,
            "heavily disagreeing rows are also predictable, got {}",
            lo
        );

        let balanced = vec![1.0; 8];
        let mid = table_correlations(
            &[vec![0, 1, 0, 1, 0, 1, 0, 1], vec![0, 1, 0, 1, 1, 0, 1, 0]],
            &balanced,
            &[2, 2],
            &[true, true],
        )[0]
        .correlation;
        assert!(
            mid < 1e-9,
            "balanced agreement and disagreement cancel, got {}",
            mid
        );
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
