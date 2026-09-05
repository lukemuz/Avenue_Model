# Monotonic fitting implementation notes

Status: numerical kernel tested; no public monotonic Plan term yet. This is ongoing
work toward the smooth/constrained-effects requirement, not feature acceptance.

## Exact table update

For an ordered log-link table, hold the other terms fixed and write each mean as
`mu_i = q_i * exp(beta_r)`. With observation weight `w_i` and variance power
`1 <= p <= 2`, accumulate per band:

- `A_r = sum(w_i * y_i * q_i^(1-p))`
- `B_r = sum(w_i * q_i^(2-p))`

The derivative of half-deviance is
`B_r * exp((2-p)*beta_r) - A_r * exp((1-p)*beta_r)`.
It is increasing, and its root is `log(A_r/B_r)`. When adjacent roots violate the
declared order, pool their A and B statistics and repeat. This solves the convex
ordered subproblem exactly. Common finite coefficient bounds clip the pooled roots.
This applies to Poisson, Gamma and Tweedie without replacing their losses with a
least-squares approximation. For zero actuals the lower bound supplies the optimum.

`src/glm/isotonic.rs` implements this adjacent-block algorithm using log statistics
and log-sum-exp pooling. No observation-sized arrays are needed. The kernel requires
positive expected weight in each supplied band; empty-band behavior is deliberately
left to fitting integration. It is private and does not change existing fits.

The existing table update's sufficient statistics include the current table factor.
To recover the statistics above, use:

- `log(A_r) = log(numer_r) + (p-1)*beta_r`
- `log(B_r) = log(denom_r) + (p-2)*beta_r`

## Integration gates still open

1. Add an explicit ordered band term/direction to Plan, resolution and serialization;
   persist the declaration alongside the scoring factors and expose it in review.
2. Select a solver that honors the constraint. Joint updates and acceleration must
   preserve feasibility, including normalization and supplied initial factors.
3. Define empty-band and missing-value behavior. Row locks and penalties need either
   a correct constrained solve or explicit rejection before fitting.
4. Use constrained optimality checks: pooled active bounds can have nonzero individual
   scores at the optimum, so the current unconstrained score criterion is insufficient.
5. Withhold ordinary unconstrained coefficient inference with an explicit explanation
   until valid constrained uncertainty is implemented; audit effective parameter counts.
6. Verify multi-term fits against an independent constrained optimizer, known-shape
   recovery, both directions, weights/offsets, and exact export/reload predictions.
   Check edits that violate a declared constraint in the workbook review workflow.

## Kernel evidence

Three Rust tests pass. The main test enumerates 256 four-band actual vectors,
unequal expected weights, both directions and two coefficient-bound ranges. For
each case it independently enumerates every contiguous partition and checks that
the kernel attains the best feasible loss for powers 1, 1.5 and 2 (3,072 comparisons).
Additional tests distinguish pooled rates from sorted or averaged log factors and
check extreme log statistics, zero actuals and invalid inputs. This evidence covers
the numerical block update only, not the integration gates above.
