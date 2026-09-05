# Monotonic fitting implementation notes

Status: connected to Plan, fitting, review and export. See
[usage and limits](MONOTONIC_EFFECTS.md). Continuous smooth terms remain separate work.

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
and log-sum-exp pooling. No observation-sized arrays are needed. Empty bands carry
zero statistics and extend adjacent supported values without affecting their fit.

The existing table update's sufficient statistics include the current table factor.
To recover the statistics above, use:

- `log(A_r) = log(numer_r) + (p-1)*beta_r`
- `log(B_r) = log(denom_r) + (p-2)*beta_r`

## Constrained optimality

Within each flat increasing block, feasible infinitesimal changes decrease a prefix
or increase a suffix. Scores are negative loss derivatives, so every prefix score
must be nonnegative and every suffix score nonpositive at the optimum. A strict
boundary separates blocks; a singleton therefore requires its ordinary score to
vanish. Decreasing constraints reverse the signs. The maximum positive directional
score, using the fitter's existing reference-scale normalization, measures convergence.
This avoids declaring failure merely because individual pooled-band scores are nonzero.

The table solver applies exact block updates and normalization. Monotonic tables never
enter unconstrained pair solves or global IRLS, and extrapolation is disabled for these
fits. Parameter-count-based inference is not computed; Plan structural dimensions are
not interpreted as effective degrees of freedom. Numerical guards are not added to
the statistical feasible cone. Penalties, row locks and constrained uncertainty remain
future work with explicit rejection/withholding today.

## Kernel evidence

The main kernel test enumerates 256 four-band actual vectors,
unequal expected weights, both directions and two coefficient-bound ranges. For
each case it independently enumerates every contiguous partition and checks that
the kernel attains the best feasible loss for powers 1, 1.5 and 2 (3,072 comparisons).
Additional tests distinguish pooled rates from sorted or averaged log factors and
check extreme log statistics, zero actuals, empty-band extension, directional scores
and invalid inputs. Integration tests cover weighted offsets, both normalizations,
multi-term fits against independent SLSQP, serialization and edited-order findings.
