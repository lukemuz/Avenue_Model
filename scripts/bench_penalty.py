"""What a penalty costs Avenue, and what it costs glum.

Two claims are under test.

**Regularisation is close to free here.** A penalty enters only the `O(n_rows)`
arithmetic between the two `O(n)` passes over the data, so the per-sweep cost should be
unmeasurable. Sweep counts do move — a penalty changes the problem — so the honest
measure is time per sweep, reported alongside the total.

**L1 is where the two engines diverge structurally.** A ridge is a diagonal addition for
both: glum adds it to a `p x p` matrix, Avenue adds it to a scalar. A lasso is not.
glum must abandon its Cholesky factorisation and switch to coordinate descent over the
whole design; Avenue's algorithm *is* coordinate descent, so a soft threshold replaces a
division and nothing else changes.

The comparison is only meaningful because the two solve the same problem: verified
coefficient-for-coefficient by `check_penalty.py`, including which levels get zeroed.

Usage:  python scripts/bench_penalty.py [--dataset fremtpl] [--repeats 3]
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import polars as pl

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from bench_memory import PeakMemory

MAX_ITER = 2000
AVENUE_TOL = 1e-9
GLUM_TOL = 1e-9


# ------------------------------------------------------------------ data

def load_fremtpl():
    from bench_fremtpl import load_fremtpl as load, prepare
    codes, levels, y, exposure = prepare(load(), wide=False)
    return codes, levels, y.astype(np.float64), exposure, "poisson", "poisson"


def load_census():
    from bench_real import load_census as load, prepare_census
    codes, levels, y, _ = prepare_census(load())
    return codes, levels, y.astype(np.float64), None, "binary", "binomial"


def load_correlated(rows=200_000, tables=25, rho=0.28, seed=7):
    """The conditioning cliff: many tables sharing one direction.

    This is the case the README calls the wrong tool for — `table_conditioning` climbs
    with the table count and an unpenalised backfit needs hundreds of sweeps. A ridge
    bounds the coordinate-descent contraction factor, so it should help here, and that
    would make penalisation a performance feature rather than only a modelling one.
    """
    rng = np.random.default_rng(seed)
    k = 8
    shared = rng.normal(size=rows)
    codes, levels = {}, {}
    eta = np.zeros(rows)
    for t in range(tables):
        z = rho * shared + np.sqrt(1.0 - rho * rho) * rng.normal(size=rows)
        c = np.clip(((z + 3.0) / 6.0 * k).astype(np.int32), 0, k - 1)
        name = f"f{t}"
        codes[name] = c
        levels[name] = k
        eta += np.linspace(-0.15, 0.15, k)[c]
    y = rng.poisson(np.exp(eta + 0.5)).astype(np.float64)
    return codes, levels, y, None, "poisson", "poisson"


def load_housing():
    from bench_housing import load_housing as load, prepare
    codes, levels, price = prepare(load(), None)
    return codes, levels, price.astype(np.float64), None, "gamma", "gamma"


def load_taxi():
    from bench_real import load_taxi as load, prepare_taxi
    codes, levels, y, _ = prepare_taxi(load())
    return codes, levels, y.astype(np.float64), None, "gamma", "gamma"


DATASETS = {
    "fremtpl": ("freMTPL2 / claim count", load_fremtpl),
    "census": ("census_income / >50k", load_census),
    "housing": ("house_sales / price", load_housing),
    "taxi": ("nyc_taxi / fare", load_taxi),
    "correlated": ("25 correlated tables / synthetic", load_correlated),
}


# ------------------------------------------------------------------ engines

def timed(fn, repeats):
    """Best of `repeats`, with the peak allocation of one untimed run alongside."""
    best, payload = float("inf"), None
    for _ in range(repeats):
        start = time.perf_counter()
        payload = fn()
        best = min(best, time.perf_counter() - start)
    with PeakMemory() as memory:
        fn()
    return best, memory.peak_mb, payload


def run_avenue(codes, levels, y, weight, family, alpha, l1_ratio, repeats,
               solver="table"):
    from avenue_model import RatingModel, fit_glm_with_diagnostics, GLMOptions

    frame = {n: c.astype(np.int32) for n, c in codes.items()}
    frame["y"] = y
    if weight is not None:
        frame["w"] = weight
    df = pl.DataFrame(frame)
    tables = [pl.DataFrame({"Rating_Factor": [0.0]})]
    for name, k in levels.items():
        tables.append(pl.DataFrame({
            name: np.arange(k, dtype=np.int32),
            "Rating_Factor": np.zeros(k),
        }))
    model = RatingModel(tables, family)
    options = GLMOptions(
        max_iterations=MAX_ITER, tolerance=AVENUE_TOL,
        alpha=alpha, l1_ratio=l1_ratio, compute_standard_errors=False,
        solver=solver,
    )

    def fit():
        return fit_glm_with_diagnostics(
            model, df, "y", weight_col=("w" if weight is not None else None),
            options=options)

    seconds, peak, result = timed(fit, repeats)
    diag = result.diagnostics
    return dict(engine=f"avenue[{solver}]", seconds=seconds, peak=peak,
                iters=diag.iterations, converged=diag.converged)


def run_glum(codes, y, weight, glum_family, alpha, l1_ratio, solver, repeats):
    import glum

    X = pd.DataFrame({n: pd.Categorical(c) for n, c in codes.items()})

    def fit():
        m = glum.GeneralizedLinearRegressor(
            family=glum_family, alpha=alpha, l1_ratio=l1_ratio,
            fit_intercept=True, max_iter=MAX_ITER, gradient_tol=GLUM_TOL,
            drop_first=True, solver=solver, scale_predictors=False)
        m.fit(X, y, sample_weight=weight)
        return m

    try:
        seconds, peak, model = timed(fit, repeats)
    except Exception as exc:  # a solver that cannot take this penalty
        return dict(engine=f"glum[{solver}]", seconds=float("nan"),
                    peak=float("nan"), iters=0, converged=False,
                    note=type(exc).__name__)
    return dict(engine=f"glum[{solver}]", seconds=seconds, peak=peak,
                iters=int(getattr(model, "n_iter_", 0)), converged=True)


# ------------------------------------------------------------------ report

def report(rows):
    print(f"  {'engine':<16} {'penalty':<14} {'fit s':>9} {'peak MB':>9} "
          f"{'sweeps':>7} {'ms/sweep':>9}")
    print("  " + "-" * 70)
    for label, r in rows:
        note = "" if r.get("converged", True) else "  (not converged)"
        if r.get("note"):
            note = f"  ({r['note']})"
        per = (1000.0 * r["seconds"] / r["iters"]) if r["iters"] else float("nan")
        print(f"  {r['engine']:<16} {label:<14} {r['seconds']:>9.3f} "
              f"{r['peak']:>9.1f} {r['iters']:>7} {per:>9.2f}{note}")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="fremtpl",
                    choices=list(DATASETS) + ["all"])
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--alpha", type=float, default=1e-4)
    ap.add_argument("--engine", default="all",
                    choices=["all", "table", "global", "glum"],
                    help="Run one engine for cleaner process-isolated memory readings")
    args = ap.parse_args()

    real = ["fremtpl", "census", "housing", "taxi"]
    names = real if args.dataset == "all" else [args.dataset]
    for name in names:
        label, loader = DATASETS[name]
        codes, levels, y, weight, family, glum_family = loader()
        n_params = 1 + sum(k - 1 for k in levels.values())
        print(f"\n{label}: {len(y):,} rows, {len(levels)} tables, "
              f"{n_params} parameters, alpha = {args.alpha:g}\n")

        rows = []
        for pen_label, alpha, ratio in [
            ("none", 0.0, 0.0),
            ("ridge", args.alpha, 0.0),
            ("elastic-net", args.alpha, 0.5),
            ("lasso", args.alpha, 1.0),
        ]:
            avenue_solvers = (["table", "global"] if args.engine == "all"
                               else [args.engine] if args.engine in {"table", "global"}
                               else [])
            for solver in avenue_solvers:
                rows.append((pen_label, run_avenue(
                    codes, levels, y, weight, family, alpha, ratio, args.repeats,
                    solver=solver)))
            # glum's Cholesky path cannot take an L1 term; it falls back to
            # coordinate descent, which is the point of the comparison.
            if args.engine in {"all", "glum"}:
                solver = "irls-cd" if ratio > 0.0 else "irls-ls"
                rows.append((pen_label, run_glum(
                    codes, y, weight, glum_family, alpha, ratio, solver,
                    args.repeats)))
        report(rows)

    return 0


if __name__ == "__main__":
    sys.exit(main())
