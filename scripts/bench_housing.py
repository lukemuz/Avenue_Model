"""Benchmark Avenue against glum on the other real dataset in glum's own suite.

`bench_fremtpl.py` fits the French motor data, which glum's `wide-insurance` problem is
built from. That dataset turned out to be close to a worst case for coordinate descent:
`Area` is a six-band rebanding of `Density`, and that single near-duplicate pair cost 12x
the sweeps on its own.

glum's suite contains one real dataset that is not motor insurance — `intermediate-housing`,
from OpenML's `house_sales` (King County, WA). It is a useful third point because its
features are *correlated but not aliased*: the strongest pair is `sqft_living` against
`grade` at 0.76, where the insurance pair was effectively deterministic. If the sweep
count tracks that difference, correlation strength is the thing that governs cost and the
French motor result is a tail case rather than the typical one.

Two deliberate departures from glum's own problem definition, both so that the two
engines solve the *same* problem and the comparison measures solvers rather than models:

* glum fits the ten drivers as raw numeric columns. Here they are banded, and both
  engines are given the banded design. Banding is what a rating plan does, and it is also
  the only way Avenue's tables and glum's one-hot columns describe the same model.
* glum's housing problems carry `alpha = 0.001`. Avenue has no regularisation yet, so
  both are fitted unpenalised.

Usage:
    python scripts/bench_housing.py
    python scripts/bench_housing.py --diagnose    # convergence rate and drop-one
    python scripts/bench_housing.py --rows 1000000

The dataset is downloaded once from OpenML and cached next to this script.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_memory import measured  # noqa: E402

CACHE = os.path.join(os.path.dirname(__file__), ".house_sales.parquet")

MAX_ITER = 5000
AVENUE_TOL = 1e-10
GLUM_TOL = 1e-10
AGREEMENT_TOL = 1e-7
SEED = 20260826

# glum's `intermediate-housing` feature set, verbatim.
FEATURES = [
    "bedrooms", "bathrooms", "sqft_living", "floors", "waterfront",
    "view", "condition", "grade", "yr_built", "yr_renovated",
]


def load_housing() -> pd.DataFrame:
    """OpenML `house_sales`, cached locally after the first download."""
    if os.path.exists(CACHE):
        return pd.read_parquet(CACHE)

    from sklearn.datasets import fetch_openml

    print("Downloading house_sales from OpenML (once)...")
    bunch = fetch_openml(name="house_sales", version=1, as_frame=True, parser="auto")
    df = bunch.frame
    df.to_parquet(CACHE)
    print(f"Cached to {CACHE}")
    return df


def prepare(df: pd.DataFrame, rows: int | None) -> tuple[dict[str, np.ndarray], dict[str, int], np.ndarray]:
    """Bands each driver into rating-table levels.

    Low-cardinality drivers keep their own values as levels; the continuous ones are cut
    at quantiles, which is how a rating plan bands a driver with no natural breaks.
    """
    df = df.copy()
    if rows is not None and rows != len(df):
        # glum oversamples with replacement when a problem is asked for more rows than
        # the dataset has; this follows that so the row count can be swept.
        rng = np.random.default_rng(SEED)
        df = df.iloc[rng.integers(0, len(df), size=rows)].reset_index(drop=True)

    price = pd.to_numeric(df["price"]).to_numpy(dtype=float)

    quantiles = np.linspace(0, 1, 17)[1:-1]

    codes: dict[str, np.ndarray] = {}
    for name in FEATURES:
        v = pd.to_numeric(df[name], errors="coerce").to_numpy(dtype=float)
        values, counts = np.unique(v, return_counts=True)

        if values.size <= 16:
            level = pd.Categorical(v).codes.astype(np.int64)
        elif counts.max() > 0.5 * len(v):
            # A driver can pile most of its mass on one value: `yr_renovated` is 0 for
            # the 96% of houses never renovated. Every quantile edge then lands on that
            # value and the driver bands into a single level, which is a table wholly
            # confounded with the intercept - not a rating factor at all. Give the
            # dominant value its own level and band what is left around it.
            dominant = values[counts.argmax()]
            rest = v[v != dominant]
            edges = np.unique(np.nanquantile(rest, quantiles))
            level = np.digitize(v, edges) + 1
            level[v == dominant] = 0
        else:
            edges = np.unique(np.nanquantile(v, quantiles))
            level = np.digitize(v, edges)
        codes[name] = np.asarray(level, dtype=np.int64)

    # Compact to 0..k-1 so every table row sees exposure.
    levels: dict[str, int] = {}
    for name, v in codes.items():
        uniques, compacted = np.unique(v, return_inverse=True)
        codes[name] = compacted.astype(np.int64)
        levels[name] = len(uniques)

    return codes, levels, price


def build_frame(codes, levels, y):
    frame = {n: c.astype(np.float64) for n, c in codes.items()}
    frame["y"] = y
    tables = [pl.DataFrame({"Rating_Factor": [0.0]})]
    for name, k in levels.items():
        tables.append(pl.DataFrame({
            name: np.arange(k, dtype=np.float64),
            "Rating_Factor": np.zeros(k),
        }))
    return pl.DataFrame(frame), tables


def best_of(fn, repeats: int):
    best, payload = float("inf"), None
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn()
        elapsed = time.perf_counter() - t0
        if elapsed < best:
            best, payload = elapsed, out
    return best, payload


def run_avenue(codes, levels, y, family, repeats, accelerate=True):
    from avenue_model import RatingModel, fit_glm_with_diagnostics, GLMOptions

    prep_seconds, (df, tables) = best_of(lambda: build_frame(codes, levels, y), repeats)
    model = RatingModel(tables, family)
    options = GLMOptions(max_iterations=MAX_ITER, tolerance=AVENUE_TOL,
                         compute_standard_errors=False, accelerate=accelerate)

    fit_seconds, result = best_of(
        lambda: fit_glm_with_diagnostics(model, df, "y", options=options), repeats)
    fitted, diag = result.model, result.diagnostics

    note = f"max|score|={diag.max_gradient:.1e}"
    if accelerate:
        note += f"  jumps={diag.accelerated_steps}"
    return dict(
        engine="avenue" if accelerate else "avenue[plain]",
        prep=prep_seconds, fit=fit_seconds, iters=diag.iterations,
        converged=diag.converged, mu=fitted.predict(df).to_series(0).to_numpy(),
        note=note,
    )


def run_glum(codes, y, family, repeats, solver):
    import glum

    prep_seconds, X = best_of(
        lambda: pd.DataFrame({n: pd.Categorical(c) for n, c in codes.items()}), repeats)

    def fit():
        m = glum.GeneralizedLinearRegressor(
            family=family, alpha=0.0, fit_intercept=True, max_iter=MAX_ITER,
            gradient_tol=GLUM_TOL, drop_first=True, solver=solver,
        )
        m.fit(X, y)
        return m

    fit_seconds, model = best_of(fit, repeats)
    return dict(
        engine=f"glum[{solver}]", prep=prep_seconds, fit=fit_seconds,
        iters=int(getattr(model, "n_iter_", 0)), converged=True,
        mu=model.predict(X), note=None,
    )


def run_statsmodels(codes, levels, y, family, repeats):
    """The independent oracle. Only affordable because this dataset is small.

    statsmodels densifies the design matrix and takes a pseudo-inverse of it on every
    IRLS iteration. At 21,613 rows that is cheap enough to run, which makes this the one
    place a third implementation can arbitrate between Avenue and glum on real data.
    """
    import statsmodels.api as sm

    families = {
        "gamma": sm.families.Gamma(link=sm.families.links.Log()),
        "gaussian": sm.families.Gaussian(),
    }

    def prep():
        blocks = [np.ones((len(y), 1))]
        for name, k in levels.items():
            c = codes[name]
            block = np.zeros((len(y), k - 1))
            mask = c > 0
            block[np.flatnonzero(mask), c[mask] - 1] = 1.0
            blocks.append(block)
        return np.hstack(blocks)

    prep_seconds, X = best_of(prep, repeats)

    def fit():
        return sm.GLM(y, X, family=families[family]).fit(maxiter=MAX_ITER, tol=1e-10)

    fit_seconds, result = best_of(fit, repeats)
    return dict(
        engine="statsmodels", prep=prep_seconds, fit=fit_seconds,
        iters=int(result.fit_history["iteration"]), converged=bool(result.converged),
        mu=np.asarray(result.fittedvalues, dtype=float), note=None,
    )


def report(label, n_rows, n_params, results):
    print(f"\n  {label}  ({n_rows:,} rows, {n_params:,} parameters)")
    print(f"  {'engine':<16}{'prep':>9}{'fit':>9}{'total':>9}{'peak MB':>10}{'iters':>7}"
          f"{'vs reference':>15}")
    print(f"  {'-' * 76}")

    # statsmodels first where it ran: a third implementation is the only reference that
    # is neither of the two engines being compared.
    reference = next((r for r in results if r["engine"] == "statsmodels"), None)
    if reference is None:
        reference = next(
            (r for r in results if r["engine"].startswith("glum")), results[0])
    rms = float(np.sqrt(np.mean(reference["mu"] ** 2)))

    problems = []
    for r in results:
        d = float(np.max(np.abs(r["mu"] - reference["mu"])) / rms)
        r["disagreement"] = None if r is reference else d
        agreement = "reference" if r is reference else f"{d:.1e}"
        flag = ""
        if r is not reference and d > AGREEMENT_TOL:
            flag = "  DISAGREES"
            problems.append(f"{label}: {r['engine']} disagrees by {d:.2e}")
        if not r["converged"]:
            flag += "  DID NOT CONVERGE"
            problems.append(f"{label}: {r['engine']} did not converge")
        if r["note"]:
            flag += f"  [{r['note']}]"
        memory = f"{r['peak_mb']:.0f}" if r.get("peak_mb") is not None else "-"
        print(f"  {r['engine']:<16}{r['prep']:>9.3f}{r['fit']:>9.3f}"
              f"{r['prep'] + r['fit']:>9.3f}{memory:>10}{r['iters']:>7}"
              f"{agreement:>15}{flag}")
    return problems


def to_json(family, n_rows, n_params, results):
    return {
        "dataset": "house_sales",
        "variant": "banded",
        "family": family,
        "n_rows": n_rows,
        "n_parameters": n_params,
        "engines": [{k: v for k, v in r.items() if k != "mu"} for r in results],
    }


def tail_rate(history) -> float:
    """Geometric rate of the gradient over the last half of the fit."""
    h = np.asarray(history)
    seg = h[len(h) // 2:]
    seg = seg[seg > 0]
    if len(seg) < 5:
        return float("nan")
    return float(np.exp(np.polyfit(np.arange(len(seg)), np.log(seg), 1)[0]))


def diagnose(codes, levels, y, family):
    """How fast does the sweep converge, and which tables are responsible?

    The same analysis that found `Area` and `Density` on the French motor data. A drop-one
    sweep is the cheapest way to tell a model that is merely correlated from one that
    carries a near-duplicate pair.
    """
    from avenue_model import RatingModel, fit_glm_with_diagnostics, GLMOptions

    def fit(drop=()):
        c = {k: v for k, v in codes.items() if k not in drop}
        l = {k: v for k, v in levels.items() if k not in drop}
        df, tables = build_frame(c, l, y)
        options = GLMOptions(max_iterations=MAX_ITER,
                             tolerance=AVENUE_TOL, compute_standard_errors=False,
                             accelerate=False)
        d = fit_glm_with_diagnostics(RatingModel(tables, family), df, "y", options=options).diagnostics
        return d

    print(f"\n  convergence shape, {family}, unaccelerated")
    full = fit()
    rho = tail_rate(full.gradient_history)
    decade = np.log(10) / -np.log(rho) if rho < 1 else float("inf")
    print(f"    full model: {full.iterations} sweeps, converged={full.converged}, "
          f"tail rho={rho:.4f} ({decade:.1f} sweeps per decade)")

    print("    drop-one:")
    for name in FEATURES:
        d = fit(drop=(name,))
        print(f"      without {name:<14} {d.iterations:5d} sweeps   "
              f"rho={tail_rate(d.gradient_history):.4f}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--rows", type=int, default=None,
                        help="oversample to this many rows, as glum's suite does")
    parser.add_argument("--diagnose", action="store_true",
                        help="report the convergence rate and a drop-one table sweep")
    parser.add_argument("--json", type=str, default=None)
    args = parser.parse_args()

    raw = load_housing()
    print(f"\nhouse_sales: {len(raw):,} sales, {len(FEATURES)} drivers")

    codes, levels, y = prepare(raw, args.rows)
    n_params = 1 + sum(k - 1 for k in levels.values())
    print(f"banded to {n_params} parameters: "
          + ", ".join(f"{n}={k}" for n, k in levels.items()))

    if args.diagnose:
        diagnose(codes, levels, y, "gamma")
        return 0

    problems = []
    collected = []
    for family in ("gamma", "gaussian"):
        results = [
            measured(lambda: run_avenue(codes, levels, y, family, args.repeats,
                                        accelerate=True)),
            measured(lambda: run_avenue(codes, levels, y, family, args.repeats,
                                        accelerate=False)),
        ]
        for solver in ("irls-ls", "irls-cd"):
            try:
                results.append(measured(
                    lambda s=solver: run_glum(codes, y, family, args.repeats, s)))
            except Exception as exc:
                print(f"  glum[{solver}] failed: {type(exc).__name__}: {exc}")
        try:
            results.append(measured(
                lambda: run_statsmodels(codes, levels, y, family, args.repeats)))
        except Exception as exc:
            print(f"  statsmodels failed: {type(exc).__name__}: {exc}")
        problems += report(f"{family} / price", len(y), n_params, results)
        collected.append(to_json(family, len(y), n_params, results))

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(collected, fh, indent=2)
        print(f"\nWrote {args.json}")

    if problems:
        print("\nPROBLEMS:")
        for p in problems:
            print(f"  {p}")
        return 1

    print("\nAll engines agreed on fitted means.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
