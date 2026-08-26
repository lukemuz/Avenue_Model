"""Benchmark Avenue against glum on the dataset glum benchmarks itself.

`bench_glm.py` uses synthetic data, which I designed - and a benchmark whose author
also designed the data is worth distrusting. Two specific biases it carries:
independently drawn rating factors give a near-orthogonal design, which is the best
case for coordinate descent; and the response is drawn from exactly the model being
fitted.

The French Motor Third-Party Liability dataset (freMTPL2freq) removes both. It is the
standard public dataset for insurance GLM work, it is what glum's own `wide-insurance`
benchmark is built from, and its rating factors are correlated the way real ones are -
driver age with bonus-malus, vehicle age with power, density with region.

The feature engineering follows glum's French motor tutorial: cap claim counts and
exposure, then band the continuous drivers. Banding is not a concession made for
Avenue's benefit; it is what a rating plan does, and both engines fit the same banded
design.

Usage:
    python scripts/bench_fremtpl.py            # tutorial-scale and wide variants
    python scripts/bench_fremtpl.py --solver-sweep   # also test glum's other solvers

The dataset is downloaded once from OpenML and cached next to this script.
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import pandas as pd
import polars as pl

CACHE = os.path.join(os.path.dirname(__file__), ".fremtpl2freq.parquet")

MAX_ITER = 500
# Avenue's tolerance is now on the score, like glum's, so these mean roughly the same
# thing. Avenue scales its score by the total absolute residual to keep the threshold
# independent of the response's units; glum's is absolute on a weight-normalised
# objective. Close enough that the agreement check can arbitrate.
AVENUE_TOL = 1e-10
GLUM_TOL = 1e-10
AGREEMENT_TOL = 1e-7


def load_fremtpl() -> pd.DataFrame:
    """freMTPL2freq, cached locally after the first download."""
    if os.path.exists(CACHE):
        return pd.read_parquet(CACHE)

    from sklearn.datasets import fetch_openml

    print("Downloading freMTPL2freq from OpenML (once; ~55 MB)...")
    bunch = fetch_openml(data_id=41214, as_frame=True, parser="auto")
    df = bunch.frame
    df.to_parquet(CACHE)
    print(f"Cached to {CACHE}")
    return df


def prepare(df: pd.DataFrame, wide: bool) -> tuple[dict[str, np.ndarray], dict[str, int], np.ndarray, np.ndarray]:
    """Bands and codes, following glum's tutorial preprocessing.

    Returns (codes, levels, claim_count, exposure). `wide` bands the continuous
    drivers far more finely, which is how a real plan with credible geography and
    a full bonus-malus scale ends up several hundred parameters wide.
    """
    df = df.copy()

    # The tutorial's caps: a handful of extreme records otherwise dominate.
    df["ClaimNb"] = df["ClaimNb"].astype(float).clip(upper=4)
    df["Exposure"] = df["Exposure"].astype(float).clip(lower=1e-3, upper=1)

    def band(series: pd.Series, edges) -> np.ndarray:
        return np.digitize(series.to_numpy(dtype=float), edges)

    def categorical(series: pd.Series) -> np.ndarray:
        return pd.Categorical(series).codes.astype(np.int64)

    if wide:
        veh_age = band(df["VehAge"], np.arange(0, 31, 1))
        driv_age = band(df["DrivAge"], np.arange(18, 91, 1))
        bonus = band(df["BonusMalus"], np.arange(50, 231, 2))
        density = band(np.log(df["Density"].astype(float)), np.linspace(0, 11, 60))
        power = band(df["VehPower"], np.arange(4, 16, 1))
    else:
        veh_age = band(df["VehAge"], [1, 2, 3, 5, 7, 10, 15, 20])
        driv_age = band(df["DrivAge"], [21, 26, 31, 41, 51, 61, 71, 81])
        bonus = band(df["BonusMalus"], [51, 55, 60, 70, 80, 90, 100, 120, 150])
        density = band(np.log(df["Density"].astype(float)), [2, 3, 4, 5, 6, 7, 8, 9, 10])
        power = band(df["VehPower"], [5, 6, 7, 8, 9, 10, 12])

    codes = {
        "veh_age": veh_age,
        "driv_age": driv_age,
        "bonus_malus": bonus,
        "density": density,
        "veh_power": power,
        "area": categorical(df["Area"]),
        "veh_brand": categorical(df["VehBrand"]),
        "veh_gas": categorical(df["VehGas"]),
        "region": categorical(df["Region"]),
    }

    # Compact each factor's codes to 0..k-1 so a table row exists for every value
    # and no row is left with zero exposure.
    levels = {}
    for name, values in codes.items():
        uniques, compacted = np.unique(values, return_inverse=True)
        codes[name] = compacted.astype(np.int64)
        levels[name] = len(uniques)

    return (
        codes,
        levels,
        df["ClaimNb"].to_numpy(dtype=float),
        df["Exposure"].to_numpy(dtype=float),
    )


def best_of(fn, repeats: int):
    best, payload = float("inf"), None
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn()
        elapsed = time.perf_counter() - t0
        if elapsed < best:
            best, payload = elapsed, out
    return best, payload


def run_avenue(codes, levels, y, exposure, repeats, standard_errors):
    from avenue_model import RatingModel, fit_glm_with_diagnostics, GLMOptions

    log_exposure = np.log(exposure)

    def prep():
        frame = {n: c.astype(np.float64) for n, c in codes.items()}
        frame["y"] = y
        frame["log_exposure"] = log_exposure
        tables = [pl.DataFrame({"Rating_Factor": [0.0]})]
        for name, k in levels.items():
            tables.append(pl.DataFrame({
                name: np.arange(k, dtype=np.float64),
                "Rating_Factor": np.zeros(k),
            }))
        return pl.DataFrame(frame), RatingModel(tables, "poisson")

    prep_seconds, (df, model) = best_of(prep, repeats)

    options = GLMOptions(
        objective="poisson", max_iterations=MAX_ITER, tolerance=AVENUE_TOL,
        compute_standard_errors=standard_errors,
    )

    def fit():
        return fit_glm_with_diagnostics(
            model, df, "y", offset_col="log_exposure", options=options)

    fit_seconds, (fitted, diag) = best_of(fit, repeats)
    mu = fitted.predict(df).to_series(0).to_numpy() * exposure

    note = f"max|score|={diag.max_gradient:.1e}"
    if standard_errors and diag.inference_error is not None:
        note += f"  no SEs: {diag.inference_error[:60]}"

    return dict(
        engine="avenue+se" if standard_errors else "avenue",
        prep=prep_seconds, fit=fit_seconds, iters=diag.iterations,
        converged=diag.converged, mu=mu, note=note,
    )


def run_glum(codes, y, exposure, repeats, solver):
    import glum

    log_exposure = np.log(exposure)

    def prep():
        return pd.DataFrame({n: pd.Categorical(c) for n, c in codes.items()})

    prep_seconds, X = best_of(prep, repeats)

    def fit():
        m = glum.GeneralizedLinearRegressor(
            family="poisson", alpha=0.0, fit_intercept=True, max_iter=MAX_ITER,
            gradient_tol=GLUM_TOL, drop_first=True, solver=solver,
        )
        m.fit(X, y, offset=log_exposure)
        return m

    fit_seconds, model = best_of(fit, repeats)
    return dict(
        engine=f"glum[{solver}]", prep=prep_seconds, fit=fit_seconds,
        iters=int(getattr(model, "n_iter_", 0)), converged=True,
        mu=model.predict(X, offset=log_exposure), note=None,
    )


def report(label, n_rows, n_params, results):
    print(f"\n  {label}  ({n_rows:,} rows, {n_params:,} parameters)")
    print(f"  {'engine':<16}{'prep':>9}{'fit':>9}{'total':>9}{'iters':>7}"
          f"{'vs reference':>15}")
    print(f"  {'-' * 66}")

    reference = next((r for r in results if r["engine"].startswith("glum")), results[0])
    rms = float(np.sqrt(np.mean(reference["mu"] ** 2)))

    problems = []
    for r in results:
        d = float(np.max(np.abs(r["mu"] - reference["mu"])) / rms)
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
        print(f"  {r['engine']:<16}{r['prep']:>9.3f}{r['fit']:>9.3f}"
              f"{r['prep'] + r['fit']:>9.3f}{r['iters']:>7}{agreement:>15}{flag}")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--solver-sweep", action="store_true",
                        help="time every glum solver, not just its default choice")
    parser.add_argument("--tutorial-only", action="store_true",
                        help="skip the wide-band variant, which is slow for glum")
    args = parser.parse_args()

    raw = load_fremtpl()
    print(f"\nfreMTPL2freq: {len(raw):,} policies")

    # glum picks irls-ls for an unpenalised fit. irls-cd is the same family of
    # algorithm Avenue uses, so timing only against irls-ls would compare
    # algorithms and call the difference an implementation win.
    solvers = ["irls-ls", "irls-cd"] if args.solver_sweep else ["irls-ls", "irls-cd"]

    problems = []
    variants = [False] if args.tutorial_only else [False, True]
    for wide in variants:
        codes, levels, y, exposure = prepare(raw, wide)
        n_params = 1 + sum(k - 1 for k in levels.values())
        label = "wide bands   " if wide else "tutorial bands"

        results = [run_avenue(codes, levels, y, exposure, args.repeats, False)]
        for solver in solvers:
            try:
                results.append(run_glum(codes, y, exposure, args.repeats, solver))
            except Exception as exc:
                print(f"  glum[{solver}] failed: {type(exc).__name__}: {exc}")
        results.append(run_avenue(codes, levels, y, exposure, args.repeats, True))

        problems += report(label, len(y), n_params, results)

    if problems:
        print("\nPROBLEMS:")
        for p in problems:
            print(f"  {p}")
        return 1

    print("\nAll engines agreed on fitted means.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
