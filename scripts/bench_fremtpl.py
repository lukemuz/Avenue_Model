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

`Area` and `Density` are the same geography banded twice, and their first canonical
correlation is high enough that backfitting between them crawls. Someone building this
plan would notice and drop one, so every design is run twice: as the tutorial writes it,
and again with `Area` removed. The two rows together separate Avenue's aliasing penalty
from its baseline speed, and say what the comparison looks like on a plan that has been
through review. glum's factorisation is indifferent to the redundancy, so the reduced
design should cost it only the parameters it no longer carries.

Usage:
    python scripts/bench_fremtpl.py            # tutorial and wide bands, each with
                                               # and without the redundant Area table
    python scripts/bench_fremtpl.py --drop ""  # keep every table
    python scripts/bench_fremtpl.py --solver-sweep   # add glum's irls-cd solver, which
                                               # is worth seeing on the tutorial design
                                               # and painfully slow on the wide one

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

CACHE = os.path.join(os.path.dirname(__file__), ".fremtpl2freq.parquet")

MAX_ITER = 500
# Avenue's tolerance is now on the score, like glum's, so these mean roughly the same
# thing. Avenue scales its score by the total absolute residual to keep the threshold
# independent of the response's units; glum's is absolute on a weight-normalised
# objective. Close enough that the agreement check can arbitrate.
AVENUE_TOL = 1e-10
GLUM_TOL = 1e-10
AGREEMENT_TOL = 1e-7

# Avenue's threshold for updating two tables as one block, mirrored from
# `glm::redundancy::NEAR_ALIAS`. Used here only to annotate the correlation report.
NEAR_ALIAS = 0.9


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


def prepare(
    df: pd.DataFrame, wide: bool, drop: tuple[str, ...] = ()
) -> tuple[dict[str, np.ndarray], dict[str, int], np.ndarray, np.ndarray]:
    """Bands and codes, following glum's tutorial preprocessing.

    Returns (codes, levels, claim_count, exposure). `wide` bands the continuous
    drivers far more finely, which is how a real plan with credible geography and
    a full bonus-malus scale ends up several hundred parameters wide. `drop` removes
    tables by name before coding, for the redundancy variants.
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

    for name in drop:
        if name not in codes:
            raise ValueError(f"unknown table {name!r}; have {sorted(codes)}")
        del codes[name]

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


# Factors that are labels rather than bands of a continuous driver. Avenue reads the
# distinction off the dtype: `Int32` is a category code matched exactly, `Float64` is a
# band's upper bound matched by binary search. Same fit either way on data compacted to
# `0..k-1`; the code is half the width, which is the point.
CATEGORICAL = {"area", "veh_brand", "veh_gas", "region"}


def build_avenue(codes, levels, y, exposure):
    """The frame and rating model Avenue fits: an intercept table plus one per factor."""
    from avenue_model import RatingModel

    def dtype_of(name):
        return np.int32 if name in CATEGORICAL else np.float64

    frame = {n: codes[n].astype(dtype_of(n)) for n in levels}
    frame["y"] = y
    frame["log_exposure"] = np.log(exposure)
    tables = [pl.DataFrame({"Rating_Factor": [0.0]})]
    for name, k in levels.items():
        tables.append(pl.DataFrame({
            name: np.arange(k, dtype=dtype_of(name)),
            "Rating_Factor": np.zeros(k),
        }))
    return pl.DataFrame(frame), RatingModel(tables, "poisson")


def report_correlations(codes, levels, y, exposure, top: int = 3) -> None:
    """The table pairs that share the most information, worst first.

    This is what identifies `Area` as the table to drop, rather than it being asserted:
    the figure is the first canonical correlation between two tables' levels, which is
    also the factor by which the direction they share survives each fitting sweep.
    """
    from avenue_model import table_correlations

    df, model = build_avenue(codes, levels, y, exposure)
    names = ["intercept"] + list(levels)
    print(f"  most shared information between tables"
          f" (Avenue solves a pair jointly above {NEAR_ALIAS}):")
    for first, second, rho in table_correlations(model, df)[:top]:
        flag = "  near-aliased" if rho >= NEAR_ALIAS else ""
        print(f"    {names[first]:>13} / {names[second]:<13} rho = {rho:.3f}{flag}")


def run_avenue(codes, levels, y, exposure, repeats, standard_errors):
    from avenue_model import fit_glm_with_diagnostics, GLMOptions

    prep_seconds, (df, model) = best_of(
        lambda: build_avenue(codes, levels, y, exposure), repeats)

    options = GLMOptions(
        max_iterations=MAX_ITER, tolerance=AVENUE_TOL,
        compute_standard_errors=standard_errors,
    )

    def fit():
        return fit_glm_with_diagnostics(
            model, df, "y", offset_col="log_exposure", options=options)

    fit_seconds, result = best_of(fit, repeats)
    fitted, diag = result.model, result.diagnostics
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
    print(f"  {'engine':<16}{'prep':>9}{'fit':>9}{'total':>9}{'peak MB':>10}{'iters':>7}"
          f"{'vs reference':>15}")
    print(f"  {'-' * 76}")

    reference = next((r for r in results if r["engine"].startswith("glum")), results[0])
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


def to_json(label, n_rows, n_params, dropped, results):
    return {
        "dataset": "freMTPL2freq",
        "variant": label.strip(),
        "family": "poisson",
        "dropped_tables": list(dropped),
        "n_rows": n_rows,
        "n_parameters": n_params,
        "engines": [
            {k: v for k, v in r.items() if k != "mu"} for r in results
        ],
    }


def report_drop_effect(label, dropped, full, reduced) -> None:
    """Side by side: the same engines on the design with and without the extra table.

    The reduced design is a different model, so this is not a like-for-like accuracy
    comparison and the fitted means are not expected to match. What it measures is what
    the redundancy costs each engine - which for a factorising solver should be about
    the parameters it saves, and for a coordinate method can be most of the fit.
    """
    print(f"\n  {label}: cost of carrying {', '.join(dropped)}")
    print(f"  {'engine':<16}{'with':>9}{'without':>9}{'saved':>9}{'iterations':>18}")
    print(f"  {'-' * 61}")
    for r in full:
        m = next((x for x in reduced if x["engine"] == r["engine"]), None)
        if m is None or not m["fit"]:
            continue
        iters = f"{r['iters']} -> {m['iters']}"
        print(f"  {r['engine']:<16}{r['fit']:>9.3f}{m['fit']:>9.3f}"
              f"{r['fit'] / m['fit']:>8.2f}x{iters:>18}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--solver-sweep", action="store_true",
                        help="time every glum solver, not just its default choice")
    parser.add_argument("--tutorial-only", action="store_true",
                        help="skip the wide-band variant, which is slow for glum")
    parser.add_argument("--drop", default="area",
                        help="comma-separated tables to remove in the reduced variant "
                             "of each design; empty keeps every table")
    parser.add_argument("--json", type=str, default=None)
    args = parser.parse_args()

    raw = load_fremtpl()
    print(f"\nfreMTPL2freq: {len(raw):,} policies")

    # glum picks irls-ls for an unpenalised fit, so that is the default comparison.
    # irls-cd is the same family of algorithm Avenue uses and is worth seeing - timing
    # only against irls-ls compares algorithms and calls the difference an
    # implementation win - but on the wide design it takes 21 minutes to hit its
    # iteration cap and still miss by 1.5e-3, so it is opt-in rather than automatic.
    solvers = ["irls-ls", "irls-cd"] if args.solver_sweep else ["irls-ls"]

    dropped = tuple(n for n in (x.strip() for x in args.drop.split(",")) if n)
    widths = [False] if args.tutorial_only else [False, True]

    problems = []
    collected = []
    for wide in widths:
        # Same design twice: as the tutorial writes it, then without the redundant
        # table. Both are fitted by every engine, so each row is internally
        # like-for-like even though the two rows are different models.
        by_design = {}
        for drop in ([(), dropped] if dropped else [()]):
            codes, levels, y, exposure = prepare(raw, wide, drop)
            n_params = 1 + sum(k - 1 for k in levels.values())
            label = ("wide bands" if wide else "tutorial bands")
            label += f" - {','.join(drop)}" if drop else ""
            label = label.ljust(16)

            if not drop:
                print(f"\n{label.strip()}:")
                try:
                    report_correlations(codes, levels, y, exposure)
                except Exception as exc:  # a diagnostic must not stop the benchmark
                    print(f"  correlations unavailable: {type(exc).__name__}: {exc}")

            results = [measured(
                lambda: run_avenue(codes, levels, y, exposure, args.repeats, False))]
            for solver in solvers:
                try:
                    results.append(measured(
                        lambda s=solver: run_glum(codes, y, exposure, args.repeats, s)))
                except Exception as exc:
                    print(f"  glum[{solver}] failed: {type(exc).__name__}: {exc}")
            results.append(measured(
                lambda: run_avenue(codes, levels, y, exposure, args.repeats, True)))

            problems += report(label, len(y), n_params, results)
            collected.append(to_json(label, len(y), n_params, drop, results))
            by_design[drop] = results

        if dropped:
            report_drop_effect("wide bands" if wide else "tutorial bands", dropped,
                               by_design[()], by_design[dropped])

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
