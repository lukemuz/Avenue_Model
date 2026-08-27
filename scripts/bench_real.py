"""Two more real datasets, chosen to cover what the existing real-data benchmarks do not.

`bench_fremtpl.py` and `bench_housing.py` are both modest in size and between them cover
Poisson, Gamma and Gaussian. Two gaps remain, and each of these datasets closes one:

* **`taxi`** - NYC yellow taxi trips, ~2.9M rows after cleaning, Gamma on the fare. The
  largest *real* problem in the suite, and the only one with high-cardinality geography:
  260 pickup zones and 261 dropoff zones, which is what a credible territory table
  actually looks like. Synthetic width sweeps have covered that shape; nothing real had.
* **`census`** - US census income, 48,842 rows, Binomial on whether income exceeds $50k.
  The suite had **no real-data coverage of the Binomial family at all**, and the logit
  link is the one path where the coordinate update is a Newton step that can overshoot
  rather than an exact `ln(A/E)` minimiser. statsmodels runs on this one as the oracle.

Both report `table_conditioning` alongside the timings. That is the figure that decides
whether a coordinate method is a good idea on a given plan - see
`src/glm/README.md#performance` - and reporting it on real data is how the claim that
badly conditioned plans are rare gets tested rather than assumed.

The usual gate applies: every engine's fitted means are compared before any timing is
reported, and a disagreement is printed as a failure rather than as a win.

Usage:
    python scripts/bench_real.py --dataset taxi
    python scripts/bench_real.py --dataset census
    python scripts/bench_real.py --dataset both --json out.json

Each dataset is downloaded once and cached next to this script.
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

HERE = os.path.dirname(os.path.abspath(__file__))
TAXI_CACHE = os.path.join(HERE, ".yellow_tripdata_2024_01.parquet")
TAXI_URL = ("https://d37ci6vzurychx.cloudfront.net/trip-data/"
            "yellow_tripdata_2024-01.parquet")
CENSUS_CACHE = os.path.join(HERE, ".adult.parquet")

MAX_ITER = 500
AVENUE_TOL = 1e-10
GLUM_TOL = 1e-10
AGREEMENT_TOL = 1e-7

# statsmodels materialises a dense float64 design matrix, so it runs only where that is
# affordable. Its value is as an independent oracle, not as a speed rival.
STATSMODELS_MAX_ELEMENTS = 2e7


# ----------------------------------------------------------------- the datasets

def band(values: np.ndarray, edges) -> np.ndarray:
    return np.digitize(np.asarray(values, dtype=float), edges)


def compact(codes: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    """Renumber each factor to 0..k-1 so every table row sees some exposure."""
    levels = {}
    out = {}
    for name, values in codes.items():
        uniques, compacted = np.unique(values, return_inverse=True)
        out[name] = compacted.astype(np.int64)
        levels[name] = len(uniques)
    return out, levels


def load_taxi() -> pd.DataFrame:
    if not os.path.exists(TAXI_CACHE):
        import urllib.request

        print("Downloading NYC yellow taxi trips for 2024-01 (once; ~50 MB)...")
        urllib.request.urlretrieve(TAXI_URL, TAXI_CACHE)
        print(f"Cached to {TAXI_CACHE}")
    return pd.read_parquet(TAXI_CACHE)


def prepare_taxi(df: pd.DataFrame):
    """Fare as a severity model: Gamma with a log link, banded like a rating plan.

    The cleaning is the minimum that makes the response admissible - a Gamma needs a
    positive response - plus caps on the tails, which every published analysis of this
    data applies and which are the same caps the French motor tutorial uses in spirit.

    Rows missing `passenger_count` or `RatecodeID` go too, for the same reason as in
    `prepare_census`: the two are missing on exactly the same 4.7% of rows, so giving
    missing its own level makes their indicators the same column. `table_correlations`
    reported three pairs at rho = 1.0000 on that design, and glum could not factor it -
    `LinAlgError: Matrix is singular` - which would have left Avenue as the only engine
    able to run, and a benchmark with one engine in it is not a benchmark.
    """
    keep = (
        (df["fare_amount"] > 0) & (df["fare_amount"] < 250)
        & (df["trip_distance"] > 0) & (df["trip_distance"] < 100)
        & df["passenger_count"].notna() & df["RatecodeID"].notna()
    )
    df = df[keep]

    pickup = df["tpep_pickup_datetime"]
    duration = (df["tpep_dropoff_datetime"] - pickup).dt.total_seconds() / 60.0

    codes = {
        "pickup_zone": df["PULocationID"].to_numpy(),
        "dropoff_zone": df["DOLocationID"].to_numpy(),
        "hour": pickup.dt.hour.to_numpy(),
        "weekday": pickup.dt.dayofweek.to_numpy(),
        "distance": band(df["trip_distance"], [0.5, 1, 1.5, 2, 3, 4, 5, 7, 10, 15, 20, 30]),
        "duration": band(duration.fillna(0.0), [3, 5, 8, 12, 18, 25, 35, 50, 75]),
        "passengers": band(df["passenger_count"], [0, 1, 2, 3, 4, 5]),
        "rate_code": df["RatecodeID"].to_numpy(),
        "payment": df["payment_type"].to_numpy(),
        "vendor": df["VendorID"].to_numpy(),
    }
    codes, levels = compact(codes)
    return codes, levels, df["fare_amount"].to_numpy(dtype=float), None


def load_census() -> pd.DataFrame:
    if not os.path.exists(CENSUS_CACHE):
        from sklearn.datasets import fetch_openml

        print("Downloading the census income data from OpenML (once)...")
        bunch = fetch_openml(data_id=1590, as_frame=True, parser="auto")
        bunch.frame.to_parquet(CENSUS_CACHE)
        print(f"Cached to {CENSUS_CACHE}")
    return pd.read_parquet(CENSUS_CACHE)


def prepare_census(df: pd.DataFrame):
    """Income over $50k as a Binomial fit, with the continuous drivers banded.

    Complete cases only - 45,222 rows of 48,842 - which is the conventional treatment of
    this dataset and here also a necessary one. Giving missing values their own level, as
    a rating table would, makes `workclass` and `occupation` **exactly aliased**: the two
    are missing on the same rows, so their missing-value indicators are the same column.
    `table_correlations` reports that pair at rho = 1.0000, Avenue fits it anyway because
    its design is over-parameterised regardless, and glum cannot factor it at all -
    `LinAlgError: Matrix is singular`. A benchmark only one engine can run is not a
    benchmark.

    What survives is more interesting than what was removed: `marital_status` and
    `relationship` sit at 0.9865, over the `NEAR_ALIAS` threshold, so this is a second
    real dataset - after the French motor `Density`/`Area` pair - where the joint pair
    solve has something to do.
    """
    categoricals = [c for c in df.columns if str(df[c].dtype) == "category"]
    df = df.dropna(subset=categoricals)

    def categorical(column: str) -> np.ndarray:
        return pd.Categorical(df[column].astype("object")).codes.astype(np.int64)

    codes = {
        "age": band(df["age"], [25, 30, 35, 40, 45, 50, 55, 60, 65, 70]),
        "workclass": categorical("workclass"),
        "education": categorical("education"),
        "marital_status": categorical("marital-status"),
        "occupation": categorical("occupation"),
        "relationship": categorical("relationship"),
        "race": categorical("race"),
        "sex": categorical("sex"),
        "hours": band(df["hours-per-week"], [20, 30, 35, 40, 45, 50, 60]),
        "capital_gain": band(df["capital-gain"], [1, 3000, 5000, 7500, 15000]),
        "capital_loss": band(df["capital-loss"], [1, 1500, 2000]),
        "native_country": categorical("native-country"),
    }
    codes, levels = compact(codes)
    y = (df["class"].astype(str).str.strip() == ">50K").to_numpy(dtype=float)
    return codes, levels, y, None


# Which factors are genuine categories rather than bands of a continuous driver.
#
# The distinction is what Avenue reads off the dtype: an `Int32` column is a category
# code matched exactly, a `Float64` one is a band's upper bound matched by binary search.
# Both give the same fit on data already compacted to `0..k-1`, so the choice is about
# saying what the data is — and about memory, since a band costs 8 bytes and a code 4.
DATASETS = {
    "taxi": {
        "label": "nyc_taxi / fare",
        "load": load_taxi,
        "prepare": prepare_taxi,
        "family": "gamma",
        "glum_family": "gamma",
        # `distance`, `duration` and `passengers` are bands; the rest are labels.
        "categorical": {"pickup_zone", "dropoff_zone", "rate_code", "payment",
                        "vendor", "weekday", "hour"},
    },
    "census": {
        "label": "census_income / >50k",
        "load": load_census,
        "prepare": prepare_census,
        "family": "binary",
        "glum_family": "binomial",
        # `age`, `hours`, `capital_gain` and `capital_loss` are bands.
        "categorical": {"workclass", "education", "marital_status", "occupation",
                        "relationship", "race", "sex", "native_country"},
    },
}


# ------------------------------------------------------------------ the engines

def best_of(fn, repeats: int):
    best, payload = float("inf"), None
    for _ in range(repeats):
        started = time.perf_counter()
        out = fn()
        elapsed = time.perf_counter() - started
        if elapsed < best:
            best, payload = elapsed, out
    return best, payload


def build_avenue(codes, levels, y, family, categorical=(), encoding="natural"):
    """The frame and rating model Avenue fits.

    `encoding` decides the dtypes. "natural" gives each factor the one its data calls
    for — `Int32` for the names in `categorical`, `Float64` for the bands — and is what
    the benchmark runs. "float" and "int32" force one throughout, which is what
    `encoding_check` uses to prove the choice cannot change the fit.
    """
    from avenue_model import RatingModel

    def dtype_of(name):
        if encoding == "float":
            return np.float64
        if encoding == "int32":
            return np.int32
        return np.int32 if name in categorical else np.float64

    frame = {name: codes[name].astype(dtype_of(name)) for name in levels}
    frame["y"] = y
    tables = [pl.DataFrame({"Rating_Factor": [0.0]})]
    for name, k in levels.items():
        tables.append(pl.DataFrame({
            name: np.arange(k, dtype=dtype_of(name)),
            "Rating_Factor": np.zeros(k),
        }))
    return pl.DataFrame(frame), RatingModel(tables, family)


def encoding_check(codes, levels, y, family, repeats):
    """Fit the same design as bands and as categories, and compare.

    Worth running because the natural way to write a categorical factor is an integer
    code, while every table in this suite is built as a Float64 band. That made the
    categorical lookup the one path no benchmark covered, and it was 7.8x slower than
    the banded one for the identical model. Two things are checked: the fits agree, and
    neither encoding is quietly the slow one.
    """
    from avenue_model import fit_glm_with_diagnostics, GLMOptions

    full = GLMOptions(objective=family, max_iterations=MAX_ITER,
                      tolerance=AVENUE_TOL, compute_standard_errors=False)
    # Matching happens once, so it is a fixed cost and a full fit dilutes it: on census
    # it is 24% of a 36-sweep fit but 92% of a one-sweep fit. Timing one sweep is what
    # makes this check sensitive enough to catch a regression in the lookup itself.
    single = GLMOptions(objective=family, max_iterations=1,
                        tolerance=AVENUE_TOL, compute_standard_errors=False)
    out = {}
    for label, encoding in (("Float64 bands", "float"), ("Int32 categories", "int32")):
        df, model = build_avenue(codes, levels, y, family, encoding=encoding)
        seconds, (fitted, diag) = best_of(
            lambda: fit_glm_with_diagnostics(model, df, "y", options=full), repeats)
        fixed, _ = best_of(
            lambda: fit_glm_with_diagnostics(model, df, "y", options=single), repeats)
        out[label] = (seconds, fixed, diag.iterations,
                      fitted.predict(df).to_series(0).to_numpy())

    (b_s, b_fx, b_it, b_mu) = out["Float64 bands"]
    (c_s, c_fx, c_it, c_mu) = out["Int32 categories"]
    drift = float(np.max(np.abs(c_mu - b_mu) / np.maximum(np.abs(b_mu), 1e-12)))
    ratio = c_fx / b_fx
    print(f"\n  encoding check   Float64 bands {b_s:.3f}s ({b_it} sweeps, "
          f"{b_fx * 1000:.0f}ms fixed)   Int32 categories {c_s:.3f}s ({c_it} sweeps, "
          f"{c_fx * 1000:.0f}ms fixed)   fixed-cost ratio {ratio:.2f}x")

    problems = []
    # Far tighter than AGREEMENT_TOL: these are not two engines that happen to agree,
    # they are the same arithmetic on the same data. Anything but noise means the two
    # lookups disagreed about which row an observation falls in.
    if drift > 1e-12:
        problems.append(f"encodings disagree by {drift:.2e}")
    # A lookup should not care which of two equivalent representations it was handed.
    # The categorical path once cost 7.8x the banded one here, which this would catch.
    if not 0.5 < ratio < 1.6:
        problems.append(f"encodings differ in fixed cost by "
                        f"{max(ratio, 1 / ratio):.1f}x")
    if not problems:
        print(f"  encodings agree to {drift:.1e}")
    return problems


def run_avenue(codes, levels, y, family, repeats, standard_errors, categorical=()):
    from avenue_model import fit_glm_with_diagnostics, GLMOptions

    prep_seconds, (df, model) = best_of(
        lambda: build_avenue(codes, levels, y, family, categorical), repeats)

    options = GLMOptions(objective=family, max_iterations=MAX_ITER,
                         tolerance=AVENUE_TOL,
                         compute_standard_errors=standard_errors)

    def fit():
        return fit_glm_with_diagnostics(model, df, "y", options=options)

    fit_seconds, (fitted, diag) = best_of(fit, repeats)

    note = f"max|score|={diag.max_gradient:.1e}"
    if standard_errors and diag.inference_error is not None:
        note += f"  no SEs: {diag.inference_error[:50]}"

    return dict(
        engine="avenue+se" if standard_errors else "avenue",
        prep=prep_seconds, fit=fit_seconds, iters=diag.iterations,
        converged=diag.converged,
        mu=fitted.predict(df).to_series(0).to_numpy(),
        note=note, conditioning=diag.table_conditioning,
    )


def run_glum(codes, y, glum_family, repeats, solver):
    import glum

    prep_seconds, X = best_of(
        lambda: pd.DataFrame({n: pd.Categorical(c) for n, c in codes.items()}), repeats)

    def fit():
        model = glum.GeneralizedLinearRegressor(
            family=glum_family, alpha=0.0, fit_intercept=True, max_iter=MAX_ITER,
            gradient_tol=GLUM_TOL, drop_first=True, solver=solver)
        model.fit(X, y)
        return model

    fit_seconds, model = best_of(fit, repeats)
    return dict(
        engine=f"glum[{solver}]", prep=prep_seconds, fit=fit_seconds,
        iters=int(getattr(model, "n_iter_", 0)), converged=True,
        mu=np.asarray(model.predict(X), dtype=np.float64), note=None,
        conditioning=None,
    )


def run_statsmodels(codes, levels, y, family, repeats):
    import statsmodels.api as sm

    n_params = 1 + sum(k - 1 for k in levels.values())
    elements = len(y) * n_params
    if elements > STATSMODELS_MAX_ELEMENTS:
        return dict(
            engine="statsmodels",
            skipped=f"{elements / 1e6:.0f}M-element dense design matrix",
        )

    families = {
        "gamma": sm.families.Gamma(link=sm.families.links.Log()),
        "binary": sm.families.Binomial(),
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
        return sm.GLM(y, X, family=families[family]).fit(maxiter=MAX_ITER, tol=1e-12)

    fit_seconds, result = best_of(fit, repeats)
    return dict(
        engine="statsmodels", prep=prep_seconds, fit=fit_seconds,
        iters=int(result.fit_history["iteration"]), converged=bool(result.converged),
        mu=np.asarray(result.fittedvalues, dtype=np.float64), note=None,
        conditioning=None,
    )


# ------------------------------------------------------------------- the runner

def report(label, n_rows, n_params, results):
    print(f"\n  {label}  ({n_rows:,} rows, {n_params:,} parameters)")
    print(f"  {'engine':<16}{'prep':>9}{'fit':>9}{'total':>9}{'peak MB':>10}{'iters':>7}"
          f"{'vs reference':>15}")
    print(f"  {'-' * 76}")

    live = [r for r in results if "skipped" not in r]
    # statsmodels first when it ran: a third implementation is a better reference than
    # either of the two being compared.
    reference = next((r for r in live if r["engine"] == "statsmodels"), None)
    if reference is None:
        reference = next((r for r in live if r["engine"].startswith("glum")), None)
    rms = float(np.sqrt(np.mean((reference or live[0])["mu"] ** 2)))

    problems = []
    if reference is None:
        # Checking Avenue against Avenue proves nothing, and reporting "all engines
        # agreed" off the back of it would be worse than reporting nothing at all.
        reference = live[0]
        problems.append(
            f"{label}: no independent engine fitted this design, so the timings below "
            f"are unvalidated")
    for r in results:
        if "skipped" in r:
            print(f"  {r['engine']:<16}{'skipped':>9}   ({r['skipped']})")
            continue
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


def run_dataset(name, repeats, with_statsmodels, solvers, check_encodings=True):
    spec = DATASETS[name]
    codes, levels, y, _ = spec["prepare"](spec["load"]())
    n_params = 1 + sum(k - 1 for k in levels.values())
    family = spec["family"]

    print(f"\n{spec['label']}: {len(y):,} rows, {len(levels)} tables, "
          f"{n_params:,} parameters")
    print("  " + ", ".join(f"{n}={k}" for n, k in levels.items()))

    categorical = spec.get("categorical", set())
    results = [measured(lambda: run_avenue(codes, levels, y, family, repeats,
                                           False, categorical))]
    for solver in solvers:
        try:
            results.append(measured(
                lambda s=solver: run_glum(codes, y, spec["glum_family"], repeats, s)))
        except Exception as exc:
            print(f"  glum[{solver}] failed: {type(exc).__name__}: {exc}")
    if with_statsmodels:
        try:
            results.append(measured(
                lambda: run_statsmodels(codes, levels, y, family, repeats)))
        except Exception as exc:
            print(f"  statsmodels failed: {type(exc).__name__}: {exc}")
    results.append(measured(lambda: run_avenue(codes, levels, y, family, repeats,
                                               True, categorical)))

    conditioning = next(
        (r["conditioning"] for r in results if r.get("conditioning") is not None), None)
    if conditioning is not None:
        print(f"  tables share a common direction at {conditioning:.1f} of a possible "
              f"{len(levels)}"
              f"{'  - expect many sweeps' if conditioning > 10 else ''}")

    problems = report(spec["label"], len(y), n_params, results)
    if check_encodings:
        problems += [f"{spec['label']}: {p}"
                     for p in encoding_check(codes, levels, y, family, repeats)]
    payload = {
        "dataset": name,
        "label": spec["label"],
        "family": family,
        "n_rows": len(y),
        "n_parameters": n_params,
        "n_tables": len(levels),
        "table_conditioning": conditioning,
        "engines": [{k: v for k, v in r.items() if k != "mu"} for r in results],
    }
    return problems, payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="both", choices=("taxi", "census", "both"))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--no-statsmodels", action="store_true",
                        help="skip the oracle, which is slow on the wider designs")
    parser.add_argument("--solver-sweep", action="store_true",
                        help="also time glum's irls-cd. Left out by default because it "
                             "does not converge on either of these designs and dominates "
                             "the runtime: 254s to reach its 500-iteration cap on the "
                             "census fit, against glum's own irls-ls at 0.8s")
    parser.add_argument("--no-encoding-check", action="store_true",
                        help="skip refitting each design with Int32 category codes "
                             "instead of Float64 bands. That check exists because the "
                             "categorical lookup is the one matching path no other "
                             "benchmark exercises")
    parser.add_argument("--json", type=str, default=None)
    args = parser.parse_args()

    names = ("taxi", "census") if args.dataset == "both" else (args.dataset,)
    # glum picks irls-ls for an unpenalised fit, so that is the comparison by default.
    solvers = ("irls-ls", "irls-cd") if args.solver_sweep else ("irls-ls",)
    problems, collected = [], []
    for name in names:
        found, payload = run_dataset(name, args.repeats, not args.no_statsmodels,
                                     solvers, not args.no_encoding_check)
        problems += found
        collected.append(payload)

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
