"""One large fit per engine, at a size where the absolute times matter on their own.

The rest of the suite finishes in under two seconds, which makes the ratios between
engines easy to dismiss. This runs a single fit of a 20-million-row book against a plan
with a few hundred parameters, and reports what each engine costs in seconds and
gigabytes to produce the same answer.

**One engine per process.** The two representations do not fit in memory together: the
polars frame Avenue matches against is one float64 column per table over 20M rows, 16 GB
by itself at 100 tables. Each child regenerates the same book from the same seed, fits
once, and writes its fitted means to disk; the parent checks they agree before reporting
any timing.

Three shapes, because the comparison is sensitive to which one you pick:

* **`--tables 100 --levels 6`** (default) puts 501 parameters in many small tables.
  Backfitting touches each observation once per table, `O(n*T)`; a blockwise sandwich
  product has to form `X'WX` from every *pair* of blocks, `O(n*T^2)`. This is the shape
  that separates the two.
* **`--tables 5 --levels 101`** puts the *same* 501 parameters in a few wide tables,
  where `T^2` is 25 rather than 10,000. If the quadratic story is right, most of the gap
  should disappear here - so this is the control, and it is meant to be run.
* **`--correlation 0.6`** loads every factor on a shared latent driver instead of drawing
  them independently. Independent factors are the best case for coordinate descent and
  the standing bias of the synthetic suite; this is the check on it.

Usage:
    python scripts/bench_large.py                          # 20M rows, 100 small tables
    python scripts/bench_large.py --tables 5 --levels 101  # same parameters, few tables
    python scripts/bench_large.py --correlation 0.6        # correlated factors
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import threading
import time

import numpy as np

SEED = 20260827
TOL = 1e-10
# A backstop, not a budget. A well-conditioned plan converges in single-digit sweeps
# whatever this says, but a correlated one legitimately needs thousands - 100 tables
# sharing a driver at a pairwise 0.28 reaches the tolerance on sweep 1119 - and capping
# it lower would report "did not converge" for a fit that was still working.
MAX_ITER = 5000
SCRATCH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".bench_large")


class Peak:
    """Whole-process high-water RSS in MB, sampled from a background thread."""

    def __init__(self) -> None:
        import psutil

        self._process = psutil.Process()
        self._stop = threading.Event()
        self._peak = 0
        self.mb = 0.0
        self._thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self) -> None:
        while not self._stop.is_set():
            self._peak = max(self._peak, self._process.memory_info().rss)
            self._stop.wait(0.05)

    def __enter__(self) -> "Peak":
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)
        self.mb = self._peak / 1e6


# --------------------------------------------------------------------- the data

def generate(rows: int, tables: int, levels: int, correlation: float):
    """The same book of business in both children, from the same seed.

    Codes are kept in the narrowest integer type that holds them, so the raw data is a
    couple of gigabytes rather than sixteen. Avenue's frame widens them to float64 later;
    that is a property of its matching path, not of the data.
    """
    rng = np.random.default_rng(SEED)
    names = [f"factor_{i:03d}" for i in range(tables)]
    dtype = np.int8 if levels <= 127 else np.int16

    # Scaled so that a hundred-table plan does not accumulate a wildly larger linear
    # predictor than a five-table one - otherwise the shapes are different problems.
    spread = 0.35 / (tables / 5.0) ** 0.5
    eta = np.full(rows, -2.3)

    latent = None
    edges = None
    if correlation > 0.0:
        # One shared driver behind every factor, which is what makes real rating
        # factors correlated: age tracks bonus-malus, density tracks region.
        latent = rng.standard_normal(rows, dtype=np.float32)
        from scipy.stats import norm

        # Equal-probability bands of a standard normal, so each level keeps its share
        # of the exposure however strong the correlation is.
        edges = norm.ppf(np.arange(1, levels) / levels).astype(np.float32)

    codes = {}
    for name in names:
        if latent is None:
            c = rng.integers(0, levels, size=rows, dtype=dtype)
        else:
            x = rng.standard_normal(rows, dtype=np.float32)
            x *= np.float32(np.sqrt(1.0 - correlation))
            x += np.float32(np.sqrt(correlation)) * latent
            c = np.digitize(x, edges).astype(dtype)
            del x
        effects = rng.normal(0.0, spread, size=levels)
        effects -= effects[0]
        eta += effects[c]
        codes[name] = c

    del latent
    exposure = rng.uniform(0.05, 1.0, size=rows)
    np.exp(eta, out=eta)
    eta *= exposure
    y = rng.poisson(eta).astype(np.float64)
    del eta
    gc.collect()
    return codes, y, exposure


# ------------------------------------------------------------------ the engines

def run_avenue(codes, y, exposure, levels):
    import polars as pl
    from avenue_model import RatingModel, fit_glm_with_diagnostics, GLMOptions

    started = time.perf_counter()
    frame = pl.DataFrame({"y": y, "log_exposure": np.log(exposure)})
    for name, c in codes.items():
        # Int32, not Float64: these factors are unordered draws, so a category code is
        # what they are, and it is half the width of a band's upper bound. Four bytes is
        # as narrow as the matching path goes — the codes themselves are int8/int16.
        # One column at a time, so the widened copy is one column, not the whole frame.
        frame = frame.with_columns(pl.Series(name, c, dtype=pl.Int32))
    tables = [pl.DataFrame({"Rating_Factor": [0.0]})]
    for name in codes:
        tables.append(pl.DataFrame({
            name: np.arange(levels, dtype=np.int32),
            "Rating_Factor": np.zeros(levels),
        }))
    model = RatingModel(tables, "poisson")
    prep = time.perf_counter() - started

    options = GLMOptions(max_iterations=MAX_ITER, tolerance=TOL,
                         compute_standard_errors=False)
    started = time.perf_counter()
    result = fit_glm_with_diagnostics(
        model, frame, "y", offset_col="log_exposure", options=options)
    fitted, diag = result.model, result.diagnostics
    fit = time.perf_counter() - started

    mu = fitted.predict(frame).to_series(0).to_numpy() * exposure
    return prep, fit, diag.iterations, bool(diag.converged), mu, diag.table_conditioning


def run_glum(codes, y, exposure, levels):
    import glum
    import pandas as pd

    started = time.perf_counter()
    # Categoricals are glum's fast path: tabmat keeps them as a CategoricalMatrix
    # rather than densifying to dummies.
    X = pd.DataFrame({n: pd.Categorical(c) for n, c in codes.items()}, copy=False)
    log_exposure = np.log(exposure)
    prep = time.perf_counter() - started

    started = time.perf_counter()
    model = glum.GeneralizedLinearRegressor(
        family="poisson", alpha=0.0, fit_intercept=True, max_iter=MAX_ITER,
        gradient_tol=TOL, drop_first=True, solver="irls-ls")
    model.fit(X, y, offset=log_exposure)
    fit = time.perf_counter() - started

    mu = np.asarray(model.predict(X, offset=log_exposure), dtype=np.float64)
    return prep, fit, int(model.n_iter_), True, mu, None


# ------------------------------------------------------------------- the runner

def child(args) -> int:
    with Peak() as peak:
        codes, y, exposure = generate(args.rows, args.tables, args.levels,
                                      args.correlation)
        runner = {"avenue": run_avenue, "glum": run_glum}[args.engine]
        prep, fit, iters, converged, mu, conditioning = runner(
            codes, y, exposure, args.levels)
        os.makedirs(SCRATCH, exist_ok=True)
        np.save(os.path.join(SCRATCH, f"mu_{args.engine}.npy"), mu)
    print("RESULT " + json.dumps({
        "engine": args.engine, "prep": prep, "fit": fit, "iterations": iters,
        "converged": converged, "peak_rss_mb": peak.mb,
        "table_conditioning": conditioning,
    }))
    return 0


def parent(args) -> int:
    n_params = 1 + args.tables * (args.levels - 1)
    design = "independent factors" if args.correlation <= 0 else (
        f"factors correlated through a shared latent driver, rho = {args.correlation}")
    print(f"\n{args.rows:,} rows, {args.tables} tables of {args.levels} levels, "
          f"{n_params:,} parameters")
    print(f"Poisson with an exposure offset, {design}")
    print("one engine per process; peak RSS is the whole process\n")

    results = []
    for engine in ("avenue", "glum"):
        cmd = [sys.executable, os.path.abspath(__file__), "--engine", engine,
               "--rows", str(args.rows), "--tables", str(args.tables),
               "--levels", str(args.levels), "--correlation", str(args.correlation)]
        env = dict(os.environ, PYTHONWARNINGS="ignore")
        started = time.perf_counter()
        proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
        wall = time.perf_counter() - started

        payload = None
        for line in proc.stdout.splitlines():
            if line.startswith("RESULT "):
                payload = json.loads(line[len("RESULT "):])
        if payload is None:
            tail = (proc.stderr or proc.stdout).strip().splitlines()
            print(f"  {engine:<10}FAILED  {tail[-1] if tail else proc.returncode}")
            return 1

        results.append(payload)
        flag = "" if payload["converged"] else "  DID NOT CONVERGE"
        print(f"  {payload['engine']:<10}{payload['prep']:>8.1f}s prep"
              f"{payload['fit']:>9.1f}s fit{payload['iterations']:>5} iters"
              f"{payload['peak_rss_mb'] / 1000:>8.1f} GB peak"
              f"{wall:>8.1f}s wall{flag}")

    avenue, glum = results
    # Avenue's own read on the design, next to what the design cost it. 1.0 is orthogonal
    # tables; the table count is every table carrying the same information.
    if avenue.get("table_conditioning") is not None:
        print(f"\n  tables share a common direction at "
              f"{avenue['table_conditioning']:.1f} of a possible {args.tables}")

    a = np.load(os.path.join(SCRATCH, "mu_avenue.npy"))
    g = np.load(os.path.join(SCRATCH, "mu_glum.npy"))
    disagreement = float(np.max(np.abs(a - g)) / float(np.sqrt(np.mean(g ** 2))))
    print(f"\n  fitted means agree to {disagreement:.1e} (max deviation over RMS)")
    if disagreement > 1e-7:
        print("  DISAGREE - the timings above are not comparable")
        return 1

    # Per iteration as well as per fit: the iteration counts differ, and only one of
    # these two ratios is independent of how many sweeps the design happened to need.
    avenue_per = avenue["fit"] / avenue["iterations"]
    glum_per = glum["fit"] / glum["iterations"]
    print(f"  glum fit / avenue fit:            {glum['fit'] / avenue['fit']:.1f}x")
    print(f"  per iteration:                    {glum_per / avenue_per:.1f}x"
          f"   ({avenue_per:.1f}s vs {glum_per:.1f}s)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=20_000_000)
    parser.add_argument("--tables", type=int, default=100)
    parser.add_argument("--levels", type=int, default=6)
    parser.add_argument("--correlation", type=float, default=0.0,
                        help="load every factor on a shared latent driver, in [0, 1)")
    parser.add_argument("--engine", choices=("avenue", "glum"), default=None,
                        help="run one engine in this process; used by the parent")
    args = parser.parse_args()
    return child(args) if args.engine else parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
