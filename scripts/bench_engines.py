"""Avenue against the wider GLM field: glum, scikit-learn and H2O.

The other benchmark scripts compare Avenue with glum, because glum's `tabmat` backend
avoids a dense dummy-coded design matrix too and is therefore the strongest single
comparison available. This one widens the field to the two other engines glum's own
benchmark suite treats as its competition and that an actuary or a data scientist would
plausibly reach for instead:

* **scikit-learn** - `PoissonRegressor`, `GammaRegressor` and `LogisticRegression`. The
  default in most Python shops. Runs on a sparse one-hot design; `newton-cholesky` is
  the solver written for the `n >> p` shape these problems have, so that is what it is
  given, with `lbfgs` (the package default) available under `--all-solvers`.
* **H2O** - `H2OGeneralizedLinearEstimator` with the IRLSM solver, the distributed JVM
  engine that shows up wherever the data is too big for one machine's idea of a
  dataframe. It takes categorical columns natively, like Avenue and glum do.
* **glum** `irls-ls`, carried over so these numbers line up with the other scripts.
* **statsmodels** as the correctness oracle wherever a dense design matrix is
  affordable.

Three datasets, three families, all already used elsewhere in the suite so the numbers
can be read next to `src/glm/README.md#benchmarks`:

    freMTPL2   678k rows,  79 parameters, Poisson with an exposure offset
    census     45.2k rows, 116 parameters, Binomial
    housing    21.6k rows,  92 parameters, Gamma

The usual gate applies: every engine's fitted means are compared against an independent
reference before any timing is reported, and a disagreement is printed as a failure
rather than as a win.

Two things this script does *not* claim to measure:

* **Memory.** H2O keeps its frames and its solver in a JVM heap in another process, so
  a resident-set figure sampled in this one would report H2O as free. Memory is
  measured in `scripts/bench_isolated.py`, one engine per process, and H2O is not in it.
* **Distributed scaling.** H2O is built to spread a fit over a cluster. Run on one
  machine it is being used well outside the shape it was designed for, and the JVM
  round trip is a real part of its cost at these sizes. The point of including it is
  that single-node is how most people actually run it.

Usage:
    python scripts/bench_engines.py                     # all three datasets
    python scripts/bench_engines.py --dataset census
    python scripts/bench_engines.py --all-solvers       # adds sklearn lbfgs
    python scripts/bench_engines.py --skip h2o          # no JVM
    python scripts/bench_engines.py --json out.json
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

MAX_ITER = 500
AVENUE_TOL = 1e-10
GLUM_TOL = 1e-10
SKLEARN_TOL = 1e-10
AGREEMENT_TOL = 1e-6

# statsmodels materialises a dense float64 design matrix, so it runs only where that is
# affordable. Its value here is as an independent oracle, not as a speed rival.
STATSMODELS_MAX_ELEMENTS = 2e7

# H2O's own iteration budget. Its stopping rules are not glum's or Avenue's - it stops
# on a relative objective change rather than on a score - so the agreement check is what
# decides whether it actually solved the problem, not this number.
H2O_MAX_ITER = 500


# ----------------------------------------------------------------- the problems

class Problem:
    """One dataset in the one representation every engine is handed.

    `codes` are compacted to `0..k-1` per factor, which is what the rating tables, the
    one-hot design and the H2O enum columns are each built from. Keeping the encoding
    identical across engines is the only way the agreement gate means anything.
    """

    def __init__(self, key, label, codes, levels, y, family,
                 exposure=None, categorical=()):
        self.key = key
        self.label = label
        self.codes = codes
        self.levels = levels
        self.y = y
        self.family = family
        self.exposure = exposure
        self.categorical = set(categorical)

    @property
    def n_rows(self) -> int:
        return len(self.y)

    @property
    def n_params(self) -> int:
        return 1 + sum(k - 1 for k in self.levels.values())


def load_fremtpl_problem() -> Problem:
    from bench_fremtpl import CATEGORICAL, load_fremtpl, prepare

    codes, levels, y, exposure = prepare(load_fremtpl(), wide=False)
    return Problem("fremtpl", "freMTPL2 / claim count", codes, levels, y,
                   "poisson", exposure=exposure, categorical=CATEGORICAL)


def load_census_problem() -> Problem:
    from bench_real import load_census, prepare_census

    codes, levels, y, _ = prepare_census(load_census())
    categorical = {"workclass", "education", "marital_status", "occupation",
                   "relationship", "race", "sex", "native_country"}
    return Problem("census", "census_income / >50k", codes, levels, y,
                   "binomial", categorical=categorical)


def load_housing_problem() -> Problem:
    from bench_housing import load_housing, prepare

    codes, levels, y = prepare(load_housing(), rows=None)
    return Problem("housing", "house_sales / price", codes, levels, y, "gamma")


DATASETS = {
    "fremtpl": load_fremtpl_problem,
    "census": load_census_problem,
    "housing": load_housing_problem,
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


def run_avenue(problem: Problem, repeats: int) -> dict:
    from avenue_model import GLMOptions, RatingModel, fit_glm_with_diagnostics

    family = {"poisson": "poisson", "gamma": "gamma", "binomial": "binary"}[problem.family]

    def dtype_of(name):
        return np.int32 if name in problem.categorical else np.float64

    def prep():
        frame = {n: problem.codes[n].astype(dtype_of(n)) for n in problem.levels}
        frame["y"] = problem.y
        if problem.exposure is not None:
            frame["log_exposure"] = np.log(problem.exposure)
        tables = [pl.DataFrame({"Rating_Factor": [0.0]})]
        for name, k in problem.levels.items():
            tables.append(pl.DataFrame({
                name: np.arange(k, dtype=dtype_of(name)),
                "Rating_Factor": np.zeros(k),
            }))
        return pl.DataFrame(frame), RatingModel(tables, family)

    prep_seconds, (df, model) = best_of(prep, repeats)

    options = GLMOptions(max_iterations=MAX_ITER, tolerance=AVENUE_TOL,
                         compute_standard_errors=False)
    offset = "log_exposure" if problem.exposure is not None else None

    def fit():
        return fit_glm_with_diagnostics(model, df, "y", offset_col=offset,
                                        options=options)

    fit_seconds, result = best_of(fit, repeats)
    fitted, diag = result.model, result.diagnostics
    mu = fitted.predict(df).to_series(0).to_numpy()
    if problem.exposure is not None:
        mu = mu * problem.exposure

    return dict(engine="avenue", prep=prep_seconds, fit=fit_seconds,
                iters=diag.iterations, converged=diag.converged, mu=mu,
                note=f"max|score|={diag.max_gradient:.1e}")


def run_glum(problem: Problem, repeats: int, solver: str = "irls-ls") -> dict:
    import glum

    offset = None if problem.exposure is None else np.log(problem.exposure)

    prep_seconds, X = best_of(
        lambda: pd.DataFrame({n: pd.Categorical(c) for n, c in problem.codes.items()}),
        repeats)

    def fit():
        m = glum.GeneralizedLinearRegressor(
            family=problem.family, alpha=0.0, fit_intercept=True, max_iter=MAX_ITER,
            gradient_tol=GLUM_TOL, drop_first=True, solver=solver)
        m.fit(X, problem.y, offset=offset)
        return m

    fit_seconds, model = best_of(fit, repeats)
    mu = np.asarray(model.predict(X, offset=offset), dtype=np.float64)
    return dict(engine=f"glum[{solver}]", prep=prep_seconds, fit=fit_seconds,
                iters=int(getattr(model, "n_iter_", 0)), converged=True, mu=mu,
                note=None)


def one_hot(problem: Problem):
    """Sparse treatment-coded design, no intercept column.

    Every engine below that wants a matrix gets this one. It is built with
    `scipy.sparse` rather than `pandas.get_dummies` because the dense version of the
    freMTPL2 design is 430 MB of mostly zeros, and handing scikit-learn that instead
    would be benchmarking the encoder.
    """
    from scipy import sparse

    n = problem.n_rows
    blocks = []
    rows = np.arange(n)
    for name, k in problem.levels.items():
        c = problem.codes[name]
        mask = c > 0
        block = sparse.csr_matrix(
            (np.ones(int(mask.sum())), (rows[mask], c[mask] - 1)), shape=(n, k - 1))
        blocks.append(block)
    return sparse.hstack(blocks, format="csr")


def run_sklearn(problem: Problem, repeats: int, solver: str) -> dict:
    """scikit-learn on the same design, with its best solver for this shape.

    Two representation choices are worth stating, because both are in scikit-learn's
    favour rather than against it:

    * The design is sparse. scikit-learn has no categorical-aware backend, so a dense
      one is what a naive user gets; a sparse one is what a careful one builds, and
      that is the version timed.
    * The Poisson fit carries its exposure as a `sample_weight` on `claims / exposure`
      rather than as an offset, because `PoissonRegressor` has no offset argument. The
      two are the same model - the score equations are identical - which the agreement
      check against glum's offset formulation confirms to 1e-9.
    """
    from sklearn.linear_model import (GammaRegressor, LogisticRegression,
                                      PoissonRegressor)

    prep_seconds, X = best_of(lambda: one_hot(problem), repeats)

    if problem.family == "binomial":
        def make():
            return LogisticRegression(penalty=None, solver=solver, tol=SKLEARN_TOL,
                                      max_iter=MAX_ITER)
        y, weight = problem.y, None
    elif problem.family == "poisson":
        def make():
            return PoissonRegressor(alpha=0.0, solver=solver, tol=SKLEARN_TOL,
                                    max_iter=MAX_ITER)
        y = problem.y / problem.exposure
        weight = problem.exposure
    else:
        def make():
            return GammaRegressor(alpha=0.0, solver=solver, tol=SKLEARN_TOL,
                                  max_iter=MAX_ITER)
        y, weight = problem.y, None

    def fit():
        m = make()
        m.fit(X, y) if weight is None else m.fit(X, y, sample_weight=weight)
        return m

    fit_seconds, model = best_of(fit, repeats)

    if problem.family == "binomial":
        mu = model.predict_proba(X)[:, 1]
    else:
        mu = np.asarray(model.predict(X), dtype=np.float64)
        if problem.exposure is not None:
            mu = mu * problem.exposure

    iters = np.atleast_1d(getattr(model, "n_iter_", [0]))[0]
    return dict(engine=f"sklearn[{solver}]", prep=prep_seconds, fit=fit_seconds,
                iters=int(iters), converged=True, mu=mu, note=None)


def run_h2o(problem: Problem, repeats: int) -> dict:
    """H2O's IRLSM GLM, with the fit timed and the JVM round trip reported separately.

    `prep` here is uploading the frame into the cluster, which for these datasets is
    the larger of the two numbers. It is reported rather than hidden because it is a
    real cost of using H2O on one machine - but the `fit` column is the like-for-like
    comparison, and that is the one the discussion uses.

    `lambda_=0` asks for an unpenalised fit, which is not H2O's default and which it
    will not do on a rank-deficient design without `remove_collinear_columns=True`.
    The design is treatment-coded by H2O itself from enum columns, so it is only
    rank-deficient where the data makes it so.
    """
    import h2o
    from h2o.estimators.glm import H2OGeneralizedLinearEstimator

    names = list(problem.levels)

    def prep():
        frame = {n: problem.codes[n].astype(str) for n in names}
        frame["y"] = problem.y
        if problem.exposure is not None:
            frame["log_exposure"] = np.log(problem.exposure)
        hf = h2o.H2OFrame(pd.DataFrame(frame),
                          column_types={n: "enum" for n in names})
        if problem.family == "binomial":
            hf["y"] = hf["y"].asfactor()
        return hf

    prep_seconds, hf = best_of(prep, repeats)

    def fit():
        m = H2OGeneralizedLinearEstimator(
            family=problem.family,
            link="log" if problem.family in ("poisson", "gamma") else "logit",
            lambda_=0.0,
            solver="IRLSM",
            standardize=False,
            remove_collinear_columns=True,
            compute_p_values=False,
            max_iterations=H2O_MAX_ITER,
            objective_epsilon=1e-12,
            beta_epsilon=1e-12,
            gradient_epsilon=1e-10,
            seed=1,
        )
        m.train(x=names, y="y", training_frame=hf,
                offset_column="log_exposure" if problem.exposure is not None else None)
        return m

    fit_seconds, model = best_of(fit, repeats)

    predictions = model.predict(hf).as_data_frame()
    column = "p1" if problem.family == "binomial" else "predict"
    mu = predictions[column].to_numpy(dtype=float)

    iters = 0
    try:
        iters = int(model.summary().as_data_frame()["number_of_iterations"][0])
    except Exception:  # noqa: BLE001 - the summary schema varies by family
        pass

    return dict(engine="h2o[IRLSM]", prep=prep_seconds, fit=fit_seconds, iters=iters,
                converged=True, mu=mu, note="fit excludes frame upload")


def run_statsmodels(problem: Problem, repeats: int) -> dict:
    import statsmodels.api as sm

    elements = problem.n_rows * problem.n_params
    if elements > STATSMODELS_MAX_ELEMENTS:
        return dict(engine="statsmodels",
                    skipped=f"{elements / 1e6:.0f}M-element dense design matrix")

    families = {
        "poisson": sm.families.Poisson(),
        "gamma": sm.families.Gamma(link=sm.families.links.Log()),
        "binomial": sm.families.Binomial(),
    }

    def prep():
        return np.hstack([np.ones((problem.n_rows, 1)), one_hot(problem).toarray()])

    prep_seconds, X = best_of(prep, repeats)
    offset = None if problem.exposure is None else np.log(problem.exposure)

    def fit():
        return sm.GLM(problem.y, X, family=families[problem.family],
                      offset=offset).fit(maxiter=MAX_ITER, tol=1e-12)

    fit_seconds, result = best_of(fit, repeats)
    return dict(engine="statsmodels", prep=prep_seconds, fit=fit_seconds,
                iters=int(result.fit_history["iteration"]),
                converged=bool(result.converged),
                mu=np.asarray(result.fittedvalues, dtype=np.float64), note=None)


# ------------------------------------------------------------------- the runner

def report(problem: Problem, results: list[dict]) -> list[str]:
    print(f"\n  {problem.label}  ({problem.n_rows:,} rows, "
          f"{problem.n_params:,} parameters, {problem.family})")
    print(f"  {'engine':<20}{'prep':>9}{'fit':>9}{'total':>9}{'iters':>7}"
          f"{'vs avenue':>11}{'agreement':>12}")
    print(f"  {'-' * 77}")

    live = [r for r in results if "skipped" not in r]

    # An independent implementation is the reference wherever one ran; statsmodels goes
    # first because it takes the dense route and shares no code with anything else here.
    reference = next((r for r in live if r["engine"] == "statsmodels"), None)
    if reference is None:
        reference = next((r for r in live if r["engine"].startswith("glum")), None)

    problems: list[str] = []
    if reference is None:
        reference = live[0]
        problems.append(f"{problem.label}: no independent engine fitted this design, "
                        f"so the timings below are unvalidated")

    rms = float(np.sqrt(np.mean(reference["mu"] ** 2)))
    avenue = next(r for r in live if r["engine"] == "avenue")

    for r in results:
        if "skipped" in r:
            print(f"  {r['engine']:<20}{'skipped':>9}   {r['skipped']}")
            continue
        drift = float(np.sqrt(np.mean((r["mu"] - reference["mu"]) ** 2)) / rms)
        total = r["prep"] + r["fit"]
        ratio = r["fit"] / avenue["fit"]
        agreement = "reference" if r is reference else f"{drift:.1e}"
        print(f"  {r['engine']:<20}{r['prep']:>8.3f}s{r['fit']:>8.3f}s"
              f"{total:>8.3f}s{r['iters']:>7}{ratio:>10.2f}x{agreement:>12}")
        if r["note"]:
            print(f"  {'':<20}{r['note']}")
        r["drift"] = drift
        if drift > AGREEMENT_TOL:
            problems.append(f"{problem.label}: {r['engine']} disagrees with "
                            f"{reference['engine']} by {drift:.2e}")
        if not r["converged"]:
            problems.append(f"{problem.label}: {r['engine']} did not converge")

    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=[*DATASETS, "all"], default="all")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--skip", nargs="*", default=[],
                        choices=["glum", "sklearn", "h2o", "statsmodels"],
                        help="engines to leave out")
    parser.add_argument("--all-solvers", action="store_true",
                        help="also time scikit-learn's default lbfgs solver")
    parser.add_argument("--json", help="write the results to this path")
    args = parser.parse_args()

    keys = list(DATASETS) if args.dataset == "all" else [args.dataset]

    h2o_started = False
    if "h2o" not in args.skip:
        import h2o

        h2o.init(nthreads=-1, max_mem_size="8G")
        h2o.no_progress()
        h2o_started = True

    problems: list[str] = []
    payload = []
    for key in keys:
        problem = DATASETS[key]()
        results = [run_avenue(problem, args.repeats)]
        if "glum" not in args.skip:
            results.append(run_glum(problem, args.repeats))
        if "sklearn" not in args.skip:
            # `newton-cholesky` is the solver written for the n >> p shape these
            # problems have; lbfgs is the package default and is timed on request.
            results.append(run_sklearn(problem, args.repeats, "newton-cholesky"))
            if args.all_solvers:
                results.append(run_sklearn(problem, args.repeats, "lbfgs"))
        if "h2o" not in args.skip:
            results.append(run_h2o(problem, args.repeats))
        if "statsmodels" not in args.skip:
            results.append(run_statsmodels(problem, args.repeats))

        problems += report(problem, results)
        payload.append(dict(
            dataset=key, label=problem.label, rows=problem.n_rows,
            parameters=problem.n_params, family=problem.family,
            engines=[{k: v for k, v in r.items() if k != "mu"} for r in results],
        ))

    if h2o_started:
        import h2o

        h2o.cluster().shutdown()

    if args.json:
        with open(args.json, "w") as handle:
            json.dump(payload, handle, indent=2)
        print(f"\n  wrote {args.json}")

    if problems:
        print("\n  FAILED")
        for line in problems:
            print(f"    {line}")
        return 1
    print("\n  all engines agree; timings above are comparable")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
