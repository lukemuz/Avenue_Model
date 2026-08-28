"""Avenue against the wider GLM field: glum, scikit-learn and H2O, penalised and not.

The other benchmark scripts compare Avenue with glum, because glum's `tabmat` backend
avoids a dense dummy-coded design matrix too and is therefore the strongest single
comparison available. This one widens the field to the two other engines glum's own
benchmark suite treats as its competition, and that an actuary or a data scientist would
plausibly reach for instead:

* **scikit-learn** - `PoissonRegressor`, `GammaRegressor` and `LogisticRegression`. The
  default in most Python shops. Runs on a sparse one-hot design; `newton-cholesky` is
  the solver written for the `n >> p` shape these problems have, so that is what it is
  given, with `lbfgs` (the package default) available under `--all-solvers`.
* **H2O** - `H2OGeneralizedLinearEstimator` with the IRLSM solver, the distributed JVM
  engine that shows up wherever the data is too big for one machine's idea of a
  dataframe. It takes categorical columns natively, like Avenue and glum do.
* **glum** carried over so these numbers line up with the other scripts: `irls-ls`
  unpenalised, and `auto` under a penalty, which is what a user actually gets.
* **statsmodels** as the correctness oracle on unpenalised fits wherever a dense design
  matrix is affordable.

Three datasets, three families, all already used elsewhere in the suite so the numbers
can be read next to `src/glm/README.md#benchmarks`:

    freMTPL2   678k rows,  79 parameters, Poisson, exposure as a prior weight
    census     45.2k rows, 116 parameters, Binomial
    housing    21.6k rows,  92 parameters, Gamma

### Penalties

glum's own published benchmark is **entirely penalised** - every problem in its
`results.csv` carries a ridge, a lasso or an elastic net - so an unpenalised-only
comparison never meets it on its own ground. This script runs `--penalty none ridge
lasso` by default.

Four engines mean four penalty conventions, and a comparison of differently-scaled
objectives is worthless however carefully it is timed. glum's is taken as the reference,

    deviance / (2 * sum of weights)  +  alpha * ( l1_ratio * |b|_1
                                                  + (1 - l1_ratio)/2 * |b|_2^2 )

and the others are mapped onto it:

| engine | mapping |
|---|---|
| Avenue | `alpha`, `l1_ratio` - already glum's, verified coefficient-wise by `check_penalty.py` |
| scikit-learn GLMs | `alpha` matches; **L1 does not exist** for Poisson or Gamma |
| scikit-learn logistic | `C = 1 / (n * alpha)`, because its data term is a sum where glum's is a mean |
| H2O | `lambda_ = alpha`, `alpha = l1_ratio`, `standardize=False`, and a pre-coded design because its Python API cannot hold out a reference level |

The mapping is asserted rather than trusted: the agreement gate compares fitted means
across engines on every row of every table, penalised rows included, and a mismatched
penalty scale shows up there as a disagreement rather than as a win. It is a two-tier
gate, because "wrong" and "stopped sooner than everyone else" are different findings -
see `AGREEMENT_TOL` and `LOOSE_TOL`, and the `~` marker in the agreement column.
statsmodels sits out the penalised rows - its `fit_regularized` is a different objective
again, and its value here was only ever as an independent oracle.

For the same reason every engine gets the **weighted** formulation of the Poisson
problem - `claims / exposure` with `exposure` as a prior weight - rather than the offset
formulation the other scripts use. The two are the same model unpenalised, but they
normalise the deviance by different totals, so under a penalty they are different
problems. `scripts/bench_fremtpl.py` keeps the offset formulation.

**Threads are measured, not assumed.** These engines parallelise through three unrelated
mechanisms - rayon for Avenue, OpenMP inside `tabmat` for glum, BLAS for scikit-learn and
statsmodels, the JVM for H2O - and on a small core count they do not all want the same
number of threads. On a four-core machine glum's census fit takes 0.52 s pinned to one
thread and 3.5 s on four: a small sandwich product oversubscribing four cores, which is a
property of the machine rather than of glum. Reporting whichever setting happened to be
in the environment would be reporting noise, and picking one would quietly favour
whichever engine liked it.

So the default run fits every problem twice - once with the thread environment pinned to
one, once unpinned - and reports **each engine's own best**, with the setting that
produced it. `--threads 1` or `--threads 0` runs a single pass when the breakdown itself
is what is wanted.

Two things this script does *not* claim to measure:

* **Memory.** H2O keeps its frames and its solver in a JVM heap in another process, so
  a resident-set figure sampled in this one would report H2O as free. Memory is
  measured in `scripts/bench_isolated.py`, one engine per process, and H2O is not in it.
* **Distributed scaling.** H2O is built to spread a fit over a cluster. Run on one
  machine it is being used well outside the shape it was designed for, and the JVM
  round trip is a real part of its cost at these sizes. The point of including it is
  that single-node is how most people actually run it. The JVM also takes its thread
  count from `h2o.init` rather than from the environment, so both passes give it every
  core and its two figures differ only by noise.

Usage:
    python scripts/bench_engines.py                       # 3 datasets x 3 penalties
    python scripts/bench_engines.py --dataset census
    python scripts/bench_engines.py --penalty none        # unpenalised only
    python scripts/bench_engines.py --threads 1           # one pinned pass only
    python scripts/bench_engines.py --all-solvers         # adds scikit-learn's lbfgs
    python scripts/bench_engines.py --skip h2o            # no JVM
    python scripts/bench_engines.py --json out.json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time

import numpy as np
import pandas as pd
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

MAX_ITER = 500
AVENUE_TOL = 1e-10
GLUM_TOL = 1e-10
SKLEARN_TOL = 1e-10
# Two tiers, because "this engine is wrong" and "this engine stopped earlier than the
# others" are different findings and only one of them is a failure. Below AGREEMENT_TOL
# the engines solved the same problem. Between the two, an engine reached a looser
# optimum than the rest - its answer is usable and its time is optimistic, which is
# reported as a caveat and marked in the table rather than counted as a clean win.
# Above LOOSE_TOL something is wrong with the problem being posed and the run fails.
AGREEMENT_TOL = 1e-6
LOOSE_TOL = 1e-2

# statsmodels materialises a dense float64 design matrix, so it runs only where that is
# affordable. Its value here is as an independent oracle, not as a speed rival.
STATSMODELS_MAX_ELEMENTS = 2e7

# H2O's own iteration budget. Its stopping rules are not glum's or Avenue's - it stops
# on a relative objective change rather than on a score - so the agreement check is what
# decides whether it actually solved the problem, not this number.
H2O_MAX_ITER = 500

PENALTIES = {
    "none": 0.0,
    "ridge": 0.0,
    "lasso": 1.0,
    "net": 0.5,
}

# Every variable the four engines read for their thread counts, so one pass can be
# pinned. H2O is not among them: the JVM takes its count from `h2o.init`.
THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
               "NUMEXPR_NUM_THREADS", "RAYON_NUM_THREADS", "POLARS_MAX_THREADS")


# ----------------------------------------------------------------- the problems

class Problem:
    """One dataset in the one representation every engine is handed.

    `codes` are compacted to `0..k-1` per factor, which is what the rating tables, the
    sparse one-hot design and the H2O enum columns are each built from, and level 0 is
    the reference level everywhere. Keeping both the encoding and the reference level
    identical across engines is what makes the agreement gate mean anything - and under
    a ridge it is load-bearing rather than cosmetic, because dropping a different level
    is a different penalised problem.
    """

    def __init__(self, key, label, codes, levels, y, family,
                 weight=None, categorical=()):
        self.key = key
        self.label = label
        self.codes = codes
        self.levels = levels
        self.y = y
        self.family = family
        self.weight = weight
        self.categorical = set(categorical)

    @property
    def n_rows(self) -> int:
        return len(self.y)

    @property
    def n_params(self) -> int:
        return 1 + sum(k - 1 for k in self.levels.values())


def load_fremtpl_problem() -> Problem:
    from bench_fremtpl import CATEGORICAL, load_fremtpl, prepare

    codes, levels, claims, exposure = prepare(load_fremtpl(), wide=False)
    return Problem("fremtpl", "freMTPL2 / claim frequency", codes, levels,
                   claims / exposure, "poisson", weight=exposure,
                   categorical=CATEGORICAL)


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


def skipped(engine: str, reason: str) -> dict:
    return dict(engine=engine, skipped=reason)


def run_avenue(problem: Problem, repeats: int, alpha: float, l1_ratio: float) -> dict:
    from avenue_model import GLMOptions, RatingModel, fit_glm_with_diagnostics

    family = {"poisson": "poisson", "gamma": "gamma",
              "binomial": "binary"}[problem.family]

    def dtype_of(name):
        return np.int32 if name in problem.categorical else np.float64

    def prep():
        frame = {n: problem.codes[n].astype(dtype_of(n)) for n in problem.levels}
        frame["y"] = problem.y
        if problem.weight is not None:
            frame["w"] = problem.weight
        tables = [pl.DataFrame({"Rating_Factor": [0.0]})]
        for name, k in problem.levels.items():
            tables.append(pl.DataFrame({
                name: np.arange(k, dtype=dtype_of(name)),
                "Rating_Factor": np.zeros(k),
            }))
        return pl.DataFrame(frame), RatingModel(tables, family)

    prep_seconds, (df, model) = best_of(prep, repeats)

    options = GLMOptions(max_iterations=MAX_ITER, tolerance=AVENUE_TOL,
                         alpha=alpha, l1_ratio=l1_ratio,
                         compute_standard_errors=False)
    weight = "w" if problem.weight is not None else None

    def fit():
        return fit_glm_with_diagnostics(model, df, "y", weight_col=weight,
                                        options=options)

    fit_seconds, result = best_of(fit, repeats)
    fitted, diag = result.model, result.diagnostics

    return dict(engine="avenue", prep=prep_seconds, fit=fit_seconds,
                iters=diag.iterations, converged=diag.converged,
                mu=fitted.predict(df).to_series(0).to_numpy(),
                note=f"max|score|={diag.max_gradient:.1e}")


def run_glum(problem: Problem, repeats: int, alpha: float, l1_ratio: float,
             solver: str) -> dict:
    import glum

    prep_seconds, X = best_of(
        lambda: pd.DataFrame({n: pd.Categorical(c) for n, c in problem.codes.items()}),
        repeats)

    def fit():
        m = glum.GeneralizedLinearRegressor(
            family=problem.family, alpha=alpha, l1_ratio=l1_ratio,
            fit_intercept=True, max_iter=MAX_ITER, gradient_tol=GLUM_TOL,
            drop_first=True, solver=solver, scale_predictors=False)
        m.fit(X, problem.y, sample_weight=problem.weight)
        return m

    fit_seconds, model = best_of(fit, repeats)
    return dict(engine=f"glum[{solver}]", prep=prep_seconds, fit=fit_seconds,
                iters=int(np.max(getattr(model, "n_iter_", 0))), converged=True,
                mu=np.asarray(model.predict(X), dtype=np.float64), note=None)


def one_hot(problem: Problem):
    """Sparse treatment-coded design, level 0 dropped, no intercept column.

    Every engine below that wants a matrix gets this one. It is built with
    `scipy.sparse` rather than `pandas.get_dummies` because the dense version of the
    freMTPL2 design is 430 MB of mostly zeros, and handing scikit-learn that instead
    would be benchmarking the encoder.
    """
    from scipy import sparse

    n = problem.n_rows
    rows = np.arange(n)
    blocks = []
    for name, k in problem.levels.items():
        c = problem.codes[name]
        mask = c > 0
        blocks.append(sparse.csr_matrix(
            (np.ones(int(mask.sum())), (rows[mask], c[mask] - 1)), shape=(n, k - 1)))
    return sparse.hstack(blocks, format="csr")


def run_sklearn(problem: Problem, repeats: int, alpha: float, l1_ratio: float,
                solver: str) -> dict:
    """scikit-learn on the same design, with its best solver for this shape.

    Two representation choices are worth stating, because both are in scikit-learn's
    favour rather than against it: the design is sparse, which is what a careful user
    builds and not what `get_dummies` hands a careless one; and the solver is the one
    written for `n >> p` rather than the package default.

    What is *not* in its favour is a gap in the library. `PoissonRegressor` and
    `GammaRegressor` take an L2 penalty and nothing else - there is no L1 or elastic net
    for a non-Gaussian GLM in scikit-learn at all - so those rows are skipped rather
    than faked with a different objective.
    """
    from sklearn.linear_model import (GammaRegressor, LogisticRegression,
                                      PoissonRegressor)

    name = f"sklearn[{solver}]"
    if l1_ratio > 0.0 and alpha > 0.0 and problem.family != "binomial":
        return skipped(name, f"no L1 penalty for a {problem.family} GLM in scikit-learn")

    prep_seconds, X = best_of(lambda: one_hot(problem), repeats)

    if problem.family == "binomial":
        # Its data term is a sum where glum's is a weight-normalised mean, so the
        # penalty has to absorb the row count: C = 1 / (n * alpha).
        if alpha == 0.0:
            kwargs = dict(penalty=None, solver=solver)
        elif l1_ratio == 0.0:
            kwargs = dict(penalty="l2", C=1.0 / (problem.n_rows * alpha), solver=solver)
        else:
            # saga is the only solver that takes an L1 or elastic-net logistic penalty.
            kwargs = dict(penalty="elasticnet" if l1_ratio < 1 else "l1",
                          C=1.0 / (problem.n_rows * alpha), solver="saga",
                          l1_ratio=l1_ratio if l1_ratio < 1 else None)
            name = "sklearn[saga]"

        def make():
            return LogisticRegression(tol=SKLEARN_TOL, max_iter=MAX_ITER, **kwargs)
    elif problem.family == "poisson":
        def make():
            return PoissonRegressor(alpha=alpha, solver=solver, tol=SKLEARN_TOL,
                                    max_iter=MAX_ITER)
    else:
        def make():
            return GammaRegressor(alpha=alpha, solver=solver, tol=SKLEARN_TOL,
                                  max_iter=MAX_ITER)

    def fit():
        model = make()
        if problem.weight is None:
            model.fit(X, problem.y)
        else:
            model.fit(X, problem.y, sample_weight=problem.weight)
        return model

    fit_seconds, model = best_of(fit, repeats)
    mu = (model.predict_proba(X)[:, 1] if problem.family == "binomial"
          else np.asarray(model.predict(X), dtype=np.float64))

    # scikit-learn warns rather than raises when it runs out of iterations, and a run
    # that spent its whole budget and stopped is not a fit anyone should be timing.
    iters = int(np.atleast_1d(getattr(model, "n_iter_", [0]))[0])
    return dict(engine=name, prep=prep_seconds, fit=fit_seconds, iters=iters,
                converged=iters < MAX_ITER, mu=mu, note=None)


def run_h2o(problem: Problem, repeats: int, alpha: float, l1_ratio: float) -> dict:
    """H2O's IRLSM GLM, with the fit timed and the JVM round trip reported separately.

    `prep` here is uploading the frame into the cluster, which for these datasets is
    the larger of the two numbers. It is reported rather than hidden because it is a
    real cost of using H2O on one machine - but the `fit` column is the like-for-like
    comparison, and that is the one the discussion uses.

    Three details make the comparison honest rather than approximate.

    `standardize=False`, because H2O standardises by default, which rescales every dummy
    column and with it the penalty each coefficient feels.

    Unpenalised, the factors go in as **enum columns** - H2O's native categorical path
    and its fastest - with the levels as zero-padded strings, because H2O orders levels
    lexicographically and drops the first for a GLM. Unpadded, `"10"` sorts before `"2"`
    and H2O silently holds out a different reference level.

    Penalised, they go in as the **same sparse treatment-coded design scikit-learn
    gets**, which is slower for H2O and is not a choice made to flatter it. The reason
    is that H2O's Python GLM API exposes no `use_all_factor_levels` - the REST endpoint
    rejects it outright - and with a penalty H2O keeps every level of every enum, giving
    102 coefficients where the design has 92. That is not a slower solution to the same
    problem, it is a different penalised problem: the fitted means come out 3.6e-3 from
    glum's, ten times further than the coded design's, because a ridge on an
    over-parameterised expansion is not a ridge on a treatment-coded one. Pre-coding is
    the only way to ask H2O the question the other three engines are being asked.
    """
    import h2o
    from h2o.estimators.glm import H2OGeneralizedLinearEstimator
    from scipy import sparse

    names = list(problem.levels)
    width = max(len(str(k)) for k in problem.levels.values())
    coded = alpha > 0.0

    def prep():
        response = {"y": problem.y}
        if problem.weight is not None:
            response["w"] = problem.weight
        if coded:
            design = h2o.H2OFrame(sparse.csr_matrix(one_hot(problem)))
            hf = design.cbind(h2o.H2OFrame(pd.DataFrame(response)))
            columns = design.columns
        else:
            frame = {n: np.char.zfill(problem.codes[n].astype(str), width)
                     for n in names}
            frame.update(response)
            hf = h2o.H2OFrame(pd.DataFrame(frame),
                              column_types={n: "enum" for n in names})
            columns = names
        if problem.family == "binomial":
            hf["y"] = hf["y"].asfactor()
        return hf, columns

    prep_seconds, (hf, columns) = best_of(prep, repeats)

    def fit():
        m = H2OGeneralizedLinearEstimator(
            family=problem.family,
            link="log" if problem.family in ("poisson", "gamma") else "logit",
            lambda_=alpha,
            alpha=l1_ratio,
            solver="IRLSM",
            standardize=False,
            # Only legal, and only needed, on an unpenalised rank-deficient design.
            remove_collinear_columns=(alpha == 0.0),
            compute_p_values=False,
            max_iterations=H2O_MAX_ITER,
            objective_epsilon=1e-12,
            beta_epsilon=1e-12,
            gradient_epsilon=1e-10,
            seed=1,
        )
        m.train(x=columns, y="y", training_frame=hf,
                weights_column="w" if problem.weight is not None else None)
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

    note = "fit excludes frame upload"
    if coded:
        note += "; pre-coded design, see the docstring"
    return dict(engine="h2o[IRLSM]", prep=prep_seconds, fit=fit_seconds, iters=iters,
                converged=True, mu=mu, note=note)


def run_statsmodels(problem: Problem, repeats: int, alpha: float,
                    l1_ratio: float) -> dict:
    import statsmodels.api as sm

    if alpha > 0.0:
        # `fit_regularized` optimises a third objective again, and statsmodels is here
        # as an oracle rather than as a rival. Penalised agreement is arbitrated
        # between the engines that share glum's parameterisation.
        return skipped("statsmodels", "penalised objective is not glum's")

    elements = problem.n_rows * problem.n_params
    if elements > STATSMODELS_MAX_ELEMENTS:
        return skipped("statsmodels",
                       f"{elements / 1e6:.0f}M-element dense design matrix")

    families = {
        "poisson": sm.families.Poisson(),
        "gamma": sm.families.Gamma(link=sm.families.links.Log()),
        "binomial": sm.families.Binomial(),
    }

    def prep():
        return np.hstack([np.ones((problem.n_rows, 1)), one_hot(problem).toarray()])

    prep_seconds, X = best_of(prep, repeats)

    def fit():
        return sm.GLM(problem.y, X, family=families[problem.family],
                      var_weights=problem.weight).fit(maxiter=MAX_ITER, tol=1e-12)

    fit_seconds, result = best_of(fit, repeats)
    return dict(engine="statsmodels", prep=prep_seconds, fit=fit_seconds,
                iters=int(result.fit_history["iteration"]),
                converged=bool(result.converged),
                mu=np.asarray(result.fittedvalues, dtype=np.float64), note=None)


# --------------------------------------------------------------- one whole pass

def run_pass(args, threads: int) -> tuple[list[dict], list[str]]:
    """Every dataset and penalty, at one thread setting, in this process."""
    keys = list(DATASETS) if args.dataset == "all" else [args.dataset]

    h2o_started = False
    if "h2o" not in args.skip:
        import h2o

        h2o.init(nthreads=-1, max_mem_size="8G")
        h2o.no_progress()
        h2o_started = True

    payload: list[dict] = []
    problems: list[str] = []
    for key in keys:
        problem = DATASETS[key]()
        for penalty in args.penalty:
            l1_ratio = PENALTIES[penalty]
            alpha = 0.0 if penalty == "none" else args.alpha

            results = [run_avenue(problem, args.repeats, alpha, l1_ratio)]
            if "glum" not in args.skip:
                # `irls-ls` is glum's unpenalised default and the comparison the other
                # scripts make; under a penalty `auto` is what a user actually gets,
                # and it is glum's own choice of solver rather than ours.
                solver = "irls-ls" if alpha == 0.0 else "auto"
                results.append(run_glum(problem, args.repeats, alpha, l1_ratio, solver))
            if "sklearn" not in args.skip:
                results.append(
                    run_sklearn(problem, args.repeats, alpha, l1_ratio,
                                "newton-cholesky"))
                if args.all_solvers:
                    results.append(
                        run_sklearn(problem, args.repeats, alpha, l1_ratio, "lbfgs"))
            if "h2o" not in args.skip:
                results.append(run_h2o(problem, args.repeats, alpha, l1_ratio))
            if "statsmodels" not in args.skip:
                results.append(run_statsmodels(problem, args.repeats, alpha, l1_ratio))

            problems += check(problem, penalty, results)
            payload.append(dict(
                dataset=key, label=problem.label, penalty=penalty, alpha=alpha,
                l1_ratio=l1_ratio, threads=threads, rows=problem.n_rows,
                parameters=problem.n_params, family=problem.family,
                engines=[{k: v for k, v in r.items() if k != "mu"} for r in results],
            ))

    if h2o_started:
        import h2o

        h2o.cluster().shutdown()

    return payload, problems


def check(problem: Problem, penalty: str, results: list[dict]) -> list[str]:
    """Fill in each engine's drift from an independent reference, and flag the bad ones.

    A wrong answer arriving quickly is the failure mode this whole script exists to
    rule out, so nothing is timed until the fitted means line up.
    """
    live = [r for r in results if "skipped" not in r]
    where = f"{problem.label} / {penalty}"

    # An independent implementation is the reference wherever one ran; statsmodels goes
    # first because it takes the dense route and shares no code with anything else here.
    reference = next((r for r in live if r["engine"] == "statsmodels"), None)
    if reference is None:
        reference = next((r for r in live if r["engine"].startswith("glum")), None)

    problems: list[str] = []
    if reference is None:
        reference = live[0]
        problems.append(f"{where}: no independent engine fitted this design, so the "
                        f"timings are unvalidated")

    rms = float(np.sqrt(np.mean(reference["mu"] ** 2)))
    for r in live:
        r["drift"] = float(np.sqrt(np.mean((r["mu"] - reference["mu"]) ** 2)) / rms)
        r["reference"] = r is reference
        if r["drift"] > LOOSE_TOL:
            problems.append(f"{where}: {r['engine']} disagrees with "
                            f"{reference['engine']} by {r['drift']:.2e}")
        if not r["converged"]:
            problems.append(f"{where}: {r['engine']} did not converge")
    return problems


# ------------------------------------------------------------------- the runner

def merge(passes: list[list[dict]]) -> list[dict]:
    """Per engine, keep the pass that fitted it fastest."""
    merged: dict[tuple, dict] = {}
    order: list[tuple] = []
    for payload in passes:
        for block in payload:
            key = (block["dataset"], block["penalty"])
            if key not in merged:
                merged[key] = {k: v for k, v in block.items() if k != "engines"}
                merged[key]["engines"] = {}
                order.append(key)
            engines = merged[key]["engines"]
            for r in block["engines"]:
                previous = engines.get(r["engine"])
                if previous is None or (
                    "skipped" not in r and (
                        "skipped" in previous or r["fit"] < previous["fit"])):
                    engines[r["engine"]] = dict(r, threads=block["threads"])
    out = []
    for key in order:
        block = merged[key]
        block["engines"] = list(block["engines"].values())
        out.append(block)
    return out


def report(blocks: list[dict], show_threads: bool) -> None:
    for block in blocks:
        penalty = block["penalty"]
        detail = ("unpenalised" if penalty == "none"
                  else f"{penalty}, alpha={block['alpha']:g}")
        print(f"\n  {block['label']}  ({block['rows']:,} rows, "
              f"{block['parameters']:,} parameters, {block['family']}, {detail})")
        header = (f"  {'engine':<24}{'prep':>9}{'fit':>9}{'iters':>7}"
                  f"{'vs avenue':>11}{'agreement':>12}")
        if show_threads:
            header += f"{'threads':>9}"
        print(header)
        print(f"  {'-' * (len(header) - 2)}")

        engines = block["engines"]
        avenue = next(r for r in engines if r["engine"] == "avenue")
        for r in engines:
            if "skipped" in r:
                print(f"  {r['engine']:<24}{'-':>9}   {r['skipped']}")
                continue
            if r.get("reference"):
                agreement = "reference"
            else:
                # A trailing ~ means this engine stopped short of the shared optimum,
                # so its time bought a looser answer than everyone else's.
                agreement = (f"{r['drift']:.1e}"
                             + ("~" if r["drift"] > AGREEMENT_TOL else ""))
            line = (f"  {r['engine']:<24}{r['prep']:>8.3f}s{r['fit']:>8.3f}s"
                    f"{r['iters']:>7}{r['fit'] / avenue['fit']:>10.2f}x"
                    f"{agreement:>12}")
            if show_threads:
                line += f"{('all' if r['threads'] == 0 else r['threads']):>9}"
            print(line)
            if r.get("note"):
                print(f"  {'':<24}{r['note']}")


def child(args, threads: int, path: str) -> list[str]:
    """Re-run this script in a subprocess with the thread environment pinned.

    A subprocess rather than `threadpoolctl` because rayon and Polars read their thread
    counts once, when the extension module is first used, and no in-process switch
    reaches them afterwards.
    """
    environment = dict(os.environ)
    if threads:
        environment.update({name: str(threads) for name in THREAD_VARS})
    else:
        for name in THREAD_VARS:
            environment.pop(name, None)

    command = [sys.executable, os.path.abspath(__file__),
               "--dataset", args.dataset, "--repeats", str(args.repeats),
               "--alpha", str(args.alpha), "--penalty", *args.penalty,
               "--threads", str(threads), "--json", path, "--quiet"]
    if args.skip:
        command += ["--skip", *args.skip]
    if args.all_solvers:
        command.append("--all-solvers")

    completed = subprocess.run(command, env=environment)
    if completed.returncode not in (0, 1):
        raise SystemExit(f"the {threads or 'unpinned'}-thread pass failed "
                         f"({completed.returncode})")
    with open(path) as handle:
        return json.load(handle)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=[*DATASETS, "all"], default="all")
    parser.add_argument("--penalty", nargs="+", choices=list(PENALTIES),
                        default=["none", "ridge", "lasso"])
    parser.add_argument("--alpha", type=float, default=1e-4,
                        help="penalty strength on glum's scale")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--threads", type=int, default=None,
                        help="run a single pass at this thread count (0 = unpinned)")
    parser.add_argument("--skip", nargs="*", default=[],
                        choices=["glum", "sklearn", "h2o", "statsmodels"],
                        help="engines to leave out")
    parser.add_argument("--all-solvers", action="store_true",
                        help="also time scikit-learn's default lbfgs solver")
    parser.add_argument("--quiet", action="store_true",
                        help="write the JSON without printing a table")
    parser.add_argument("--json", help="write the results to this path")
    args = parser.parse_args()

    if args.threads is None:
        # Two passes, each engine reported at whichever suited it.
        with tempfile.TemporaryDirectory() as directory:
            passes = [child(args, threads,
                            os.path.join(directory, f"pass{threads}.json"))
                      for threads in (1, 0)]
        blocks = merge(passes)
        problems = [f"{block['label']} / {block['penalty']}: {r['engine']} disagrees "
                    f"by {r['drift']:.2e}" for payload in passes for block in payload
                    for r in block["engines"] if r.get("drift", 0) > LOOSE_TOL]
        problems += [f"{block['label']} / {block['penalty']}: {r['engine']} did not "
                     f"converge" for payload in passes for block in payload
                     for r in block["engines"]
                     if "skipped" not in r and not r["converged"]]
        report(blocks, show_threads=True)
    else:
        payload, problems = run_pass(args, args.threads)
        blocks = merge([payload])
        if not args.quiet:
            report(blocks, show_threads=False)

    if args.json:
        with open(args.json, "w") as handle:
            json.dump(blocks if args.threads is None else payload, handle, indent=2)
        if not args.quiet:
            print(f"\n  wrote {args.json}")

    loose = sorted({f"{block['label']} / {block['penalty']}: {r['engine']} reached a "
                    f"looser optimum ({r['drift']:.1e}), so its time is optimistic"
                    for block in blocks for r in block["engines"]
                    if AGREEMENT_TOL < r.get("drift", 0) <= LOOSE_TOL})

    if not args.quiet:
        if loose:
            print("\n  marked ~ above:")
            for line in loose:
                print(f"    {line}")
        if problems:
            print("\n  FAILED")
            for line in problems:
                print(f"    {line}")
        else:
            print("\n  every engine solved the same problem; timings are comparable")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
