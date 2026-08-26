"""Benchmark Avenue's table GLM against glum and statsmodels.

Avenue fits on rating tables and never builds a design matrix. glum and
statsmodels both need one. That difference is the whole point of the library, so
this script times two things separately and reports both:

    prep  - turning a raw dataframe into whatever the engine needs to fit
    fit   - the solve itself, given prepared inputs

Reporting only `fit` flatters glum and statsmodels (someone else paid for their
design matrix); reporting only `total` flatters Avenue. Both are printed.

A fast wrong answer is not a fit, so every engine's fitted means are compared
against the reference engine before any timing is reported. A case where the
engines disagree is printed as a failure, not as a win.

Known biases in this benchmark
------------------------------

The data here is synthetic and was written by the same person as the fitter, so it
should be read with suspicion. Two biases are measured rather than assumed, and both
favour Avenue:

1. **Independent rating factors.** Every factor is an independent uniform draw, which
   makes the design near-orthogonal - the best case for coordinate descent, and unlike
   real rating data where driver age tracks bonus-malus and density tracks region.
   Measured at 81 parameters and 500k rows, correlating the factors (rho = 0.6 through
   a shared latent driver) took Avenue from 8 sweeps to 32, a 3.9x slowdown, while glum
   moved from 5 IRLS iterations to 6. On that correlated problem at matched accuracy
   glum wins 0.226s to 0.635s.

2. **The response is drawn from the model being fitted**, so there is no
   misspecification for either engine to struggle against.

A third effect is not a bias so much as a caveat: glum's iteration count on wide
problems is sensitive to the data, and it has been observed taking anywhere from 7 to
23 IRLS iterations on nominally similar 5,000-parameter fits. Single wide-case ratios
from this script are therefore not stable enough to quote.

`bench_fremtpl.py` addresses all three by running the same comparison on the French
Motor Third-Party Liability data, which is what glum's own benchmark suite uses.

Usage:
    python scripts/bench_glm.py               # full suite
    python scripts/bench_glm.py --quick       # small sizes only
    python scripts/bench_glm.py --width       # design-matrix width sweep
    python scripts/bench_glm.py --json out.json
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import sys
import threading
import time
from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd
import polars as pl

SEED = 20260826

# Avenue and glum both stop on the largest absolute score, so these now mean roughly
# the same thing; statsmodels tests the deviance and is not directly comparable.
#
# They are deliberately tighter than any of the defaults. Timing engines at different
# accuracies is not a comparison: at its looser settings glum stops ~1e-5 from the
# optimum, which is fine for rating work but would let it post a faster time for less
# work.
AVENUE_TOL = 1e-10
GLUM_TOL = 1e-10
STATSMODELS_TOL = 1e-10
MAX_ITER = 200

# Agreement threshold on fitted means, relative to the reference engine. Set well below
# what any rating application would care about, so it fails on a real defect rather
# than on the last couple of digits.
AGREEMENT_TOL = 1e-7

# statsmodels materialises a dense float64 design matrix and takes a pseudo-inverse of
# it on every IRLS iteration, so it costs minutes and gigabytes on the larger cases.
# Its value here is as an independent correctness oracle, not as a speed rival, and one
# size is enough for that - so run it only where it is cheap. Above this many design
# matrix elements it is skipped, and glum becomes the reference the others are checked
# against.
STATSMODELS_MAX_ELEMENTS = 2e7


# --------------------------------------------------------------- the problem

#: A personal-auto-shaped rating structure: name -> number of levels. 81 parameters.
RATING_FACTORS = {
    "driver_age": 12,
    "vehicle_age": 10,
    "territory": 40,
    "vehicle_symbol": 15,
    "credit_band": 8,
}


def deep_factors(territory: int, symbol: int) -> dict[str, int]:
    """The same five factors, but with high-cardinality geography and symbol.

    Widens the design matrix without adding tables. Avenue should barely notice:
    an observation still falls in exactly one row of each table, so the per-sweep
    cost is unchanged and only the standard-error inversion grows.
    """
    factors = dict(RATING_FACTORS)
    factors["territory"] = territory
    factors["vehicle_symbol"] = symbol
    return factors


def many_factors(count: int, levels: int = 6) -> dict[str, int]:
    """Many small factors, as an interaction-heavy plan tends to produce.

    Widens the design matrix by adding tables, which for Avenue means one more pass
    over the data per sweep - linear, where the design-matrix solve is cubic in the
    parameter count.
    """
    return {f"factor_{i:03d}": levels for i in range(count)}


@dataclass
class Dataset:
    """One generated book of business, shared verbatim by every engine."""

    n_rows: int
    factors: dict[str, int]  # factor name -> number of levels
    codes: dict[str, np.ndarray]  # factor name -> integer level per row
    y: np.ndarray
    exposure: np.ndarray
    log_exposure: np.ndarray

    @property
    def n_parameters(self) -> int:
        return 1 + sum(k - 1 for k in self.factors.values())


def make_dataset(n_rows: int, family: str, factors: dict[str, int] | None = None) -> Dataset:
    """Generate a book of business with real signal in every rating factor.

    The response is drawn from the family being fitted, so the fit is
    well-posed and every engine converges from a sensible starting point.
    """
    factors = dict(RATING_FACTORS) if factors is None else factors
    rng = np.random.default_rng(SEED + n_rows + len(factors))

    codes = {
        name: rng.integers(0, levels, size=n_rows)
        for name, levels in factors.items()
    }

    # True log-scale effects. Scaled down as the plan grows so that a wide model does
    # not accumulate a wildly larger linear predictor than a narrow one - otherwise
    # the families would be fitting different-shaped problems, not the same problem at
    # different widths.
    spread = 0.35 / max(1.0, (len(factors) / 5.0) ** 0.5)
    eta = np.full(n_rows, -2.3)
    for name, levels in factors.items():
        effects = rng.normal(0.0, spread, size=levels)
        effects -= effects[0]
        eta += effects[codes[name]]

    exposure = rng.uniform(0.05, 1.0, size=n_rows)

    if family == "poisson":
        y = rng.poisson(np.exp(eta) * exposure).astype(np.float64)
    elif family == "gamma":
        mu = np.exp(eta + 6.0)
        y = rng.gamma(shape=2.0, scale=mu / 2.0)
    elif family == "tweedie":
        mu = np.exp(eta)
        # Poisson number of claims x gamma severity = Tweedie in (1, 2).
        counts = rng.poisson(mu * exposure * 3.0)
        severity = rng.gamma(shape=2.0, scale=200.0, size=n_rows)
        y = counts * severity / np.maximum(exposure, 1e-9)
    elif family == "gaussian":
        y = eta * 100.0 + rng.normal(0.0, 25.0, size=n_rows)
    else:
        raise ValueError(f"unknown family {family!r}")

    return Dataset(
        n_rows=n_rows,
        factors=factors,
        codes=codes,
        y=y.astype(np.float64),
        exposure=exposure,
        log_exposure=np.log(exposure),
    )


@dataclass
class Case:
    """One (family, size) benchmark cell."""

    family: str
    n_rows: int
    #: Poisson carries exposure as an offset; the rest carry it as a weight.
    use_offset: bool
    use_weight: bool
    tweedie_power: float = 1.5
    #: Rating structure. Defaults to the personal-auto shape in RATING_FACTORS.
    factors: dict[str, int] | None = None
    #: Overrides the generated label, for the width sweep.
    name: str | None = None

    @property
    def structure(self) -> dict[str, int]:
        return dict(RATING_FACTORS) if self.factors is None else self.factors

    @property
    def n_parameters(self) -> int:
        return 1 + sum(k - 1 for k in self.structure.values())

    @property
    def label(self) -> str:
        if self.name is not None:
            return self.name
        return f"{self.family}/{self.n_rows:,}"


# --------------------------------------------------------------- the results

@dataclass
class Timing:
    engine: str
    prep_seconds: float | None = None
    fit_seconds: float | None = None
    iterations: int | None = None
    converged: bool | None = None
    mu: np.ndarray | None = field(default=None, repr=False)
    #: Peak resident memory added by this engine, in MB. See `PeakMemory`.
    peak_memory_mb: float | None = None
    #: Anything that qualifies how this row should be read.
    note: str | None = None
    skipped: str | None = None
    error: str | None = None
    #: Max relative deviation of fitted means from the reference engine.
    disagreement: float | None = None

    @property
    def total_seconds(self) -> float | None:
        if self.prep_seconds is None or self.fit_seconds is None:
            return None
        return self.prep_seconds + self.fit_seconds

    def to_json(self) -> dict:
        d = asdict(self)
        d.pop("mu")
        d["total_seconds"] = self.total_seconds
        return d


class PeakMemory:
    """Peak resident memory an engine adds, over the baseline when it started.

    Sampled from a background thread rather than read before and after, because the
    interesting number is the high-water mark - a design matrix that is built, used and
    freed still has to fit in RAM.

    Two caveats worth knowing when reading the result. CPython does not always return
    freed memory to the OS, so an engine that runs after a greedy one can show a
    smaller figure than it would alone; and the shared dataset is allocated before any
    of this, so it is excluded from every engine's total.
    """

    INTERVAL_SECONDS = 0.002

    def __init__(self) -> None:
        import psutil

        self._process = psutil.Process()
        self.peak_mb = 0.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._baseline = 0.0
        self._peak_bytes = 0

    def _sample(self) -> None:
        while not self._stop.is_set():
            rss = self._process.memory_info().rss
            if rss > self._peak_bytes:
                self._peak_bytes = rss
            self._stop.wait(self.INTERVAL_SECONDS)

    def __enter__(self) -> "PeakMemory":
        gc.collect()
        self._baseline = self._process.memory_info().rss
        self._peak_bytes = self._baseline
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self.peak_mb = max(0.0, (self._peak_bytes - self._baseline) / 1e6)


def best_of(fn, repeats: int):
    """Run `fn` `repeats` times and keep the fastest.

    Minimum rather than mean: we want the engine's speed, not the operating
    system's scheduling noise. The returned payload is from the fastest run.
    """
    best_seconds = float("inf")
    best_payload = None
    for _ in range(repeats):
        start = time.perf_counter()
        payload = fn()
        elapsed = time.perf_counter() - start
        if elapsed < best_seconds:
            best_seconds, best_payload = elapsed, payload
    return best_seconds, best_payload


# ---------------------------------------------------------------- the engines

def run_avenue(data: Dataset, case: Case, repeats: int, standard_errors: bool) -> Timing:
    from avenue_model import RatingModel, fit_glm_with_diagnostics, GLMOptions

    objective = {"gaussian": "regression"}.get(case.family, case.family)
    name = "avenue+se" if standard_errors else "avenue"

    def prep():
        frame = {name: codes.astype(np.float64) for name, codes in data.codes.items()}
        frame["y"] = data.y
        if case.use_weight:
            frame["w"] = data.exposure
        if case.use_offset:
            frame["log_exposure"] = data.log_exposure
        df = pl.DataFrame(frame)

        # An intercept table (one row, no features) plus one step table per
        # factor. Level j is the row with inclusive upper bound j.
        tables = [pl.DataFrame({"Rating_Factor": [0.0]})]
        for factor, levels in data.factors.items():
            tables.append(pl.DataFrame({
                factor: np.arange(levels, dtype=np.float64),
                "Rating_Factor": np.zeros(levels),
            }))
        return df, RatingModel(tables, objective)

    prep_seconds, (df, model) = best_of(prep, repeats)

    options = GLMOptions(
        objective=objective,
        max_iterations=MAX_ITER,
        tolerance=AVENUE_TOL,
        tweedie_power=case.tweedie_power,
        compute_standard_errors=standard_errors,
    )

    def fit():
        return fit_glm_with_diagnostics(
            model,
            df,
            "y",
            weight_col="w" if case.use_weight else None,
            offset_col="log_exposure" if case.use_offset else None,
            options=options,
        )

    fit_seconds, (fitted, diag) = best_of(fit, repeats)

    # Avenue declines to compute standard errors above a parameter cap. Without this
    # the widest cases post a suspiciously cheap "avenue+se" time that is cheap only
    # because no standard errors were produced.
    note = None
    if standard_errors and diag.inference_error is not None:
        note = f"no SEs: {diag.inference_error}"

    mu = np.asarray(fitted.predict(df).to_series(0).to_numpy(), dtype=np.float64)
    if case.use_offset:
        # `predict` returns the tables' own mean; the offset is a property of the
        # observation, not the model, so Avenue leaves it out. Add it back to
        # compare like for like with statsmodels' `fittedvalues`, which includes
        # it. Every offset case here is log-link, so that is a multiplication.
        mu = mu * data.exposure

    return Timing(
        engine=name,
        prep_seconds=prep_seconds,
        fit_seconds=fit_seconds,
        iterations=diag.iterations,
        converged=diag.converged,
        mu=mu,
        note=note,
    )


def run_glum(data: Dataset, case: Case, repeats: int) -> Timing:
    import glum

    if case.family == "tweedie":
        family = glum.TweedieDistribution(case.tweedie_power)
    else:
        family = {"gaussian": "normal"}.get(case.family, case.family)

    def prep():
        # Pandas categoricals are glum's fast path: tabmat keeps them as a
        # CategoricalMatrix instead of densifying to dummies.
        return pd.DataFrame({
            name: pd.Categorical(codes) for name, codes in data.codes.items()
        })

    prep_seconds, X = best_of(prep, repeats)

    def fit():
        model = glum.GeneralizedLinearRegressor(
            family=family,
            alpha=0.0,
            fit_intercept=True,
            max_iter=MAX_ITER,
            gradient_tol=GLUM_TOL,
            drop_first=True,
        )
        model.fit(
            X,
            data.y,
            sample_weight=data.exposure if case.use_weight else None,
            offset=data.log_exposure if case.use_offset else None,
        )
        return model

    fit_seconds, model = best_of(fit, repeats)

    mu = model.predict(
        X,
        offset=data.log_exposure if case.use_offset else None,
    )
    return Timing(
        engine="glum",
        prep_seconds=prep_seconds,
        fit_seconds=fit_seconds,
        iterations=int(getattr(model, "n_iter_", 0)) or None,
        converged=True,
        mu=np.asarray(mu, dtype=np.float64),
    )


def run_statsmodels(data: Dataset, case: Case, repeats: int) -> Timing:
    import statsmodels.api as sm

    elements = data.n_rows * data.n_parameters
    if elements > STATSMODELS_MAX_ELEMENTS:
        return Timing(
            engine="statsmodels",
            skipped=f"{elements / 1e6:.0f}M-element dense design matrix; "
                    f"oracle role covered at 100k",
        )

    families = {
        "poisson": sm.families.Poisson(),
        "gamma": sm.families.Gamma(link=sm.families.links.Log()),
        "gaussian": sm.families.Gaussian(),
        "tweedie": sm.families.Tweedie(
            link=sm.families.links.Log(), var_power=case.tweedie_power
        ),
    }

    def prep():
        # Dense treatment-coded dummies - the design matrix Avenue never builds.
        blocks = [np.ones((data.n_rows, 1))]
        for name, levels in data.factors.items():
            codes = data.codes[name]
            block = np.zeros((data.n_rows, levels - 1))
            mask = codes > 0
            block[np.flatnonzero(mask), codes[mask] - 1] = 1.0
            blocks.append(block)
        return np.hstack(blocks)

    prep_seconds, X = best_of(prep, repeats)

    def fit():
        model = sm.GLM(
            data.y,
            X,
            family=families[case.family],
            offset=data.log_exposure if case.use_offset else None,
            var_weights=data.exposure if case.use_weight else None,
        )
        return model.fit(maxiter=MAX_ITER, tol=STATSMODELS_TOL)

    fit_seconds, result = best_of(fit, repeats)

    return Timing(
        engine="statsmodels",
        prep_seconds=prep_seconds,
        fit_seconds=fit_seconds,
        iterations=int(result.fit_history["iteration"]),
        converged=bool(result.converged),
        mu=np.asarray(result.fittedvalues, dtype=np.float64),
    )


# ----------------------------------------------------------------- the runner

def check_agreement(timings: list[Timing]) -> str | None:
    """Fill in each engine's deviation from the reference engine's fitted means.

    Coefficients are not comparable - Avenue's table model is deliberately
    over-parameterised, so its intercept and levels differ from a treatment-coded
    design by a constant per table. Fitted means are the invariant.

    Reported as max absolute deviation over the RMS of the reference means, which
    is meaningful whether or not the fitted values are positive.
    """
    # Prefer an engine that is not us: checking Avenue against Avenue proves nothing.
    # statsmodels first (a third implementation, and the one the Rust test suite is
    # pinned to), then glum, and only fall back to whatever ran if neither did.
    reference = None
    for preferred in ("statsmodels", "glum"):
        reference = next(
            (t for t in timings if t.engine == preferred and t.mu is not None), None
        )
        if reference is not None:
            break
    if reference is None:
        reference = next((t for t in timings if t.mu is not None), None)
    if reference is None:
        return None

    # Scale by the RMS of the reference means, not pointwise by |mu|. Identity-link
    # fitted values pass through zero, where a pointwise relative error is unbounded
    # no matter how good the fit is - it reports a disagreement of 1e1 for two fits
    # whose deviances agree to 1e-15.
    scale = float(np.sqrt(np.mean(reference.mu**2)))
    if scale <= 0.0:
        scale = 1.0
    for timing in timings:
        if timing.mu is None:
            continue
        timing.disagreement = float(np.max(np.abs(timing.mu - reference.mu)) / scale)

    return reference.engine


def format_table(case: Case, timings: list[Timing], reference_engine: str) -> str:
    lines = [
        "",
        f"  {case.label}  ({case.n_rows:,} rows, {case.n_parameters:,} parameters,"
        f" {len(case.structure)} tables)",
        f"  {'engine':<14}{'prep':>9}{'fit':>9}{'total':>9}{'peak MB':>10}"
        f"{'iters':>7}{'vs ' + reference_engine:>14}",
        f"  {'-' * 72}",
    ]
    for t in timings:
        if t.skipped:
            lines.append(f"  {t.engine:<14}{'skipped':>10}   ({t.skipped})")
            continue
        if t.error:
            lines.append(f"  {t.engine:<14}{'FAILED':>10}   {t.error}")
            continue

        agreement = "reference" if t.engine == reference_engine else (
            f"{t.disagreement:.1e}" if t.disagreement is not None else "-"
        )
        flag = ""
        if t.converged is False:
            flag = "  DID NOT CONVERGE"
        elif t.disagreement is not None and t.disagreement > AGREEMENT_TOL:
            flag = "  DISAGREES"
        if t.note:
            flag += f"  [{t.note}]"

        memory = f"{t.peak_memory_mb:.0f}" if t.peak_memory_mb is not None else "-"
        lines.append(
            f"  {t.engine:<14}"
            f"{t.prep_seconds:>9.3f}"
            f"{t.fit_seconds:>9.3f}"
            f"{t.total_seconds:>9.3f}"
            f"{memory:>10}"
            f"{t.iterations if t.iterations is not None else '-':>7}"
            f"{agreement:>14}"
            f"{flag}"
        )
    return "\n".join(lines)


def run_case(case: Case, repeats: int) -> tuple[list[Timing], str]:
    data = make_dataset(case.n_rows, case.family, case.structure)

    engines = [
        ("avenue", lambda: run_avenue(data, case, repeats, standard_errors=False)),
        ("glum", lambda: run_glum(data, case, repeats)),
        ("statsmodels", lambda: run_statsmodels(data, case, repeats)),
        ("avenue+se", lambda: run_avenue(data, case, repeats, standard_errors=True)),
    ]

    timings = []
    for name, engine in engines:
        try:
            with PeakMemory() as mem:
                timing = engine()
            timing.peak_memory_mb = mem.peak_mb
            timings.append(timing)
        except Exception as exc:  # a failing engine must not hide the others
            timings.append(Timing(engine=name, error=f"{type(exc).__name__}: {exc}"))

    return timings, check_agreement(timings) or "-"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="small sizes only")
    parser.add_argument("--width", action="store_true",
                        help="sweep design-matrix width at a fixed row count")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--json", type=str, default=None)
    args = parser.parse_args()

    def poisson(n_rows: int, factors=None, name=None) -> Case:
        return Case("poisson", n_rows, use_offset=True, use_weight=False,
                    factors=factors, name=name)

    if args.quick:
        cases = [
            Case("poisson", 100_000, use_offset=True, use_weight=False),
            Case("gamma", 100_000, use_offset=False, use_weight=True),
        ]
    elif args.width:
        # Two ways for a design matrix to get wide, at a fixed number of rows.
        # Avenue's per-sweep cost should be flat in the first and linear in the
        # second; the design-matrix solve is cubic in the parameter count in both.
        n = 500_000
        cases = [
            poisson(n, name="narrow          "),
            poisson(n, deep_factors(400, 150), "deep levels     "),
            poisson(n, deep_factors(1_500, 500), "deeper levels   "),
            poisson(n, deep_factors(4_000, 1_000), "deepest levels  "),
            poisson(n, many_factors(25), "many tables     "),
            poisson(n, many_factors(75), "more tables     "),
            poisson(n, many_factors(150), "most tables     "),
        ]
    else:
        cases = [
            # Family sweep at a fixed size.
            Case("poisson", 1_000_000, use_offset=True, use_weight=False),
            Case("gamma", 1_000_000, use_offset=False, use_weight=True),
            Case("tweedie", 1_000_000, use_offset=False, use_weight=True),
            Case("gaussian", 1_000_000, use_offset=False, use_weight=True),
            # Size sweep on the flagship insurance case.
            Case("poisson", 100_000, use_offset=True, use_weight=False),
            Case("poisson", 5_000_000, use_offset=True, use_weight=False),
        ]

    import glum
    import statsmodels

    print("Avenue table GLM vs glum vs statsmodels")
    print(f"  python       {sys.version.split()[0]}")
    print(f"  platform     {platform.platform()}")
    print(f"  processor    {platform.processor()}")
    print(f"  glum         {glum.__version__}")
    print(f"  statsmodels  {statsmodels.__version__}")
    print(f"  numpy        {np.__version__}")
    print(f"  cores        {os.cpu_count()} logical (every engine left at its default"
          f" threading)")
    print(f"  repeats      {args.repeats} (fastest reported)")
    print("  note         statsmodels' IRLS solves each weighted least squares step")
    print("               with a pseudo-inverse (SVD). That is its default and what a")
    print("               user gets out of the box, and it is most of the gap below.")

    all_results = []
    failures = []
    for case in cases:
        timings, reference = run_case(case, args.repeats)
        print(format_table(case, timings, reference))

        for t in timings:
            if t.error:
                failures.append(f"{case.label}: {t.engine} failed - {t.error}")
            elif t.converged is False:
                failures.append(f"{case.label}: {t.engine} did not converge")
            elif t.disagreement is not None and t.disagreement > AGREEMENT_TOL:
                failures.append(
                    f"{case.label}: {t.engine} disagrees with {reference} "
                    f"by {t.disagreement:.2e}"
                )

        all_results.append({
            "case": asdict(case),
            "reference": reference,
            "timings": [t.to_json() for t in timings],
        })

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(all_results, fh, indent=2)
        print(f"\nWrote {args.json}")

    if failures:
        print("\nPROBLEMS:")
        for f in failures:
            print(f"  {f}")
        return 1

    print("\nAll engines agreed on fitted means.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
