"""Check Avenue's penalised fits against glum on the same objective.

The two engines parameterise a categorical factor differently — Avenue carries an
intercept *and* a factor per level, glum drops a reference level — so raw coefficients
are not comparable. What is comparable, and what the penalty is defined on, is the
contrast of each level against the level glum dropped. Avenue shrinks toward its base
level; glum shrinks toward zero, which *is* its dropped reference. Same problem.

glum minimises

    deviance / (2 * sum(w))  +  alpha * (l1_ratio * |b|_1 + (1 - l1_ratio)/2 * |b|^2)

so `alpha` should mean the same number in both. That is the claim under test here, and
it is a claim about a constant that is easy to be quietly wrong about — hence checking
it across several alphas rather than one, since a mis-scaled alpha would still produce
a plausible-looking fit at any single point.

Usage:  python scripts/check_penalty.py [--family poisson] [--rows 20000]
"""

import argparse
import sys

import numpy as np
import polars as pl

MAX_ITER = 20000
TOL = 1e-11


def make_data(rows, family, seed=0):
    """A design with levels of deliberately uneven weight and one nearly-null factor.

    The uneven weight is what makes shrinkage visible: a thin level moves a long way
    and a heavy one barely moves, so a penalty that was being applied uniformly, or
    against the wrong anchor, shows up as a disagreement rather than as noise.
    """
    rng = np.random.default_rng(seed)
    levels = {"a": 6, "b": 4, "c": 8}
    codes = {}
    # Skewed level frequencies, so some levels are thin.
    for name, k in levels.items():
        p = np.linspace(1.0, 6.0, k)
        p = p / p.sum()
        codes[name] = rng.choice(k, size=rows, p=p).astype(np.int32)

    eta = (
        np.linspace(-0.4, 0.5, levels["a"])[codes["a"]]
        + np.linspace(0.0, 0.3, levels["b"])[codes["b"]]
        # `c` carries almost no signal, which is what a lasso should find and drop.
        + np.linspace(0.0, 0.02, levels["c"])[codes["c"]]
    )

    if family == "poisson":
        y = rng.poisson(np.exp(eta + 0.7)).astype(np.float64)
    elif family == "gamma":
        y = rng.gamma(shape=3.0, scale=np.exp(eta + 0.7) / 3.0)
    elif family == "gaussian":
        y = eta * 4.0 + 10.0 + rng.normal(0.0, 1.5, rows)
    else:
        raise SystemExit(f"unknown family {family}")

    w = rng.gamma(shape=4.0, scale=0.25, size=rows)
    return codes, levels, y, w


def fit_avenue(codes, levels, y, w, family, alpha, l1_ratio):
    from avenue_model import RatingModel, fit_glm_with_diagnostics, GLMOptions

    frame = {name: codes[name] for name in levels}
    frame["y"] = y
    frame["w"] = w
    tables = [pl.DataFrame({"Rating_Factor": [0.0]})]
    for name, k in levels.items():
        tables.append(
            pl.DataFrame(
                {
                    name: np.arange(k, dtype=np.int32),
                    "Rating_Factor": np.zeros(k),
                }
            )
        )
    model = RatingModel(tables, family)
    options = GLMOptions(
        objective=family,
        max_iterations=MAX_ITER,
        tolerance=TOL,
        alpha=alpha,
        l1_ratio=l1_ratio,
        compute_standard_errors=False,
    )
    fitted, diag = fit_glm_with_diagnostics(
        model, pl.DataFrame(frame), "y", weight_col="w", options=options
    )
    out = {}
    for t, name in enumerate(levels, start=1):
        f = np.asarray(fitted.model_tables()[t]["Rating_Factor"])
        out[name] = f - f[0]  # contrasts against the base level
    return out, diag


def fit_glum(codes, levels, y, w, family, alpha, l1_ratio):
    from glum import GeneralizedLinearRegressor

    # One-hot with the first level dropped: exactly the parameterisation Avenue's
    # contrasts are measured in.
    cols = []
    index = {}
    for name, k in levels.items():
        start = len(cols)
        for level in range(1, k):
            cols.append((codes[name] == level).astype(np.float64))
        index[name] = (start, k)
    X = np.column_stack(cols)

    kwargs = dict(
        family=family,
        alpha=alpha,
        l1_ratio=l1_ratio,
        fit_intercept=True,
        gradient_tol=1e-10,
        max_iter=10000,
        # scale_predictors=True would rescale the penalty per column; Avenue penalises
        # the coefficients as they are, so it must stay off for the two objectives to
        # be the same one.
        scale_predictors=False,
    )
    model = GeneralizedLinearRegressor(**kwargs)
    model.fit(X, y, sample_weight=w)

    out = {}
    for name, (start, k) in index.items():
        out[name] = np.concatenate([[0.0], model.coef_[start : start + k - 1]])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="poisson",
                    choices=["poisson", "gamma", "gaussian"])
    ap.add_argument("--rows", type=int, default=20000)
    ap.add_argument("--tol", type=float, default=2e-5)
    args = ap.parse_args()

    codes, levels, y, w = make_data(args.rows, args.family)
    print(f"{args.family}, {args.rows} rows, "
          f"levels {', '.join(f'{k}={v}' for k, v in levels.items())}\n")

    cases = [
        ("ridge", 0.0),
        ("elastic net", 0.5),
        ("lasso", 1.0),
    ]
    alphas = [1e-5, 1e-4, 1e-3, 1e-2]

    worst_overall = 0.0
    failures = 0
    for label, ratio in cases:
        print(f"{label} (l1_ratio = {ratio})")
        print(f"  {'alpha':>8}  {'max |diff|':>11}  {'zeros A/g':>10}  {'sweeps':>6}")
        for alpha in alphas:
            av, diag = fit_avenue(codes, levels, y, w, args.family, alpha, ratio)
            gl = fit_glum(codes, levels, y, w, args.family, alpha, ratio)
            worst = 0.0
            zeros_a = zeros_g = 0
            for name in levels:
                worst = max(worst, float(np.max(np.abs(av[name] - gl[name]))))
                zeros_a += int(np.sum(av[name][1:] == 0.0))
                zeros_g += int(np.sum(gl[name][1:] == 0.0))
            worst_overall = max(worst_overall, worst)
            flag = "" if worst <= args.tol else "   <-- MISMATCH"
            if worst > args.tol:
                failures += 1
            converged = "" if diag.converged else " (not converged)"
            print(f"  {alpha:>8.0e}  {worst:>11.2e}  {zeros_a:>4}/{zeros_g:<5}"
                  f"  {diag.iterations:>6}{flag}{converged}")
        print()

    print(f"worst disagreement anywhere: {worst_overall:.3e} "
          f"(tolerance {args.tol:.0e})")
    if failures:
        print(f"FAILED: {failures} of {len(cases) * len(alphas)} cases disagree")
        return 1
    print("OK: Avenue and glum agree on every penalised fit")
    return 0


if __name__ == "__main__":
    sys.exit(main())
