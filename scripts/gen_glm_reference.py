"""Generate reference GLM fits with statsmodels and emit them as Rust constants.

Avenue's rating tables are deliberately over-parameterised (an intercept table plus
every level of every factor), so raw coefficients are not comparable to statsmodels'
treatment-coded ones. Two things ARE invariant to the parameterisation, and those are
what we pin:

  * fitted values (mu_hat) for every row
  * within-table level contrasts, beta_j - beta_0

Run:  python scripts/gen_glm_reference.py
Emits: src/tests/glm_reference_data.rs
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import numpy as np
import statsmodels.api as sm

SEED = 20260826
OUT = Path(__file__).resolve().parents[1] / "src" / "tests" / "glm_reference_data.rs"

# Level counts for the two factors used by every dataset below.
N_X1, N_X2 = 3, 4
N_ROWS = 240


def make_design(rng: np.random.Generator):
    """A balanced-ish two-factor design shared by all families.

    x1 in {1,2,3}, x2 in {1,2,3,4}. Both are stored as f64 so they land in Avenue's
    numeric (threshold) tables with bounds [1,2,inf] and [1,2,3,inf].
    """
    x1 = rng.integers(1, N_X1 + 1, N_ROWS).astype(float)
    x2 = rng.integers(1, N_X2 + 1, N_ROWS).astype(float)
    # Prior weights: vary them so weighted paths are genuinely exercised.
    weight = rng.uniform(0.5, 3.0, N_ROWS).round(4)
    return x1, x2, weight


def dummies(x: np.ndarray, n_levels: int) -> np.ndarray:
    """Treatment coding: drop level 1, columns for levels 2..n."""
    return np.column_stack([(x == lv).astype(float) for lv in range(2, n_levels + 1)])


def design_matrix(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    return np.column_stack(
        [np.ones(len(x1)), dummies(x1, N_X1), dummies(x2, N_X2)]
    )


def true_eta(x1: np.ndarray, x2: np.ndarray, b0: float, b1: list, b2: list) -> np.ndarray:
    eta = np.full(len(x1), b0)
    for lv, b in zip(range(2, N_X1 + 1), b1):
        eta += b * (x1 == lv)
    for lv, b in zip(range(2, N_X2 + 1), b2):
        eta += b * (x2 == lv)
    return eta


def fit(family, X, y, weight, offset=None):
    model = sm.GLM(y, X, family=family, var_weights=weight, offset=offset)
    res = model.fit(tol=1e-13, maxiter=200, scale=1.0)
    return res


def contrasts(res):
    """params[1:] are already the level contrasts vs level 1, in table order:
    x1 levels 2..3, then x2 levels 2..4."""
    p = res.params
    x1_c = [0.0] + list(p[1 : N_X1])
    x2_c = [0.0] + list(p[N_X1 : N_X1 + N_X2 - 1])
    return x1_c, x2_c


def _full(v: float) -> str:
    return f"{v:.17e}"


def _short(v: float) -> str:
    """Compact literal that is still unambiguously an f64 to rustc."""
    return f"{v:.1f}" if float(v).is_integer() else f"{v:g}"


def rust_vec(vals, per_line=6, indent=8, fmt=_full):
    body = []
    vals = list(vals)
    for i in range(0, len(vals), per_line):
        chunk = ", ".join(fmt(v) for v in vals[i : i + per_line])
        body.append(" " * indent + chunk + ",")
    return "\n".join(body)


def rust_short(vals, per_line=16, indent=8):
    """Compact form for inputs that are small integers or 4-dp rounded."""
    return rust_vec(vals, per_line=per_line, indent=indent, fmt=_short)


def emit_case(name, x1, x2, y, weight, offset, res, extra_doc=""):
    x1_c, x2_c = contrasts(res)
    mu = res.fittedvalues
    off = offset if offset is not None else np.zeros(len(y))

    # Inputs that are exactly representable in short form stay short, to keep the
    # generated file small; anything continuous keeps full round-trip precision.
    y_fmt = rust_short(y) if np.all(y == np.floor(y)) else rust_vec(y)
    off_fmt = rust_short(off) if np.all(off == 0.0) else rust_vec(off)

    def arr(const_name, n, body, doc=None):
        lines = []
        if doc:
            lines.append(f"    /// {doc}")
        lines.append(f"    pub const {const_name}: [f64; {n}] = [")
        lines.append(body)
        lines.append("    ];")
        return "\n".join(lines)

    parts = [
        f"/// {name}{(' - ' + extra_doc) if extra_doc else ''}",
        f"/// statsmodels {sm.__version__}; deviance = {res.deviance:.12e}",
        f"pub struct {name};",
        "",
        f"impl {name} {{",
        arr("X1", len(x1), rust_short(x1)),
        arr("X2", len(x2), rust_short(x2)),
        arr("Y", len(y), y_fmt),
        arr("WEIGHT", len(weight), rust_short(weight)),
        arr("OFFSET", len(off), off_fmt),
        arr("MU", len(mu), rust_vec(mu),
            "Fitted means from statsmodels, one per row."),
        arr("X1_CONTRASTS", len(x1_c), rust_vec(x1_c),
            "Level contrasts within table x1, relative to level 1 (which is 0.0)."),
        arr("X2_CONTRASTS", len(x2_c), rust_vec(x2_c),
            "Level contrasts within table x2, relative to level 1 (which is 0.0)."),
        f"    pub const DEVIANCE: f64 = {res.deviance:.17e};",
        "}",
        "",
    ]
    return "\n".join(parts)


def main():
    rng = np.random.default_rng(SEED)
    x1, x2, weight = make_design(rng)
    X = design_matrix(x1, x2)

    blocks = []

    # ---- Gaussian / identity ------------------------------------------------
    eta = true_eta(x1, x2, 10.0, [1.5, -2.0], [0.5, 3.0, -1.0])
    y = eta + rng.normal(0, 1.0, N_ROWS)
    res = fit(sm.families.Gaussian(sm.families.links.Identity()), X, y, weight)
    blocks.append(emit_case("GaussianTwoFactor", x1, x2, y, weight, None, res,
                            "identity link, prior weights"))

    # ---- Poisson / log ------------------------------------------------------
    eta = true_eta(x1, x2, 0.8, [0.4, -0.3], [0.15, 0.6, -0.25])
    y = rng.poisson(np.exp(eta)).astype(float)
    res = fit(sm.families.Poisson(sm.families.links.Log()), X, y, weight)
    blocks.append(emit_case("PoissonTwoFactor", x1, x2, y, weight, None, res,
                            "log link, prior weights"))

    # ---- Poisson / log with offset -----------------------------------------
    exposure = rng.uniform(0.2, 2.0, N_ROWS).round(4)
    offset = np.log(exposure)
    eta = true_eta(x1, x2, 0.8, [0.4, -0.3], [0.15, 0.6, -0.25])
    y = rng.poisson(exposure * np.exp(eta)).astype(float)
    res = fit(sm.families.Poisson(sm.families.links.Log()), X, y,
              np.ones(N_ROWS), offset=offset)
    blocks.append(emit_case("PoissonOffset", x1, x2, y, np.ones(N_ROWS), offset, res,
                            "log link, log(exposure) offset, unit weights"))

    # ---- Gamma / log --------------------------------------------------------
    eta = true_eta(x1, x2, 2.0, [0.3, -0.5], [0.2, 0.7, -0.4])
    shape = 4.0
    y = rng.gamma(shape, np.exp(eta) / shape)
    res = fit(sm.families.Gamma(sm.families.links.Log()), X, y, weight)
    blocks.append(emit_case("GammaTwoFactor", x1, x2, y, weight, None, res,
                            "log link, prior weights"))

    # ---- Binomial / logit ---------------------------------------------------
    eta = true_eta(x1, x2, -0.5, [0.8, -0.6], [0.3, 1.2, -0.9])
    y = rng.binomial(1, 1.0 / (1.0 + np.exp(-eta))).astype(float)
    res = fit(sm.families.Binomial(sm.families.links.Logit()), X, y, weight)
    blocks.append(emit_case("BinaryTwoFactor", x1, x2, y, weight, None, res,
                            "logit link, prior weights"))

    # ---- Tweedie(1.5) / log -------------------------------------------------
    p = 1.5
    eta = true_eta(x1, x2, 1.0, [0.35, -0.45], [0.25, 0.55, -0.3])
    mu = np.exp(eta)
    # Compound Poisson-Gamma draw with variance power p.
    phi = 1.0
    lam = mu ** (2 - p) / (phi * (2 - p))
    alpha = (2 - p) / (p - 1)
    beta_scale = phi * (p - 1) * mu ** (p - 1)
    n_claims = rng.poisson(lam)
    y = np.array([
        rng.gamma(alpha * n, beta_scale[i]) if n > 0 else 0.0
        for i, n in enumerate(n_claims)
    ])
    res = fit(sm.families.Tweedie(sm.families.links.Log(), var_power=p), X, y, weight)
    blocks.append(emit_case("TweedieTwoFactor", x1, x2, y, weight, None, res,
                            "log link, var_power=1.5, prior weights"))

    header = textwrap.dedent(f"""\
        // @generated by scripts/gen_glm_reference.py — DO NOT EDIT BY HAND.
        //
        // Reference GLM fits produced by statsmodels {sm.__version__} (numpy {np.__version__}).
        // Seed = {SEED}. Regenerate with:
        //
        //     python scripts/gen_glm_reference.py
        //
        // Avenue's table parameterisation is over-parameterised relative to statsmodels'
        // treatment coding, so we pin the two things that do not depend on the
        // parameterisation: fitted means per row, and level contrasts within each table.
        #![allow(dead_code)]

        /// Level thresholds for the x1 table: values 1, 2, 3 map to rows 0, 1, 2.
        pub const X1_BOUNDS: [f64; {N_X1}] = [1.0, 2.0, f64::INFINITY];
        /// Level thresholds for the x2 table: values 1, 2, 3, 4 map to rows 0, 1, 2, 3.
        pub const X2_BOUNDS: [f64; {N_X2}] = [1.0, 2.0, 3.0, f64::INFINITY];

        """)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(header + "\n".join(blocks), encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
