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
    # Leave `scale` at the family default: 1 for Poisson and Binomial, Pearson
    # chi-squared over residual degrees of freedom for Gaussian, Gamma and Tweedie.
    # Scale does not affect the fitted mean, only the standard errors.
    model = sm.GLM(y, X, family=family, var_weights=weight, offset=offset)
    res = model.fit(tol=1e-13, maxiter=200)
    return res


def contrasts(res):
    """params[1:] are already the level contrasts vs level 1, in table order:
    x1 levels 2..3, then x2 levels 2..4."""
    p = res.params
    x1_c = [0.0] + list(p[1 : N_X1])
    x2_c = [0.0] + list(p[N_X1 : N_X1 + N_X2 - 1])
    return x1_c, x2_c


def standard_errors(res):
    """Standard errors laid out the way Avenue reports them: the intercept, then
    each table's rows with the base level pinned at exactly 0."""
    b = res.bse
    intercept = b[0]
    x1_se = [0.0] + list(b[1 : N_X1])
    x2_se = [0.0] + list(b[N_X1 : N_X1 + N_X2 - 1])
    return intercept, x1_se, x2_se


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
    int_se, x1_se, x2_se = standard_errors(res)
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
        arr("X1_SE", len(x1_se), rust_vec(x1_se),
            "Standard errors of the x1 contrasts; the base level is exactly 0."),
        arr("X2_SE", len(x2_se), rust_vec(x2_se),
            "Standard errors of the x2 contrasts; the base level is exactly 0."),
        f"    pub const DEVIANCE: f64 = {res.deviance:.17e};",
        f"    pub const INTERCEPT_SE: f64 = {int_se:.17e};",
        "    /// Dispersion statsmodels used: 1 for Poisson and Binomial, Pearson",
        "    /// chi-squared over residual degrees of freedom otherwise.",
        f"    pub const SCALE: f64 = {res.scale:.17e};",
        f"    pub const DF_RESID: f64 = {res.df_resid:.17e};",
        "    /// Log-likelihood. Meaningless for Tweedie, whose density has no closed",
        "    /// form; statsmodels substitutes an approximation there and Avenue does not.",
        f"    pub const LLF: f64 = {res.llf:.17e};",
        f"    pub const AIC: f64 = {res.aic:.17e};",
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

    # ---- Linear variate ----------------------------------------------------
    #
    # A five-row age table whose factors are tied to one slope. The equivalent GLM
    # design is an ordinary continuous covariate whose value, for each record, is the
    # value attached to the age band that record falls in. So statsmodels fits
    # [intercept, x1 dummies, z] and its coefficient on z is the slope Avenue should
    # recover; each row's factor is that slope times (value[r] - value[base]).
    age_bounds = [20.0, 30.0, 40.0, 50.0, float("inf")]
    age_values = [20.0, 30.0, 40.0, 50.0, 65.0]

    def band_of(x: float) -> int:
        for i, b in enumerate(age_bounds):
            if x <= b:
                return i
        return len(age_bounds) - 1

    rng_v = np.random.default_rng(SEED + 7)
    x1v = rng_v.integers(1, N_X1 + 1, N_ROWS).astype(float)
    age = rng_v.uniform(18.0, 80.0, N_ROWS).round(4)
    wv = rng_v.uniform(0.5, 3.0, N_ROWS).round(4)
    z = np.array([age_values[band_of(a)] for a in age])

    Xv = np.column_stack([np.ones(N_ROWS), dummies(x1v, N_X1), z])
    eta = 0.5 + 0.4 * (x1v == 2) - 0.3 * (x1v == 3) + 0.02 * z
    yv = rng_v.poisson(np.exp(eta)).astype(float)

    res_v = fit(sm.families.Poisson(sm.families.links.Log()), Xv, yv, wv)
    p, b = res_v.params, res_v.bse
    x1_c = [0.0] + list(p[1:N_X1])
    x1_se_v = [0.0] + list(b[1:N_X1])

    variate_block = "\n".join([
        "/// LinearVariate - log link; a 3-level step factor plus a 5-row age table",
        "/// whose factors are constrained to a single slope.",
        f"/// statsmodels {sm.__version__}; deviance = {res_v.deviance:.12e}",
        "pub struct LinearVariate;",
        "",
        "impl LinearVariate {",
        "    /// Inclusive upper bounds for the age table's rows.",
        "    pub const AGE_BOUNDS: [f64; 5] = [20.0, 30.0, 40.0, 50.0, f64::INFINITY];",
        "    /// What each row is worth on the age scale. The last stands in for the",
        "    /// open-ended top band.",
        "    pub const AGE_VALUES: [f64; 5] = [20.0, 30.0, 40.0, 50.0, 65.0];",
        f"    pub const X1: [f64; {N_ROWS}] = [", rust_short(x1v), "    ];",
        f"    pub const AGE: [f64; {N_ROWS}] = [", rust_vec(age), "    ];",
        f"    pub const Y: [f64; {N_ROWS}] = [", rust_short(yv), "    ];",
        f"    pub const WEIGHT: [f64; {N_ROWS}] = [", rust_short(wv), "    ];",
        "    /// Fitted means from statsmodels.",
        f"    pub const MU: [f64; {N_ROWS}] = [", rust_vec(res_v.fittedvalues), "    ];",
        "    /// Step-table level contrasts, relative to level 1.",
        f"    pub const X1_CONTRASTS: [f64; {len(x1_c)}] = [", rust_vec(x1_c), "    ];",
        f"    pub const X1_SE: [f64; {len(x1_se_v)}] = [", rust_vec(x1_se_v), "    ];",
        "    /// Coefficient on the age value: the slope the variate table encodes.",
        f"    pub const SLOPE: f64 = {p[-1]:.17e};",
        f"    pub const SLOPE_SE: f64 = {b[-1]:.17e};",
        f"    pub const DEVIANCE: f64 = {res_v.deviance:.17e};",
        f"    pub const INTERCEPT_SE: f64 = {b[0]:.17e};",
        f"    pub const SCALE: f64 = {res_v.scale:.17e};",
        f"    pub const DF_RESID: f64 = {res_v.df_resid:.17e};",
        "}",
        "",
    ])
    blocks.append(variate_block)

    # ---- Quadratic variate -------------------------------------------------
    #
    # Same age table, but the factors follow a curve rather than a line. The
    # equivalent GLM design carries both z and z^2, where z is the band value. Avenue
    # solves the two powers jointly on a rescaled basis; the raw-scale coefficients it
    # reports must match statsmodels' two coefficients here.
    rng_q = np.random.default_rng(SEED + 11)
    x1q = rng_q.integers(1, N_X1 + 1, N_ROWS).astype(float)
    ageq = rng_q.uniform(18.0, 80.0, N_ROWS).round(4)
    wq = rng_q.uniform(0.5, 3.0, N_ROWS).round(4)
    zq = np.array([age_values[band_of(a)] for a in ageq])

    Xq = np.column_stack([np.ones(N_ROWS), dummies(x1q, N_X1), zq, zq ** 2])
    # A U shape in age: cheap in the middle, dearer at both ends.
    eta = 0.9 + 0.35 * (x1q == 2) - 0.25 * (x1q == 3) - 0.06 * zq + 0.0007 * zq ** 2
    yq = rng_q.poisson(np.exp(eta)).astype(float)

    res_q = fit(sm.families.Poisson(sm.families.links.Log()), Xq, yq, wq)
    pq, bq = res_q.params, res_q.bse
    x1_cq = [0.0] + list(pq[1:N_X1])
    x1_seq = [0.0] + list(bq[1:N_X1])

    quadratic_block = "\n".join([
        "/// QuadraticVariate - log link; a 3-level step factor plus a 5-row age table",
        "/// whose factors are constrained to a degree-2 polynomial.",
        f"/// statsmodels {sm.__version__}; deviance = {res_q.deviance:.12e}",
        "pub struct QuadraticVariate;",
        "",
        "impl QuadraticVariate {",
        "    pub const AGE_BOUNDS: [f64; 5] = [20.0, 30.0, 40.0, 50.0, f64::INFINITY];",
        "    pub const AGE_VALUES: [f64; 5] = [20.0, 30.0, 40.0, 50.0, 65.0];",
        f"    pub const X1: [f64; {N_ROWS}] = [", rust_short(x1q), "    ];",
        f"    pub const AGE: [f64; {N_ROWS}] = [", rust_vec(ageq), "    ];",
        f"    pub const Y: [f64; {N_ROWS}] = [", rust_short(yq), "    ];",
        f"    pub const WEIGHT: [f64; {N_ROWS}] = [", rust_short(wq), "    ];",
        "    /// Fitted means from statsmodels.",
        f"    pub const MU: [f64; {N_ROWS}] = [", rust_vec(res_q.fittedvalues), "    ];",
        f"    pub const X1_CONTRASTS: [f64; {len(x1_cq)}] = [", rust_vec(x1_cq), "    ];",
        f"    pub const X1_SE: [f64; {len(x1_seq)}] = [", rust_vec(x1_seq), "    ];",
        "    /// Raw-scale polynomial coefficients: [beta_1 on z, beta_2 on z^2].",
        f"    pub const COEFFICIENTS: [f64; 2] = [{pq[-2]:.17e}, {pq[-1]:.17e}];",
        f"    pub const DEVIANCE: f64 = {res_q.deviance:.17e};",
        f"    pub const INTERCEPT_SE: f64 = {bq[0]:.17e};",
        f"    pub const SCALE: f64 = {res_q.scale:.17e};",
        f"    pub const DF_RESID: f64 = {res_q.df_resid:.17e};",
        "}",
        "",
    ])
    blocks.append(quadratic_block)

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
