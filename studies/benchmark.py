"""Reproduce published-wheel benchmarks against glum in isolated processes.

Download inputs explicitly with ``--download --data-dir /path``. The runner never
installs packages. Run ``--help`` for the case and thread grids. Raw predictions
stay in the output directory; only compact JSON evidence belongs in the repo.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
               "NUMEXPR_NUM_THREADS", "RAYON_NUM_THREADS", "POLARS_MAX_THREADS")
REAL_CASES = ("motor", "motor_wide", "taxi", "census", "housing_gamma", "housing_gaussian")
CASES = (*REAL_CASES, "tweedie", "tweedie_ridge", "tweedie_elastic", "tweedie_lasso",
         "large_few", "large_many")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def download(data_dir):
    from sklearn.datasets import fetch_openml
    import urllib.request
    data_dir.mkdir(parents=True, exist_ok=True)
    for name, query in [("motor", {"data_id": 41214}), ("severity", {"data_id": 41215}),
                        ("census", {"data_id": 1590}), ("housing", {"name": "house_sales", "version": 1})]:
        path = data_dir / f"{name}.parquet"
        if not path.exists():
            fetch_openml(**query, as_frame=True, parser="auto", data_home=str(data_dir / "openml")).frame.to_parquet(path)
    taxi = data_dir / "taxi.parquet"
    if not taxi.exists():
        urllib.request.urlretrieve("https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet", taxi)


def prepare_motor(df: pd.DataFrame, wide: bool, drop: tuple[str, ...]=()) -> tuple[dict[str, np.ndarray], dict[str, int], np.ndarray, np.ndarray]:
    import numpy as np
    import pandas as pd
    df = df.copy()
    df['ClaimNb'] = df['ClaimNb'].astype(float).clip(upper=4)
    df['Exposure'] = df['Exposure'].astype(float).clip(lower=0.001, upper=1)

    def band(series: pd.Series, edges) -> np.ndarray:
        return np.digitize(series.to_numpy(dtype=float), edges)

    def categorical(series: pd.Series) -> np.ndarray:
        return pd.Categorical(series).codes.astype(np.int64)
    if wide:
        veh_age = band(df['VehAge'], np.arange(0, 31, 1))
        driv_age = band(df['DrivAge'], np.arange(18, 91, 1))
        bonus = band(df['BonusMalus'], np.arange(50, 231, 2))
        density = band(np.log(df['Density'].astype(float)), np.linspace(0, 11, 60))
        power = band(df['VehPower'], np.arange(4, 16, 1))
    else:
        veh_age = band(df['VehAge'], [1, 2, 3, 5, 7, 10, 15, 20])
        driv_age = band(df['DrivAge'], [21, 26, 31, 41, 51, 61, 71, 81])
        bonus = band(df['BonusMalus'], [51, 55, 60, 70, 80, 90, 100, 120, 150])
        density = band(np.log(df['Density'].astype(float)), [2, 3, 4, 5, 6, 7, 8, 9, 10])
        power = band(df['VehPower'], [5, 6, 7, 8, 9, 10, 12])
    codes = {'veh_age': veh_age, 'driv_age': driv_age, 'bonus_malus': bonus, 'density': density, 'veh_power': power, 'area': categorical(df['Area']), 'veh_brand': categorical(df['VehBrand']), 'veh_gas': categorical(df['VehGas']), 'region': categorical(df['Region'])}
    for name in drop:
        if name not in codes:
            raise ValueError(f'unknown table {name!r}; have {sorted(codes)}')
        del codes[name]
    levels = {}
    for name, values in codes.items():
        uniques, compacted = np.unique(values, return_inverse=True)
        codes[name] = compacted.astype(np.int64)
        levels[name] = len(uniques)
    return (codes, levels, df['ClaimNb'].to_numpy(dtype=float), df['Exposure'].to_numpy(dtype=float))
FEATURES = ['bedrooms', 'bathrooms', 'sqft_living', 'floors', 'waterfront', 'view', 'condition', 'grade', 'yr_built', 'yr_renovated']

def prepare_housing(df: pd.DataFrame, rows: int | None) -> tuple[dict[str, np.ndarray], dict[str, int], np.ndarray]:
    import numpy as np
    import pandas as pd
    df = df.copy()
    if rows is not None and rows != len(df):
        rng = np.random.default_rng(20260826)
        df = df.iloc[rng.integers(0, len(df), size=rows)].reset_index(drop=True)
    price = pd.to_numeric(df['price']).to_numpy(dtype=float)
    quantiles = np.linspace(0, 1, 17)[1:-1]
    codes: dict[str, np.ndarray] = {}
    for name in FEATURES:
        v = pd.to_numeric(df[name], errors='coerce').to_numpy(dtype=float)
        values, counts = np.unique(v, return_counts=True)
        if values.size <= 16:
            level = pd.Categorical(v).codes.astype(np.int64)
        elif counts.max() > 0.5 * len(v):
            dominant = values[counts.argmax()]
            rest = v[v != dominant]
            edges = np.unique(np.nanquantile(rest, quantiles))
            level = np.digitize(v, edges) + 1
            level[v == dominant] = 0
        else:
            edges = np.unique(np.nanquantile(v, quantiles))
            level = np.digitize(v, edges)
        codes[name] = np.asarray(level, dtype=np.int64)
    levels: dict[str, int] = {}
    for name, v in codes.items():
        uniques, compacted = np.unique(v, return_inverse=True)
        codes[name] = compacted.astype(np.int64)
        levels[name] = len(uniques)
    return (codes, levels, price)

def band(values: np.ndarray, edges) -> np.ndarray:
    import numpy as np
    import pandas as pd
    return np.digitize(np.asarray(values, dtype=float), edges)

def compact(codes: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    import numpy as np
    import pandas as pd
    levels = {}
    out = {}
    for name, values in codes.items():
        uniques, compacted = np.unique(values, return_inverse=True)
        out[name] = compacted.astype(np.int64)
        levels[name] = len(uniques)
    return (out, levels)

def prepare_taxi(df: pd.DataFrame):
    import numpy as np
    import pandas as pd
    keep = (df['fare_amount'] > 0) & (df['fare_amount'] < 250) & (df['trip_distance'] > 0) & (df['trip_distance'] < 100) & df['passenger_count'].notna() & df['RatecodeID'].notna()
    df = df[keep]
    pickup = df['tpep_pickup_datetime']
    duration = (df['tpep_dropoff_datetime'] - pickup).dt.total_seconds() / 60.0
    codes = {'pickup_zone': df['PULocationID'].to_numpy(), 'dropoff_zone': df['DOLocationID'].to_numpy(), 'hour': pickup.dt.hour.to_numpy(), 'weekday': pickup.dt.dayofweek.to_numpy(), 'distance': band(df['trip_distance'], [0.5, 1, 1.5, 2, 3, 4, 5, 7, 10, 15, 20, 30]), 'duration': band(duration.fillna(0.0), [3, 5, 8, 12, 18, 25, 35, 50, 75]), 'passengers': band(df['passenger_count'], [0, 1, 2, 3, 4, 5]), 'rate_code': df['RatecodeID'].to_numpy(), 'payment': df['payment_type'].to_numpy(), 'vendor': df['VendorID'].to_numpy()}
    codes, levels = compact(codes)
    return (codes, levels, df['fare_amount'].to_numpy(dtype=float), None)

def prepare_census(df: pd.DataFrame):
    import numpy as np
    import pandas as pd
    categoricals = [c for c in df.columns if str(df[c].dtype) == 'category']
    df = df.dropna(subset=categoricals)

    def categorical(column: str) -> np.ndarray:
        return pd.Categorical(df[column].astype('object')).codes.astype(np.int64)
    codes = {'age': band(df['age'], [25, 30, 35, 40, 45, 50, 55, 60, 65, 70]), 'workclass': categorical('workclass'), 'education': categorical('education'), 'marital_status': categorical('marital-status'), 'occupation': categorical('occupation'), 'relationship': categorical('relationship'), 'race': categorical('race'), 'sex': categorical('sex'), 'hours': band(df['hours-per-week'], [20, 30, 35, 40, 45, 50, 60]), 'capital_gain': band(df['capital-gain'], [1, 3000, 5000, 7500, 15000]), 'capital_loss': band(df['capital-loss'], [1, 1500, 2000]), 'native_country': categorical('native-country')}
    codes, levels = compact(codes)
    y = (df['class'].astype(str).str.strip() == '>50K').to_numpy(dtype=float)
    return (codes, levels, y, None)

def generate_large(rows: int, tables: int, levels: int, correlation: float):
    import numpy as np
    import pandas as pd
    rng = np.random.default_rng(20260827)
    names = [f'factor_{i:03d}' for i in range(tables)]
    dtype = np.int8 if levels <= 127 else np.int16
    spread = 0.35 / (tables / 5.0) ** 0.5
    eta = np.full(rows, -2.3)
    latent = None
    edges = None
    if correlation > 0.0:
        latent = rng.standard_normal(rows, dtype=np.float32)
        from scipy.stats import norm
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
    return (codes, y, exposure)


def load_case(case, data_dir):
    import numpy as np
    import pandas as pd
    weights = offset = None
    alpha = ratio = 0.0
    sources = []
    audit = {}
    if case.startswith("large_"):
        tables, width = (5, 101) if case == "large_few" else (100, 6)
        codes, y, exposure = generate_large(20_000_000, tables, width, 0.0)
        levels = {k: width for k in codes}
        offset = np.log(exposure)
        family = "poisson"
        audit = {"seed": 20260827, "independent_synthetic_factors": True}
    elif case in ("motor", "motor_wide") or case.startswith("tweedie"):
        sources = [data_dir / "motor.parquet"]
        raw = pd.read_parquet(sources[0])
        codes, levels, y, exposure = prepare_motor(raw, wide=case == "motor_wide")
        family = "poisson"
        offset = np.log(exposure)
        audit = {"claim_count_cap": 4, "exposure_clip": [0.001, 1.0]}
        if case.startswith("tweedie"):
            sources.append(data_dir / "severity.parquet")
            claims = pd.read_parquet(sources[-1])
            assert raw.IDpol.is_unique
            amounts = claims.groupby("IDpol").ClaimAmount.sum()
            loss = raw.IDpol.map(amounts).fillna(0).to_numpy(dtype=float)
            weights = raw.Exposure.to_numpy(dtype=float)
            assert np.all(weights > 0) and np.all(loss >= 0)
            y, offset, family = loss / weights, None, "tweedie"
            alpha = 0.0 if case == "tweedie" else 0.1
            ratio = {"tweedie_elastic": 0.5, "tweedie_lasso": 1.0}.get(case, 0.0)
            audit = {"positive_loss_rows": int(np.sum(loss > 0)), "total_loss": float(loss.sum()),
                     "orphan_claim_rows_excluded": int((~claims.IDpol.isin(raw.IDpol)).sum()),
                     "target": "observed claim amount per policy / raw exposure", "caps": None}
    elif case in ("taxi", "census"):
        sources = [data_dir / f"{case}.parquet"]
        prepare = prepare_taxi if case == "taxi" else prepare_census
        codes, levels, y, _ = prepare(pd.read_parquet(sources[0]))
        family = "gamma" if case == "taxi" else "binary"
    else:
        sources = [data_dir / "housing.parquet"]
        codes, levels, y = prepare_housing(pd.read_parquet(sources[0]), None)
        family = case.removeprefix("housing_")
    return codes, levels, y, weights, offset, family, alpha, ratio, audit, sources


def worker(args):
    # Set pool sizes before importing numerical packages, including indirect imports.
    for key in THREAD_VARS:
        os.environ[key] = str(args.threads[0])
    import importlib.metadata
    import resource
    import warnings
    import numpy as np
    import pandas as pd
    import polars as pl
    import glum
    import avenue_model
    from avenue_model import GLMOptions, RatingModel, fit_glm_with_diagnostics
    from threadpoolctl import threadpool_info

    case, engine = args.cases[0], args.engines[0]
    origin = Path(avenue_model.__file__).resolve()
    assert not origin.is_relative_to(ROOT / "python"), "Install the published wheel first"
    dist = importlib.metadata.distribution("avenue_model")
    direct = json.loads(dist.read_text("direct_url.json") or "{}")
    assert not direct.get("dir_info", {}).get("editable"), "Editable installs are not release evidence"
    assert args.wheel is not None, "Pass the installed wheel for payload verification"
    import zipfile
    with zipfile.ZipFile(args.wheel) as archive:
        payload = [n for n in archive.namelist() if n.startswith("avenue_model/") and not n.endswith("/")]
        assert any(n.endswith((".so", ".pyd")) for n in payload)
        for name in payload:
            assert hashlib.sha256(archive.read(name)).hexdigest() == digest(dist.locate_file(name)), name
    codes, levels, y, weights, offset, family, alpha, ratio, audit, sources = load_case(case, args.data_dir)
    # Match the reference coding exactly; no predictor scaling or intercept penalty.
    tolerance = args.tolerance if args.tolerance is not None else (
        1e-11 if case == "taxi" else (1e-9 if case.startswith("tweedie") else 1e-10))
    if args.tolerance is None and engine.startswith("avenue"):
        if case in ("motor_wide", "census", "housing_gamma"):
            tolerance = 1e-12
        elif case.startswith("tweedie"):
            tolerance = 1e-11
    requested_solver = "table" if engine == "avenue_table" else "global"

    def prepare():
        if engine.startswith("avenue"):
            frame = pl.DataFrame({"y": y})
            for name, c in codes.items():
                frame = frame.with_columns(pl.Series(name, c, dtype=pl.Int32))
            if weights is not None:
                frame = frame.with_columns(pl.Series("weight", weights))
            if offset is not None:
                frame = frame.with_columns(pl.Series("offset", offset))
            tables = [pl.DataFrame({"Rating_Factor": [0.0]})]
            tables += [pl.DataFrame({k: np.arange(n, dtype=np.int32), "Rating_Factor": np.zeros(n)}) for k, n in levels.items()]
            return frame, RatingModel(tables, family)
        return pd.DataFrame({k: pd.Categorical(v, categories=np.arange(levels[k])) for k, v in codes.items()}, copy=False)

    def fit(prepared):
        if engine.startswith("avenue"):
            opts = GLMOptions(max_iterations=5000, tolerance=tolerance, tweedie_power=1.5,
                              alpha=alpha, l1_ratio=ratio, compute_standard_errors=False,
                              solver=requested_solver, normalization="base_level")
            result = fit_glm_with_diagnostics(prepared[1], prepared[0], "y", options=opts,
                                             weight_col="weight" if weights is not None else None,
                                             offset_col="offset" if offset is not None else None)
            d = result.diagnostics
            return result, {"converged": bool(d.converged), "iterations": d.iterations,
                            "max_gradient": d.max_gradient, "solver": requested_solver,
                            "gradient_tail": list(d.gradient_history)[-5:]}
        model = glum.GeneralizedLinearRegressor(
            family=glum.TweedieDistribution(1.5) if family == "tweedie" else {"binary": "binomial", "gaussian": "normal"}.get(family, family),
            alpha=alpha, l1_ratio=ratio, drop_first=True, fit_intercept=True,
            max_iter=5000, gradient_tol=tolerance, scale_predictors=False, solver="auto")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model.fit(prepared, y, sample_weight=weights, offset=offset)
        messages = [str(w.message) for w in caught]
        converged = model.n_iter_ < 5000 and not any("converg" in s.lower() for s in messages)
        return model, {"converged": converged, "iterations": int(model.n_iter_),
                       "solver": getattr(model, "_solver", "auto"), "warnings": messages}

    record = {"case": case, "engine": engine, "threads": args.threads[0], "rows": len(y),
              "parameters": 1 + sum(n - 1 for n in levels.values()), "tables": len(levels),
              "family": family, "power": 1.5 if family == "tweedie" else None,
              "alpha": alpha, "l1_ratio": ratio, "tolerance": tolerance, "standard_errors": False,
              "audit": audit, "input_sha256": {p.name: digest(p) for p in sources},
              "wheel_sha256": digest(args.wheel), "package_origin": str(origin),
              "versions": {k: importlib.metadata.version(k) for k in ["avenue_model", "glum", "numpy", "scipy", "pandas", "polars", "tabmat"]},
              "python": platform.python_version(), "platform": platform.platform(),
              "threadpools": threadpool_info(), "polars_threads": pl.thread_pool_size(), "runs": []}
    # Cold fit is the warmup for timings and a clean-process memory measurement.
    # Record the OS high-water mark through fit, before prediction or subsequent fits.
    # Python sampling threads miss temporary native allocations while the GIL is held.
    for repeat in range(args.repeats + 1):
        gc.collect()
        started = time.perf_counter(); prepared = prepare(); prep_seconds = time.perf_counter() - started
        started = time.perf_counter(); model, diagnostics = fit(prepared); fit_seconds = time.perf_counter() - started
        if repeat == 0:
            record["peak_process_rss_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
            record["cold_fit_seconds"] = fit_seconds
        record["diagnostics"] = diagnostics
        print(f"{case} {engine} {args.threads[0]} threads run {repeat}: {fit_seconds:.4f}s, {diagnostics}", flush=True)
        if not diagnostics["converged"]:
            record["status"] = "not_converged"
            write(args.output / "result.json", record)
            return
        if repeat:
            record["runs"].append({"prep_seconds": prep_seconds, "fit_seconds": fit_seconds,
                                   "total_seconds": prep_seconds + fit_seconds})
        if repeat < args.repeats:
            del model, prepared
    record["median"] = {k: statistics.median(r[k] for r in record["runs"]) for k in record["runs"][0]}
    predictions = []
    for _ in range(args.repeats):
        started = time.perf_counter()
        if engine.startswith("avenue"):
            mu = model.model.predict(prepared[0]).to_series().to_numpy()
            if offset is not None:
                mu = mu * np.exp(offset)
        else:
            mu = np.asarray(model.predict(prepared, offset=offset))
        predictions.append(time.perf_counter() - started)
    assert np.isfinite(mu).all()
    record["predict_seconds"] = predictions
    record["median"]["predict_seconds"] = statistics.median(predictions)
    np.save(args.output / "predictions.npy", mu)
    if alpha:
        if engine.startswith("avenue"):
            coeff = np.concatenate([t["Rating_Factor"].to_numpy()[1:] - t["Rating_Factor"][0] for t in model.model.model_tables()[1:]])
        else:
            coeff = np.asarray(model.coef_)
        from sklearn.metrics import mean_tweedie_deviance
        deviance = mean_tweedie_deviance(y, mu, power=1.5, sample_weight=weights)
        record["objective"] = deviance / 2 + alpha * (ratio * np.abs(coeff).sum() + (1-ratio) / 2 * np.square(coeff).sum())
        record["nonzero_coefficients"] = int(np.sum(np.abs(coeff) > 1e-10))
        np.save(args.output / "coefficients.npy", coeff)
    record["status"] = "ok"
    write(args.output / "result.json", record)


def aggregate(output):
    import numpy as np
    records = []
    for path in sorted(output.glob("*/result.json")):
        r = json.loads(path.read_text())
        r["job"] = path.parent.name
        records.append(r)
    checks = []
    for a in records:
        if not a["engine"].startswith("avenue") or a["status"] != "ok":
            continue
        refs = [g for g in records if g["case"] == a["case"] and g["engine"] == "glum" and g["threads"] == a["threads"] and g["status"] == "ok"]
        if not refs:
            continue
        g = refs[0]
        av = np.load(output / a["job"] / "predictions.npy", mmap_mode="r")
        gv = np.load(output / g["job"] / "predictions.npy", mmap_mode="r")
        maximum, ss, failed, max_relative = 0., 0., 0, 0.
        for start in range(0, len(av), 100_000):
            x, y = av[start:start+100_000], gv[start:start+100_000]
            error = np.abs(x-y)
            maximum = max(maximum, float(error.max()))
            max_relative = max(max_relative, float(np.max(error / np.maximum(np.abs(y), 1e-12))))
            ss += float(np.square(y).sum())
            failed += int(np.sum(error > 1e-6 + 2e-6*np.abs(y)))
        check = {"avenue_job": a["job"], "glum_job": g["job"], "max_error_over_reference_rms": maximum / max((ss/len(av))**.5, 1e-12),
                 "max_relative_error": max_relative, "prediction_failures": failed, "atol": 1e-6, "rtol": 2e-6}
        passed = failed == 0 and check["max_error_over_reference_rms"] < 1e-7
        if a["alpha"]:
            ac = np.load(output / a["job"] / "coefficients.npy")
            gc_ = np.load(output / g["job"] / "coefficients.npy")
            check["max_coefficient_difference"] = float(np.max(np.abs(ac-gc_)))
            check["objective_difference"] = abs(a["objective"]-g["objective"])
            passed &= check["max_coefficient_difference"] < 1e-5 and check["objective_difference"] < 1e-8*max(1,abs(g["objective"]))
        check["passed"] = bool(passed)
        checks.append(check)
    evidence = {"records": records, "agreement": checks,
                "errors": {p.parent.name: json.loads(p.read_text()) for p in sorted(output.glob("*/error.json"))},
                "source_sha256": {str(p.relative_to(ROOT)): digest(p) for p in [Path(__file__)]},
                "memory_method": "Linux OS whole-process peak RSS through the first fit; includes interpreter, imported libraries, source-data loading, common preparation and engine preparation; excludes subsequent prediction/timing repeats",
                "timing_method": "median of requested repeats after one full-size warmup; source loading and common banding excluded; engine preparation reported separately; sequential fresh processes"}
    write(output / "results.json", evidence)
    return evidence


def validate(evidence, args):
    """Reject incomplete grids, failed fits and missing or failed comparisons."""
    expected = {f"{case}-{engine}-{threads}" for case in args.cases
                for threads in args.threads for engine in args.engines}
    records = {r["job"]: r for r in evidence["records"]}
    problems = [f"missing job: {job}" for job in sorted(expected - records.keys())]
    problems += [f"unexpected job: {job}" for job in sorted(records.keys() - expected)]
    problems += [f"worker error: {job}" for job in evidence["errors"]]
    problems += [f"fit failed: {job}" for job, r in records.items() if r["status"] != "ok"]
    if "glum" in args.engines:
        checks = {c["avenue_job"]: c for c in evidence["agreement"]}
        for job, r in records.items():
            if r["engine"].startswith("avenue") and not checks.get(job, {}).get("passed", False):
                problems.append(f"agreement failed or missing: {job}")
    for problem in problems:
        print(problem, file=sys.stderr)
    if problems:
        raise SystemExit(1)
    print(f"Validated {len(records)} fits and {len(evidence['agreement'])} comparisons")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--output", type=Path)
    p.add_argument("--wheel", type=Path)
    p.add_argument("--download", action="store_true")
    p.add_argument("--aggregate", action="store_true")
    p.add_argument("--check", action="store_true", help="Validate the requested grid when aggregating")
    p.add_argument("--cases", nargs="+", choices=CASES, default=list(REAL_CASES))
    p.add_argument("--engines", nargs="+", choices=["avenue", "avenue_table", "glum"], default=["avenue", "glum"])
    p.add_argument("--threads", nargs="+", type=int, default=[os.cpu_count() or 1])
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--tolerance", type=float, help="Override both engines' gradient tolerance")
    p.add_argument("--timeout", type=float, help="Seconds per worker (default: 600 real, 3600 large)")
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = p.parse_args()
    if args.download:
        download(args.data_dir)
        return
    if args.output is None:
        p.error("--output is required")
    if args.repeats < 1 or any(t < 1 for t in args.threads):
        p.error("repeats and thread counts must be positive")
    if args.tolerance is not None and not 0 < args.tolerance < float("inf"):
        p.error("tolerance must be finite and positive")
    if args.timeout is not None and not 0 < args.timeout < float("inf"):
        p.error("timeout must be finite and positive")
    args.output.mkdir(parents=True, exist_ok=True)
    if args.aggregate:
        evidence = aggregate(args.output)
        if args.check:
            validate(evidence, args)
        return
    if args.worker:
        worker(args)
        return
    for case in args.cases:
        for threads in args.threads:
            for engine in args.engines:
                job = args.output / f"{case}-{engine}-{threads}"
                job.mkdir(exist_ok=False)  # never silently reuse an old result
                command = [sys.executable, str(Path(__file__).resolve()), "--worker", "--data-dir", str(args.data_dir),
                           "--output", str(job), "--cases", case, "--engines", engine,
                           "--threads", str(threads), "--repeats", str(args.repeats)]
                if args.wheel:
                    command += ["--wheel", str(args.wheel)]
                if args.tolerance is not None:
                    command += ["--tolerance", str(args.tolerance)]
                print("RUN", job.name, flush=True)
                timeout = args.timeout or (3600 if case.startswith("large_") else 600)
                with (job / "run.log").open("w") as log:
                    try:
                        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
                    except subprocess.TimeoutExpired:
                        write(job / "error.json", {"timeout_seconds": timeout, "command": command})
                        print("TIMEOUT", job.name, flush=True)
                        aggregate(args.output)
                        continue
                if result.returncode:
                    write(job / "error.json", {"returncode": result.returncode, "command": command,
                                               "log_tail": (job / "run.log").read_text().splitlines()[-12:]})
                    print("ERROR", job.name, flush=True)
                else:
                    status = json.loads((job / "result.json").read_text())
                    print(status["status"], status.get("median"), flush=True)
                aggregate(args.output)
    validate(aggregate(args.output), args)


if __name__ == "__main__":
    main()
