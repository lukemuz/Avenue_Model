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


def load_case(case, data_dir):
    import numpy as np
    import pandas as pd
    sys.path.insert(0, str(ROOT / "scripts"))
    import bench_fremtpl as motor
    import bench_housing as housing
    import bench_real as real
    import bench_large as large
    weights = offset = None
    alpha = ratio = 0.0
    sources = []
    audit = {}
    if case.startswith("large_"):
        tables, width = (5, 101) if case == "large_few" else (100, 6)
        codes, y, exposure = large.generate(20_000_000, tables, width, 0.0)
        levels = {k: width for k in codes}
        offset = np.log(exposure)
        family = "poisson"
        audit = {"seed": large.SEED, "independent_synthetic_factors": True}
    elif case in ("motor", "motor_wide") or case.startswith("tweedie"):
        sources = [data_dir / "motor.parquet"]
        raw = pd.read_parquet(sources[0])
        codes, levels, y, exposure = motor.prepare(raw, wide=case == "motor_wide")
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
        spec = real.DATASETS[case]
        codes, levels, y, _ = spec["prepare"](pd.read_parquet(sources[0]))
        family = spec["family"]
    else:
        sources = [data_dir / "housing.parquet"]
        codes, levels, y = housing.prepare(pd.read_parquet(sources[0]), None)
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
                "source_sha256": {str(p.relative_to(ROOT)): digest(p) for p in [Path(__file__), *sorted((ROOT / "scripts").glob("bench_*.py"))]},
                "memory_method": "Linux OS whole-process peak RSS through the first fit; includes interpreter, imported libraries, source-data loading, common preparation and engine preparation; excludes subsequent prediction/timing repeats",
                "timing_method": "median of requested repeats after one full-size warmup; source loading and common banding excluded; engine preparation reported separately; sequential fresh processes"}
    selection = output / "selected_sources.json"
    if selection.exists():
        evidence["selected_sources"] = json.loads(selection.read_text())
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
    p.add_argument("--avenue-results", nargs="+", type=Path,
                   help="Combine existing Avenue results; later directories take precedence")
    p.add_argument("--glum-results", nargs="+", type=Path,
                   help="Combine existing glum results; later directories take precedence")
    p.add_argument("--cases", nargs="+", choices=CASES, default=list(REAL_CASES))
    p.add_argument("--engines", nargs="+", choices=["avenue", "avenue_table", "glum"], default=["avenue", "glum"])
    p.add_argument("--threads", nargs="+", type=int, default=[4, 32])
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
    if args.avenue_results or args.glum_results:
        if not (args.avenue_results and args.glum_results):
            p.error("combining results requires both --avenue-results and --glum-results")
        selected = {}
        for case in args.cases:
            for threads in args.threads:
                for engine in args.engines:
                    job = f"{case}-{engine}-{threads}"
                    sources = args.avenue_results if engine.startswith("avenue") else args.glum_results
                    source = next((root / job for root in reversed(sources) if (root / job).is_dir()), None)
                    if source is None:
                        p.error(f"missing source for {job}")
                    (args.output / job).symlink_to(source.resolve(), target_is_directory=True)
                    selected[job] = str(source.resolve())
        write(args.output / "selected_sources.json", selected)
        validate(aggregate(args.output), args)
        return
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
