"""Peak memory per engine, one engine per process.

`bench_memory.PeakMemory` samples RSS inside a single process, which is enough to see a
design matrix appear but carries an ordering artifact: CPython does not return freed
memory to the OS, so an engine that runs after a greedy one allocates inside the pool
its predecessor left behind and reports a peak of roughly zero. That is an artifact of
the harness, not a property of the engine.

This script removes the artifact by giving each engine its own process and reporting the
whole process's high-water mark - data, design matrix, solver and all. That is a
different number from the in-process one and a more honest one to compare across
engines: it answers "how much memory does it take to fit this model", which is the
question someone sizing a machine actually has.

Usage:
    python scripts/bench_isolated.py --sweep --json out.json
    python scripts/bench_isolated.py --dataset housing --family gamma --engine avenue
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ENGINES = ("avenue", "glum-ls", "glum-cd", "statsmodels")


class WholeProcessPeak:
    """High-water mark of this process's RSS, in MB, from start to stop."""

    INTERVAL_SECONDS = 0.002

    def __init__(self) -> None:
        import psutil

        self._process = psutil.Process()
        self._stop = threading.Event()
        self._peak = 0
        self.peak_mb = 0.0
        self._thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self) -> None:
        while not self._stop.is_set():
            rss = self._process.memory_info().rss
            if rss > self._peak:
                self._peak = rss
            self._stop.wait(self.INTERVAL_SECONDS)

    def __enter__(self) -> "WholeProcessPeak":
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)
        self.peak_mb = self._peak / 1e6


# ------------------------------------------------------------------ the child

def fit_once(dataset: str, family: str, rows: int | None, engine: str) -> dict:
    """Load the data and run exactly one engine, in this process."""
    if dataset == "synthetic":
        import bench_glm as B

        case = B.Case(family, rows,
                      use_offset=(family == "poisson"),
                      use_weight=(family != "poisson"))
        data = B.make_dataset(rows, family, case.structure)
        runners = {
            "avenue": lambda: B.run_avenue(data, case, 1, standard_errors=False),
            "glum-ls": lambda: B.run_glum(data, case, 1),
            # bench_glm leaves glum on its own solver choice, so there is no separate
            # irls-cd cell here - it would be the same run under a different name.
            "glum-cd": None,
            "statsmodels": lambda: B.run_statsmodels(data, case, 1),
        }
        n_rows, n_params = data.n_rows, data.n_parameters

    elif dataset == "fremtpl":
        import bench_fremtpl as B

        codes, levels, y, exposure = B.prepare(B.load_fremtpl(), wide=False)
        runners = {
            "avenue": lambda: B.run_avenue(codes, levels, y, exposure, 1, False),
            "glum-ls": lambda: B.run_glum(codes, y, exposure, 1, "irls-ls"),
            "glum-cd": lambda: B.run_glum(codes, y, exposure, 1, "irls-cd"),
            "statsmodels": None,
        }
        n_rows, n_params = len(y), 1 + sum(k - 1 for k in levels.values())

    elif dataset == "housing":
        import bench_housing as B

        codes, levels, y = B.prepare(B.load_housing(), rows)
        runners = {
            "avenue": lambda: B.run_avenue(codes, levels, y, family, 1),
            "glum-ls": lambda: B.run_glum(codes, y, family, 1, "irls-ls"),
            "glum-cd": lambda: B.run_glum(codes, y, family, 1, "irls-cd"),
            "statsmodels": lambda: B.run_statsmodels(codes, levels, y, family, 1),
        }
        n_rows, n_params = len(y), 1 + sum(k - 1 for k in levels.values())

    else:
        raise ValueError(f"unknown dataset {dataset!r}")

    runner = runners.get(engine)
    if runner is None:
        return {"skipped": f"{engine} not run on {dataset}"}

    started = time.perf_counter()
    result = runner()
    elapsed = time.perf_counter() - started

    # bench_glm returns a Timing dataclass; the other two return plain dicts.
    if hasattr(result, "fit_seconds"):
        fit_seconds, iters = result.fit_seconds, result.iterations
        if result.skipped:
            return {"skipped": result.skipped}
    else:
        fit_seconds, iters = result["fit"], result["iters"]

    return {
        "n_rows": n_rows,
        "n_parameters": n_params,
        "fit_seconds": fit_seconds,
        "wall_seconds": elapsed,
        "iterations": iters,
    }


def child_main(args) -> int:
    with WholeProcessPeak() as peak:
        payload = fit_once(args.dataset, args.family, args.rows, args.engine)
    payload["peak_rss_mb"] = peak.peak_mb
    print("RESULT " + json.dumps(payload))
    return 0


# ----------------------------------------------------------------- the parent

#: (dataset, family, rows, label). Sized so every cell finishes in reasonable time.
SWEEP = [
    ("synthetic", "poisson", 1_000_000, "synthetic poisson 1M"),
    ("synthetic", "gamma", 1_000_000, "synthetic gamma 1M"),
    ("synthetic", "poisson", 5_000_000, "synthetic poisson 5M"),
    ("synthetic", "poisson", 100_000, "synthetic poisson 100k"),
    ("fremtpl", "poisson", None, "freMTPL2 tutorial bands"),
    ("housing", "gamma", None, "house_sales gamma"),
]


def spawn(dataset, family, rows, engine) -> dict:
    cmd = [sys.executable, os.path.abspath(__file__),
           "--dataset", dataset, "--family", family, "--engine", engine]
    if rows is not None:
        cmd += ["--rows", str(rows)]
    env = dict(os.environ, PYTHONWARNINGS="ignore")
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT "):
            return json.loads(line[len("RESULT "):])
    tail = (proc.stderr or proc.stdout).strip().splitlines()
    return {"error": tail[-1] if tail else f"exit {proc.returncode}"}


def sweep_main(args) -> int:
    print("Peak memory per engine, one engine per process")
    print("  reported figure is the whole process's high-water RSS: data, design")
    print("  matrix, solver and interpreter together.\n")

    collected = []
    for dataset, family, rows, label in SWEEP:
        print(f"  {label}")
        cells = []
        for engine in ENGINES:
            result = spawn(dataset, family, rows, engine)
            result.update(dataset=dataset, family=family, engine=engine, label=label)
            cells.append(result)

            if "skipped" in result:
                print(f"    {engine:<14}{'skipped':>10}   ({result['skipped']})")
            elif "error" in result:
                print(f"    {engine:<14}{'FAILED':>10}   {result['error'][:70]}")
            else:
                print(f"    {engine:<14}{result['peak_rss_mb']:>9.0f} MB"
                      f"{result['fit_seconds']:>9.3f}s fit"
                      f"{result['iterations'] if result['iterations'] else '-':>6} iters")
        collected.append({"label": label, "dataset": dataset, "family": family,
                          "rows": rows, "cells": cells})
        print()

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(collected, fh, indent=2)
        print(f"Wrote {args.json}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", action="store_true",
                        help="run every engine on every case, each in its own process")
    parser.add_argument("--dataset", default="housing",
                        choices=("synthetic", "fremtpl", "housing"))
    parser.add_argument("--family", default="gamma")
    parser.add_argument("--rows", type=int, default=None)
    parser.add_argument("--engine", default="avenue", choices=ENGINES)
    parser.add_argument("--json", type=str, default=None)
    args = parser.parse_args()

    return sweep_main(args) if args.sweep else child_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
