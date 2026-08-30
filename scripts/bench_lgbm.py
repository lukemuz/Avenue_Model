"""Does a converted booster predict what the booster predicted, and how big is it?

The claim the LightGBM conversion rests on is not a speed claim, so this script is not a
benchmark in the sense the others are. It checks the two things the claim is made of:

* **Exactness.** `FittedModel.from_lgbm_json` changes the representation and nothing
  else. The gate here is far tighter than the one the GLM benchmarks use against glum,
  because this is not two engines that ought to agree — it is one set of predictions
  re-expressed, so anything above floating-point noise is a bug.
* **Size.** Neither table count nor table size is about tree count, and the two grow at
  very different rates. Count is the number of distinct feature *combinations* the
  ensemble uses; rows are the cross product of every threshold along a path, so they
  grow far faster - on freMTPL2 at a fixed 30 trees, depth 2 to 4 takes the count from
  5 to 14 while the rows go from 57 to 4,145. Rows are what decides whether a person
  can read the result, and depth is the blunt lever on both.

`estimate_num_tables` is checked against the conversion it predicts, since it exists to
be called on hyperparameter trials where doing the conversion would be too slow — a
cheap estimate that disagreed with the real thing would quietly misdirect every search
that used it.

Usage:
    python scripts/bench_lgbm.py                  # freMTPL2, depth sweep
    python scripts/bench_lgbm.py --trees 200
    python scripts/bench_lgbm.py --json out.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# One prediction in 1e12 - far below anything a modelling decision turns on, and far
# above the 1e-15 a correct conversion actually achieves, so a real regression trips it.
EXACTNESS_TOL = 1e-12


def build(rows: int | None):
    from bench_fremtpl import load_fremtpl, prepare

    codes, levels, y, exposure = prepare(load_fremtpl(), wide=False)
    names = list(levels)
    if rows is not None:
        codes = {n: c[:rows] for n, c in codes.items()}
        y, exposure = y[:rows], exposure[:rows]
    design = np.column_stack([codes[n] for n in names]).astype(np.float64)
    frame = pl.DataFrame({n: codes[n].astype(np.float64) for n in names})
    return names, design, frame, y, exposure


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trees", type=int, default=100)
    # Deliberately stopping at 4. Depth 6 on this data converts correctly but into
    # tables with millions of rows, which takes minutes and is the illegible case this
    # script exists to warn about - pass --depths 6 to watch it happen.
    parser.add_argument("--depths", type=int, nargs="*", default=[2, 3, 4])
    parser.add_argument("--rows", type=int, default=None)
    parser.add_argument("--json", help="write the results to this path")
    args = parser.parse_args()

    import lightgbm as lgb

    from avenue_model import FittedModel, estimate_num_tables

    names, design, frame, y, exposure = build(args.rows)
    print(f"\n  freMTPL2, {len(y):,} rows, {len(names)} features, "
          f"{args.trees} trees, Poisson\n")
    print(f"  {'depth':>6}{'tables':>8}{'estimate':>10}{'rows':>10}{'widest':>9}"
          f"{'convert':>9}{'predict':>9}{'max rel drift':>15}")
    print(f"  {'-' * 79}")

    problems: list[str] = []
    payload = []

    for depth in args.depths:
        dataset = lgb.Dataset(design, label=y / exposure, weight=exposure,
                              feature_name=names)
        booster = lgb.train(
            dict(objective="poisson", max_depth=depth, num_leaves=2 ** depth,
                 learning_rate=0.1, min_data_in_leaf=200, verbose=-1),
            dataset, num_boost_round=args.trees)
        reference = booster.predict(design)
        dumped = json.dumps(booster.dump_model())

        estimated = estimate_num_tables(dumped)

        started = time.perf_counter()
        converted = FittedModel.from_lgbm_json(dumped, consolidation="max")
        convert_seconds = time.perf_counter() - started

        started = time.perf_counter()
        predicted = converted.predict(frame).to_series(0).to_numpy()
        predict_seconds = time.perf_counter() - started

        actual = len(converted.table_names)
        heights = [table.height for table in converted.rating_tables()]
        drift = float(np.max(np.abs(predicted - reference)
                             / np.maximum(np.abs(reference), 1e-12)))

        print(f"  {depth:>6}{actual:>8}{estimated:>10}{sum(heights):>10,}"
              f"{max(heights):>9,}{convert_seconds:>8.2f}s{predict_seconds:>8.2f}s"
              f"{drift:>15.1e}")

        if drift > EXACTNESS_TOL:
            problems.append(f"depth {depth}: conversion changed predictions by {drift:.2e}")
        if estimated != actual:
            problems.append(
                f"depth {depth}: estimate_num_tables said {estimated}, conversion "
                f"produced {actual}")

        payload.append(dict(depth=depth, trees=args.trees, tables=actual,
                            estimated=estimated, rows=sum(heights),
                            widest=max(heights), convert_seconds=convert_seconds,
                            predict_seconds=predict_seconds, drift=drift))

    print(f"\n  Every row above has the same {args.trees} trees, so neither column is "
          f"about tree count.\n  Table *count* is the number of distinct feature "
          f"combinations the ensemble uses.\n  Table *rows* is what actually costs you: "
          f"a path over d features becomes the cross\n  product of every threshold on "
          f"each of them, so rows grow far faster than count,\n  and they are what "
          f"decides whether a person can read the result.")

    if args.json:
        with open(args.json, "w") as handle:
            json.dump(payload, handle, indent=2)
        print(f"\n  wrote {args.json}")

    if problems:
        print("\n  FAILED")
        for line in problems:
            print(f"    {line}")
        return 1
    print(f"\n  conversion is exact to {EXACTNESS_TOL:.0e}; "
          f"estimate_num_tables agrees with the conversion on every row")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
