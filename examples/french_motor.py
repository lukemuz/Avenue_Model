"""Claim frequency on the French motor data, as rating tables you could file.

This is the insurance case study from *GBMs as Factor Tables* (Muzynoski, 2025), reduced
to one script. It does the whole loop:

    tune for accuracy AND table count  ->  pick a point on the frontier
    ->  train  ->  convert exactly  ->  write CSVs someone can read

freMTPL2 is the standard actuarial benchmark for claim frequency: 678,013 policies, nine
rating factors, claim counts over an exposure. The model is Poisson on claims per unit
exposure, with exposure as a prior weight, which is how a frequency model is normally
written and how Avenue's own `Plan.frequency` expresses it.

What to look at in the output:

* The **frontier**. Each row is a model. Reading down it, table count falls and
  cross-validated loss rises — but far more slowly than you would expect, which is the
  finding the paper rests on. Interpretability is much cheaper than it looks.
* The **two selected models**, scored on held-out data. `--max-tables` picks the most
  accurate model that fits in a table budget; the other is the best cross-validated
  model regardless of size.
* The **CSVs**, written to `french_motor_tables/`. That is the model. Add the intercept
  to one factor from each table and exponentiate.

Published comparison (test-set mean Poisson deviance, lower is better):

    Random Forest  0.6898     EBM  0.5994     Ours, interpretable  0.5934     Ours, best CV  0.5834

Those figures came from `avenue-lightgbm`, the fork that adds `interaction_penalty` and
`interaction_complexity` — penalties aimed at the table count itself rather than at tree
size. This script does not require it. On stock LightGBM the frontier is driven by depth
and leaf count alone, which are blunter instruments for the same job, and even a 30-trial
run lands in the same neighbourhood: ten tables at roughly 0.59, already past the
published EBM. The fork buys finer control over where on that frontier you can sit, not
the ability to be on it. The script says which case it is in before it starts.

    pip install avenue-lightgbm      # https://github.com/lukemuz/avenue-lightgbm

Usage:
    python examples/french_motor.py
    python examples/french_motor.py --trials 60 --max-tables 12
    python examples/french_motor.py --out-dir my_plan
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
import polars as pl

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".fremtpl2freq.parquet")
SEED = 20260829


def load() -> pd.DataFrame:
    if os.path.exists(CACHE):
        return pd.read_parquet(CACHE)
    from sklearn.datasets import fetch_openml

    print("  Downloading freMTPL2freq from OpenML (once; ~55 MB)...")
    frame = fetch_openml(data_id=41214, as_frame=True, parser="auto").frame
    frame.to_parquet(CACHE)
    return frame


def prepare(df: pd.DataFrame):
    """The tutorial preprocessing: cap the tails, then hand the drivers over raw.

    The bands are deliberately *not* chosen here. A GLM needs its bands up front because
    its factors are levels; a booster picks its own split points, and choosing them by
    hand in advance would be doing the model's job badly. The bands in the converted
    rating tables are the ones the trees found.
    """
    df = df.copy()
    df["ClaimNb"] = df["ClaimNb"].astype(float).clip(upper=4)
    df["Exposure"] = df["Exposure"].astype(float).clip(lower=1e-3, upper=1)

    features = {
        "veh_age": df["VehAge"].to_numpy(dtype=float),
        "driv_age": df["DrivAge"].to_numpy(dtype=float),
        "bonus_malus": df["BonusMalus"].to_numpy(dtype=float),
        "density": np.log(df["Density"].to_numpy(dtype=float)),
        "veh_power": df["VehPower"].to_numpy(dtype=float),
        "area": pd.Categorical(df["Area"]).codes.astype(float),
        "veh_brand": pd.Categorical(df["VehBrand"]).codes.astype(float),
        "veh_gas": pd.Categorical(df["VehGas"]).codes.astype(float),
        "region": pd.Categorical(df["Region"]).codes.astype(float),
    }
    return (features,
            df["ClaimNb"].to_numpy(dtype=float),
            df["Exposure"].to_numpy(dtype=float))


def poisson_deviance(claims, frequency, exposure) -> float:
    """Mean Poisson deviance on the frequency scale, weighted by exposure.

    The convention matters more than it looks. Scored per policy on counts instead, a
    constant model on this data reads 0.330; scored this way it reads 0.625, and the
    published figures this script compares against are on the second scale. The
    difference is the normalisation - dividing by the number of policies rather than by
    total exposure, which averages 0.53 here - not the model. sklearn's implementation
    is used rather than a local one so there is nothing to get subtly wrong.
    """
    from sklearn.metrics import mean_poisson_deviance

    return float(mean_poisson_deviance(claims / exposure, frequency,
                                       sample_weight=exposure))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=30,
                        help="tuning trials (default 30; the paper used 100)")
    parser.add_argument("--max-tables", type=float, default=10,
                        help="table budget for the interpretable model")
    parser.add_argument("--out-dir", default="french_motor_tables")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    import lightgbm as lgb

    from avenue_model import FittedModel, supports_interaction_penalties, tune_lgbm

    forked = supports_interaction_penalties()
    print("\n  LightGBM:", "avenue-lightgbm — interaction penalties available" if forked
          else "stock — interaction penalties NOT available, see this file's docstring")

    features, claims, exposure = prepare(load())
    names = list(features)
    design = np.column_stack([features[n] for n in names])

    rng = np.random.default_rng(args.seed)
    holdout = rng.random(len(claims)) < 0.25
    train, test = ~holdout, holdout
    print(f"  freMTPL2: {train.sum():,} train rows, {test.sum():,} held out, "
          f"{len(names)} features")

    dataset = lgb.Dataset(design[train], label=claims[train] / exposure[train],
                          weight=exposure[train], feature_name=names)
    base = {"objective": "poisson", "verbose": -1, "num_iterations": 400,
            "min_data_in_leaf": 200}

    print(f"\n  Tuning {args.trials} trials on cross-validated loss and table count...")
    result = tune_lgbm(dataset, base, n_trials=args.trials, nfold=3, seed=args.seed)
    print(result.summary())

    try:
        interpretable = result.select(max_tables=args.max_tables)
    except ValueError as exc:
        print(f"\n  {exc}")
        interpretable = result.frontier[0]
        print(f"  Falling back to the smallest found: {interpretable.tables:.0f} tables.")

    frame = pl.DataFrame({n: features[n][test] for n in names})
    rows = []

    for label, trial in (("interpretable", interpretable), ("best CV", result.best_cv)):
        booster = lgb.train({**base, **trial.params, "num_iterations": trial.num_iterations},
                            dataset)
        dumped = json.dumps(booster.dump_model())
        converted = FittedModel.from_lgbm_json(dumped, consolidation="max")

        predicted = converted.predict(frame).to_series(0).to_numpy()
        reference = booster.predict(design[test])
        drift = float(np.max(np.abs(predicted - reference)
                             / np.maximum(np.abs(reference), 1e-12)))
        if drift > 1e-12:
            print(f"  FAILED: conversion changed predictions by {drift:.2e}")
            return 1

        deviance = poisson_deviance(claims[test], predicted, exposure[test])
        rows.append((label, len(converted.table_names), deviance, drift, converted))

    print(f"\n  Held-out results ({test.sum():,} policies)")
    print(f"  {'model':<16}{'tables':>8}{'Poisson deviance':>20}{'conversion drift':>20}")
    print(f"  {'-' * 64}")
    for label, tables, deviance, drift, _ in rows:
        print(f"  {label:<16}{tables:>8}{deviance:>20.4f}{drift:>20.1e}")
    print(f"  {'EBM (paper)':<16}{'—':>8}{0.5994:>20.4f}")
    print(f"  {'RF (paper)':<16}{'—':>8}{0.6898:>20.4f}")
    print("\n  The paper's rows are on the same metric but its own train/test split and"
          "\n  tuning budget, so read them as the neighbourhood to land in rather than"
          "\n  as a target to match exactly.")

    smallest = rows[0][4]
    smallest.with_response("frequency", exposure="exposure",
                           exposure_role="weight").to_workbook().save_csv_dir(args.out_dir)
    listing = sorted(os.listdir(args.out_dir))
    print(f"\n  Wrote the interpretable model to {args.out_dir}/ — "
          f"{len(listing)} files, this is the model:")
    for name in listing[:6]:
        print(f"    {name}")
    if len(listing) > 6:
        print(f"    ... and {len(listing) - 6} more")

    if not forked:
        print("\n  Note: with stock LightGBM the frontier is driven by depth and leaf "
              "count\n  alone. Install avenue-lightgbm for the penalties that target "
              "table count\n  directly, which is what the published figures used.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
