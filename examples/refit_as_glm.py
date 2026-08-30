"""File a GLM, not a booster: let LightGBM choose the bands, then fit them by IRLS.

The objection to converting a gradient booster into rating tables is rarely that the
tables are wrong. It is that a filing reviewer is being asked to accept a tree ensemble,
and the tables carry no standard errors, no reference levels and no inference of any
kind — the factors are whatever the boosting arrived at.

There is a way around that which costs almost nothing. A converted model is a set of
rating tables, and a rating table is a *shape*: which bands, which levels, which
interactions. Hand those shapes to `Plan.given()` and the factors are re-estimated by
the GLM engine. What comes out is an ordinary Poisson GLM — Wald standard errors, a
reference row at relativity 1.0, the same `report()` and `validate()` as any other
fitted model — whose banding happened to be chosen by a booster rather than by hand.

Algorithmic band selection is not exotic; CART-derived banding has been in actuarial
practice for years. This is that idea with a data-driven band chooser, and the model
that gets estimated and reviewed is a GLM.

One caveat that this script cannot measure and you should not skip. The bands were
selected adaptively from the same data the GLM is then fitted on, so the Wald standard
errors below are *conditional on that selected structure*: they do not account for the
uncertainty in having chosen it, and are narrower than an honest accounting would give.
Where inference has to hold up, split the sample - select the shapes on one part, then
estimate and infer on a part the selection never saw. Nothing here does that for you.

What this script measures, over several random splits so the gap is not one draw:

    GBM converted        the booster's own factors
    GLM refit            same shapes, factors re-estimated, unpenalised
    GLM refit + ridge    same, with a small L2
    hand-built GLM       bands chosen by a person, as a baseline

The headline from the run this was written against: refitting costs about 0.2% of
holdout deviance, a whisker of ridge recovers it entirely, and both beat hand-chosen
bands by around 2.8%.

One trade-off worth stating plainly: a penalised fit does not produce standard errors.
The ridge column is the better model; the unpenalised column is the one that provides
conventional GLM inference, subject to the caveat above and to whatever the applicable
filing requirements actually are. The gap between them is small enough that the choice
is not painful either way.

Requires the `tuning` extra; `avenue-lightgbm` is optional and only sharpens the
band selection.

Usage:
    python examples/refit_as_glm.py
    python examples/refit_as_glm.py --splits 5 --out-dir filed_plan
"""

from __future__ import annotations

import argparse
import json
import os
import warnings

import numpy as np
import pandas as pd
import polars as pl

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".fremtpl2freq.parquet")
SEED = 20260829

CATEGORICALS = ["Area", "VehBrand"]
NUMERICS = ["DrivAge", "BonusMalus", "VehAge"]


def load() -> pd.DataFrame:
    if os.path.exists(CACHE):
        return pd.read_parquet(CACHE)
    from sklearn.datasets import fetch_openml

    print("  Downloading freMTPL2freq from OpenML (once; ~55 MB)...")
    frame = fetch_openml(data_id=41214, as_frame=True, parser="auto").frame
    frame.to_parquet(CACHE)
    return frame


def prepare(df: pd.DataFrame):
    df = df.copy()
    df["ClaimNb"] = df["ClaimNb"].astype(float).clip(upper=4)
    df["Exposure"] = df["Exposure"].astype(float).clip(lower=1e-3, upper=1)
    df["frequency"] = df["ClaimNb"] / df["Exposure"]

    names: dict[str, list[str]] = {}
    for column in CATEGORICALS:
        coded = pd.Categorical(df[column])
        # Int32 and declared categorical to LightGBM, which is what makes the converted
        # table one row per level rather than a band across codes - and therefore what
        # `with_categories` can put names back on.
        df[column + "_c"] = coded.codes.astype(np.int32)
        names[column + "_c"] = [str(level) for level in coded.categories]
    return df, names


def deviance(actual, predicted, exposure) -> float:
    from sklearn.metrics import mean_poisson_deviance

    return float(mean_poisson_deviance(actual, predicted, sample_weight=exposure))


def shapes_of(converted, features: list[str]):
    """The converted model's tables, reduced to the columns that define their shape.

    `Rating_Factor` comes along because `given` wants a complete table; the factors in
    it are then re-estimated and nothing of the booster's arithmetic survives into the
    fitted model except which rows exist.
    """
    out = []
    for table in converted.rating_tables():
        columns = [c for c in table.columns if c in features]
        if columns:
            out.append(table.select(columns + ["Rating_Factor"]))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=200)
    parser.add_argument("--out-dir", default="refit_plan")
    args = parser.parse_args()

    from avenue_model import (FittedModel, GLMOptions, Plan, resolve_lightgbm,
                              supports_interaction_penalties)

    lgb, build = resolve_lightgbm()
    forked = supports_interaction_penalties()
    print(f"\n  LightGBM build: {build} — interaction penalties "
          f"{'available' if forked else 'unavailable (bands will be coarser)'}")

    df, level_names = prepare(load())
    features = NUMERICS + [c + "_c" for c in CATEGORICALS]
    columns = features + ["Exposure", "frequency"]

    booster_params = dict(objective="poisson", max_depth=3, num_leaves=8,
                          learning_rate=0.15, min_data_in_leaf=400, verbose=-1,
                          seed=5, deterministic=True, num_threads=1)
    if forked:
        booster_params.update(interaction_penalty=60.0, interaction_complexity=200.0)

    results: dict[str, list[float]] = {}
    keep = None

    for split in range(args.splits):
        rng = np.random.default_rng(SEED + split)
        held = rng.random(len(df)) < 0.25
        train, test = df[~held], df[held]
        frame_train = pl.from_pandas(train[columns])
        frame_test = pl.from_pandas(test[columns])
        score = lambda p: deviance(test["frequency"].to_numpy(),  # noqa: E731
                                   p, test["Exposure"].to_numpy())

        dataset = lgb.Dataset(train[features].to_numpy(float),
                              label=train["frequency"].to_numpy(),
                              weight=train["Exposure"].to_numpy(),
                              feature_name=features,
                              categorical_feature=[c + "_c" for c in CATEGORICALS],
                              params={"feature_pre_filter": False})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            booster = lgb.train(booster_params, dataset, num_boost_round=args.rounds)

        converted = FittedModel.from_lgbm_json(json.dumps(booster.dump_model()),
                                               consolidation="max")
        results.setdefault("GBM converted", []).append(
            score(converted.predict(frame_test).to_series(0).to_numpy()))

        def plan_from_shapes():
            plan = Plan.frequency("Exposure")
            for index, table in enumerate(shapes_of(converted, features)):
                plan = plan.given(f"t{index}", table)
            return plan

        refit = plan_from_shapes().fit(frame_train, "frequency")
        results.setdefault("GLM refit", []).append(
            score(refit.predict(frame_test).to_series(0).to_numpy()))

        ridge = plan_from_shapes().fit(frame_train, "frequency",
                                       options=GLMOptions(alpha=1e-6, l1_ratio=0.0))
        results.setdefault("GLM refit + ridge", []).append(
            score(ridge.predict(frame_test).to_series(0).to_numpy()))

        hand = (Plan.frequency("Exposure")
                .banded("DrivAge", breaks=[21, 26, 31, 41, 51, 61, 71, 81])
                .banded("BonusMalus", breaks=[51, 55, 60, 70, 80, 90, 100, 120, 150])
                .banded("VehAge", breaks=[1, 3, 5, 7, 10, 15])
                .categorical("Area_c").categorical("VehBrand_c")
                ).fit(frame_train, "frequency")
        results.setdefault("hand-built GLM", []).append(
            score(hand.predict(frame_test).to_series(0).to_numpy()))

        if keep is None:
            keep = (refit, frame_test)

    print(f"\n  Holdout mean Poisson deviance over {args.splits} random 75/25 splits\n")
    header = "".join(f"{f'split {i}':>10}" for i in range(args.splits))
    print(f"  {'model':<20}{header}{'mean':>10}")
    print(f"  {'-' * (20 + 10 * (args.splits + 1))}")
    baseline = float(np.mean(results["GBM converted"]))
    for name, values in results.items():
        mean = float(np.mean(values))
        gap = "" if name == "GBM converted" else f"   {(mean / baseline - 1) * 100:+.2f}%"
        print(f"  {name:<20}" + "".join(f"{v:>10.4f}" for v in values)
              + f"{mean:>10.4f}{gap}")

    refit, frame_test = keep
    print("\n  What the refit adds, and the booster cannot:\n")
    for table in refit.rating_tables():
        if "Standard_Error" in table.columns and table.height > 2:
            errors = table["Standard_Error"].to_numpy()
            if not np.all(np.isnan(errors)):
                shown = [c for c in table.columns
                         if c in ("Rating_Factor", "Standard_Error", "Status",
                                  "Relativity") or c in features]
                print(table.select(shown).head(4))
                break

    named = refit
    for column, levels in level_names.items():
        if any(column in t.columns for t in refit.rating_tables()):
            named = named.with_categories({column: levels})
    named.to_workbook().save_csv_dir(args.out_dir)
    print(f"\n  Wrote the refitted GLM to {args.out_dir}/ — "
          f"{len(os.listdir(args.out_dir))} files.")
    print("  This is a Poisson GLM with standard errors and reference levels. The only "
          "\n  thing the booster contributed is which bands exist - which is also why "
          "those\n  standard errors are conditional on a structure chosen from this same "
          "data.\n  Split the sample if the inference has to hold up.")

    validation = refit.validate(frame_test)
    print(f"\n  Held-out A/E {validation.ae_ratio:.4f}, Gini {validation.gini:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
