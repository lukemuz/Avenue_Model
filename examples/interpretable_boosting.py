"""Tune a booster, inspect its exact tables, then refit those tables as a GLM.

Run: python examples/interpretable_boosting.py --output /tmp/avenue-tables
Requires Avenue's [tuning] extra. Uses the Avenue LightGBM backend when installed;
stock LightGBM runs the same example without the additional interaction penalties.
"""
import argparse
from pathlib import Path

import numpy as np
import polars as pl

from avenue_model import GLMOptions, Plan, Workbook, from_booster, resolve_lightgbm, tune_lgbm


def run(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)

    # 1. Small synthetic portfolio: each row is one full year of coverage.
    rng = np.random.default_rng(47)
    age = rng.integers(18, 81, 1200).astype(float)
    vehicle_age = rng.integers(0, 21, 1200).astype(float)
    expected_claims = np.exp(-1.5 + 0.6 * (age < 25) + 0.3 * (vehicle_age > 10))
    data = pl.DataFrame({"age": age, "vehicle_age": vehicle_age,
                         "claims": rng.poisson(expected_claims).astype(float)})
    train, holdout = data[:1000], data[1000:]
    features = ["age", "vehicle_age"]
    quotes = holdout.select(features)

    # 2. Avenue tunes predictive loss and the number of resulting tables together.
    # The installed backend supplies Dataset/train; Avenue owns the search and tables.
    backend, backend_name = resolve_lightgbm()
    dataset = backend.Dataset(train.select(features).to_numpy(),
                              label=train["claims"].to_numpy(), feature_name=features)
    search = tune_lgbm(
        dataset,
        {"objective": "poisson", "num_iterations": 25, "max_depth": 2,
         "num_leaves": 3, "min_data_in_leaf": 50, "num_threads": 2,
         "verbosity": -1, "seed": 47},
        n_trials=5, seed=47,
        tunable=["learning_rate", "interaction_penalty", "interaction_complexity"],
        space={"learning_rate": (0.05, 0.2), "interaction_penalty": (0.0, 0.05),
               "interaction_complexity": (0.0, 0.1)},
    )
    print(f"Training backend: {backend_name}")
    print(f"Interaction penalties enabled: {search.tuned_interaction_penalties}")
    print(search.summary())
    selected = search.select(max_tables=4)  # Mean CV table count; inspect the final fit too.
    booster = backend.train({**selected.params, "num_iterations": selected.num_iterations}, dataset)

    # 3. Conversion changes the representation and preserves the booster's predictions.
    conversion = from_booster(booster, quotes)
    assert conversion.parity["status"] == "passed", conversion.parity
    model = conversion.model
    print("Final table sizes:", conversion.metadata["complexity"])
    for name, table in model.rating_tables_by_name().items():
        print(name, table.head(5), sep="\n")
    print("Predicted annual claim counts:", model.predict(quotes.head(5)), sep="\n")
    conversion.save(output / "boosted_tables")
    reloaded = Workbook.load_csv_dir(str(output / "boosted_tables")).to_model()
    np.testing.assert_allclose(reloaded.predict(quotes).to_series(),
                               model.predict(quotes).to_series(), atol=1e-12, rtol=1e-12)
    print("Booster → tables → CSV reload: predictions agree.")

    # 4. Optional: keep the learned bands/interactions, estimate new GLM coefficients.
    # This is a new statistical fit: its predictions can differ from the booster.
    plan = Plan("poisson")
    for name, table in model.rating_tables_by_name().items():
        predictors = [column for column in table.columns if column in features]
        if predictors:
            plan = plan.given(name, table.select(predictors + ["Rating_Factor"]))
    refit = plan.fit(train, "claims", GLMOptions(alpha=1e-4, l1_ratio=0.0, max_iterations=500))
    assert refit.converged, "GLM refit did not converge"
    refit.to_workbook().save_csv_dir(str(output / "refitted_glm"))
    print("GLM refit converged; saved boosted_tables/ and refitted_glm/ in", output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="A new directory for editable tables")
    run(parser.parse_args().output)
