# LightGBM as rating tables

How a gradient booster becomes a set of rating tables, how to keep those tables small
enough that someone will read them, and what it takes to turn the result into a GLM.

The short version lives in the [README](../README.md#convert-a-booster); this is
the working detail. The method and the case studies behind it are in *GBMs as Factor
Tables: Achieving Both Transparency and Interpretability Without Approximation*
(Muzynoski, 2025), [PDF](https://avenue-analytics.com/research/avenue-analytics-methodology.pdf).

**Contents:** [Table size](#making-the-tables-small-enough-to-read) ·
[Tuning](#tuning-for-interpretability) · [Category names](#naming-the-levels-behind-the-codes) ·
[Refitting as a GLM](#refit-the-selected-table-structure-as-a-glm)

---

### Making the tables small enough to read

A supported booster converts without approximation, but its tables can still be
too large to review. Two quantities decide that, and neither is the tree count — on freMTPL2 at a fixed
30 trees:

| max depth | tables | rows across them | widest table |
|---:|---:|---:|---:|
| 2 | 5 | 57 | 28 |
| 3 | 11 | 397 | 192 |
| 4 | 14 | 4,145 | 960 |

Table *count* is the number of distinct feature combinations the ensemble uses. *Rows*
are the cross product of every threshold along a path, so they grow much faster, and they
are what decides whether anyone can read the result. Both are modelling choices rather
than facts of the data, and `scripts/bench_lgbm.py` reproduces the table above.

```python
from avenue_model import estimate_num_tables

estimate_num_tables(json.dumps(booster.dump_model()))   # -> 20
```

`estimate_num_tables` reads a LightGBM dump and returns the number of consolidated tables
the conversion would produce, without doing the conversion. It is cheap enough to call on
every trial of a hyperparameter search, which is exactly what it is for:
[`avenue_model.tune_lgbm`](#tuning-for-interpretability) optimises cross-validated loss
and table count together and returns the Pareto frontier, so the trade-off is chosen
rather than stumbled into.

Depth and leaf count are the blunt levers and work with stock LightGBM.
[`avenue-lightgbm`](https://github.com/lukemuz/avenue-lightgbm), a small fork, adds two
that target the table count directly:

| parameter | effect |
|---|---|
| `interaction_penalty` | penalises a split whose feature combination is new to the ensemble |
| `interaction_complexity` | penalises each feature newly introduced within one tree |

Both default to zero. In the recorded experiment,
holding every other parameter fixed on the French motor data, `interaction_penalty` alone
takes a 39-table model to 12 for 0.1% of cross-validated loss, and to 5 for 0.7%.

| `interaction_penalty` | tables | cv Poisson |
|---:|---:|---:|
| 0 | 39 | 0.310534 |
| 10 | 12 | 0.310831 |
| 100 | 5 | 0.312757 |

The paper reports a selected four-feature model with held-out deviance of 0.5934,
compared with 0.5994 for EBM in that experiment. These results use a different
evaluation setup from the penalty sweep above.

### Tuning for interpretability

`tune_lgbm` runs an Optuna study against both objectives at once and hands back the
Pareto frontier of predictive loss and table count:

```python
from avenue_model import tune_lgbm

result = tune_lgbm(dataset, {"objective": "poisson"}, n_trials=50)
print(result.summary())

trial = result.select(max_tables=10)     # screen by mean CV table count
booster = lgb.train({**trial.params, "num_iterations": trial.num_iterations}, dataset)
```

`result.frontier` is sorted by table count, `result.best_cv` ignores size entirely, and
`result.trials` retains every trial. Each trial's `fold_complexity` records the actual
maximum-consolidation artifact for each CV fold at `num_iterations`: table count,
total rows, largest table, largest interaction order and coefficient cells. The text
summary shows mean rows across folds, the largest individual table across folds and
the maximum interaction order alongside mean table count and loss. Thus equal table
counts no longer hide different row counts. The Pareto objectives remain loss and
table count; row count does not silently change the selection rule.

These measurements require conversion of each selected fold prefix. The recorded
`conversion_seconds` includes dump creation, conversion and extraction of table data;
it is not scoring latency. Models are released after each fold's measurement. Large
artifacts can make this materially more expensive than the standalone inexpensive
`estimate_num_tables` helper. Conversion errors propagate instead of becoming invented
complexity scores. No prediction-parity claim is inferred from measuring structure.
`coefficient_cells` counts stored factor values, not independent fitted parameters.
Final full-data refits can differ from these CV artifacts and need separate review.
Measure scoring performance on your representative quotes outside the tuning loop.

`select(max_tables=...)` screens the mean CV table count at the selected iteration.
It does not enforce a limit on the final converted artifact. Inspect the final
`from_booster` result's `metadata['complexity']` and apply your delivery requirements
([conversion guide](lightgbm.md#conversion-support-and-verification)). `trial.fold_tables` retains
the fold distribution; a constant booster is a valid one-table artifact. When
the LightGBM in play is stock, the two interaction penalties are dropped from the search
with a warning instead of being tuned silently — LightGBM ignores an unknown parameter
with only a log line, so a search over one would otherwise spend its whole budget on a
knob wired to nothing.

The fork is packaged two ways — importable as `avenue_lightgbm` beside stock LightGBM, or
as `lightgbm` replacing it — so no import name is hardcoded. `resolve_lightgbm(dataset)`
returns the module and its name, taking the answer from the `Dataset` you pass whenever
you pass one: the two builds ship separate compiled libraries and separate `Dataset`
classes, so the one that built your frame is the only one that can train on it.

### Naming the levels behind the codes

LightGBM is handed numbers, so a converted model knows category codes and not what they
stood for — and a rating table whose column reads `3` is not something anyone can file.
Supply the names and the workbook writes level text instead:

```python
codes = pd.Categorical(df["VehBrand"])
booster_input["VehBrand"] = codes.codes            # what LightGBM sees

converted = converted.with_categories({"VehBrand": list(codes.categories)})
converted.to_workbook().save_csv_dir("plan")
```

```text
VehBrand,Relativity
(any other level),0.9078
B1,0.9176
B10,0.9368
B12,1.3073
```

A level's position in the list is its code; pass a `{code: name}` dict instead when the
codes are not contiguous. Naming is presentation only — the model matches on the code
either way, and the predictions are bit-identical before and after.

Which shape a category table takes is decided when the booster is trained, and **both
are good options**:

| how the feature is given to LightGBM | converted table | names apply |
|---|---|---|
| `categorical_feature=[...]`, integer-coded | `Int32`, one row per level, plus a `-999` wildcard | yes |
| a plain number | `Float64` band over the codes — a *grouping* of levels | no, a range is not one level |

Passing category codes as plain numbers imposes an ordering: splits group adjacent
codes, so results depend on how codes were assigned. Use this deliberately and compare
on held-out data. `with_categories` applies to the first shape, where each code
identifies a single level.

### Refit the selected table structure as a GLM

Conversion preserves the booster factors. Refitting estimates new GLM factors on
the same table structure, with reference levels and inference for supported fits.

A rating table is a *shape*: which bands, which levels, which interactions. Hand the
converted shapes to `Plan.given()` and the factors are re-estimated by the GLM engine:

```python
plan = Plan.frequency("Exposure")
for i, table in enumerate(converted.rating_tables()):
    plan = plan.given(f"t{i}", table)
filed = plan.fit(train, "frequency")
```

What comes out is an ordinary Poisson GLM — Wald standard errors, a reference row at
relativity 1.0, the same `report()` and `validate()` as any fitted model — whose banding
happened to be chosen by a booster rather than by hand. Algorithmic band selection is
ordinary practice; this is that idea with a data-driven band chooser.

Mean holdout Poisson deviance over three random splits of the French motor data
(`examples/refit_as_glm.py` reproduces it):

| | mean | vs the booster |
|---|---:|---:|
| GBM converted | 0.5858 | — |
| GLM refit | 0.5869 | +0.19% |
| GLM refit + ridge `alpha=1e-6` | 0.5858 | +0.02% |
| GLM with hand-chosen bands | 0.6019 | +2.75% |

In this experiment, GLM refitting retained most of the booster performance, and a
small ridge penalty closed the remaining gap. Both refits improved on the hand-banded
baseline.

The unpenalized refit provides Wald standard errors conditional on its table
structure. They do not account for the booster selecting that structure, and they
do not describe the ridge estimates. Penalized fits omit standard errors.

See the [executable stock/fork study](../examples/booster_pricing_study.py) for tuning through raw-quote reload and an audited rate change.

## Conversion support and verification

```python
from avenue_model import from_booster

# quote_predictors is a Polars frame, using the numeric inputs/codes of the booster.
result = from_booster(booster, quote_predictors, consolidation="max")
print(result.metadata)  # module/version, objective, parameters, dump fingerprint
print(result.parity)    # status, errors, failed rows, unmatched/nonfinite counts
if result.parity["status"] != "passed":
    raise ValueError("Review conversion discrepancies before using this model")
result.save("converted_plan")  # editable workbook plus conversion.json evidence
model = result.model
```

`from_booster(booster)` also works without a dataset. Its parity status is
`not_verified` and explicitly says numerical verification was not performed.
`FittedModel.from_lgbm_json()` remains available as a lower-level constructor; it
performs no parity check. A successful report establishes agreement on the supplied
inputs only. Include representative business and threshold/rare-category probes.

The report uses `abs(actual - expected) <= atol + rtol * abs(expected)`, with both
tolerances defaulting to 1e-12. Relative error is summarized over nonzero reference
means. Failed rows are zero-based positions in the supplied frame. Conversion metadata
and parity are saved separately from the editable scoring workbook; the evidence
belongs to the original artifact, not to later manual edits. The saved dump fingerprint
identifies the source model without storing training data.

### Structural counts

`metadata['complexity']` records `tables`, `total_rows`, `largest_table` and
`largest_interaction_order` after consolidation, including explicit missing routes.
Callers can compare these measurements with their own delivery limits. They measure
the artifact, not peak memory, scoring latency or statistical rank.

Supported objectives are regression/gaussian, Poisson, Gamma, Tweedie, and binary
with unit sigmoid. Multiclass, averaged ensembles, linear leaves and unsupported
split operators are rejected. The binary objective's serialized options preserve
its logit link. Inputs for this entry point are numeric booster values/codes;
`with_categories` can attach external labels to the resulting model.

Numerical null/NaN routes are represented by explicit missing-only rows, whose numeric
bound is `NaN`. Finite inputs cannot match those rows. LightGBM `missing_type="NaN"`
uses its declared default direction; `missing_type="None"` treats missing numerical
input as zero, as the booster does. Categorical nulls in Int32 input columns follow
the complement/wildcard route. Integer codes remain the boundary requirement for
categorical boosters.

`zero_as_missing` is unsupported and rejected because it uses a separate near-zero
routing rule. Constant-only boosters produce an intercept artifact in both modes.
Preprocessing is explicit caller code; see the booster example.

Converted step-table workbooks use format version 2 so older readers cannot
silently interpret a missing-only bound as an ordinary unconstrained bound.
The current reader also accepts version-1 artifacts. JSON encodes the bound as the string `"NaN"`; CSV writes `NaN`.
These are explicit rating-table matching rows, not nonfinite prediction values.
