# LightGBM as rating tables

How a gradient booster becomes a set of rating tables, how to keep those tables small
enough that someone will read them, and what it takes to turn the result into a GLM.

The short version lives in the [README](../README.md#exact-lightgbm-conversion); this is
the working detail. The method and the case studies behind it are in *GBMs as Factor
Tables: Achieving Both Transparency and Interpretability Without Approximation*
(Muzynoski, 2025), [PDF](https://avenue-analytics.com/research/avenue-analytics-methodology.pdf).

**Contents:** [Table size](#making-the-tables-small-enough-to-read) ·
[Tuning](#tuning-for-interpretability) · [Category names](#naming-the-levels-behind-the-codes) ·
[Refitting as a GLM](#refit-the-selected-table-structure-as-a-glm)

---

### Making the tables small enough to read

A booster converts exactly whatever its shape, but not every exact model is a readable
one. Two quantities decide that, and neither is the tree count — on freMTPL2 at a fixed
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

Both are zero by default and cost nothing when unused. Their effect is large and cheap:
holding every other parameter fixed on the French motor data, `interaction_penalty` alone
takes a 39-table model to 12 for 0.1% of cross-validated loss, and to 5 for 0.7%.

| `interaction_penalty` | tables | cv Poisson |
|---:|---:|---:|
| 0 | 39 | 0.310534 |
| 10 | 12 | 0.310831 |
| 100 | 5 | 0.312757 |

That is the trade-off the paper is about, and why it is worth searching rather than
assuming. Its selected model — four features — beats EBM out of sample (0.5934 against
0.5994), and its best cross-validated one reaches 0.5834.

### Tuning for interpretability

`tune_lgbm` runs an Optuna study against both objectives at once and hands back the
frontier, so the trade-off is chosen rather than discovered afterwards:

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
`coefficient_cells` counts stored factor values, not independent fitted parameters:
`statistical_rank` is null and `support_status` is `not_measured`.
Support/rank require the corresponding data and fitting specification. Final full-data
refits can differ from these CV artifacts and still need separate review.

Pass `scoring_data=quotes` to measure scoring cost for every selected fold artifact.
Use a nonempty Polars frame with the booster's numeric predictors/category codes;
feature columns are selected in booster order. For example:

```python
result = tune_lgbm(dataset, {"objective": "poisson"}, n_trials=10,
                   scoring_data=encoded_quotes)
```

Each fold scores the same quote frame three times using public `predict`, including
input preparation, matching and returned-array materialization. `scoring_seconds` is
the median of those three calls; `scoring_samples_seconds` retains each observation,
including the first. The text summary's `mean score ms` averages the fold medians.
Without quote data this field stays null and the summary displays `unknown`.

Every scored vector must agree with that fold booster's **selected prefix** at
`atol=rtol=1e-12`, with finite outputs, before the cost is accepted. A mismatch raises
an error. The record includes quote count, predictor order, data fingerprint, Polars
version, maximum absolute parity error and tolerances. Parity only covers these quotes.
Reference scoring, parity checks and fingerprinting are outside the timed calls.

Choose a representative batch size and record your hardware/thread settings when
comparing timings. These observations are not isolated memory measurements, prepared
cache timings, or universal latency estimates. Timing adds three public predictions
per fold, and does not enter either Pareto objective. The data are used for scoring
measurements, not training or loss evaluation; this is not an additional validation
score or training-support estimate.

`select(max_tables=...)` screens the mean CV table count at the selected iteration.
It does not enforce a limit on the final converted artifact. `trial.fold_tables` retains
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

Integer codes passed as plain numbers are a standard and often better choice, especially
at high cardinality where set-based splits overfit; the resulting table groups adjacent
codes into a band, which is a real modelling choice rather than a defect. Names can only
stand in for a single level, so `with_categories` applies to the first shape.

### Refit the selected table structure as a GLM

The usual objection to a converted model is not that the tables are wrong — it is that
a reviewer is being handed a tree ensemble, and the tables carry no standard errors and
no reference levels. There is a way around that which costs almost nothing.

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

Refitting costs a fifth of a percent, a whisker of ridge recovers it, and both beat bands
chosen by hand by nearly 3%.

The unpenalised refit also carries Wald standard errors and reference levels, which a
converted model does not. A penalised fit omits the standard errors, so the ridge row
is the better model and the unpenalised row is the one to quote errors from.

See the [executable stock/fork study](BOOSTER_STUDY.md) for tuning through raw-quote reload and an audited rate change.
