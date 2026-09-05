# Executable stock/fork booster study

Install the tuning extra, then run from a checkout with a new output directory:

```sh
python examples/booster_pricing_study.py --output /tmp/booster-study
```

The script uses the installed build selected by `resolve_lightgbm`. Run it in separate
stock and fork environments to compare them. It records module/version and CPU use;
this study makes no GPU capability claim. Stock drops unsupported interaction-penalty
search parameters with a warning; the fork searches them. The three-trial, twenty-round
search is intentionally small enough for CI, not an adequate production tuning budget.

The study generates and loads a labeled synthetic auto CSV, or accepts `--data` with
the schema used by the ordinary auto study. The target is claims per exposure, weighted
by exposure. Preparation is audited without capping. Region labels are learned on
training data and explicitly encoded for the booster, then restored in the exported
model so raw quote labels work after reload.

The final year is held out before tuning. Training policies define reproducible grouped
inner CV folds. LightGBM CV uses its Dataset's shared bin construction; this is not the
fold-local bin-resolution guarantee of Avenue's GLM selection. The final year does not
participate in Dataset construction. For stricter fold-local booster preprocessing,
construct and train each fold's Dataset separately.

Artifacts include split records, trial parameters, per-fold table counts, a booster,
conversion parity evidence, named rating workbooks, a model report, quote explanations,
and a common weighted Poisson-loss comparison against a converged GLM. Reloaded raw
quote predictions must agree with the booster at `atol=rtol=1e-12`. A 5% intercept edit
produces factor/change exhibits and a fresh validation report. Boosters have no GLM
convergence certificate or classical coefficient inference; their comparison eligibility
is therefore unknown rather than silently asserted. Review their predictive evidence
explicitly. Optional GLM refitting is demonstrated separately in `refit_as_glm.py`;
its data-selected structure does not justify unconditional Wald inference.

The complexity report distinguishes mean CV table count and its fold distribution from
final table count, total rows and largest table. It records one warm batch timing as an
observation, not a speed comparison. CV complexity is computed at the selected boosting
iteration. `select(max_tables=...)` screens the mean CV proxy; it does not enforce a
final-artifact limit. Check the actual converted tables for any required resource limit.

Verified on stock 4.7.0 and fork 4.6.0.99: both selected one boosting round, converted to
two tables with five total rows, and passed 1,000 held-out quote parity checks and workbook
reload. This low-signal synthetic run establishes workflow mechanics, not an accuracy
advantage for boosting or fork penalties.
