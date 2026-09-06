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
convergence certificate or classical coefficient inference. After the selected schedule
and parity checks finish, the tutorial explicitly supplies `training_status="completed"`
to comparison. The booster becomes eligible under the same holdout loss criterion while
its `converged` value stays null. Completion is distinct from predictive quality and
never overrides a reported failure.

Add `--refit-glm` to run the optional booster-structure Poisson GLM branch:

```sh
python examples/booster_pricing_study.py --output /tmp/booster-refit-study --refit-glm
```

This uses `Plan.given` to re-estimate supported table rows on the pre-2022 training
population, with a new intercept and a prespecified ridge penalty (`alpha=1e-4`,
`l1_ratio=0`). It retains the pre-fit parameter count and support warnings in
`refit_check.json`, requires convergence, and compares the new model on the same
untouched final year as the GLM and booster. Penalized rows with no training exposure
use the reference relativity and remain flagged as unsupported; refitting does not create
evidence for those rows. A report, quote explanations, portfolio/region change tables,
and an analytical `refit_bundle` preserve the fit, split, validation and lineage.
Raw-label quote predictions must survive bundle reload at `atol=rtol=1e-12`.

Refitting changes predictions and is not exact conversion. The selected structure
depends on training outcomes; neither classical post-selection confidence intervals
nor uncertainty about the structure selection is supplied. Penalized coefficient
standard errors are withheld. The fixed penalty is an example specification, not a
tuned optimum. Selecting it using CV on this already learned structure would omit
structure-selection uncertainty and leakage from that performance estimate; evaluating
the full procedure requires learning the booster structure within each training fold.
Ordinary CI and the installed-wheel acceptance runner exercise this optional branch.

The complexity report distinguishes mean CV table count and its fold distribution from
final table count, total rows and largest table. It records one warm batch timing as an
observation, not a speed comparison. CV complexity is computed at the selected boosting
iteration. `select(max_tables=...)` screens the mean CV proxy; it does not enforce a
final-artifact limit. A caller can explicitly enforce final structural limits with
`from_booster(..., resource_limits={'tables': N, 'total_rows': M})`; the tutorial
itself imposes no arbitrary hard cap.
The tutorial also passes the first 256 training predictor rows as a common scoring
batch to tuning. Every fold retains three public-call timings and selected-prefix
booster parity; the tuning summary shows the mean of fold median times. These timing
rows do not change the selection objectives or supply a new validation metric.

Verified on stock 4.7.0 and fork 4.6.0.99: both selected one boosting round, converted to
two tables with five total rows, and passed 1,000 held-out quote parity checks and workbook
reload. This low-signal synthetic run establishes workflow mechanics, not an accuracy
advantage for boosting or fork penalties.
