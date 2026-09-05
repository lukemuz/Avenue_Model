"""Tune a LightGBM model for accuracy *and* for how many rating tables it becomes.

Table count is an inexpensive complexity proxy. It grows with the distinct feature
combinations in an ensemble, but does not describe table rows, support or scoring cost.
Conversion supports a subset of booster semantics; use `from_booster` with quote data
to retain numerical parity evidence for the supplied inputs.

`tune_lgbm` runs an Optuna study with two objectives — cross-validated loss and the
mean consolidated table count — and returns the Pareto frontier, so the trade-off is
chosen rather than stumbled into. Each selected fold prefix is converted with maximum
consolidation to measure table count, total rows, largest table and interaction order.
Conversion cost is recorded separately. These are actual fold artifacts, not promises
about a later full-data refit or estimates of statistical rank and training support.

Two of the levers only exist in `avenue-lightgbm`, a small fork that adds penalties
aimed at the table count directly rather than at tree size:

    interaction_penalty      penalises a split whose feature combination is new to the
                             ensemble — the quantity that *is* the table count
    interaction_complexity   penalises each feature newly introduced within one tree,
                             preferring main effects and low-order interactions

Both default to zero and cost nothing when unused. When the LightGBM in play is stock,
they are dropped from the search with a warning rather than tuned silently — an unknown
parameter is warned about and ignored by LightGBM, so tuning one that does not exist
would otherwise burn the whole budget on a knob wired to nothing.

The fork is packaged two ways, importable as either `avenue_lightgbm` or `lightgbm`, so
nothing here hardcodes an import name — see `resolve_lightgbm`.

The method and the case studies behind it are in *GBMs as Factor Tables* (Muzynoski,
2025), <https://avenue-analytics.com/research/avenue-analytics-methodology.pdf>.

Requires the `tuning` extra:

    pip install "avenue-model[tuning]"
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

__all__ = ["TuningResult", "resolve_lightgbm", "supports_interaction_penalties",
           "tune_lgbm"]

# The fork is packaged two ways in the wild, and which one is installed decides both
# the import name and whether the penalties exist at all:
#
#   avenue_lightgbm   distribution `avenue-lightgbm`, coexists with stock LightGBM
#   lightgbm          distribution `lightgbm`, replaces it as a drop-in
#
# Guessing wrong is silent rather than loud - a Dataset built by one build cannot be
# trained by the other, and probing the wrong module reports the wrong answer about the
# penalties - so nothing here imports a module by a hardcoded name.
LIGHTGBM_MODULES = ("avenue_lightgbm", "lightgbm")


_EXTRA_HINT = (
    "avenue_model.tune_lgbm needs lightgbm and optuna. "
    'Install them with: pip install "avenue-model[tuning]"'
)

# The two parameters that only exist in the avenue-lightgbm fork.
INTERACTION_PARAMS = ("interaction_penalty", "interaction_complexity")

# Ranges from the tuning used for the paper's case studies. `interaction_penalty` runs
# far wider than `interaction_complexity` because it is compared against a split gain
# whose scale depends on the objective, while the complexity term is a divisor.
DEFAULT_SPACE: dict[str, tuple[float, float]] = {
    "num_leaves": (3, 6),
    "max_depth": (2, 4),
    "learning_rate": (0.01, 0.6),
    "num_iterations": (50, 5000),
    "min_data_in_leaf": (1, 1000),
    "min_gain_to_split": (0.0, 1000.0),
    "feature_fraction": (0.1, 1.0),
    "bagging_fraction": (0.1, 1.0),
    "lambda_l1": (0.0, 20.0),
    "lambda_l2": (0.0, 20.0),
    "interaction_penalty": (0.0, 1000.0),
    "interaction_complexity": (0.0, 500.0),
}

_INT_PARAMS = frozenset({"num_leaves", "max_depth", "num_iterations", "min_data_in_leaf"})

# What LightGBM calls the validation curve for each objective.
_DEFAULT_METRIC = {
    "binary": "binary_logloss",
    "multiclass": "multi_logloss",
    "regression": "l2",
    "poisson": "poisson",
    "gamma": "gamma",
    "tweedie": "tweedie",
    "huber": "huber",
    "quantile": "quantile",
    "mape": "mape",
    "l1": "l1",
}


def resolve_lightgbm(dataset=None):
    """The LightGBM module to use, and its name.

    When a `Dataset` is given its own defining module wins, always. The two builds ship
    separate compiled libraries and separate `Dataset` classes, so a frame built by one
    cannot be trained by the other, and preferring the fork here because it happens to
    be installed would break a caller who deliberately built with stock LightGBM.

    With no `Dataset` to go on, the fork is preferred: installing `avenue_lightgbm`
    alongside stock LightGBM is a deliberate act, and the penalties are the reason to
    do it.
    """
    import importlib

    if dataset is not None:
        name = type(dataset).__module__.split(".")[0]
        if name in LIGHTGBM_MODULES:
            return importlib.import_module(name), name
        # Not a LightGBM Dataset at all, or a build under an unfamiliar name; fall
        # through rather than guessing, so the error names the real problem.

    for name in LIGHTGBM_MODULES:
        try:
            return importlib.import_module(name), name
        except ImportError:
            continue
    raise ImportError(f"{_EXTRA_HINT} (missing: no LightGBM found; tried "
                      f"{' and '.join(LIGHTGBM_MODULES)})")


def _require_deps(dataset=None):
    try:
        import optuna  # noqa: F401
    except ImportError as exc:  # pragma: no cover - exercised by the error path only
        raise ImportError(f"{_EXTRA_HINT} (missing: {exc.name})") from exc
    lightgbm, name = resolve_lightgbm(dataset)
    return lightgbm, optuna, name


class _LogProbe:
    """Collects LightGBM's own log lines, which is the only place it reports a bad name."""

    def __init__(self) -> None:
        self.lines: list[str] = []

    def _record(self, message: object) -> None:
        self.lines.append(str(message))

    info = warning = error = debug = _record


def supports_interaction_penalties(dataset=None) -> bool:
    """Does the LightGBM in play accept the interaction penalties?

    Pass the `Dataset` you intend to train on, so the answer is about the build that
    will actually run rather than about whichever module imports first.

    LightGBM does not raise on an unrecognised parameter — it logs
    ``Unknown parameter: <name>`` and carries on with the default, so a caller who sets
    `interaction_penalty` against stock LightGBM gets a silently unchanged model. That
    log line is the only reliable signal: `booster.params` merely echoes the dict it was
    handed and reports the key as present either way.

    So the probe trains a one-round booster on six rows with the parameter set, and
    reads LightGBM's log rather than its return value.
    """
    lightgbm, _, module_name = _require_deps(dataset)

    import numpy as np

    probe = _LogProbe()
    try:
        # register_logger has no getter, so the stock logger is restored by re-registering
        # lightgbm's module-level default rather than by saving the current one.
        lightgbm.register_logger(probe)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # ndarray, not a list: LightGBM rejects a plain list of lists with
            # "Data list can only be of ndarray or Sequence", and a probe that dies
            # on its own input reports "unsupported" for every build there is.
            data = lightgbm.Dataset(
                np.array([[0.0], [1.0], [0.0], [1.0], [0.0], [1.0]]),
                label=np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0]),
            )
            lightgbm.train(
                {"objective": "regression", "num_leaves": 2, "verbose": 1,
                 "min_data_in_leaf": 1, "min_data_in_bin": 1,
                 "interaction_penalty": 1.0},
                data,
                num_boost_round=1,
            )
    except Exception as exc:
        # Failing closed is the safe default - it only drops the penalties from a
        # search - but doing it silently is the exact failure this probe exists to
        # catch, so it is never silent.
        warnings.warn(
            f"could not determine whether {module_name} supports the interaction "
            f"penalties ({type(exc).__name__}: {exc}); assuming it does not",
            RuntimeWarning, stacklevel=2)
        return False
    finally:
        import logging

        # register_logger has no getter, so the default is restored by name rather
        # than by saving whatever was there.
        lightgbm.register_logger(logging.getLogger(module_name))

    return not any("Unknown parameter" in line and "interaction_penalty" in line
                   for line in probe.lines)


def _model_json(booster, num_iteration=None) -> str:
    return json.dumps(booster.dump_model(num_iteration=num_iteration))


@dataclass
class Trial:
    """One hyperparameter configuration, scored on both objectives."""

    params: dict[str, Any]
    cv_loss: float
    tables: float
    num_iterations: int
    fold_tables: list[float] = field(default_factory=list)
    fold_complexity: list[dict[str, Any]] = field(default_factory=list)

    def dominates(self, other: "Trial") -> bool:
        no_worse = self.cv_loss <= other.cv_loss and self.tables <= other.tables
        strictly_better = self.cv_loss < other.cv_loss or self.tables < other.tables
        return no_worse and strictly_better


@dataclass
class TuningResult:
    """Every trial, and the non-dominated ones.

    `frontier` is sorted by table count. Fewer tables do not necessarily mean fewer
    rows or easier review; inspect fold_complexity as well. The study does not decide
    the trade-off for you.
    """

    trials: list[Trial]
    metric: str
    tuned_interaction_penalties: bool
    lightgbm: str = "lightgbm"
    study: Any = field(default=None, repr=False)

    @property
    def frontier(self) -> list[Trial]:
        keep = [t for t in self.trials
                if not any(o.dominates(t) for o in self.trials if o is not t)]
        return sorted(keep, key=lambda t: (t.tables, t.cv_loss))

    @property
    def best_cv(self) -> Trial:
        return min(self.trials, key=lambda t: t.cv_loss)

    def select(self, max_tables: float | None = None) -> Trial:
        """Pick by mean CV table count; this does not constrain the final artifact."""
        candidates = self.frontier
        if max_tables is not None:
            within = [t for t in candidates if t.tables <= max_tables]
            if not within:
                cheapest = min(candidates, key=lambda t: t.tables)
                raise ValueError(
                    f"no configuration reached {max_tables} tables or fewer; the "
                    f"smallest found was {cheapest.tables:.0f}. Widen the search, or "
                    f"raise the budget."
                )
            candidates = within
        return min(candidates, key=lambda t: t.cv_loss)

    def summary(self) -> str:
        lines = [
            f"  {len(self.trials)} trials, {len(self.frontier)} on the frontier"
            f"   metric: {self.metric}   build: {self.lightgbm}",
        ]
        if not self.tuned_interaction_penalties:
            lines.append(f"  interaction penalties were NOT tuned - {self.lightgbm} "
                         f"does not accept them")
        lines.append(f"  {'mean tables':>12}{'mean rows':>12}{'largest':>10}{'order':>7}{'mean score ms':>15}{'cv loss':>14}   parameters")
        for t in self.frontier:
            shown = {k: v for k, v in t.params.items() if k in DEFAULT_SPACE}
            rendered = ", ".join(
                f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}"
                for k, v in sorted(shown.items()))
            if t.fold_complexity:
                mean_rows = sum(c['total_rows'] for c in t.fold_complexity) / len(t.fold_complexity)
                largest = max(c['largest_table'] for c in t.fold_complexity)
                order = max(c['largest_interaction_order'] for c in t.fold_complexity)
                complexity = f"{mean_rows:>12.1f}{largest:>10}{order:>7}"
            else:
                complexity = f"{'unknown':>12}{'unknown':>10}{'unknown':>7}"
            times = [c.get('scoring_seconds') for c in t.fold_complexity]
            score = (f"{1000 * sum(times) / len(times):.3f}"
                     if times and all(v is not None for v in times) else 'unknown')
            lines.append(f"  {t.tables:>12.2f}{complexity}{score:>15}{t.cv_loss:>14.6f}   {rendered}")
        return "\n".join(lines)


def tune_lgbm(
    dataset,
    params: dict[str, Any],
    *,
    tunable: Sequence[str] | None = None,
    space: dict[str, tuple[float, float]] | None = None,
    n_trials: int = 50,
    timeout: float | None = None,
    nfold: int = 3,
    folds: Iterable | None = None,
    metric: str | None = None,
    seed: int | None = None,
    callback: Callable[[Trial], None] | None = None,
    scoring_data=None,
) -> TuningResult:
    """Search for boosters that are both accurate and small enough to read.

    Args:
        dataset: a `lightgbm.Dataset` to cross-validate on.
        params: base LightGBM parameters. Must include `objective`. Anything not named
            in `tunable` is held fixed.
        tunable: parameter names to search. Defaults to the shape parameters plus, when
            the fork is installed, the two interaction penalties.
        space: overrides for the search ranges, merged over `DEFAULT_SPACE`.
        n_trials: Optuna trials. Ignored when `timeout` is given.
        timeout: wall-clock budget in seconds, in place of a trial count.
        nfold: cross-validation folds; ignored when `folds` is given.
        folds: an explicit splitter, for grouped or time-ordered data.
        metric: LightGBM validation metric. Defaults by objective.
        seed: sampler seed, for a reproducible search.
        callback: called with each completed `Trial`, for progress reporting.
        scoring_data: optional nonempty Polars quote frame with numeric booster
            values/category codes. Every selected fold prefix is scored three times
            on this same frame, with booster parity checked before recording timings.
            These observations do not enter the loss/table-count selection objectives.

    Returns:
        A `TuningResult`. Use `.frontier` to see the trade-off, `.select(max_tables=N)`
        to screen by mean CV table count (not a final-artifact limit), and `.best_cv` for the most accurate configuration
        regardless of size.
    """
    lightgbm, optuna, module_name = _require_deps(dataset)
    if scoring_data is not None:
        import polars as pl
        if not isinstance(scoring_data, pl.DataFrame):
            raise TypeError('scoring_data must be a Polars DataFrame of numerical booster inputs')
        if not scoring_data.height:
            raise ValueError('scoring_data must contain at least one quote')
        scoring_data = scoring_data.clone()

    if "objective" not in params:
        raise ValueError("params must include 'objective'")
    objective_name = params["objective"]
    metric = metric or _DEFAULT_METRIC.get(objective_name)
    if metric is None:
        raise ValueError(
            f"no default metric for objective {objective_name!r}; pass metric=...")

    search_space = {**DEFAULT_SPACE, **(space or {})}

    forked = supports_interaction_penalties(dataset)
    if tunable is None:
        tunable = ["num_leaves", "max_depth", "learning_rate", "min_data_in_leaf",
                   "feature_fraction", "lambda_l2"]
        if forked:
            tunable = [*tunable, *INTERACTION_PARAMS]
    tunable = list(tunable)

    asked_for = [p for p in tunable if p in INTERACTION_PARAMS]
    if asked_for and not forked:
        warnings.warn(
            f"{', '.join(asked_for)} not supported by the installed LightGBM, so they "
            f"are dropped from the search. LightGBM ignores unknown parameters with a "
            f"log warning rather than an error, so tuning them here would spend the "
            f"budget on knobs wired to nothing. The build in use is {module_name!r}; "
            f"install avenue-lightgbm to use them: "
            f"https://github.com/lukemuz/avenue-lightgbm",
            RuntimeWarning,
            stacklevel=2,
        )
        tunable = [p for p in tunable if p not in INTERACTION_PARAMS]

    unknown = set(tunable) - set(search_space)
    if unknown:
        raise ValueError(
            f"no search range for {sorted(unknown)}; pass space={{...}} to add one")

    # LightGBM pre-filters features using the first `min_data_in_leaf` it is given and
    # caches that decision on the Dataset, then refuses any later trial that lowers the
    # value: "Reducing min_data_in_leaf with feature_pre_filter=true may cause
    # unexpected behaviour". Tuning the parameter therefore requires turning the
    # pre-filter off, which has to happen before the Dataset is constructed — by the
    # time the first trial has run it is too late.
    if "min_data_in_leaf" in tunable:
        try:
            dataset.params = {**(dataset.params or {}), "feature_pre_filter": False}
        except AttributeError:  # pragma: no cover - older lightgbm layouts
            pass
        if getattr(dataset, "_handle", getattr(dataset, "handle", None)) is not None:
            raise ValueError(
                "this Dataset is already constructed, so LightGBM's feature pre-filter "
                "is fixed to the min_data_in_leaf it was built with and later trials "
                "cannot lower it. Build it with "
                "lgb.Dataset(..., params={'feature_pre_filter': False}), or drop "
                "'min_data_in_leaf' from `tunable`.")

    # Conversion comes from the compiled engine; importing it here rather
    # than at module scope keeps this module importable from a source checkout.
    from .avenue_model import FittedModel
    import time

    # Materialize one-shot fold iterators once so every trial sees the same split.
    if folds is not None and not hasattr(folds, 'split'):
        folds = list(folds)
        if not folds:
            raise ValueError('folds must contain at least one train/validation pair')

    stratified = objective_name in ("binary", "multiclass")
    curve_key = f"valid {metric}-mean"
    trials: list[Trial] = []

    def run_trial(trial) -> tuple[float, float]:
        trial_params = dict(params)
        for name in tunable:
            low, high = search_space[name]
            if name in _INT_PARAMS:
                trial_params[name] = trial.suggest_int(name, int(low), int(high))
            else:
                trial_params[name] = trial.suggest_float(name, float(low), float(high))

        cv_args = dict(params=trial_params, train_set=dataset, metrics=metric,
                       stratified=stratified, return_cvbooster=True)
        if seed is not None:
            cv_args["seed"] = seed
        if folds is not None:
            cv_args["folds"] = folds
        else:
            cv_args["nfold"] = nfold

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = lightgbm.cv(**cv_args)

        curve = result[curve_key]
        # The best round rather than the last: `lgb.cv` reports the whole curve, and
        # taking its minimum is equivalent to early stopping without the extra knob.
        best_round = int(min(range(len(curve)), key=curve.__getitem__))
        cv_loss = float(curve[best_round])

        # Complexity must describe the same boosting prefix as the selected loss.
        # Constant ensembles legitimately have one intercept table. Invalid dumps
        # raise their real error instead of becoming an invented complexity score.
        complexity = []
        for booster in result["cvbooster"].boosters:
            started = time.perf_counter()
            converted = FittedModel.from_lgbm_json(_model_json(booster, best_round + 1), consolidation='max')
            artifact = converted.to_workbook(scale='factor').tables
            rows = [table.height for table in artifact]
            orders = [table.width - 1 for table in artifact]
            complexity.append({'tables': len(artifact), 'total_rows': sum(rows),
                               'largest_table': max(rows), 'largest_interaction_order': max(orders),
                               'coefficient_cells': sum(rows),
                               'conversion_seconds': time.perf_counter() - started,
                               'consolidation': 'max', 'num_iterations': best_round + 1,
                               'support_status': 'not_measured', 'statistical_rank': None,
                               'scoring_seconds': None})
            if scoring_data is not None:
                import numpy as np
                from .splitting import _fingerprint
                predictors = scoring_data.select(booster.feature_name())
                if any(not column.dtype.is_numeric() for column in predictors):
                    raise TypeError('scoring_data predictors must contain numerical booster values/category codes')
                reference = np.asarray(booster.predict(predictors.to_numpy(), num_iteration=best_round + 1))
                if reference.shape != (predictors.height,) or not np.isfinite(reference).all():
                    raise ValueError('Scoring-cost reference must contain one finite mean per quote')
                samples = []
                max_error = 0.
                for _ in range(3):
                    started = time.perf_counter()
                    actual = converted.predict(predictors).to_numpy().reshape(-1)
                    samples.append(time.perf_counter() - started)
                    if actual.shape != reference.shape or not np.isfinite(actual).all() or not np.allclose(actual, reference, atol=1e-12, rtol=1e-12):
                        raise ValueError('Converted fold failed scoring-data parity; scoring cost is not accepted')
                    max_error = max(max_error, float(np.max(np.abs(actual - reference))))
                complexity[-1].update(scoring_seconds=float(np.median(samples)),
                    scoring_samples_seconds=samples, scoring_rows=predictors.height,
                    scoring_data_fingerprint=_fingerprint(predictors),
                    scoring_polars_version=pl.__version__,
                    scoring_feature_names=booster.feature_name(),
                    scoring_method='median of three public batch calls, including first call',
                    scoring_parity={'status': 'passed', 'atol': 1e-12, 'rtol': 1e-12,
                                    'max_absolute_error': max_error})
            del converted, artifact
        counts = [float(c['tables']) for c in complexity]
        tables = sum(counts) / len(counts)

        record = Trial(params=dict(trial_params), cv_loss=cv_loss, tables=tables,
                       num_iterations=best_round + 1, fold_tables=counts, fold_complexity=complexity)
        trials.append(record)
        if callback is not None:
            callback(record)
        return cv_loss, tables

    # Before create_study, which announces itself at INFO on the way in.
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(multivariate=len(tunable) > 1, seed=seed)
    study = optuna.create_study(directions=["minimize", "minimize"], sampler=sampler)
    if timeout is not None:
        study.optimize(run_trial, timeout=timeout)
    else:
        study.optimize(run_trial, n_trials=n_trials)

    return TuningResult(trials=trials, metric=metric,
                        tuned_interaction_penalties=forked, lightgbm=module_name,
                        study=study)
