"""Explicit, fold-local GLM grid evaluation with retained failures and common loss."""
from dataclasses import dataclass
import json
import math

import polars as pl

from .avenue_model import GLMOptions, Plan
from .comparison import Candidate, compare_models
from .splitting import Fold, SplitSpec, _fingerprint


@dataclass(frozen=True)
class GLMTrial:
    """An unresolved Plan and JSON-compatible keyword arguments for GLMOptions."""
    plan: Plan
    options: dict | None = None


@dataclass
class GLMSelection:
    summary: pl.DataFrame
    history: pl.DataFrame
    metadata: dict
    recommended: str | None

    def refit(self, data):
        """Refit the selected specification on the original full selection dataset.

        This is a new fit, not an independent performance estimate. Validate on a
        separate final holdout. Nonconvergence raises instead of returning a winner.
        """
        if self.recommended is None:
            raise ValueError('No trial converged and scored successfully in every fold')
        if _fingerprint(data) != self.metadata['population_fingerprint']:
            raise ValueError('Refit data differs from the original selection dataset')
        spec = self.metadata['trials'][self.recommended]
        model = Plan.from_json(spec['plan_json']).fit(
            data, self.metadata['target'], GLMOptions(**spec['options']))
        if model.converged is not True:
            raise ValueError('Selected full-data refit did not converge; inspect or revise the specification')
        return model

    def save(self, directory):
        """Save review tables and declarative trial/fold specifications, not data rows."""
        from pathlib import Path
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=False)
        self.summary.write_csv(path / 'summary.csv')
        self.history.write_csv(path / 'history.csv')
        (path / 'selection.json').write_text(json.dumps(
            {**self.metadata, 'recommended': self.recommended}, indent=2,
            sort_keys=True, allow_nan=False) + '\n')


def select_glm(data, trials, *, target, unit, metric, split, weight=None,
               tweedie_power=1.5):
    """Evaluate named GLMTrial specifications on identical held-out fold populations.

    ``split`` accepts a SplitSpec or a sequence of already resolved Fold objects.
    Plans resolve categories and bands only within each training fold. Trial names
    break exact loss ties in insertion order. Validation loss is pooled by weight,
    not an unweighted average of fold means. All folds must converge and score for
    a trial to be eligible. No statistical claim of post-selection coverage is made.
    """
    if not isinstance(data, pl.DataFrame) or not data.height:
        raise ValueError('Selection requires a nonempty Polars DataFrame')
    if not trials:
        raise ValueError('At least one GLMTrial is required')
    specifications = {}
    for name, trial in trials.items():
        if not isinstance(name, str) or not name or not isinstance(trial, GLMTrial):
            raise ValueError('Trials require nonempty names and GLMTrial values')
        if not isinstance(trial.plan, Plan):
            raise TypeError('GLMTrial.plan must be an unresolved Plan')
        options = {} if trial.options is None else trial.options
        # Snapshot caller dictionaries before fitting; no executable factories.
        options = json.loads(json.dumps(options, allow_nan=False))
        if not isinstance(options, dict):
            raise TypeError('GLMTrial.options must be a dictionary of GLMOptions arguments')
        if 'tweedie_power' in options:
            raise ValueError('Set fitting Tweedie power on the trial Plan, not GLMOptions')
        GLMOptions(**options)
        specifications[name] = {'plan_json': trial.plan.to_json(), 'options': options}
    folds = split.split(data) if isinstance(split, SplitSpec) else list(split)
    if not folds or any(not isinstance(fold, Fold) for fold in folds):
        raise ValueError('split must resolve to at least one Fold')
    frames = []
    seen_validation = set()
    for fold in folds:
        train, validation = fold.frames(data)
        if seen_validation.intersection(fold.validation_rows):
            raise ValueError('Validation rows overlap between folds; use one nonoverlapping evaluation partition')
        seen_validation.update(fold.validation_rows)
        # Check loss, unit, target and weights before attributing failures to trials.
        compare_models(validation, {'check': Candidate([1.] * validation.height, unit, True)},
                       target=target, unit=unit, metric=metric, weight=weight,
                       tweedie_power=tweedie_power)
        frames.append((train, validation))
    records = []
    summaries = []
    for name, spec in specifications.items():
        trial_records = []
        for fold, (train, validation) in zip(folds, frames):
            record = {'trial': name, 'fold': fold.name, 'split_id': fold.split_id,
                      'status': 'failed', 'converged': None, 'eligible': False,
                      'train_rows': train.height, 'validation_rows': validation.height,
                      'weight': None, 'mean_loss': None, 'ae_ratio': None,
                      'n_parameters': None, 'iterations': None, 'max_gradient': None,
                      'findings_json': None, 'error': None}
            try:
                model = Plan.from_json(spec['plan_json']).fit(train, target, GLMOptions(**spec['options']))
                report = model.report()
                fit = report.fit_summary
                record.update(converged=model.converged,
                              n_parameters=fit.get('n_parameters'), iterations=fit.get('iterations'),
                              max_gradient=fit.get('max_gradient'),
                              findings_json=json.dumps(report.findings, allow_nan=False))
                result = compare_models(validation, {'fit': Candidate(model, unit)},
                                        target=target, unit=unit, metric=metric,
                                        weight=weight, tweedie_power=tweedie_power)
                score = result.summary.row(0, named=True)
                record.update({key: score[key] for key in (
                    'status', 'eligible', 'weight', 'mean_loss', 'ae_ratio', 'error')})
            except (ValueError, TypeError, OverflowError, pl.exceptions.PolarsError) as error:
                record['error'] = str(error)
            trial_records.append(record)
        records.extend(trial_records)
        eligible = all(r['eligible'] for r in trial_records)
        scored = all(r['status'] == 'scored' for r in trial_records)
        support = math.fsum(r['weight'] for r in trial_records) if scored else None
        loss = math.fsum(r['weight'] * r['mean_loss'] for r in trial_records) / support if scored else None
        summaries.append({'trial': name, 'eligible': eligible, 'folds': len(folds),
                          'converged_folds': sum(r['converged'] is True for r in trial_records),
                          'scored_folds': sum(r['status'] == 'scored' for r in trial_records),
                          'validation_rows': len(seen_validation), 'weight': support,
                          'mean_loss': loss})
    candidates = [row for row in summaries if row['eligible']]
    recommended = min(candidates, key=lambda row: row['mean_loss'])['trial'] if candidates else None
    return GLMSelection(
        pl.DataFrame(summaries, schema_overrides={'weight': pl.Float64, 'mean_loss': pl.Float64}),
        pl.DataFrame(records, schema_overrides={
            'weight': pl.Float64, 'mean_loss': pl.Float64, 'ae_ratio': pl.Float64,
            'converged': pl.Boolean, 'error': pl.String, 'findings_json': pl.String,
            'max_gradient': pl.Float64,
            'n_parameters': pl.Int64, 'iterations': pl.Int64}),
        {'schema_version': 1, 'trials': specifications,
         'folds': [json.loads(fold.to_json()) for fold in folds],
         'population_fingerprint': _fingerprint(data), 'target': target,
         'unit': unit, 'metric': metric, 'weight': weight,
         'tweedie_power': tweedie_power if metric == 'tweedie' else None,
         'aggregation': 'pooled validation loss weighted by support',
         'uncertainty': 'selection scores; no post-selection confidence intervals'}, recommended)
