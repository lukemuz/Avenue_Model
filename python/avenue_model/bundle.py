"""Versioned source evidence alongside an explicitly editable scoring workbook."""
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import tempfile

from .avenue_model import FittedModel, Plan, Workbook


def _evidence(value):
    # Preserve unavailable inference and infinite bounds without nonstandard JSON.
    if isinstance(value, float) and not math.isfinite(value):
        return {'nonfinite': str(value)}
    if isinstance(value, dict):
        return {k: _evidence(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_evidence(v) for v in value]
    return value


def _write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def _hashes(directory):
    return {p.relative_to(directory).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(directory.rglob('*')) if p.is_file()}


@dataclass(frozen=True)
class ModelBundle:
    """Source evidence never becomes fitting diagnostics on the current scorer."""
    model: FittedModel
    source_model: FittedModel
    source_evidence: dict
    changed_files: tuple

    @property
    def edited(self):
        return bool(self.changed_files)

    @property
    def source_plan(self):
        plan = self.source_evidence['plan']
        return None if plan is None else Plan.from_json(json.dumps(plan))


def save_bundle(model, directory, *, fit_options=None, training_id=None, fold=None,
                validation_data=None, validation_id=None, preprocessing=None,
                unit=None, lineage=None):
    """Save to a new directory; optional context must be declarative JSON data.

    Fit options and dataset identifiers are caller assertions, not inferred records.
    No data rows are persisted. Validation is evaluated now when data is supplied.
    This version accepts individual FittedModel objects, not composition graphs.
    """
    if not isinstance(model, FittedModel):
        raise TypeError('save_bundle requires a FittedModel; bundle components separately')
    target = Path(directory)
    if target.exists():
        raise FileExistsError(f'bundle destination already exists: {target}')
    report = model.report(validation_data)
    validation = report.validation
    validation_evidence = None
    if validation is not None:
        validation_evidence = {name: getattr(validation, name) for name in (
            'n_rows', 'unmatched_rows', 'n_scored', 'deviance', 'null_deviance',
            'pseudo_r2', 'total_actual', 'total_expected', 'ae_ratio', 'lift', 'gini')}
        validation_evidence['calibration'] = validation.calibration.to_dicts()
        validation_evidence['actual_vs_expected'] = {
            name: frame.to_dicts() for name, frame in zip(model.table_names, validation.actual_vs_expected)}
    evidence = _evidence({
        'schema_version': 1,
        'created_at': datetime.now(timezone.utc).isoformat(),
        'plan': None if model.plan is None else json.loads(model.plan.to_json()),
        'input_schema': model.input_schema,
        'fit_summary': report.fit_summary,
        'inference_summary': model.inference_summary,
        'findings': report.findings,
        'resolved': report.resolved,
        'estimates': {name: frame.to_dicts() for name, frame in model.rating_tables_by_name().items()},
        'report_markdown': report.markdown,
        'validation': validation_evidence,
        'validation_recorded': validation_data is not None,
        'caller_context': {'fit_options': fit_options, 'training_id': training_id,
                           'validation_id': validation_id, 'preprocessing': preprocessing,
                           'unit': unit, 'lineage': lineage},
        'fold': None if fold is None else json.loads(fold.to_json()),
        'split_id': None if fold is None else fold.split_id,
    })
    # Validate metadata before creating any output; never pickle arbitrary objects.
    json.dumps(evidence, allow_nan=False)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.avenue-bundle-', dir=target.parent))
    try:
        source = staging / 'source'
        source.mkdir()
        model.to_workbook(scale='factor').save_json(str(source / 'workbook.json'))
        _write(source / 'evidence.json', evidence)
        model.to_workbook(scale='factor').save_csv_dir(str(staging / 'scoring'))
        _write(staging / 'bundle.json', {
            'schema_version': 1, 'source_hashes': _hashes(source),
            'scoring_hashes': _hashes(staging / 'scoring'),
        })
        if target.exists():
            raise FileExistsError(f'bundle destination already exists: {target}')
        staging.rename(target)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return load_bundle(target)


def load_bundle(directory):
    """Verify original evidence integrity and identify edits before loading the scorer.

    SHA-256 detects accidental changes; this is not a signed authenticity claim.
    Byte-level changes, including formatting-only edits, are conservatively flagged.
    """
    directory = Path(directory)
    manifest = json.loads((directory / 'bundle.json').read_text())
    if manifest.get('schema_version') != 1:
        raise ValueError('unsupported analytical bundle schema version')
    if _hashes(directory / 'source') != manifest['source_hashes']:
        raise ValueError('bundle source integrity check failed; original evidence changed')
    evidence = json.loads((directory / 'source' / 'evidence.json').read_text())
    if evidence.get('schema_version') != 1:
        raise ValueError('unsupported source evidence schema version')
    current = _hashes(directory / 'scoring')
    original = manifest['scoring_hashes']
    changed = tuple(sorted(k for k in current.keys() | original.keys()
                           if current.get(k) != original.get(k)))
    return ModelBundle(
        Workbook.load_csv_dir(str(directory / 'scoring')).to_model(),
        Workbook.load_json(str(directory / 'source' / 'workbook.json')).to_model(),
        evidence, changed,
    )
