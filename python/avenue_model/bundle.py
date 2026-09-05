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


@dataclass(frozen=True)
class ComposedBundle:
    """A scoring graph and independently preserved analytical component bundles.

    `components` retains named child bundles, recursively. `source_model` uses their
    original workbooks; `model` uses their current editable workbooks. Component
    inference stays in each child's `source_evidence`; no joint covariance is implied.
    """
    model: object
    source_model: object
    source_evidence: dict
    components: dict
    changed_files: tuple

    @property
    def edited(self):
        """Whether any descendant scoring workbook differs from its saved source."""
        return bool(self.changed_files)


def _component_evidence_hashes(directory):
    return {name: digest for name, digest in _hashes(directory).items()
            if 'scoring' not in Path(name).parts}


def _check_graph(model, ancestors=None):
    from .composition import ComposedModel
    ancestors = set() if ancestors is None else ancestors
    if id(model) in ancestors:
        raise ValueError('Composition graph contains a cycle')
    if isinstance(model, ComposedModel):
        model.__post_init__()
        for child in model.components.values():
            _check_graph(child, ancestors | {id(model)})
    elif not isinstance(model, FittedModel):
        raise TypeError('Analytical composition leaves must be FittedModel objects')


def _save_composed_bundle(model, directory, *, fit_options, training_id, fold,
                          validation_data, validation_id, preprocessing, unit,
                          lineage, component_context, validation_options):
    _check_graph(model)
    contexts = {} if component_context is None else component_context
    if not isinstance(contexts, dict) or set(contexts) - set(model.components):
        raise ValueError('component_context must map existing component names to bundle options')
    if any(not isinstance(options, dict) for options in contexts.values()):
        raise TypeError('Each component_context value must be a dictionary of bundle options')
    if unit is not None and unit != model.unit:
        raise ValueError('Bundle unit must match the composition response unit')
    if validation_data is None and validation_options is not None:
        raise ValueError('validation_options requires validation_data')
    target = Path(directory)
    if target.exists():
        raise FileExistsError(f'bundle destination already exists: {target}')
    validation = None
    if validation_data is not None:
        compared = model.validate(validation_data, **(validation_options or {}))
        validation = {'summary': compared.summary.to_dicts(),
                      'segments': {name: table.to_dicts() for name, table in compared.segments.items()},
                      'metadata': compared.metadata}
    entries = [{'name': name} for name in model.components]
    evidence = _evidence({'schema_version': 2, 'kind': 'composition',
        'created_at': datetime.now(timezone.utc).isoformat(),
        'operation': model.operation, 'unit': model.unit, 'components': entries,
        'converged': model.converged, 'validation': validation,
        'validation_recorded': validation_data is not None,
        'inference': 'Component source evidence only; no joint composition covariance or intervals.',
        'caller_context': {'fit_options': fit_options, 'training_id': training_id,
                          'validation_id': validation_id, 'preprocessing': preprocessing,
                          'unit': unit, 'lineage': lineage},
        'fold': None if fold is None else json.loads(fold.to_json()),
        'split_id': None if fold is None else fold.split_id})
    json.dumps(evidence, allow_nan=False)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.avenue-composed-bundle-', dir=target.parent))
    try:
        source = staging / 'source'
        source.mkdir()
        _write(source / 'evidence.json', evidence)
        for index, (name, child) in enumerate(model.components.items()):
            save_bundle(child, staging / 'components' / f'component_{index}', **contexts.get(name, {}))
        _write(staging / 'bundle.json', {'schema_version': 2, 'kind': 'composition',
            'source_hashes': _hashes(source),
            'component_evidence_hashes': _component_evidence_hashes(staging / 'components')})
        if target.exists():
            raise FileExistsError(f'bundle destination already exists: {target}')
        staging.rename(target)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return load_bundle(target)


def save_bundle(model, directory, *, fit_options=None, training_id=None, fold=None,
                validation_data=None, validation_id=None, preprocessing=None,
                unit=None, lineage=None, component_context=None, validation_options=None):
    """Save to a new directory; optional context must be declarative JSON data.

    Effective fitting options are captured from the original fit. The optional
    fit_options annotation and dataset identifiers remain separate caller assertions.
    No data rows are persisted. Validation is evaluated now when data is supplied.
    Accepts an individual FittedModel or a nested ComposedModel graph. For graphs,
    component_context maps child names to keyword arguments for their save_bundle
    calls, including nested component_context and distinct validation populations.
    Root graph validation requires explicit target/metric in validation_options.
    Per-row predictions and input frames are not stored. Inference remains attached
    to original component fits, not to the loaded graph scorer.
    """
    from .composition import ComposedModel
    if isinstance(model, ComposedModel):
        return _save_composed_bundle(model, directory, fit_options=fit_options,
            training_id=training_id, fold=fold, validation_data=validation_data,
            validation_id=validation_id, preprocessing=preprocessing, unit=unit,
            lineage=lineage, component_context=component_context, validation_options=validation_options)
    if not isinstance(model, FittedModel):
        raise TypeError('save_bundle requires a FittedModel or ComposedModel')
    if component_context is not None or validation_options is not None:
        raise ValueError('component_context and validation_options apply only to composed models')
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
    from .inference import term_tests
    try:
        joint = term_tests(model)
        joint_evidence = {'status': 'recorded', 'table': joint.table.to_dicts(), 'metadata': joint.metadata}
    except ValueError as error:
        joint_evidence = {'status': 'unavailable', 'reason': str(error)}
    evidence = _evidence({
        'schema_version': 1,
        'created_at': datetime.now(timezone.utc).isoformat(),
        'plan': None if model.plan is None else json.loads(model.plan.to_json()),
        'input_schema': model.input_schema,
        'fit_summary': report.fit_summary,
        'effective_fit_options': model.fit_options,
        'solver_used': model.solver_used,
        'inference_summary': model.inference_summary,
        'term_tests': joint_evidence,
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
    if manifest.get('schema_version') == 2:
        from .composition import ComposedModel
        if manifest.get('kind') != 'composition':
            raise ValueError('Unsupported analytical bundle kind')
        if (_hashes(directory / 'source') != manifest['source_hashes'] or
                _component_evidence_hashes(directory / 'components') != manifest['component_evidence_hashes']):
            raise ValueError('Composition bundle source integrity check failed')
        evidence = json.loads((directory / 'source' / 'evidence.json').read_text())
        if evidence.get('schema_version') != 2 or evidence.get('kind') != 'composition':
            raise ValueError('Unsupported composition source evidence schema')
        children = {}
        changed = []
        for index, entry in enumerate(evidence['components']):
            name = entry['name']
            if name in children:
                raise ValueError('Duplicate composition bundle component name')
            relative = f'components/component_{index}'
            child = load_bundle(directory / relative)
            children[name] = child
            prefix = relative if isinstance(child, ComposedBundle) else relative + '/scoring'
            changed.extend(f'{prefix}/{path}' for path in child.changed_files)
        return ComposedBundle(
            ComposedModel(evidence['operation'], {name: child.model for name, child in children.items()}, evidence['unit']),
            ComposedModel(evidence['operation'], {name: child.source_model for name, child in children.items()}, evidence['unit']),
            evidence, children, tuple(sorted(changed)))
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
