"""Conditional Poisson–Gamma partial pooling of one categorical relativity."""
from dataclasses import dataclass, field
import hashlib
import json
import math
from numbers import Real
from pathlib import Path
import tempfile

import polars as pl

from .avenue_model import FittedModel, Workbook
from .splitting import _fingerprint


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@dataclass
class CredibilityResult:
    model: FittedModel
    posterior: pl.DataFrame
    metadata: dict
    _source_identity: tuple = field(init=False, repr=False)

    def __post_init__(self):
        self._source_identity = self._identity()

    def _identity(self):
        with tempfile.TemporaryDirectory(prefix='avenue-credibility-identity-') as directory:
            path = Path(directory) / 'model.json'
            self.model.to_workbook(scale='factor').save_json(str(path))
            model = json.loads(path.read_text())
        # Workbook creation timestamps change during serialization; scoring
        # geometry, factors, encodings and response conventions must not change.
        model['manifest'].pop('created', None)
        def digest(value):
            return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()
        return (digest(model), hashlib.sha256(self.posterior.write_csv().encode()).hexdigest(),
                digest(self.metadata))

    def save(self, directory):
        """Save scoring means with separate, integrity-checked posterior evidence."""
        if self._identity() != self._source_identity:
            raise ValueError('Credibility result changed in memory; export the edited model as a Workbook without the original posterior evidence')
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=False)
        self.model.to_workbook(scale='factor').save_json(str(path / 'model.json'))
        self.posterior.write_csv(path / 'posterior.csv')
        record = {'schema_version': 1, 'metadata': self.metadata,
                  'sha256': {name: _hash(path / name) for name in ['model.json', 'posterior.csv']}}
        (path / 'credibility.json').write_text(json.dumps(record, indent=2, allow_nan=False) + '\n')

    @classmethod
    def load(cls, directory):
        path = Path(directory)
        record = json.loads((path / 'credibility.json').read_text())
        if record.get('schema_version') != 1:
            raise ValueError('Unsupported credibility evidence version')
        for name in ['model.json', 'posterior.csv']:
            if record['sha256'].get(name) != _hash(path / name):
                raise ValueError('Credibility artifact changed; load its Workbook for scoring without attributing the original posterior')
        return cls(Workbook.load_json(str(path / 'model.json')).to_model(),
                   pl.read_csv(path / 'posterior.csv', schema_overrides={'group': pl.String}), record['metadata'])


def poisson_credibility(data, baseline, *, group, counts, prior_strength, confidence=.95):
    """Pool group relativities using a prespecified Gamma(shape=a, rate=a) prior.

    Counts must be nonnegative integers. Baseline expected counts include exposure
    exactly once. Posterior Gamma parameters are a + group counts and a + baseline
    expected counts. Uncertainty is conditional on that baseline and fixed a; this
    function estimates neither baseline uncertainty nor hyperparameters. Group labels
    must be strings and cannot already be a baseline predictor. Unseen quote groups
    follow Avenue's strict unmatched policy; declared zero-support groups retain the prior.
    The returned ordinary scoring model predicts counts, with predict_rate available.
    """
    try:
        from scipy.stats import gamma
    except ImportError as exc:
        raise ImportError('Poisson credibility intervals require avenue_model[credibility] (SciPy)') from exc
    if not isinstance(data, pl.DataFrame) or not data.height:
        raise ValueError('Credibility requires a nonempty Polars DataFrame')
    if not isinstance(baseline, FittedModel) or baseline.family != 'poisson' or baseline.prediction_kind not in ('rate', 'count'):
        raise ValueError('Supply a Poisson baseline with an explicit exposure convention')
    if baseline.converged is False:
        raise ValueError('A nonconverged baseline cannot supply credibility expectations')
    if not isinstance(group, str) or not isinstance(counts, str) or group == counts:
        raise ValueError('Declare distinct group and count columns')
    if group in baseline.input_schema['predictors'] or group == baseline.input_schema['exposure']:
        raise ValueError('The pooled group must not already be a baseline predictor or exposure column')
    if not isinstance(prior_strength, Real) or isinstance(prior_strength, bool) or not math.isfinite(prior_strength) or prior_strength <= 0:
        raise ValueError('prior_strength must be finite and positive')
    if not isinstance(confidence, Real) or isinstance(confidence, bool) or not math.isfinite(confidence) or not 0 < confidence < 1:
        raise ValueError('confidence must be strictly between zero and one')
    confidence = float(confidence)
    labels = data[group]
    if labels.dtype != pl.String or labels.null_count():
        raise ValueError('Group labels must be non-null strings')
    if not data[counts].dtype.is_numeric():
        raise ValueError('Observed counts must be numeric nonnegative integers')
    observed = data[counts].to_list()
    if any(v is None or isinstance(v, bool) or not math.isfinite(v) or v < 0 or v != math.floor(v) for v in observed):
        raise ValueError('Observed counts must be finite nonnegative integers, not rates or fractional weights')
    # Float aggregation avoids silent Int64 wraparound for large grouped totals.
    observed = [float(v) for v in observed]
    expected = baseline.predict_count(data).to_series().to_list()
    if any(not math.isfinite(e) or e < 0 or (e == 0 and y > 0) for y, e in zip(observed, expected)):
        raise ValueError('Baseline expectations must be finite/nonnegative; positive counts require positive expectation')
    grouped = pl.DataFrame({'group': labels, 'counts': observed, 'expected': expected}).group_by('group').agg(
        pl.len().alias('rows'), pl.col('counts').sum(), pl.col('expected').sum()).sort('group')
    records = []
    a = float(prior_strength)
    for row in grouped.iter_rows(named=True):
        k, e = row['counts'], row['expected']
        shape, rate = a + k, a + e
        mean = shape / rate
        tail = (1-confidence)/2
        lower = gamma.ppf(tail, a=shape, scale=1/rate)
        upper = gamma.isf(tail, a=shape, scale=1/rate)
        if not all(math.isfinite(v) and v > 0 for v in [shape, rate, mean, lower, upper]):
            raise ValueError('Posterior calculation exceeds the supported numerical range')
        records.append({**row, 'unpooled_relativity': k/e if e else None,
                        'credibility_weight': e/rate, 'posterior_shape': shape, 'posterior_rate': rate,
                        'relativity': mean, 'posterior_sd': math.sqrt(shape)/rate,
                        'relativity_lower': float(lower), 'relativity_upper': float(upper),
                        'status': 'posterior' if e else 'prior_only'})
    posterior = pl.DataFrame(records, schema_overrides={'counts': pl.Float64, 'unpooled_relativity': pl.Float64})
    # Reuse the canonical workbook loader and category encoding rather than create
    # a separate scoring implementation. Reloading correctly withholds GLM inference.
    with tempfile.TemporaryDirectory(prefix='avenue-credibility-') as directory:
        path = Path(directory) / 'model.json'
        baseline.to_workbook(scale='factor').save_json(str(path))
        baseline_hash = _hash(path)
        workbook = json.loads(path.read_text())
        name = 'credibility.' + group
        if any(t['name'] == name for t in workbook['manifest']['tables']):
            raise ValueError('Credibility table name already exists in the baseline')
        workbook['manifest']['tables'].append({'name': name})
        levels = posterior['group'].to_list()
        workbook['manifest'].setdefault('encodings', {})[group] = [[level, code] for code, level in enumerate(levels)]
        workbook['tables'].append([{group: code, 'Rating_Factor': math.log(value)}
                                   for code, value in enumerate(posterior['relativity'])])
        workbook['manifest']['target'] = counts
        workbook['manifest']['exposure_role'] = 'offset'
        path.write_text(json.dumps(workbook, allow_nan=False))
        model = Workbook.load_json(str(path)).to_model()
    return CredibilityResult(model, posterior, {
        'method': 'conditional Poisson-Gamma group relativity', 'group': group, 'counts': counts,
        'prior_shape': a, 'prior_rate': a, 'prior_mean': 1., 'confidence': confidence,
        'training_fingerprint': _fingerprint(data), 'baseline_workbook_sha256': baseline_hash,
        'excluded_rows': 0, 'training_status': 'completed',
        'interpretation': 'Gamma credible intervals for group relativities conditional on a fixed baseline and prespecified prior strength; no baseline, hyperparameter or selection uncertainty',
        'prediction_unit': 'count', 'unseen_groups': 'strict unmatched error',
        'reference': 'https://statproofbook.github.io/P/poissexp-post.html',
    })
