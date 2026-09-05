"""Named response-scale composition retaining independently scored components."""
from dataclasses import dataclass
import json
import math
from pathlib import Path

import polars as pl

from .avenue_model import Workbook
from .comparison import Candidate, compare_models


@dataclass
class ComposedModel:
    operation: str
    components: dict
    unit: str = 'loss_per_exposure'

    def __post_init__(self):
        if self.operation not in ('sum', 'frequency_severity'):
            raise ValueError('Composition operation must be sum or frequency_severity')
        if not self.components or any(not isinstance(k, str) or not k or k == 'predictions' for k in self.components):
            raise ValueError('Components need nonempty names other than predictions')
        if not isinstance(self.unit, str) or not self.unit:
            raise ValueError('Declare a nonempty composition response unit')
        if self.operation == 'frequency_severity' and set(self.components) != {'frequency', 'severity'}:
            raise ValueError('Frequency-severity composition requires frequency and severity components')
        if self.operation == 'frequency_severity':
            if getattr(self.components['frequency'], 'prediction_kind', None) not in ('rate', 'count'):
                raise ValueError('Frequency component must declare a Poisson exposure convention')
            if getattr(self.components['severity'], 'family', None) != 'gamma':
                raise ValueError('Severity component must be a Gamma severity model')
        if self.operation == 'sum':
            for model in self.components.values():
                if getattr(model, 'prediction_kind', 'response') in ('rate', 'count'):
                    raise ValueError('Sum loss-cost means, not Poisson frequency rates/counts')
                if isinstance(model, ComposedModel) and model.unit != self.unit:
                    raise ValueError('Component loss-cost units differ')

    @property
    def family(self):
        return None

    @property
    def prediction_kind(self):
        return 'response'

    @property
    def converged(self):
        states = [getattr(model, 'converged', None) for model in self.components.values()]
        return False if False in states else True if all(state is True for state in states) else None

    def predict_components(self, data):
        values = {}
        for name, model in self.components.items():
            frame = (model.predict_rate(data) if self.operation == 'frequency_severity' and name == 'frequency'
                     else model.predict(data))
            if frame.width != 1 or frame.height != data.height:
                raise ValueError(f'Component {name!r} must return one mean per input row')
            series = frame.to_series()
            if any(v is None or not math.isfinite(v) or v < 0 for v in series):
                raise ValueError(f'Component {name!r} produced invalid loss-cost means')
            values[name] = series.to_list()
        return pl.DataFrame(values)

    def predict(self, data):
        components = self.predict_components(data)
        means = []
        for row in components.iter_rows():
            value = math.fsum(row) if self.operation == 'sum' else math.prod(row)
            if not math.isfinite(value):
                raise ValueError('Composed response mean is nonfinite')
            means.append(value)
        return pl.DataFrame({'predictions': pl.Series(means, dtype=pl.Float64)})

    def validate(self, data, *, target, metric, weight=None, **options):
        """Return the common comparison exhibits under an explicit evaluation metric."""
        return compare_models(data, {'composed': Candidate(self, self.unit)}, target=target,
                              unit=self.unit, metric=metric, weight=weight, **options)

    def save(self, directory):
        """Save a versioned composition manifest and independently editable components."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        entries = []
        for index, (name, model) in enumerate(self.components.items()):
            path = directory / f'component_{index}'
            nested = isinstance(model, ComposedModel)
            if nested:
                model.save(path)
            else:
                model.to_workbook().save_csv_dir(str(path))
            entries.append({'name': name, 'nested': nested})
        (directory / 'composition.json').write_text(json.dumps(
            {'schema_version': 1, 'operation': self.operation, 'unit': self.unit,
             'components': entries}, indent=2) + '\n')

    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        manifest = json.loads((directory / 'composition.json').read_text())
        if manifest['schema_version'] != 1:
            raise ValueError('Unsupported composition schema version')
        components = {}
        for index, entry in enumerate(manifest['components']):
            if entry['name'] in components:
                raise ValueError('Duplicate component name')
            path = directory / f'component_{index}'
            components[entry['name']] = cls.load(path) if entry['nested'] else Workbook.load_csv_dir(str(path)).to_model()
        return cls(manifest['operation'], components, manifest['unit'])


def frequency_severity(frequency, severity, *, unit='loss_per_exposure'):
    """Multiply a recorded Poisson frequency rate by mean claim severity.

    Count-offset frequency components are converted to rates before multiplication.
    The caller asserts a compatible severity definition/cost level and exposure unit.
    """
    if getattr(frequency, 'prediction_kind', None) not in ('rate', 'count'):
        raise ValueError('Frequency component must declare a Poisson exposure convention')
    if getattr(severity, 'family', None) != 'gamma':
        raise ValueError('Severity component must be a Gamma severity model')
    return ComposedModel('frequency_severity', {'frequency': frequency, 'severity': severity}, unit)


def sum_loss_costs(components, *, unit='loss_per_exposure'):
    """Sum named peril loss costs on the response scale, preserving component lineage.

    The caller declares compatible currency, exposure and cost-level units. No
    likelihood family is assigned to the sum. Legacy FittedModel + is unchanged.
    """
    return ComposedModel('sum', dict(components), unit)
