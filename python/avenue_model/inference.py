"""Direct, explicitly conditional coefficient intervals for supported original fits."""
from dataclasses import dataclass
import math
from statistics import NormalDist

import polars as pl
from .avenue_model import FittedModel


@dataclass(frozen=True)
class CoefficientIntervals:
    tables: dict[str, pl.DataFrame]
    metadata: dict


def coefficient_intervals(model, *, confidence=.95, dispersion='model'):
    """Normal Wald intervals on the fitted coefficient scale and log-link factors.

    ``dispersion='quasi_poisson'`` uses Pearson chi-square / residual degrees of
    freedom from the original Poisson fit. Point estimates are unchanged. Neither
    route accounts for clustering, model selection or regularization.
    """
    if not isinstance(model, FittedModel) or model.converged is not True:
        raise ValueError('Intervals require an original, converged FittedModel')
    if not isinstance(confidence, (float, int)) or not 0 < confidence < 1:
        raise ValueError('confidence must be strictly between zero and one')
    if dispersion not in ('model', 'quasi_poisson'):
        raise ValueError('dispersion must be model or quasi_poisson')
    evidence = model.inference_summary
    if not evidence:
        raise ValueError('Inference was not computed for this fit')
    if evidence['standard_errors_note']:
        raise ValueError(evidence['standard_errors_note'])
    base_dispersion = evidence['dispersion']
    if not math.isfinite(base_dispersion) or base_dispersion <= 0:
        raise ValueError('Fit dispersion must be finite and positive for Wald intervals')
    scale = base_dispersion
    if dispersion == 'quasi_poisson':
        if model.family != 'poisson':
            raise ValueError('quasi_poisson requires a Poisson fit')
        df = evidence['df_residual']
        if not math.isfinite(df) or df <= 0:
            raise ValueError('quasi_poisson requires positive residual degrees of freedom')
        scale = evidence['pearson_chi2'] / df
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError('Pearson dispersion must be finite and positive')
    multiplier = math.sqrt(scale / base_dispersion)
    z = -NormalDist().inv_cdf((1. - confidence) / 2.)
    tables = {}
    for name, table in model.rating_tables_by_name().items():
        errors, lowers, uppers, statuses = [], [], [], []
        for row in table.iter_rows(named=True):
            coefficient, error = row['Coefficient'], row['Standard_Error']
            available = (error is not None and math.isfinite(error) and error >= 0
                         and coefficient is not None and math.isfinite(coefficient))
            if available:
                adjusted = error * multiplier
                errors.append(adjusted)
                lowers.append(coefficient - z * adjusted)
                uppers.append(coefficient + z * adjusted)
                statuses.append('fixed' if error == 0 else 'wald')
            else:
                errors.append(None)
                lowers.append(None)
                uppers.append(None)
                statuses.append('unavailable')
        table = table.with_columns(
            pl.Series('Interval_Standard_Error', errors, dtype=pl.Float64),
            pl.Series('Coefficient_Lower', lowers, dtype=pl.Float64),
            pl.Series('Coefficient_Upper', uppers, dtype=pl.Float64),
            pl.Series('Interval_Status', statuses, dtype=pl.String))
        if 'Relativity' in table.columns:
            table = table.with_columns(
                pl.col('Coefficient_Lower').exp().alias('Relativity_Lower'),
                pl.col('Coefficient_Upper').exp().alias('Relativity_Upper'))
        tables[name] = table
    return CoefficientIntervals(tables, {
        'confidence': confidence, 'method': 'normal Wald', 'dispersion_method': dispersion,
        'dispersion': scale, 'source_inference': evidence,
        'interpretation': 'conditional on the fixed model structure; no selection or cluster adjustment',
        'point_estimates_changed': False,
    })
