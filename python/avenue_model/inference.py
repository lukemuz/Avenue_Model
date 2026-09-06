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


@dataclass(frozen=True)
class TermTests:
    table: pl.DataFrame
    metadata: dict


def _chi_square_survival(statistic, df):
    """Chi-square upper tail for positive integer degrees of freedom.

    Gamma's recurrence gives a finite sum for integer/half-integer shape. Evaluate
    the sum in log space, starting with exp(-x) or erfc(sqrt(x)), to avoid underflow
    at large df. No optional scientific dependency is needed for inference output.
    """
    if not isinstance(df, int) or df <= 0 or not math.isfinite(statistic) or statistic < 0:
        raise ValueError('Chi-square tails require nonnegative finite statistics and positive integer df')
    if statistic == 0:
        return 1.
    x = statistic / 2.
    if x == 0.:
        return 1.
    logs = []
    if df % 2:
        base = math.erfc(math.sqrt(x))
        if base:
            logs.append(math.log(base))
        logs.extend(-x + (k+.5)*math.log(x) - math.lgamma(k+1.5) for k in range(df//2))
    else:
        logs.extend(-x + k*math.log(x) - math.lgamma(k+1.) for k in range(df//2))
    if not logs:
        return 0.
    peak = max(logs)
    return min(1., math.exp(peak) * math.fsum(math.exp(value-peak) for value in logs))


def term_tests(model, *, dispersion='model'):
    """Joint asymptotic Wald tests of each fitted term's supported contrasts.

    Conditional on the other terms and fixed prespecified structure. A main effect
    in a hierarchical model tests that table's contrasts, not every interaction
    involving its predictor. No multiplicity, selection or small-sample correction.
    """
    if not isinstance(model, FittedModel) or model.converged is not True:
        raise ValueError('Term tests require an original, converged FittedModel')
    if dispersion not in ('model', 'quasi_poisson'):
        raise ValueError('dispersion must be model or quasi_poisson')
    if model.fit_options.get('alpha', 0.) != 0:
        raise ValueError('Joint Wald tests are unavailable for penalized fits')
    evidence = model.inference_summary
    if not evidence or evidence['covariance_method'] is None:
        raise ValueError(evidence.get('standard_errors_note') or 'Inference was not computed for this fit')
    multiplier = 1.
    if dispersion == 'quasi_poisson':
        if model.family != 'poisson' or evidence['covariance_method'] != 'model_based':
            raise ValueError('quasi_poisson requires model-based Poisson covariance; HC0/cluster cannot be rescaled')
        df = evidence['df_residual']
        if df <= 0 or not math.isfinite(df):
            raise ValueError('quasi_poisson requires positive residual degrees of freedom')
        multiplier = evidence['pearson_chi2'] / df
        if multiplier <= 0 or not math.isfinite(multiplier):
            raise ValueError('Pearson dispersion must be finite and positive')
    records = model._term_wald_statistics()
    for record in records:
        record['p_value'] = None
        if record['statistic'] is not None:
            record['statistic'] /= multiplier
            record['p_value'] = _chi_square_survival(record['statistic'], record['df'])
    schema = {'term': pl.String, 'table_index': pl.UInt64, 'df': pl.UInt64,
              'null_hypothesis': pl.String, 'excluded_rows': pl.List(pl.UInt64),
              'statistic': pl.Float64, 'status': pl.String, 'note': pl.String, 'p_value': pl.Float64}
    return TermTests(pl.DataFrame(records, schema=schema), {
        'method': 'joint Wald chi-square', 'covariance_method': evidence['covariance_method'],
        'dispersion_method': dispersion, 'covariance_multiplier': multiplier,
        'cluster_column': evidence['cluster_column'], 'n_clusters': evidence['n_clusters'],
        'interpretation': 'conditional on fixed structure and other terms; asymptotic; no selection, multiplicity or small-sample correction',
        'source_inference': evidence,
    })


def coefficient_intervals(model, *, confidence=.95, dispersion='model'):
    """Normal Wald intervals on the fitted coefficient scale and log-link factors.

    ``dispersion='quasi_poisson'`` uses Pearson chi-square / residual degrees of
    freedom from the original Poisson fit. Point estimates are unchanged. Neither
    route accounts for model selection or regularization. With the default
    dispersion='model', intervals use the fit's recorded covariance, including HC0
    or one-way clustered covariance when requested during fitting.
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
    covariance_method = evidence['covariance_method']
    base_dispersion = evidence['dispersion']
    if covariance_method == 'model_based' and (not math.isfinite(base_dispersion) or base_dispersion <= 0):
        raise ValueError('Fit dispersion must be finite and positive for Wald intervals')
    scale = base_dispersion
    if dispersion == 'quasi_poisson':
        if covariance_method != 'model_based':
            raise ValueError('quasi_poisson cannot rescale HC0 or cluster covariance')
        if model.family != 'poisson':
            raise ValueError('quasi_poisson requires a Poisson fit')
        df = evidence['df_residual']
        if not math.isfinite(df) or df <= 0:
            raise ValueError('quasi_poisson requires positive residual degrees of freedom')
        scale = evidence['pearson_chi2'] / df
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError('Pearson dispersion must be finite and positive')
    multiplier = math.sqrt(scale / base_dispersion) if covariance_method == 'model_based' else 1.
    z = -NormalDist().inv_cdf((1. - confidence) / 2.)
    tables = {}
    for name, table in model.rating_tables_by_name().items():
        generated = ['Interval_Standard_Error', 'Coefficient_Lower', 'Coefficient_Upper', 'Interval_Status']
        if 'Relativity' in table.columns:
            generated += ['Relativity_Lower', 'Relativity_Upper']
        collisions = sorted(set(generated).intersection(table.columns))
        if collisions:
            raise ValueError(f'Derived interval columns {collisions} conflict with existing columns in {name!r}; rename the predictors before creating this exhibit')
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
        'dispersion': scale if covariance_method == 'model_based' else None,
        'covariance_method': covariance_method, 'source_inference': evidence,
        'interpretation': ('conditional on fixed structure and independent clusters; no selection or small-sample adjustment'
                           if covariance_method == 'cluster_cr0' else
                           'conditional on the fixed model structure; no selection or cluster adjustment'),
        'point_estimates_changed': False,
    })
