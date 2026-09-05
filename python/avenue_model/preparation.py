"""Explicit insurance response preparation with row-level population accounting."""
from dataclasses import dataclass
import math

import polars as pl


@dataclass
class PreparedPricing:
    frequency: pl.DataFrame
    severity: pl.DataFrame
    pure_premium: pl.DataFrame
    audit: pl.DataFrame
    summary: pl.DataFrame
    columns: dict

    def experience(self, factor):
        """Experience on the common positive-exposure population; null ratios stay null."""
        exposure, claims, loss = (self.columns[k] for k in ('exposure', 'claims', 'loss'))
        return (self.frequency.group_by(factor, maintain_order=True)
                .agg(pl.len().alias('rows'), pl.col(exposure).sum().alias('exposure'),
                     pl.col(claims).sum().alias('claims'), pl.col(loss).sum().alias('loss'))
                .with_columns((pl.col('claims') / pl.col('exposure')).alias('frequency'),
                              pl.when(pl.col('claims') > 0).then(pl.col('loss') / pl.col('claims'))
                              .otherwise(None).alias('severity'),
                              (pl.col('loss') / pl.col('exposure')).alias('pure_premium')))


def prepare_pricing(data, *, exposure, claims, loss, predictors=(), invalid='raise',
                    large_loss=None):
    """Prepare frequency, positive severity and pure premium with auditable exclusions.

    Loss is explicitly caller-supplied (for example, developed/trended loss). No
    capping, development or exposure adjustment is performed. ``invalid='exclude'``
    explicitly excludes invalid source rows and records every reason. Large losses
    are flagged only. Zero exposure excludes rate records; zero loss excludes Gamma
    severity records but remains valid for frequency/pure premium.
    """
    if not isinstance(data, pl.DataFrame) or not data.height:
        raise ValueError('Preparation requires a nonempty Polars DataFrame')
    if invalid not in ('raise', 'exclude'):
        raise ValueError("invalid must be 'raise' or 'exclude'")
    if large_loss is not None and (not math.isfinite(large_loss) or large_loss <= 0):
        raise ValueError('large_loss must be a finite positive threshold')
    if len({exposure, claims, loss}) != 3:
        raise ValueError('Exposure, claims and loss must name distinct source columns')
    generated = ('avenue_row', 'avenue_frequency', 'avenue_severity', 'avenue_pure_premium')
    if any(name in data.columns for name in generated):
        raise ValueError(f'Input already contains a reserved preparation column: {generated}')
    numeric = {}
    for name in (exposure, claims, loss):
        if not data[name].dtype.is_numeric():
            raise TypeError(f'{name!r} must be numerical')
        numeric[name] = data[name].cast(pl.Float64).to_list()
    if isinstance(predictors, str):
        predictors = (predictors,)
    predictor_values = {name: data[name].to_list() for name in predictors}
    audits, rate_rows, severity_rows = [], [], []
    for row in range(data.height):
        e, c, l = (numeric[name][row] for name in (exposure, claims, loss))
        reasons, flags = [], []
        for name, value in ((exposure, e), (claims, c), (loss, l)):
            if value is None or not math.isfinite(value):
                reasons.append(f'{name}:missing_or_nonfinite')
            elif value < 0:
                reasons.append(f'{name}:negative')
        for name, values in predictor_values.items():
            value = values[row]
            if value is None or (isinstance(value, float) and not math.isfinite(value)):
                reasons.append(f'{name}:missing_predictor')
        if not reasons:
            if c != math.floor(c):
                reasons.append('claims:not_integer')
            if c == 0 and l != 0:
                reasons.append('loss_without_claims')
            if e == 0 and (c > 0 or l > 0):
                reasons.append('activity_without_exposure')
            if e == 0:
                flags.append('zero_exposure')
            if c > 0 and l == 0:
                flags.append('zero_claim_loss')
            if large_loss is not None and l >= large_loss:
                flags.append('large_loss')
        valid = not reasons
        rate = valid and e > 0
        severity = rate and c > 0 and l > 0
        rate_rows.append(bool(rate))
        severity_rows.append(bool(severity))
        audits.append({'row': row, 'valid': valid, 'frequency_included': bool(rate),
                       'severity_included': bool(severity), 'pure_premium_included': bool(rate),
                       'reasons': ', '.join(reasons), 'flags': ', '.join(flags)})
    audit = pl.DataFrame(audits)
    if invalid == 'raise' and not audit['valid'].all():
        first = next(record for record in audits if not record['valid'])
        raise ValueError(f"Invalid pricing row {first['row']}: {first['reasons']}. "
                         "Use invalid='exclude' to obtain the row-level exclusion audit.")
    source = data.with_row_index('avenue_row')
    rate = source.filter(pl.Series(rate_rows)).with_columns(
        (pl.col(claims) / pl.col(exposure)).alias('avenue_frequency'),
        (pl.col(loss) / pl.col(exposure)).alias('avenue_pure_premium'))
    severity = source.filter(pl.Series(severity_rows)).with_columns(
        (pl.col(loss) / pl.col(claims)).alias('avenue_severity'))
    summary = []
    for name, frame in (('frequency', rate), ('severity', severity), ('pure_premium', rate)):
        summary.append({'population': name, 'input_rows': data.height, 'included_rows': frame.height,
                        'excluded_rows': data.height - frame.height,
                        'exposure': float(frame[exposure].sum()), 'claims': float(frame[claims].sum()),
                        'loss': float(frame[loss].sum())})
    return PreparedPricing(rate, severity, rate.clone(), audit, pl.DataFrame(summary),
                           {'exposure': exposure, 'claims': claims, 'loss': loss})
