"""Fresh-wheel real frequency/severity study; never writes historical evaluation outputs.

Inputs: OpenML 41214 policy parquet and 41215 claim parquet. No caps or monetary
adjustments. Frequency here counts observed paid claim records, not reported ClaimNb.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import resource
import time
import warnings

import numpy as np
import pandas as pd
import polars as pl
from glum import GeneralizedLinearRegressor, TweedieDistribution
from avenue_model import GLMOptions, Plan, coefficient_intervals, term_tests, Workbook
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_tweedie_deviance


BANDS = {'age': [21., 26., 36., 51., 71.], 'vehicle_age': [1., 5., 10., 20.],
         'bonus': [51., 60., 80., 100., 150.]}
PREDICTORS = [*BANDS, 'region', 'fuel']


def write(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def shape(plan, smooth=False):
    for name, breaks in BANDS.items():
        plan = plan.spline(name, quantile=5) if smooth else plan.banded(name, breaks=breaks)
    return plan.categorical('region', base='first').categorical('fuel', base='first')


def resolved_knots(model):
    return {term['name']: term['knots'] for term in model.resolved if term.get('knots') is not None} or None


def native_frame(data, levels=None, knots=None):
    # The reference independently evaluates a SciPy cardinal basis on the resolved
    # training knots. Drop its first column: glum's intercept represents the constant.
    numeric = {}
    categories = {name: data[name].to_numpy() for name in ('region', 'fuel')}
    if knots is None:
        categories = {**{name: np.searchsorted(breaks, data[name].to_numpy(), side='left').astype(str)
                         for name, breaks in BANDS.items()}, **categories}
    else:
        from scipy.interpolate import CubicSpline
        for name, locations in knots.items():
            locations = np.asarray(locations)
            x = data[name].to_numpy()
            identity = np.eye(len(locations))
            curve = CubicSpline(locations, identity, bc_type='natural')
            basis = curve(np.clip(x, locations[0], locations[-1]))
            for mask, knot, row in [(x < locations[0], locations[0], 0),
                                    (x > locations[-1], locations[-1], -1)]:
                basis[mask] = identity[row] + (x[mask]-knot)[:, None]*curve(knot, 1)
            for j in range(1, len(locations)):
                numeric[f'{name}__spline_{j}'] = basis[:, j]
    if levels is None:
        levels = {name: sorted(set(column)) for name, column in categories.items()}
    return pd.DataFrame({**numeric, **{name: pd.Categorical(column, categories=levels[name])
                                     for name, column in categories.items()}}), levels


def run(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    begin = time.perf_counter()
    smooth = getattr(args, "smooth", False)
    fit_tolerance = getattr(args, "fit_tolerance", 1e-9)
    policies = pl.read_parquet(args.frequency).with_columns(pl.col('IDpol').cast(pl.Int64))
    losses = pl.read_parquet(args.severity).with_columns(pl.col('IDpol').cast(pl.Int64))
    if policies['IDpol'].n_unique() != policies.height:
        raise ValueError('Policy identifiers are not unique')
    if losses.filter(pl.col('ClaimAmount').is_null() | ~pl.col('ClaimAmount').is_finite() |
                     (pl.col('ClaimAmount') <= 0)).height:
        raise ValueError('Paid-claim definition requires finite positive claim amounts')
    orphan = losses.join(policies.select('IDpol'), on='IDpol', how='anti')
    orphan.write_csv(out / 'orphan_severity_records.csv')
    losses = losses.join(policies.select('IDpol'), on='IDpol', how='semi')
    aggregated = losses.group_by('IDpol').agg(pl.len().alias('paid_claims'),
                                             pl.col('ClaimAmount').sum().alias('loss'))
    joined = policies.join(aggregated, on='IDpol', how='left').with_columns(
        pl.col('paid_claims').fill_null(0), pl.col('loss').fill_null(0.))
    data = joined.select(pl.col('IDpol').alias('policy_id'), pl.col('Exposure').alias('exposure'),
        pl.col('ClaimNb').alias('reported_claims'), 'paid_claims', 'loss',
        pl.col('DrivAge').cast(pl.Float64).alias('age'), pl.col('VehAge').cast(pl.Float64).alias('vehicle_age'),
        pl.col('BonusMalus').cast(pl.Float64).alias('bonus'), pl.col('Region').cast(pl.String).alias('region'),
        pl.col('VehGas').cast(pl.String).alias('fuel'))
    audit = {'policies': data.height, 'orphan_severity_rows_excluded': orphan.height,
             'orphan_severity_loss_excluded': orphan['ClaimAmount'].sum(), 'paid_claim_records': losses.height,
             'reported_claims': int(data['reported_claims'].sum()),
             'reported_paid_count_disagreement_rows': data.filter(pl.col('reported_claims') != pl.col('paid_claims')).height,
             'loss_total': data['loss'].sum(), 'largest_claim': losses['ClaimAmount'].max(),
             'exposure_over_one_rows': data.filter(pl.col('exposure') > 1).height,
             'response_definition': 'observed positive-payment records per exposure; reported claims retained separately',
             'adjustments': 'none; no development, trend, loss cap or exposure cap'}
    write(out / 'source_audit.json', audit)
    valid = pl.all_horizontal([pl.col(c).is_not_null() & pl.col(c).is_finite()
                               for c in ['exposure', 'paid_claims', 'loss', *BANDS]]) & (pl.col('exposure') > 0)
    data.filter(~valid).write_csv(out / 'excluded_rows.csv')
    data = data.filter(valid).with_columns(
        (pl.col('paid_claims') / pl.col('exposure')).alias('avenue_frequency'),
        (pl.col('loss') / pl.col('exposure')).alias('avenue_pure_premium'))
    train_rows, holdout_rows = next(GroupShuffleSplit(n_splits=1, test_size=.25,
        random_state=20260905).split(data, groups=data['policy_id'].to_numpy()))
    train, holdout = data[train_rows], data[holdout_rows]
    data.select('policy_id').with_columns(pl.Series('holdout', np.isin(np.arange(data.height), holdout_rows))).write_csv(out / 'split.csv')
    severity = data.filter((pl.col('paid_claims') > 0) & (pl.col('loss') > 0)).with_columns(
        (pl.col('loss') / pl.col('paid_claims')).alias('avenue_severity'))
    severity_train = severity.join(train.select('policy_id'), on='policy_id', how='semi')
    severity_holdout = severity.join(holdout.select('policy_id'), on='policy_id', how='semi')
    preparation_seconds = time.perf_counter() - begin
    models, references, records = {}, {}, []
    specs = [('frequency', Plan.frequency('exposure'), 'avenue_frequency', 'exposure', train, holdout, 'poisson'),
             ('severity', Plan.severity('paid_claims'), 'avenue_severity', 'paid_claims', severity_train, severity_holdout, 'gamma'),
             ('premium', Plan.pure_premium('exposure'), 'avenue_pure_premium', 'exposure', train, holdout, TweedieDistribution(1.5))]
    for name, plan, target, weight, training, validation, family in specs:
        started = time.perf_counter()
        model = shape(plan, smooth=smooth).fit(training, target, GLMOptions(max_iterations=1000, tolerance=fit_tolerance))
        fit_seconds = time.perf_counter() - started
        (out / f'{name}_report.md').write_text(model.report(validation).markdown)
        if model.converged is not True:
            raise RuntimeError(f'{name} did not converge; see retained report')
        if not models:
            first_model_seconds = time.perf_counter() - begin
        model.to_workbook(scale='factor').save_json(str(out / f'{name}_scoring.json'))
        joint = term_tests(model)
        write(out / f'{name}_term_tests.json', {'table': joint.table.to_dicts(), 'metadata': joint.metadata})
        started = time.perf_counter()
        knots = resolved_knots(model)
        x, levels = native_frame(training, knots=knots)
        xt, _ = native_frame(validation, levels, knots=knots)
        reference_prepare_seconds = time.perf_counter() - started
        reference = GeneralizedLinearRegressor(family=family, alpha=0, drop_first=True,
                                               gradient_tol=1e-9, max_iter=1000)
        started = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            reference.fit(x, training[target].to_numpy(), sample_weight=training[weight].to_numpy())
        reference_fit_seconds = time.perf_counter() - started
        if any('converg' in str(w.message).lower() for w in caught):
            raise RuntimeError(f'Independent reference convergence warning: {caught}')
        mu_reference = reference.predict(xt)
        started = time.perf_counter()
        mu = model.predict(validation.select(PREDICTORS)).to_series().to_numpy()
        score_seconds = time.perf_counter() - started
        errors = np.abs(mu-mu_reference)
        failed = errors > 1e-6 + 2e-6*np.abs(mu_reference)
        write(out / f'{name}_parity.json', {'atol': 1e-6, 'rtol': 2e-6,
            'failed_rows': int(failed.sum()), 'holdout_rows': len(mu),
            'max_relative_difference': float(np.max(errors/np.abs(mu_reference))),
            'fit_tolerance': fit_tolerance, 'reference_gradient_tolerance': 1e-9})
        if failed.any():
            validation.select('policy_id', *PREDICTORS).with_columns(
                pl.Series('avenue', mu), pl.Series('reference', mu_reference)
            ).filter(pl.Series(failed)).write_csv(out / f'{name}_parity_failures.csv')
        np.testing.assert_allclose(mu, mu_reference, atol=1e-6, rtol=2e-6)
        intervals = coefficient_intervals(model)
        for term, table in intervals.tables.items():
            table.write_csv(out / f'{name}_{term}_intervals.csv')
        (out / f'{name}_plan.json').write_text(model.plan.to_json())
        loaded = Workbook.load_json(str(out / f'{name}_scoring.json')).to_model()
        np.testing.assert_allclose(loaded.predict(validation.select(PREDICTORS)).to_series(), mu, atol=1e-12, rtol=1e-12)
        model.explain(validation.select(PREDICTORS).head(20))['contributions'].write_csv(out / f'{name}_explanations.csv')
        models[name], references[name] = model, reference
        records.append({'model': name, 'train_rows': training.height, 'holdout_rows': validation.height,
                        'fit_seconds': fit_seconds, 'raw_score_seconds': score_seconds,
                        'reference_native_category_prepare_seconds': reference_prepare_seconds,
                        'reference_fit_seconds': reference_fit_seconds, 'reference_iterations': int(reference.n_iter_),
                        'max_relative_difference': float(np.max(np.abs(mu / mu_reference - 1))),
                        'parameters': model.report().fit_summary['n_parameters'], 'converged': model.converged,
                        'resolved_knots': knots, 'iterations': model.report().fit_summary['iterations'],
                        'covariance_method': model.inference_summary['covariance_method']})
        write(out / 'models.json', records)
    premium_knots = resolved_knots(models['premium'])
    xholdout, _ = native_frame(holdout, native_frame(train, knots=premium_knots)[1], knots=premium_knots)
    predictions = {
        'frequency_severity': models['frequency'].predict(holdout).to_series().to_numpy() * models['severity'].predict(holdout).to_series().to_numpy(),
        'tweedie': models['premium'].predict(holdout).to_series().to_numpy(),
        'glum_tweedie': references['premium'].predict(xholdout),
    }
    if args.booster:
        from booster_challenger import run_challenger
        challenger = run_challenger(train, holdout, models['frequency'], out / 'booster')
        predictions['booster_frequency_glm_severity'] = challenger.predict(holdout).to_series().to_numpy() * models['severity'].predict(holdout).to_series().to_numpy()
    y, w = holdout['avenue_pure_premium'].to_numpy(), holdout['exposure'].to_numpy()
    comparison = pl.DataFrame([{'model': name, 'ae': float(np.sum(w*y)/np.sum(w*mu)),
        'mean_tweedie_deviance': mean_tweedie_deviance(y, mu, power=1.5, sample_weight=w)}
        for name, mu in predictions.items()])
    comparison.write_csv(out / 'comparison.csv')
    edited_path = out / 'edited_premium'
    models['premium'].to_workbook().save_csv_dir(str(edited_path))
    intercept = next(edited_path.glob('*intercept.csv'))
    pl.read_csv(intercept).with_columns((pl.col('Relativity') * 1.05).alias('Relativity')).write_csv(intercept)
    edited = Workbook.load_csv_dir(str(edited_path)).to_model()
    holdout.select('policy_id').with_columns(
        models['premium'].predict(holdout).to_series().alias('before'),
        edited.predict(holdout).to_series().alias('after')).write_csv(out / 'policy_changes.csv')
    (out / 'edited_report.md').write_text(edited.report(holdout).markdown)
    np.testing.assert_allclose(edited.predict(holdout).to_series(), models['premium'].predict(holdout).to_series() * 1.05, rtol=1e-12)
    write(out / 'run.json', {'status': 'passed', 'numeric_effects': 'natural_cubic_quantile_5' if smooth else 'prespecified_bands',
          'fit_tolerance': fit_tolerance,
          'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'preparation_seconds': preparation_seconds,
          'time_to_first_valid_model_seconds': first_model_seconds,
          'total_seconds': time.perf_counter() - begin,
          'whole_process_peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
          'versions': {p: importlib.metadata.version(p) for p in ('avenue-model', 'polars', 'glum', 'pandas', 'scikit-learn')},
          'input_sha256': {str(p): hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in (args.frequency, args.severity)},
          'limitations': ['single-run workflow timings, not a controlled speed benchmark',
                         'whole-process RSS is not incremental engine memory',
                         'no temporal variable; grouped random policy holdout',
                         'observed paid-loss cost, not developed prospective ultimate loss cost',
                         'normal intervals conditional on fixed prespecified structure']})
    print(comparison)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frequency', required=True)
    parser.add_argument('--severity', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--fit-tolerance', type=float, default=1e-9, help='Avenue relative-score tolerance; reference and parity tolerances stay fixed')
    parser.add_argument('--smooth', action='store_true', help='Use exact natural cubics with five training-quantile knots for numeric effects')
    parser.add_argument('--booster', action='store_true', help='Tune and verify the installed stock/fork challenger on the same holdout')
    run(parser.parse_args())
