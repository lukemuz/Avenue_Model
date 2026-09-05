"""Profile one real spline fit in a fresh process, bound to an acceptance split."""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import resource
import subprocess
import time

import numpy as np
import polars as pl
from avenue_model import Fold, GLMOptions, Plan, prepare_pricing
from real_motor_acceptance import PREDICTORS, shape


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frequency', required=True)
    parser.add_argument('--severity', required=True)
    parser.add_argument('--split', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--iterations', type=int, default=1000)
    parser.add_argument('--no-inference', action='store_true')
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    policies = pl.read_parquet(args.frequency).with_columns(pl.col('IDpol').cast(pl.Int64))
    losses = pl.read_parquet(args.severity).with_columns(pl.col('IDpol').cast(pl.Int64))
    aggregate = losses.group_by('IDpol').agg(pl.len().alias('paid_claims'), pl.col('ClaimAmount').sum().alias('loss'))
    joined = policies.join(aggregate, on='IDpol', how='left').with_columns(pl.col('paid_claims').fill_null(0), pl.col('loss').fill_null(0.))
    data = joined.select(pl.col('IDpol').alias('policy_id'), pl.col('Exposure').alias('exposure'),
        pl.col('ClaimNb').alias('reported_claims'), 'paid_claims', 'loss',
        pl.col('DrivAge').cast(pl.Float64).alias('age'), pl.col('VehAge').cast(pl.Float64).alias('vehicle_age'),
        pl.col('BonusMalus').cast(pl.Float64).alias('bonus'), pl.col('Region').cast(pl.String).alias('region'),
        pl.col('VehGas').cast(pl.String).alias('fuel'))
    prepared = prepare_pricing(data, exposure='exposure', claims='paid_claims', loss='loss',
                               predictors=PREDICTORS, invalid='exclude', large_loss=200000.)
    # Fold.frames checks the full prepared-data fingerprint, not just row counts.
    fold = Fold.from_json(Path(args.split).read_text())
    train, holdout = fold.frames(prepared.frequency)
    setup_seconds = time.perf_counter()-start
    options = GLMOptions(max_iterations=args.iterations, tolerance=1e-11,
                         compute_standard_errors=not args.no_inference)
    start = time.perf_counter()
    model = shape(Plan.pure_premium('exposure'), smooth=True).fit(train, 'avenue_pure_premium', options)
    fit_seconds = time.perf_counter()-start
    start = time.perf_counter()
    prediction = model.predict(holdout.select(PREDICTORS)).to_numpy().reshape(-1)
    score_seconds = time.perf_counter()-start
    np.save(out/'prediction.npy', prediction)
    model.to_workbook(scale='factor').save_json(str(out/'model.json'))
    summary = model.report().fit_summary
    extension = Path(importlib.import_module('avenue_model.avenue_model').__file__)
    result = {'source_commit': subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
        'extension_sha256': hashlib.sha256(extension.read_bytes()).hexdigest(),
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'split_id': fold.split_id, 'train_rows': train.height, 'holdout_rows': holdout.height,
        'requested_iterations': args.iterations, 'inference': not args.no_inference,
        'fit_seconds': fit_seconds, 'score_seconds': score_seconds, 'setup_seconds': setup_seconds,
        'peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'converged': model.converged, 'iterations': summary['iterations'],
        'prediction_sha256': hashlib.sha256(prediction.tobytes()).hexdigest(),
        'fit_summary': summary}
    (out/'result.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
