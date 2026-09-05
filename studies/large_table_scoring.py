"""Repeat the historical four-table motor scoring workload without refitting it."""
import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import subprocess
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl
from avenue_model import Workbook, from_booster


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--quote-rows', type=int, help='Score this many leading holdout rows for a batch-size study')
    artifact = parser.add_mutually_exclusive_group()
    artifact.add_argument('--reconvert', action='store_true', help='Measure a fresh conversion, including current missing routes, instead of the historical workbook')
    artifact.add_argument('--model', type=Path, help='Score an exact saved JSON artifact against the same original booster and quotes')
    args = parser.parse_args()
    if args.repeats < 3:
        parser.error('at least three repeats are required')
    if args.quote_rows is not None and args.quote_rows < 1:
        parser.error('--quote-rows must be positive')
    args.output.mkdir(parents=True, exist_ok=False)
    source = args.evaluation / 'fork_results/fork_numeric'
    data_path = args.evaluation / 'data/freMTPL2freq.parquet'
    raw = pd.read_parquet(data_path)
    # Exactly the original feature construction, row split and training-only maps.
    # Historical response/exposure clipping does not enter this scoring-only frame.
    names = ['age', 'vehicle_age', 'bonus', 'region', 'fuel']
    frame = pl.DataFrame({
        'age': raw.DrivAge.to_numpy(dtype=float),
        'vehicle_age': raw.VehAge.to_numpy(dtype=float),
        'bonus': raw.BonusMalus.to_numpy(dtype=float),
        'region': raw.Region.astype(str).to_numpy(),
        'fuel': raw.VehGas.astype(str).to_numpy()})
    mask = np.random.default_rng(20260905).random(frame.height) < .75
    train, holdout = frame.filter(mask), frame.filter(~mask)
    categories = {n: sorted(train[n].unique().to_list()) for n in ('region', 'fuel')}
    x = np.column_stack([
        pd.Categorical(holdout[n].to_list(), categories=categories[n]).codes
        if n in categories else holdout[n].to_numpy() for n in names]).astype(float)
    if args.quote_rows is not None:
        if args.quote_rows > len(x):
            parser.error('--quote-rows exceeds the holdout population')
        x = x[:args.quote_rows]
    quotes = pl.DataFrame({n: x[:, i] for i, n in enumerate(names)})
    booster_path = source / 'selected_booster.txt'
    booster = lgb.Booster(model_file=str(booster_path))
    started = time.perf_counter()
    if args.model:
        model = Workbook.load_json(str(args.model)).to_model()
    else:
        model = from_booster(booster).model if args.reconvert else Workbook.load_csv_dir(str(source / 'selected')).to_model()
    artifact_setup_seconds = time.perf_counter() - started
    tables = model.to_workbook().tables
    if not args.reconvert and not args.model:
        assert len(tables) == 4 and sum(t.height for t in tables) == 19181
    reference = booster.predict(x, num_threads=4)
    timings = []
    for _ in range(args.repeats):
        start = time.perf_counter()
        actual = model.predict(quotes).to_numpy().reshape(-1)
        timings.append(time.perf_counter() - start)
        np.testing.assert_allclose(actual, reference, atol=1e-12, rtol=1e-12)
    np.save(args.output / 'prediction.npy', actual)
    extension = Path(importlib.import_module('avenue_model.avenue_model').__file__)
    artifact_paths = [data_path, booster_path]
    if args.model:
        artifact_paths.append(args.model)
    elif not args.reconvert:
        artifact_paths += sorted((source / 'selected').iterdir())
    model.to_workbook(scale='factor').save_json(str(args.output / 'model.json'))
    result = {
        'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'matching_sha256': digest(Path(__file__).resolve().parents[1] / 'src/glm/matching.rs'),
        'script_sha256': digest(Path(__file__)), 'extension_sha256': digest(extension),
        'artifact_source': 'supplied_workbook' if args.model else ('fresh_conversion' if args.reconvert else 'historical_workbook'),
        'artifact_setup_seconds': artifact_setup_seconds,
        'model_json_sha256': digest(args.output / 'model.json'),
        'inputs': {str(p.relative_to(args.evaluation) if p.is_relative_to(args.evaluation) else p): digest(p) for p in artifact_paths if p.is_file()},
        'versions': {p: importlib.metadata.version(p) for p in ['avenue_model', 'lightgbm', 'numpy', 'pandas', 'polars']},
        'threads': {k: os.environ.get(k) for k in ['OMP_NUM_THREADS', 'POLARS_MAX_THREADS', 'RAYON_NUM_THREADS']},
        'train_rows': train.height, 'quote_rows': quotes.height,
        'quote_sha256': hashlib.sha256(x.tobytes()).hexdigest(),
        'tables': len(tables), 'table_rows': sum(t.height for t in tables),
        'largest_table': max(t.height for t in tables),
        'seconds': timings, 'median_seconds': float(np.median(timings)),
        'max_relative_error': float(np.max(np.abs(actual / reference - 1.))),
        'prediction_sha256': hashlib.sha256(actual.tobytes()).hexdigest(),
        'process_peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'interpretation': 'Repeated public scoring including frame preparation and matching; no reusable cache. Peak RSS includes inputs, booster and workbook, not isolated scoring memory.',
        'passed': True}
    (args.output / 'result.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
