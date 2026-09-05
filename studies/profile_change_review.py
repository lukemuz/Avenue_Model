"""Measure change-review time and process high-water RSS in a fresh process.

Run each implementation separately with the same inputs and thread environment.
Outputs include complete Arrow exhibits for an exact before/after comparison.
"""
import argparse
import json
import importlib.util
import sys
from pathlib import Path
import resource
import time

import polars as pl
from avenue_model import Workbook, compare_changes


def run(args):
    compare = compare_changes
    if args.implementation:
        spec = importlib.util.spec_from_file_location('avenue_model._profile_changes', args.implementation)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        compare = module.compare_changes
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    data = pl.read_parquet(args.data).head(args.rows).select(
        pl.col('DrivAge').cast(pl.Float64).alias('age'),
        pl.col('VehAge').cast(pl.Float64).alias('vehicle_age'),
        pl.col('BonusMalus').cast(pl.Float64).alias('bonus'),
        pl.col('Region').cast(pl.String).alias('region'),
        pl.col('VehGas').cast(pl.String).alias('fuel'),
        pl.col('Exposure').alias('exposure'))
    old = Workbook.load_json(args.old).to_model()
    new = Workbook.load_csv_dir(args.new).to_model()
    before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    started = time.perf_counter()
    result = compare(old, new, data, unit='loss/exposure', weight='exposure',
                             segments=['region', 'fuel'])
    elapsed = time.perf_counter() - started
    after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    summary = {'rows': data.height, 'contribution_rows': result.contributions.height,
               'seconds': elapsed, 'peak_rss_before_kib': before, 'peak_rss_after_kib': after,
               'peak_rss_increase_kib': after-before, 'polars_version': pl.__version__,
               'totals': result.totals.to_dicts(), 'metadata': result.metadata,
               'limitations': 'Linux process high-water marks; includes input/artifact allocations, not allocator-isolated memory'}
    for name, frame in {'policies': result.policies, 'contributions': result.contributions,
                        'totals': result.totals, **result.segments}.items():
        frame.write_ipc(out / (name + '.arrow'))
    (out / 'measurement.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--implementation', help='Optional saved changes.py implementation for controlled baseline measurement')
    parser.add_argument('--data', required=True)
    parser.add_argument('--old', required=True)
    parser.add_argument('--new', required=True)
    parser.add_argument('--rows', type=int, default=169504)
    parser.add_argument('--output', required=True)
    run(parser.parse_args())
