"""Verify an installed release wheel, run all Python tests and the three tutorials."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import unittest

from readiness_acceptance import ROOT, git, installed_wheel_evidence, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wheel', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    output = args.output.resolve()
    for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                'POLARS_MAX_THREADS', 'RAYON_NUM_THREADS']:
        os.environ[key] = '4'
    record = {'schema_version': 1, 'status': 'running', 'python': sys.version,
              'platform': platform.platform(), 'machine': platform.machine(),
              'source_commit': git('rev-parse', 'HEAD'),
              'script_sha256': sha256(__file__),
              'test_files_sha256': {str(p.relative_to(ROOT)): sha256(p)
                                    for p in sorted((ROOT / 'tests').rglob('*'))
                                    if p.is_file() and p.suffix in ['.py', '.json', '.csv']},
              'steps': []}
    try:
        record['wheel'] = installed_wheel_evidence(args.wheel.resolve(strict=True))
        record['versions'] = {name: importlib.metadata.version(name) for name in
                              ['avenue_model', 'polars', 'pyarrow', 'numpy', 'pandas',
                               'scipy', 'scikit-learn', 'lightgbm', 'optuna']}
        start = time.perf_counter()
        suite = unittest.defaultTestLoader.discover(str(ROOT / 'tests'))
        with (output / 'tests.log').open('w', encoding='utf-8') as log:
            result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
        passed = result.wasSuccessful() and result.testsRun > 0 and not result.skipped
        record['steps'].append({'name': 'python_tests', 'passed': passed,
                               'tests': result.testsRun, 'skips': len(result.skipped),
                               'failures': len(result.failures), 'errors': len(result.errors),
                               'seconds': time.perf_counter() - start})
        if not passed:
            raise RuntimeError('Wheel tests failed or skipped tests; inspect tests.log')
        for name, script in [('auto', 'auto_pricing_study.py'),
                             ('homeowners', 'homeowners_perils.py'),
                             ('booster', 'booster_pricing_study.py')]:
            path = ROOT / 'examples' / script
            command = [sys.executable, str(path), '--output', str(output / name)]
            start = time.perf_counter()
            with (output / f'{name}.log').open('w', encoding='utf-8') as log:
                completed = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            record['steps'].append({'name': name, 'command': command,
                                   'script_sha256': sha256(path),
                                   'exit_code': completed.returncode,
                                   'seconds': time.perf_counter() - start})
            if completed.returncode:
                raise RuntimeError(f'{name} tutorial failed; inspect {name}.log')
        record['status'] = 'passed'
    except Exception as exc:
        record['status'] = 'failed'
        record['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        (output / 'acceptance.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
