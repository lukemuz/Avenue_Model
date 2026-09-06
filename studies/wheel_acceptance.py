"""Verify an installed release wheel, run all Python tests and the four tutorials."""
import argparse
import hashlib
import zipfile
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()


def installed_wheel_evidence(wheel):
    import avenue_model
    distribution = importlib.metadata.distribution('avenue_model')
    origin = Path(avenue_model.__file__).resolve()
    if origin.is_relative_to(ROOT / 'python'):
        raise ValueError('Acceptance requires an installed wheel, not the source package')
    direct_url = json.loads(distribution.read_text('direct_url.json') or '{}')
    if direct_url.get('dir_info', {}).get('editable'):
        raise ValueError('Acceptance refuses an editable installation')
    # Version strings alone cannot distinguish successive local 0.1.0 builds.
    # Verify every installed package payload against the supplied wheel bytes.
    verified = []
    with zipfile.ZipFile(wheel) as archive:
        for name in archive.namelist():
            if not name.startswith('avenue_model/') or name.endswith('/'):
                continue
            installed = Path(distribution.locate_file(name))
            expected = hashlib.sha256(archive.read(name)).hexdigest()
            if not installed.is_file() or sha256(installed) != expected:
                raise ValueError(f'Installed package differs from supplied wheel: {name}')
            verified.append(name)
    if not any(name.endswith('.so') or name.endswith('.pyd') for name in verified):
        raise ValueError('The supplied wheel has no verified native extension')
    return {'path': str(wheel), 'sha256': sha256(wheel), 'import_origin': str(origin),
            'installed_payload_files_verified': len(verified), 'direct_url': direct_url}


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
                             ('booster', 'booster_pricing_study.py'),
                             ('smooth', 'smooth_pricing_study.py')]:
            path = ROOT / 'examples' / script
            command = [sys.executable, str(path), '--output', str(output / name)]
            if name == 'booster':
                command.append('--refit-glm')
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
