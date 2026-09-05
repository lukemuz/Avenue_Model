"""Replay the full workflow acceptance from an installed wheel, with provenance.

Run from a fresh environment containing the wheel's test/tuning extras and glum.
Input parquet paths are the public frequency/severity sources documented in
REAL_MOTOR_ACCEPTANCE.md. This script never downloads, installs or publishes anything.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time
import zipfile


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
    distribution = metadata.distribution('avenue_model')
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
    parser.add_argument('--frequency', type=Path, required=True)
    parser.add_argument('--severity', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error('--threads must be positive')
    wheel, frequency, severity = [p.resolve(strict=True) for p in (args.wheel, args.frequency, args.severity)]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    record = {'schema_version': 1, 'status': 'running', 'started_at': datetime.now(timezone.utc).isoformat(),
              'source_commit': git('rev-parse', 'HEAD'), 'working_tree_status': git('status', '--short'),
              'python': sys.version, 'platform': platform.platform(), 'libc': platform.libc_ver(),
              'threads': args.threads, 'steps': [],
              'limitations': ['single-process workflow timings, not controlled speed comparisons',
                              'one Python/platform/dependency combination per run',
                              'successful mechanics do not establish adequate predictive calibration',
                              'this runner does not cover all improvement-plan requirements']}

    def save():
        (output / 'acceptance.json').write_text(json.dumps(record, indent=2, allow_nan=False) + '\n')

    save()
    try:
        record['wheel'] = installed_wheel_evidence(wheel)
        record['versions'] = {name: metadata.version(name) for name in
                              ['avenue_model', 'polars', 'pyarrow', 'numpy', 'pandas', 'scipy',
                               'scikit-learn', 'glum', 'lightgbm', 'optuna']}
        record['inputs_sha256'] = {str(path): sha256(path) for path in (frequency, severity)}
        names = git('ls-files', '-z', 'src', 'python', 'tests', 'examples', 'studies',
                    'Cargo.toml', 'Cargo.lock', 'pyproject.toml').split('\0')
        names.append(str(Path(__file__).resolve().relative_to(ROOT)))
        record['source_files_sha256'] = {name: sha256(ROOT / name) for name in sorted(set(names))
                                        if name and (Path(name).suffix in ('.rs', '.py', '.toml', '.lock'))}
        env = os.environ.copy()
        for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                    'POLARS_MAX_THREADS', 'RAYON_NUM_THREADS']:
            env[key] = str(args.threads)
        steps = [
            ('python_tests', ['-m', 'unittest', 'discover', '-s', 'tests', '-v']),
            ('auto', ['examples/auto_pricing_study.py', '--output', str(output / 'auto')]),
            ('homeowners', ['examples/homeowners_perils.py', '--output', str(output / 'homeowners')]),
            ('booster', ['examples/booster_pricing_study.py', '--output', str(output / 'booster')]),
            ('real_motor', ['studies/real_motor_acceptance.py', '--frequency', str(frequency),
                            '--severity', str(severity), '--booster', '--output', str(output / 'real_motor')]),
            ('real_spline', ['studies/real_motor_acceptance.py', '--frequency', str(frequency),
                             '--severity', str(severity), '--smooth', '--fit-tolerance', '1e-11',
                             '--output', str(output / 'real_spline')]),
        ]
        for name, arguments in steps:
            command = [sys.executable, *arguments]
            log = output / f'{name}.log'
            step = {'name': name, 'command': command, 'status': 'running', 'log': log.name}
            record['steps'].append(step)
            save()
            print(f'Running {name}', flush=True)
            started = time.perf_counter()
            with log.open('w') as stream:
                completed = subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
            step.update(returncode=completed.returncode, wall_seconds=time.perf_counter()-started,
                        log_sha256=sha256(log), status='passed' if completed.returncode == 0 else 'failed')
            if name == 'python_tests':
                text = log.read_text()
                count = re.search(r'Ran (\d+) tests?', text)
                step['tests_run'] = int(count[1]) if count else 0
                step['skipped'] = bool(re.search(r'\bskipped[= ]', text))
                if not step['tests_run'] or step['skipped']:
                    step['status'] = 'failed'
            save()
            if step['status'] != 'passed':
                raise RuntimeError(f'{name} did not pass completely; inspect {log}')
            print(f'Passed {name} ({step["wall_seconds"]:.2f}s)', flush=True)
        record['artifact_files_sha256'] = {
            str(path.relative_to(output)): sha256(path) for path in sorted(output.rglob('*'))
            if path.is_file() and path.name != 'acceptance.json'}
        record['status'] = 'passed'
    except Exception as error:
        record['status'] = 'failed'
        record['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        record['finished_at'] = datetime.now(timezone.utc).isoformat()
        save()


if __name__ == '__main__':
    main()
