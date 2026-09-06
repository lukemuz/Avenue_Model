# Release-wheel acceptance

The release workflow builds five platform wheels, then tests each artifact on both
CPython 3.12 and 3.13. Publication depends on every wheel-test job succeeding, as well
as the wheel and source-distribution builds.

| Built wheel | Native test runner |
|---|---|
| Linux x86-64 | `ubuntu-latest` |
| Linux aarch64 | `ubuntu-24.04-arm` |
| Windows x64 | `windows-latest` |
| macOS x86-64 | `macos-15-intel` |
| macOS aarch64 | `macos-latest` |

The ARM Linux test uses a native [GitHub-hosted ARM runner](https://docs.github.com/en/actions/reference/runners/github-hosted-runners),
even though the wheel build uses cross-compilation. macOS installs the OpenMP runtime
used by LightGBM; see its [installation guide](https://lightgbm.readthedocs.io/en/stable/Installation-Guide.html).

Each test job downloads only its corresponding wheel, installs the test and tuning extras, and runs `studies/wheel_acceptance.py`. The runner:

1. Rejects source/editable imports and verifies every installed package payload file
   against the downloaded wheel, including the native extension.
2. Runs the complete Python test suite. Failures, errors, zero discovered tests or
   skipped tests fail the gate.
3. Runs the auto, homeowners and stock-LightGBM tutorials, including their scoring,
   validation and export/reload assertions.
4. Saves a JSON record with wheel/source/test fingerprints, dependency versions,
   platform, test counts and tutorial exit codes, plus logs and generated exhibits.

Acceptance artifacts are uploaded even on test failure. Publication downloads only
wheel and sdist artifacts; acceptance logs cannot be mistaken for release packages.
No platform matrix or package publication is claimed successful merely because this
workflow is configured. A failed installation, unavailable runner or failed test blocks
publication through the job dependency.

## Local verification

The [historical local result](../studies/results/wheel_gate/) uses an earlier wheel build
installed in a fresh CPython 3.12 Linux x86-64 environment. All 143 Python tests pass
without skips, all three tutorials complete, and 15 installed payload files match the
wheel. A separate run confirms that the editable development installation is rejected.
The workflow's YAML, embedded Python commands, artifact mappings and publication
dependency were checked locally. Remote jobs, Python 3.13, other platforms, publishing,
sdist installation and dependency-range extremes remain unverified here.

To run against an already installed wheel with its extras:

```sh
python studies/wheel_acceptance.py --wheel path/to/avenue_model.whl --output wheel-acceptance
```

Use the actual wheel filename and a new output directory. This runner does not install
or publish packages. The larger real-data and fork acceptance remains available through
`studies/readiness_acceptance.py`; the release matrix's synthetic tutorials do not
substitute for real predictive-quality evidence.
