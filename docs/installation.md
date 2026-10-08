# Installation and development

## Install a release wheel

Download the wheel for your operating system and processor from
[GitHub Releases](https://github.com/lukemuz/Avenue_Model/releases).
Use Python 3.12 or 3.13:

```sh
python -m pip install ./WHEEL_FILENAME.whl
```

Choose `manylinux` for Linux, `win_amd64` for Windows x64, or `macosx` for macOS.
For Linux and macOS, choose `x86_64` for Intel/AMD or `aarch64`/`arm64` for ARM/Apple
Silicon. The `cp312-abi3` wheels work on both tested Python versions. Wheels include
the Rust engine; no compiler is required. You can pass a wheel's GitHub release
download URL directly to `pip install`.

For LightGBM conversion and tuning:

```sh
python -m pip install "./WHEEL_FILENAME.whl[tuning]"
python -m pip install avenue-lightgbm  # optional fork with interaction penalties
```

## Install from source

From a source checkout, with Python 3.12 or newer and a Rust toolchain:

```sh
python -m pip install .
python -m pip install '.[tuning]'  # optional LightGBM conversion and tuning
```

## Publish a GitHub release

The Release workflow builds wheels for Linux x86_64/ARM64, macOS Intel/Apple Silicon
and Windows x64. It tests the installed wheels on Python 3.12 and 3.13 before
attaching them and a source archive to a GitHub release. No PyPI account or publishing
credentials are required.

Run **Actions → Release → Run workflow** on `main` for a build-and-test rehearsal.
To publish, ensure `pyproject.toml` and `Cargo.toml` have the same new version, then
push a matching `v` tag. Tagged runs publish the GitHub release after all wheel tests
pass.

## Local checks

```sh
cargo test --no-default-features
maturin develop --release
python -m unittest discover -s tests
```

Python code lives in `python/avenue_model/`; Rust fitting and scoring live in `src/`.
The engine uses Polars, PyO3 and Rayon. See the [evaluation guide](../studies/README.md)
for installed-wheel checks and the [API reference guide](API_REFERENCE.md) to build
searchable documentation. Rust API documentation is available with `cargo doc --open`.
