# Building and hosting the Python API reference

The API reference is generated from an installed wheel with [pdoc](https://pdoc.dev/docs/pdoc.html).
It includes searchable Python/native API entries, runtime signatures, docstrings and
links to the workflow guides pinned to the build's source revision.

```sh
maturin build --release --out dist
python -m pip install 'dist/ACTUAL_WHEEL_FILENAME.whl[docs]'
python docs/build_api.py --wheel dist/ACTUAL_WHEEL_FILENAME.whl --output api-html
python -m http.server --directory api-html 8000
```

Substitute the generated wheel's actual filename. Open `http://localhost:8000` for
local review. Use a fresh output directory. The documentation extra installs pdoc;
tuning and test dependencies are not required to generate the reference.

The builder verifies the installed package against the supplied wheel bytes and rejects
editable/source imports. It checks that every public top-level export (apart from
submodules), public class member and annotated public field has an HTML anchor. It
also requires the search index. `build.json` records these checks, the wheel hash,
package origin, generator hash, pdoc version and source revision.

These checks prevent silently omitting an exported native class or method. They do not
prove that every docstring fully explains its parameters or statistical assumptions.
The linked workflow guides retain those meanings and limitations; semantic docstring
coverage remains an ongoing documentation task.

## CI and Pages

`.github/workflows/docs.yml` builds the wheel and reference on pull requests and main
branch updates, and uploads the generated HTML as the `api-reference` artifact. Only
a main-branch build can deploy the Pages artifact. PR builds have read-only repository
permissions; Pages write permission and the identity token are limited to the separate
deployment job. No repository write credential is persisted by checkout.

The repository's Pages source must be set to **GitHub Actions**, following
[pdoc's hosting instructions](https://pdoc.dev/docs/pdoc.html#how-can-i-host-the-documentation-i-generated).
The deployment job reports the resulting URL when it succeeds. Configuration alone is
not evidence of a working hosted site. The local build has been verified from a fresh
CPython 3.12 Linux wheel installation and visually inspected in Chromium; remote CI,
Pages settings and the deployed URL remain unverified.
