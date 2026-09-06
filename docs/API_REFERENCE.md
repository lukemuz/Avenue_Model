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
The linked user guides explain the modeling assumptions and supported combinations.

## CI artifacts

`.github/workflows/docs.yml` builds the wheel and reference on pull requests and main
branch updates, and uploads the generated HTML as the `api-reference` artifact.
The workflow uses read-only repository permissions, and no repository write credential
is persisted by checkout. It does not deploy a public site or require a PyPI release.

Use the local build or download and extract the `api-reference` workflow artifact
to browse the generated HTML. Public hosting can be added later by enabling GitHub
Pages and restoring a deployment job, following
[pdoc's hosting instructions](https://pdoc.dev/docs/pdoc.html#how-can-i-host-the-documentation-i-generated).
