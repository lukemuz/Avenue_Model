"""Build searchable API HTML from a verified installed wheel, with export coverage."""
import argparse
from html.parser import HTMLParser
import importlib.metadata
import inspect
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'studies'))
from wheel_acceptance import git, installed_wheel_evidence, sha256


class Anchors(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = set()

    def handle_starttag(self, tag, attrs):
        for name, value in attrs:
            if name == 'id':
                self.ids.add(value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wheel', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    evidence = installed_wheel_evidence(args.wheel.resolve(strict=True))
    args.output.mkdir(parents=True, exist_ok=False)
    import avenue_model
    import pdoc
    commit = git('rev-parse', 'HEAD')
    source = f'https://github.com/lukemuz/Avenue_Model/blob/{commit}/docs'
    guides = [('User guide', 'README.md'),
              ('Scoring and exposure', 'SCORING_CONTRACT.md'),
              ('Continuous splines', 'SPLINES.md'),
              ('Inference and intervals', 'COEFFICIENT_INTERVALS.md'),
              ('LightGBM tuning', 'lightgbm.md')]
    avenue_model.__doc__ += '\n\n## Workflow guides\n\n' + '\n'.join(
        f'- [{title}]({source}/{name})' for title, name in guides)
    pdoc.render.configure(docformat='google', show_source=False,
                          footer_text=f'Avenue {importlib.metadata.version("avenue_model")} · {commit[:8]}')
    pdoc.pdoc('avenue_model', output_directory=args.output)
    parsed = Anchors()
    parsed.feed((args.output / 'avenue_model.html').read_text(encoding='utf-8'))
    expected = {name for name in avenue_model.__all__
                if not inspect.ismodule(getattr(avenue_model, name))}
    for name in tuple(expected):
        obj = getattr(avenue_model, name)
        if inspect.isclass(obj):
            members = set(dir(obj)) | set(getattr(obj, '__annotations__', {}))
            expected.update(f'{name}.{member}' for member in members
                            if not member.startswith('_'))
    missing = sorted(expected - parsed.ids)
    if missing:
        raise RuntimeError(f'Public API entries missing from generated HTML: {missing}')
    if not (args.output / 'search.js').is_file():
        raise RuntimeError('Search index was not generated')
    (args.output / '.nojekyll').touch()
    record = {'source_commit': commit, 'wheel': evidence,
              'generator_sha256': sha256(__file__), 'pdoc_version': importlib.metadata.version('pdoc'),
              'public_anchors_verified': sorted(expected),
              'limitations': ['Export presence does not prove complete semantic documentation.',
                              'Generated locally; hosting requires a successful Pages deployment.']}
    (args.output / 'build.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
    print(f'API reference built; {len(expected)} public anchors verified.')


if __name__ == '__main__':
    main()
