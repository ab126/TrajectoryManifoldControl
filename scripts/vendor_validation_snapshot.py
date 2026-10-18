"""Rebuild the minimal validation snapshot from the pinned upstream Git store.

No experiments run. Sources are copied byte-for-byte; reference-case JSON values
are packed losslessly. The destination must be new or empty.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

PIN = 'e66a2b79f27f347286576f678f424625f60787b4'
URL = 'https://github.com/aroyphillips/TrajectoryManifoldControl.git'
FILES = (
    'compat.py', 'experiments/bootstrap.py', 'experiments/campaign.py',
    'experiments/distribution_ablation.py', 'experiments/figures.py',
    'experiments/e3_robustness.py', 'results_v1/initial_conditions.json',
    'results_literature/inventory/fresh_200_states.json',
    'results_v1/COLLABORATOR_BRIEF.md', 'results_v1/RESULTS.md',
    'results_v1/DOCUMENTATION_REVIEW.md', 'results_v1/STABILITY_GUARANTEE_HANDOFF.md',
    'handoff/sources/trajectory_manifold_implementation.tex',
    *(f'results_v1/models/episodes3500_seed{s}/diagnostics.json' for s in range(3)),
)
SUT = ('scripts/run_inverted_pendulum_mc.py', 'src/manifold_control.py',
       'src/inverted_pendulum.py', 'src/utils.py', 'src/__init__.py', 'scripts/__init__.py')
CASE = re.compile(r'validation/results_v1/(?:E0/cases/seed[012]_[HL]_A_\d{4}_14s|E1/cases/seed[012]_[HL]_C_\d{4}_14s)\.json$')


def extract(git_dir, output):
    if output.exists() and any(output.iterdir()):
        raise ValueError('destination must be new or empty')
    command = ['git', '--git-dir', str(git_dir.resolve())]
    def read(path):
        return subprocess.check_output([*command, 'show', f'{PIN}:{path}'])
    def digest(data):
        return hashlib.sha256(data).hexdigest()
    tracked = subprocess.check_output([*command, 'ls-tree', '-r', '--name-only', PIN], text=True).splitlines()
    output.mkdir(parents=True, exist_ok=True)
    manifest = dict(repository=URL, commit=PIN, layout='root-validation-v1', files={}, sut={},
        license='No project-level LICENSE/COPYING found; no redistribution license inferred.')
    for relative in FILES:
        data = read('validation/'+relative)
        target = output/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        manifest['files'][relative] = dict(upstream_path='validation/'+relative, sha256=digest(data))
    for path in SUT:
        manifest['sut'][path] = digest(read(path))
    records = {}; sources = {}
    for path in sorted(filter(CASE.fullmatch, tracked)):
        data = read(path)
        key = str(Path(path).relative_to('validation/results_v1'))
        records[key] = json.loads(data)
        sources[key] = dict(upstream_path=path, sha256=digest(data))
    if len(records) != 1236:
        raise ValueError(f'expected 1236 H/L cases, got {len(records)}')
    data = (json.dumps(records, indent=2, allow_nan=False)+'\n').encode()
    (output/'reference_cases.json').write_bytes(data)
    manifest['files']['reference_cases.json'] = dict(sha256=digest(data), sources=sources)
    data = (json.dumps(manifest, indent=2, allow_nan=False)+'\n').encode()
    (output/'SOURCE.json').write_bytes(data)
    print(json.dumps(dict(files=len(manifest['files'])+1, cases=len(records), manifest_sha256=digest(data))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--git-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    extract(args.git_dir, args.output)
