"""Preserve a frozen run across the validation-only layout migration.

Only integration hashes may change. Authenticate the old integration from Git,
require identical controllers/models/scoring/protocol, and copy completed cases
to a new output without modifying either historical bundles or results.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

from validation_benchmark import ROOT, SOURCE_SHA256, sha256, verify_snapshot, write_json


def check_equivalence(old, new):
    integration = {'adapter_sha256', 'evaluator_sha256', 'validation_manifest_sha256'}
    if {k:v for k,v in old.items() if k not in integration} != {k:v for k,v in new.items() if k not in integration}:
        raise ValueError('scientific configuration changed; this is not a layout-only migration')


def migrate(args):
    verify_snapshot()
    old=json.loads(args.old_bundle.read_text());new=json.loads(args.new_bundle.read_text())
    check_equivalence(old,new)
    commit=subprocess.check_output(['git','rev-parse',args.source_commit],cwd=ROOT,text=True).strip()
    for path,key in [('scripts/validation_benchmark.py','adapter_sha256'),
                     ('scripts/frozen_manifold_validation.py','evaluator_sha256')]:
        data=subprocess.check_output(['git','show',f'{commit}:{path}'],cwd=ROOT)
        if hashlib.sha256(data).hexdigest()!=old[key] or sha256(ROOT/path)!=new[key]:
            raise ValueError(f'integration source cannot be authenticated: {path}')
    before=json.loads((args.old_output/'manifest.json').read_text())
    after=json.loads((args.new_output/'manifest.json').read_text())
    if before['bundle_sha256']!=sha256(args.old_bundle) or after['bundle_sha256']!=sha256(args.new_bundle):
        raise ValueError('output does not belong to the supplied bundle')
    if {k:v for k,v in before.items() if k!='bundle_sha256'} != {k:v for k,v in after.items() if k!='bundle_sha256'}:
        raise ValueError('evaluation protocol changed')
    if any((args.new_output/name).exists() for name in ('cases','raw','summary.json','LAYOUT_MIGRATION.json')):
        raise ValueError('destination must contain only the new verify-only manifest')
    paths=[]
    for case in sorted((args.old_output/'cases').glob('*.json')):
        row=json.loads(case.read_text());index=row['ic_index']
        if row['seed']!=before['seed'] or row['ic_set']!=before['ic_set'] or row['arm'] not in before['arms'] or index not in before['indices'] or row['initial_state']!=before['initial_states'][index]:
            raise ValueError(f'case identity mismatch: {case}')
        raw=args.old_output/'raw'/f'{case.stem}.npz'
        if not raw.is_file():raise ValueError(f'missing raw trace: {raw}')
        paths.extend((case,raw))
    if (args.old_output/'summary.json').exists():paths.append(args.old_output/'summary.json')
    copied={}
    for source in paths:
        relative=source.relative_to(args.old_output);target=args.new_output/relative
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
        if sha256(source)!=sha256(target):raise ValueError(f'copy mismatch: {source}')
        copied[str(relative)]=sha256(source)
    write_json(args.new_output/'LAYOUT_MIGRATION.json',dict(
        old_bundle_sha256=sha256(args.old_bundle),new_bundle_sha256=sha256(args.new_bundle),
        original_integration_commit=commit,validation_manifest_sha256=SOURCE_SHA256,
        old_manifest_sha256=sha256(args.old_output/'manifest.json'),
        new_manifest_sha256=sha256(args.new_output/'manifest.json'),copied_sha256=copied,
        equivalence='Identical scientific bundle/protocol; only verified integration layout changed. No simulation or tuning.'))
    print(f'Preserved {len(paths)//2} completed controller cases; no historical files changed.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for flag in ('old-bundle','new-bundle','old-output','new-output'):
        p.add_argument('--'+flag,type=Path,required=True)
    p.add_argument('--source-commit',default='f41056d')
    migrate(p.parse_args())
