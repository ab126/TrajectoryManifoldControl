"""Export compact, versionable evidence without moving historical raw arrays."""
import argparse
import json
from pathlib import Path

from validation_benchmark import ROOT, PIN, sha256, write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--artifacts',type=Path,default=ROOT/'saves/validation')
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();evidence=dict(upstream=PIN,artifacts={},sources={})
    frozen=args.artifacts/'frozen-native.json'
    if frozen.exists():evidence['frozen_bundle']=dict(sha256=sha256(frozen),data=json.loads(frozen.read_text()))
    for filename in ('src/manifold_experiments.py','src/validation_metrics.py',
                     'scripts/validation_benchmark.py','scripts/run_manifold_ablation.py',
                     'scripts/train_multistep_ablation.py','scripts/frozen_manifold_validation.py',
                     'scripts/diagnose_validation_decoder.py','scripts/compare_manifold_solvers.py',
                     'scripts/summarize_validation.py','scripts/export_validation_evidence.py'):
        evidence['sources'][filename]=sha256(ROOT/filename)
    for directory in ('adapter-checks','L-full-summary','reconstructed-seed0-summary',
                      'reconstructed-seed0-C3-summary','diagnostics-seed0','static-solvers-v1',
                      'multistep-screen-v1','model-one-step-control','model-multistep-control',
                      'tuning-screen-v1','final-seed0-C','final-seed0-delay',
                      'final-seed0-mass','final-seed0-A60'):
        path=args.artifacts/directory
        if not path.exists():continue
        entry={}
        for name in ('adapter_checks.json','manifest.json','summary.json','diagnostics.json',
                     'comparison.json','training_summary.json'):
            if not (path/name).exists():continue
            data=json.loads((path/name).read_text())
            if name=='summary.json' and 'archived_comparisons' in data:
                rows=data.pop('archived_comparisons')
                data['archived_comparison_digest']=dict(n=len(rows),
                    legacy_mismatches=sum(not row['legacy_agrees'] for row in rows),
                    recovery_mismatches=sum(not row['recovery_agrees'] for row in rows),
                    max_scalar_differences={key:max((abs(row[key]) for row in rows if row[key] is not None),default=None)
                        for key in ('final_theta_difference','final_p_difference','max_abs_p_difference')})
            entry[name]=dict(sha256=sha256(path/name),data=data)
        cases=[]
        for case in sorted((path/'cases').glob('*.json')):
            row=json.loads(case.read_text())
            raw=path/'raw'/f'{case.stem}.npz'
            values={key:row.get(key) for key in ('arm','ic_index','legacy_uninterrupted','terminal_recovery',
                'periodic_terminal_recovery','terminal_max_angle_unwrapped','terminal_max_angular_speed_physical',
                'angle_settle_seconds','final_cart_position','control_energy_normalized','saturation_seconds',
                'max_abs_cart_position','fallback_events','solver_latency','controller_latency')}
            values['net_cart_displacement']=(row['final_cart_position']-row['initial_state'][0]
                if row.get('final_cart_position') is not None else None)
            # Original arms expose no custom solver counters. Derive the count
            # from saved original CEM solve records rather than treating the
            # ablation runner's zero uninstrumented counter as zero computation.
            if raw.exists() and directory.startswith(('tuning-','model-')):
                import numpy as np
                with np.load(raw) as arrays:
                    values['cem_solves']=len(arrays['loss']) if row['arm'] in ('M','M1','H','cem5','cem1') else None
                if values['cem_solves'] is not None:
                    values['forward_candidate_evaluations']=values['cem_solves']*768
                else:values['forward_candidate_evaluations']=row.get('solver_evaluations')
            cases.append(values)
        if cases:entry['cases']=cases
        evidence['artifacts'][directory]=entry
    write_json(args.output,evidence)
    print(args.output)


if __name__=='__main__':main()
