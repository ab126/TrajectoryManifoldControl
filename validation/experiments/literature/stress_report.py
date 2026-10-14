"""Exact stress counts and disturbance/latency diagnostics; no interpolation."""
import collections,json,statistics
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3];OUT=ROOT/'validation/results_literature'
METHODS=['edmd','sindy','pets','rokdeepc','deepkoopman_control','dfine_control']


def read(path):return json.loads(path.read_text())
def write(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)
def median(values):
    values=[x for x in values if x is not None]
    return statistics.median(values) if values else None


def main():
    groups=collections.defaultdict(list);plans={};references={}
    for method in METHODS:
        root=OUT/method/'stress'
        if (root/'plan.json').exists():plans[method]=read(root/'plan.json')
        for path in (root/'cases').glob('*.json'):
            r=read(path);assert r['seconds']==14.
            groups[(method,r['arm'],r['seed'],r['condition'])].append(r)
    rows=[]
    for (method,arm,seed,condition),cases in sorted(groups.items()):
        identities={(r['set'],r['index']) for r in cases};assert len(identities)==len(cases)
        cohort='E4' if method=='rokdeepc' else 'original';paired={}
        if cohort=='original':
            for reference in ['H','L']:
                matched=[]
                for r in cases:
                    name='C50' if r['set']=='C' else r['set'];key=f'seed{seed}_{reference}_{condition}_{name}_{r["index"]:04d}_14s'
                    if key not in references:
                        p=ROOT/'validation/results_v1/E3/cases'/f'{key}.json'
                        references[key]=read(p) if p.exists() else None
                    ref=references[key]
                    if ref is None:continue
                    assert ref['initial_state']==r['identity']['case']['state'],(key,'initial state mismatch')
                    assert ref['seed']==seed and ref['seconds']==14.
                    matched.append((r,ref))
                if matched:paired[reference]=dict(n=len(matched),reference_successes=sum(b['primary_success'] for a,b in matched),method_only=sum(a['primary_success'] and not b['primary_success'] for a,b in matched),reference_only=sum(b['primary_success'] and not a['primary_success'] for a,b in matched))
        scales={tuple(r['noise_std_physical']) for r in cases if 'noise_std_physical' in r};assert len(scales)<=1
        rows.append(dict(method=method,cohort=cohort,arm=arm,seed=seed,condition=condition,successes=sum(r['primary_success'] for r in cases),tested=len(cases),requested=56,complete=len(cases)==56,falls=sum(not r['no_fall'] for r in cases),nonfinite=sum(r.get('nonfinite_state',False) for r in cases),solver_rejections=sum(r['solver_failures'] for r in cases),paired=paired,noise_std_physical=list(next(iter(scales))) if scales else None,median_tail_angle_rms=median([r.get('tail_angle_rms') for r in cases]),median_tail_angular_velocity_rms_physical=median([r.get('tail_angular_velocity_rms_physical') for r in cases]),median_case_median_solver_seconds=median([r['solve_latency_quantiles'][0] for r in cases if r.get('solve_latency_quantiles')]),finite_tail_rms_cases=sum(r.get('tail_angle_rms') is not None for r in cases)))
        rows[-1]['terminal_recoveries']=sum(r['balanced'] for r in cases)
        rows[-1]['additional_terminal_recoveries']=sum(r['balanced'] and not r['primary_success'] for r in cases)
    write(OUT/'stress_counts.json',dict(rows=rows,plans=plans,interpretation='Primary no-horizontal-crossing endpoint unchanged; terminal recovery is a separate stored endpoint count. Failures retained, untested conditions explicit. Original methods pair with exact E3 H/L states/conditions; E4 has no borrowed original-cohort reference. Three training seeds are descriptive, no significance claim. Latencies describe offline fixed-step simulation.'))
    lines=['# Stress comparison checkpoint','','Matched 14-second A6/C50 cases; exact tested/requested denominators. Noise is scaled by each training cohort. E4 H/L stress references remain missing. See stress_counts.json for physical noise scales, finite-case RMS, rejections and paired discordant counts.','', '| Cohort | Method | Arm | Seed | Condition | Successes/tested | Requested | Paired H/L n |', '|---|---|---|---:|---|---:|---:|---|']
    for r in rows:
        paired=', '.join(f"{a}:{b['n']}" for a,b in r['paired'].items()) or 'missing'
        lines.append(f"| {r['cohort']} | {r['method']} | {r['arm']} | {r['seed']} | {r['condition']} | {r['successes']}/{r['tested']} | 56 | {paired} |")
    (OUT/'STRESS_COUNTS.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(stress_groups=len(rows),completed_groups=sum(r['complete'] for r in rows))))


if __name__=='__main__':main()
