"""Read-only scientific aggregation; exact arm/set identity, partial denominators."""
import collections,datetime,json,functools,os,math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3];OUT=ROOT/'validation/results_literature'
METHODS={'edmd':'original','sindy':'original','pets':'original','pets/deterministic_reference':'original','deepkoopman_control':'original','neural_first_action':'original','dfine_control':'original','smc_C200':'original','rokdeepc':'E4','dfine_control/output_only':'output_only','fresh_references':'original','rokdeepc/references':'E4'}
NATIVE_ARMS={'sindy':['CEM','CEM+L','NMPC_exact','NMPC_exact+L']}


def read(path):return json.loads(path.read_text())
def write(path,obj):
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(obj,indent=2)+'\n');temp.replace(path)
def index(row):
    if 'index' in row:return row['index']
    identity=row.get('identity',{});case=identity.get('case')
    return case['index'] if case else int(row['case'].rsplit('_',2)[1])


def descriptive_interval(successes,n):
    z=1.959963984540054;p=successes/n;denominator=1+z*z/n
    center=(p+z*z/(2*n))/denominator
    radius=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/denominator
    return [max(0.,center-radius),min(1.,center+radius)]


@functools.lru_cache(maxsize=None)
def refs(cohort,seed,name):
    if cohort=='output_only':return {}
    if cohort=='original' and name in ['A','A60']:
        original=read(ROOT/'validation/results_v1/initial_conditions.json')['sets'];seconds=60 if name=='A60' else 14
        result={}
        for i in ([0,2,5] if name=='A60' else range(6)):
            state=original['A'][i];indices=[j for j,x in enumerate(original['B']) if x==state]
            for arm in ['H','L']:
                for j in indices:
                    p=ROOT/f'validation/results_v1/E1/cases/seed{seed}_{arm}_B_{j:04d}_{seconds}s.json'
                    if p.exists():
                        row=read(p);assert row['initial_state']==state and row['seconds']==seconds
                        result[(arm,i)]=row
                p=OUT/f'long_references/cases/seed{seed}_{arm}_A_{i:04d}_{seconds}s.json'
                if p.exists():
                    row=read(p);assert row['initial_state']==state and row['seconds']==seconds
                    result[(arm,i)]=row
        return result
    if cohort=='E4':
        source_set='A' if name=='A60' else name;seconds=60 if name=='A60' else 14
        paths=(OUT/'rokdeepc/references/cases').glob(f'seed{seed}_*_{source_set}_*_{seconds}s.json')
    elif name=='F':paths=(OUT/'fresh_references/cases').glob(f'seed{seed}_*_{name}_*')
    else:
        source_set='A' if name=='A60' else name;seconds=60 if name=='A60' else 14
        paths=(ROOT/'validation/results_v1/E1/cases').glob(f'seed{seed}_*_{source_set}_*_{seconds}s.json')
    result={}
    for p in paths:
        r=read(p)
        if r['arm'] in ['H','L']:result[(r['arm'],index(r))]=r
    return result


def main():
    assert set(os.sched_getaffinity(0))<=set(range(80,128))
    original=read(ROOT/'validation/results_v1/initial_conditions.json');subset=set(original['C_subset']);groups=collections.defaultdict(list)
    for method,cohort in METHODS.items():
        for p in (OUT/method/'cases').glob('*.json'):
            r=read(p)
            if method in NATIVE_ARMS and r['arm'] not in NATIVE_ARMS[method]:continue
            if r.get('seconds',14.)!=14.:name='A60';expected=3
            else:name=r.get('set','C');expected={'A':6,'B':171,'C':200,'D':441,'F':200}.get(name,0)
            groups[(method,cohort,r['arm'],r['seed'],name,expected)].append(r)
            if name=='C' and index(r) in subset:groups[(method,cohort,r['arm'],r['seed'],'C50',50)].append(r)
    # Historical references remain separate from new planner-port cases. Read
    # only matched original C200 records; no old campaign is resumed or charged.
    for seed in range(3):
        for arm in ['H','L','M1']:
            for p in (ROOT/'validation/results_v1/E1/cases').glob(f'seed{seed}_{arm}_C_*_14s.json'):
                r=read(p);i=index(r)
                assert r['episodes']==3500 and r['seed']==seed and r['seconds']==14.
                assert r['initial_state']==original['sets']['C'][i]
                groups[('E1_reference','original',arm,seed,'C',200)].append(r)
                if i in subset:groups[('E1_reference','original',arm,seed,'C50',50)].append(r)
    counts=[]
    for (method,cohort,arm,seed,name,expected),rows in sorted(groups.items()):
        assert len({index(r) for r in rows})==len(rows),(method,arm,seed,name)
        item=dict(method=method,cohort=cohort,arm=arm,seed=seed,set=name,successes=sum(r['primary_success'] for r in rows),tested=len(rows),requested=expected,complete=len(rows)==expected,falls=sum(not r['no_fall'] for r in rows),solver_rejections=sum(r.get('solver_failures',0) for r in rows),paired={})
        item['terminal_recoveries']=sum(r['balanced'] for r in rows)
        item['additional_terminal_recoveries']=sum(r['balanced'] and not r['primary_success'] for r in rows)
        item['provenance']='read-only historical E1 original-collection reference, not merged with new planner-port identities' if method=='E1_reference' else 'literature campaign case records'
        item['descriptive_wilson95']=descriptive_interval(item['successes'],len(rows)) if name in ['C','C50','F'] else None
        item['interval_interpretation']='Per-model sampled-state binomial reference interval, descriptive only; does not pool correlated training seeds or calibrate fixed-grid probabilities.' if item['descriptive_wilson95'] else None
        references=refs(cohort,seed,'C' if name=='C50' else name)
        for refarm in ['H','L']:
            matched=[(r,references[(refarm,index(r))]) for r in rows if (refarm,index(r)) in references]
            for r,reference in matched:
                state=r.get('initial_state',r.get('identity',{}).get('case',{}).get('state'))
                if state is not None:assert state==reference['initial_state'],(method,arm,seed,name,'reference initial state')
                assert r.get('seconds',14.)==reference.get('seconds',14.),(method,arm,seed,name,'reference duration')
            if matched:item['paired'][refarm]=dict(n=len(matched),reference_successes=sum(b['primary_success'] for a,b in matched),reference_only=sum(b['primary_success'] and not a['primary_success'] for a,b in matched),method_only=sum(a['primary_success'] and not b['primary_success'] for a,b in matched),reference_cases=[b['case'] for a,b in matched])
        counts.append(item)
    write(OUT/'controller_counts.json',counts)
    lines=['# Controller comparison checkpoint','','Exact completed denominators; missing cells remain untested. No statistical significance is claimed. Original, E4 and output-only cohorts are separate. Counts are descriptive across training seeds. controller_counts.json includes per-model descriptive Wilson reference intervals for sampled random states; no seed pooling or fixed-grid probability calibration.','', '| Cohort | Controller | Seed | Set | Successes/tested | Requested | Paired H/L n |','|---|---|---:|---|---:|---:|---|']
    for r in counts:
        if r['set'] not in ['C','F','A60']:continue
        paired=', '.join(f"{a}:{p['n']}" for a,p in r['paired'].items()) or 'missing'
        lines.append(f"| {r['cohort']} | {r['method']} {r['arm']} | {r['seed']} | {r['set']} | {r['successes']}/{r['tested']} | {r['requested']} | {paired} |")
    (OUT/'CONTROLLER_COUNTS.md').write_text('\n'.join(lines)+'\n')
    recovery=['# Separate terminal-recovery counts','','Stored terminal angle/angular-speed endpoint allows earlier horizontal crossings. Primary success counts are unchanged; 14/60-second rows and training cohorts remain separate. This uses saved endpoint fields whose raw sample was independently checked, not an exhaustive raw rescore.','', '| Cohort | Controller | Seed | Set | Primary/tested | Terminal recovery/tested | Additional recoveries |','|---|---|---:|---|---:|---:|---:|']
    for r in counts:
        if r['set'] not in ['C','F','A60']:continue
        recovery.append(f"| {r['cohort']} | {r['method']} {r['arm']} | {r['seed']} | {r['set']} | {r['successes']}/{r['tested']} | {r['terminal_recoveries']}/{r['tested']} | {r['additional_terminal_recoveries']} |")
    (OUT/'TERMINAL_RECOVERY_COUNTS.md').write_text('\n'.join(recovery)+'\n')
    accounts=[dict(stage=p.parent.name,**read(p)) for p in OUT.glob('*/accounting.json')];ledger=dict(timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),stages=accounts,cpu_core_hours=sum(a['cpu_seconds'] for a in accounts)/3600,gpu_hours=sum(a.get('gpu_seconds',0.) for a in accounts)/3600,elapsed_separate=True,provisional=True,preparation_charge='inventory/preparation_conservative_charge.json; conservative unmeasured half hour',ceilings=dict(cpu_core_hours=84,gpu_hours=30));write(OUT/'resource_checkpoint.json',ledger)
    write(OUT/'reviews/results_checkpoint.json',dict(question='Review exact denominators, cohort separation and paired counts; source/adaptation evidence remains in method packets',timestamp=ledger['timestamp'],controller_counts='controller_counts.json',forecast_tables='prediction_tables.json',port_notes='sources/PORT_NOTES.md',resource_checkpoint='resource_checkpoint.json',scientific_conclusions='Native failures are retained; common-L results are explicitly attributed. Implementation errors/usage pauses are not scientific failures.',results=[r for r in counts if r['set'] in ['C','F']]))
    report=OUT/'RESULTS.md';text=report.read_text();marker='\n## Machine-derived comparison checkpoint\n'
    paragraph=f"\nUpdated {ledger['timestamp']}. [Controller counts](CONTROLLER_COUNTS.md) retain exact tested/requested denominators and paired discordant counts; original, E4 and output-only cohorts stay separate. [Forecast tables](prediction_tables.json) cover passive and forced prediction separately. [Port notes](sources/PORT_NOTES.md) disclose source changes and controller additions. Provisional ledger: {ledger['cpu_core_hours']:.3f} CPU core-hours and {ledger['gpu_hours']:.3f} GPU-hours, including the conservative preparation charge; ceilings remain 84/30. Running jobs can advance beyond this snapshot.\n"
    # Refresh this section only; later scientific checkpoints must survive.
    if marker in text:
        before,after=text.split(marker,1);next_section=after.find('\n## ')
        text=before+marker+paragraph+(after[next_section:] if next_section>=0 else '')
    else:text+=marker+paragraph
    report.write_text(text)
    print(json.dumps(dict(controller_groups=len(counts),cpu_core_hours=ledger['cpu_core_hours'],gpu_hours=ledger['gpu_hours'])))


if __name__=='__main__':main()
