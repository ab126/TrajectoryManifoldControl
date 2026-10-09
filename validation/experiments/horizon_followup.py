"""20-second prefixes only for angularly unsettled, no-fall 14-second cases."""
from campaign import *
out=OUT/'E0';rows=[]
for path in sorted((out/'cases').glob('*_14s.json')):
    case=json.loads(path.read_text())
    if case['arm'] not in ('H','Hstar') or case['balanced'] or not case['no_fall']:continue
    longer=out/'cases'/(case['case'].replace('_14s','_60s')+'.json')
    if not longer.exists():continue
    source=json.loads(longer.read_text());data=np.load(ROOT/source['raw_path'])
    n=1000;steps=data['solve_steps'];loss=data['loss'][steps<n]
    row=dict(case=case['case'].replace('_14s','_20s_prefix'),seed=case['seed'],arm=case['arm'],ic_set='A',ic_index=case['ic_index'],
        source_60s_path=source['raw_path'],selection='Not balanced but no fall at 14 s',
        **metrics(data['t'][:n+1],data['X'][:n+1],data['U'][:n],data['local'][:n],losses=loss))
    rows.append(row)
csv_write(out/'horizon_limited_20s.csv',rows)
dump(out/'horizon_followup.json',dict(selection='H/Hstar IC-A cases with balanced=False and no_fall=True at 14 s',
    method='Prefix of independently simulated same-seed 60 s run; no fresh CEM solve',
    selected_cases=len(rows),primary_successes_at_20s=sum(r['primary_success'] for r in rows),
    by_arm={arm:dict(selected=sum(r['arm']==arm for r in rows),successes=sum(r['arm']==arm and r['primary_success'] for r in rows)) for arm in ('H','Hstar')}))
print(json.dumps(json.loads((out/'horizon_followup.json').read_text())),flush=True)
