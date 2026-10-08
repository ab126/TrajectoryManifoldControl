"""Independent saved-trace scoring and original-plant replay; no new experiments."""
from common import *
import argparse
from collections import Counter
from src.inverted_pendulum import rk4_step

def score(X,steps=1000):
    physical=X*np.array([1.,np.sqrt(9.81),1.,np.sqrt(9.81)])
    complete=len(X)==steps+1
    finite=bool(np.isfinite(physical).all())
    tail=physical[-151:]
    flags={'complete':complete,'finite':finite,
           'rail':bool(finite and np.all(np.abs(physical[:,0])<=2.4)),
           'angle':bool(finite and np.all(np.abs(physical[:,2])<=np.pi/2)),
           'terminal_position':bool(finite and np.all(np.abs(tail[:,0])<=.1)),
           'terminal_cart_speed':bool(finite and np.all(np.abs(tail[:,1])<=.05)),
           'terminal_angle':bool(finite and np.all(np.abs(tail[:,2])<=.02)),
           'terminal_angular_speed':bool(finite and np.all(np.abs(tail[:,3])<=.05))}
    return all(flags.values()),flags

def audit(directory):
    directory=Path(directory)
    summary_path=directory/'summary.json'
    summary=json.loads(summary_path.read_text()) if summary_path.exists() else {'passed':False,'incomplete':True}
    rows=json.loads((directory/'cases.json').read_text()) if (directory/'cases.json').exists() else []
    initial=np.load(directory/'initial_states.npy') if (directory/'initial_states.npy').exists() else None
    audited=[]; mismatches=[]; reasons=Counter(); replay_max=0.
    config_path=directory/'config.json'
    config=json.loads(config_path.read_text()) if config_path.exists() else None
    if config is not None and initial is not None:
        expected=np.random.default_rng(indexed_seed('gate_a_cases',1.)).uniform(-1,1,(200,4))*np.array([1.5,2.5,1.3,3.5])
        if not np.array_equal(initial,expected):mismatches.append('Nominal case seed/box mismatch')
        if config.get('preflight',{}).get('starts')!=1 or config.get('dtype')!='float64':mismatches.append('Overnight starts/dtype mismatch')
    h=.02/np.sqrt(1/9.81)
    for row in rows:
        case=row['case']; path=directory/f'case_{case:03d}.npz'
        if not path.exists(): mismatches.append(f'Missing trace for case {case}'); continue
        with np.load(path,allow_pickle=False) as trace:
            X,U=trace['X'],trace['U']; branches=trace['branches']
            if X.shape!=(1001,4) or U.shape!=(1000,1):
                mismatches.append(f'Wrong trajectory shape for case {case}'); continue
            success,flags=score(X)
            if success!=row['strict_physical_success']: mismatches.append(f'Score disagreement for case {case}')
            if initial is None or not np.array_equal(X[0],initial[case]): mismatches.append(f'Initial-state disagreement for case {case}')
            if not np.array_equal(np.isfinite(U[:,0]),np.isfinite(X[:-1]).all(axis=1)):
                # The original calibration intentionally fills the suffix with NaN
                # after the first nonfinite next state; each finite input prefix
                # must still correspond to finite preceding states.
                if np.any(np.isfinite(U[:,0]) & ~np.isfinite(X[:-1]).all(axis=1)):
                    mismatches.append(f'Action after nonfinite state for case {case}')
            finite_u=U[np.isfinite(U)]
            if np.any(np.abs(finite_u)>10+1e-12): mismatches.append(f'Input clipping violation for case {case}')
            for k in range(len(branches)):
                if not (np.isfinite(X[k]).all() and np.isfinite(U[k]).all()): break
                with np.errstate(over='ignore',invalid='ignore',divide='ignore'):
                    expected=rk4_step(X[k],float(U[k,0]),h,2.)
                if not np.allclose(expected,X[k+1],atol=1e-10,rtol=1e-10,equal_nan=True):
                    mismatches.append(f'Plant replay disagreement for case {case}, sample {k}'); break
                both=np.isfinite(expected)&np.isfinite(X[k+1])
                if both.any(): replay_max=max(replay_max,float(np.max(np.abs(expected[both]-X[k+1,both]))))
            for name,value in flags.items():
                if not value: reasons[name]+=1
            if len(branches)!=int(np.sum(np.isfinite(U[:,0]))):mismatches.append(f'Missing/excess control branches for case {case}')
            if bool(row.get('positive_angle',X[0,2]>=0))!=bool(X[0,2]>=0):mismatches.append(f'Angle-sign label mismatch for case {case}')
            if len(branches)!=row.get('executed_samples',len(branches)):
                mismatches.append(f'Branch length disagreement for case {case}')
            if int(np.sum(branches=='plan'))!=row['planning_calls']:
                # Nonfinite solver fallback is also a planning call.
                if int(np.sum((branches=='plan')|(branches=='nonfinite')))!=row['planning_calls']:
                    mismatches.append(f'Planning-count disagreement for case {case}')
            first_rail=np.flatnonzero(np.abs(X[:,0])>2.4)
            first_angle=np.flatnonzero(np.abs(X[:,2])>np.pi/2)
            audited.append({'case':case,'success':success,'positive_angle':bool(X[0,2]>=0),
                'failure_flags':[name for name,value in flags.items() if not value],
                'first_rail_sample':int(first_rail[0]) if len(first_rail) else None,
                'first_angle_sample':int(first_angle[0]) if len(first_angle) else None,
                'branches':dict(Counter(branches.tolist()))})
    complete=bool(initial is not None and len(initial)==200 and len(rows)==200 and len(audited)==200 and not summary.get('incomplete',True))
    rate=float(np.mean([r['success'] for r in audited])) if audited else None
    halves=[]
    for sign in (False,True):
        values=[r['success'] for r in audited if r['positive_angle']==sign]
        halves.append(float(np.mean(values)) if values else None)
    attainability=bool(complete and rate is not None and rate>=.98 and all(p is not None and p>=.95 for p in halves))
    if complete:
        if config is None:mismatches.append('Completed calibration missing its configuration manifest')
        if not np.isclose(rate,summary.get('physical_success_rate',np.nan)): mismatches.append('Summary rate disagrees')
        if attainability!=bool(summary.get('passed')): mismatches.append('Summary Gate A flag disagrees')
    return {'audit_passed':not mismatches,'calibration_complete':complete,'Gate_A_passed':attainability and not mismatches,
            'cases_audited':len(audited),'success_rate':rate,'angle_half_rates':halves,
            'failure_counts':dict(reasons),'plant_replay_max_abs_error':replay_max,
            'mismatches':mismatches,'case_diagnostics':audited,
            'scope':'independent physical endpoint scoring and original NumPy plant replay; no model/optimizer changes or new plant acquisition'}

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('directory'); args=parser.parse_args()
    result=audit(args.directory)
    (Path(args.directory)/'independent_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='case_diagnostics'},indent=2))
    raise SystemExit(0 if result['audit_passed'] else 2)
