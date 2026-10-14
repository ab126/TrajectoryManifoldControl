"""Bounded stratified raw endpoint/opaque plant replays; no controller decisions."""
import collections,json,hashlib
import numpy as np
from common import ROOT,OUT,Budget,dump,digest,isolation,plant_interface
from stress_protocol import mismatch_step,CONDITIONS

METHODS=['edmd','sindy','pets','pets/deterministic_reference','rokdeepc',
         'deepkoopman_control','neural_first_action','dfine_control',
         'dfine_control/output_only','long_references','fresh_references','rokdeepc/references',
         'edmd/stress','sindy/stress','rokdeepc/stress',
         'deepkoopman_control/stress','dfine_control/stress']


def main():
    isolation('dfine');budget=Budget('preparation',2);budget.checkpoint()
    strata=collections.defaultdict(list);inventory=[]
    for method in METHODS:
        for p in (OUT/method/'cases').glob('*.json'):
            r=json.loads(p.read_text());key=(method,r['seed'],r['arm'],r.get('set','C'),r.get('seconds',14.),r.get('condition','nominal'),r['primary_success'])
            rank=hashlib.sha256(str(p.relative_to(ROOT)).encode()).hexdigest()
            strata[key].append((rank,p,r))
        budget.checkpoint()
    step=plant_interface();physical_step=mismatch_step();rows=[]
    for key,candidates in sorted(strata.items()):
        # One deterministic representative per seed/arm/set/duration/outcome.
        _,p,r=min(candidates);raw=ROOT/r['raw_path'];a=np.load(raw)
        X,U,t=a['X'],a['U'],a['t'];seconds=r.get('seconds',14.)
        finite=bool(np.isfinite(X).all() and np.isfinite(U).all());complete=len(U)==round(seconds/.02)
        tail=t>=seconds-3.;no_fall=bool(finite and np.all(np.abs(X[:,2])<np.pi/2))
        balanced=bool(complete and finite and np.all(np.abs(X[tail,2])<=.02) and np.all(np.abs(X[tail,3])*np.sqrt(9.81)<=.05))
        assert bool(r['primary_success'])==(balanced and no_fall),(p,'primary endpoint')
        assert bool(r['balanced'])==balanced and bool(r['no_fall'])==no_fall,(p,'endpoint fields')
        case=r.get('identity',{}).get('case',{})
        expected=case.get('state',r.get('initial_state',r.get('identity',{}).get('initial_state')))
        if expected is not None:assert np.allclose(X[0],expected,rtol=0,atol=1e-12),(p,'initial state')
        applied=a['U_applied'] if 'U_applied' in a.files else U
        condition=r.get('condition');errors=[]
        for k in sorted(set([0,len(U)//2,len(U)-1])):
            if k<0 or not np.isfinite(X[k:k+2]).all():continue
            predicted=physical_step(X[k],float(applied[k,0]),condition) if condition else step(X[k],float(applied[k,0]))
            if condition and CONDITIONS[condition]['mode']=='process':predicted=predicted+a['noise'][k]
            errors.append(float(np.max(np.abs(predicted-X[k+1]))))
        assert not errors or max(errors)<2e-6,(p,'opaque plant replay',errors)
        rows.append(dict(method=key[0],seed=key[1],arm=key[2],set=key[3],seconds=key[4],condition=key[5],primary_success=balanced and no_fall,cases_in_stratum=len(candidates),case_path=str(p.relative_to(ROOT)),case_sha256=digest(p),raw_path=r['raw_path'],raw_sha256=digest(raw),max_one_step_error=max(errors) if errors else None))
        a.close();budget.checkpoint()
    dump(OUT/'inventory/literature_replay.json',dict(passed=True,selection='one SHA256-ranked case per method/seed/arm/set/duration/condition/success stratum; deterministic and outcome-stratified, no claim of exhaustive replay',sampled_cases=len(rows),inventoried_cases=sum(len(v) for v in strata.values()),rows=rows,source_sha256=digest(__file__),simulator_sha256=digest(ROOT/'src/inverted_pendulum.py'),accounting=budget.checkpoint()))
    dump(OUT/'replay_inventory/completion.json',dict(passed=True,sampled_cases=len(rows)))


if __name__=='__main__':main()
