"""Frozen training-only PID design, bounded pilot and matched C200/fresh sets."""
import argparse
import json
import numpy as np
from scipy.optimize import minimize
from common import *
from casebook import cases
from pid import PID,command,design_cost,local_matrix,GAIN_BOUNDS,REFERENCE_BOUND


def checked_fit(seed,budget):
    data,identity=load_data(seed);X,U=data['train'];x=X[:,:-1].reshape(-1,4);xn=X[:,1:].reshape(-1,4)
    nearest=np.argsort(np.linalg.norm(x/x.std(0).clip(1e-6),axis=1))[:6000]
    reg=np.column_stack((x[nearest],U.reshape(-1,1)[nearest]));coef=np.linalg.solve(reg.T@reg+1e-6*np.eye(5),reg.T@xn[nearest]);A,B=coef[:4].T,coef[4:].T
    savedpath=ROOT/f'validation/results_v1/models/episodes3500_seed{seed}/diagnostics.json';saved=json.loads(savedpath.read_text())['local']
    np.testing.assert_allclose(A,saved['A'],rtol=1e-10,atol=1e-10);np.testing.assert_allclose(B,saved['B'],rtol=1e-10,atol=1e-10)
    A,B=np.asarray(saved['A']),np.asarray(saved['B'])
    # All starts' signs come from the fitted force->angle response, not plant equations.
    sign=np.sign(B[3,0]);assert sign!=0
    starts=[[sign*.03,sign*.3,sign*k,sign*d,sign*i] for k,d,i in [(40,20,1),(60,40,1),(80,50,2),(100,60,5),(120,80,10),(30,15,.1)]]
    initial_ids=np.argsort(np.linalg.norm(X[:,0]/x.std(0),axis=1))[:16]
    initial=X[initial_ids,0]
    root=OUT/'pid'/f'seed{seed}';root.mkdir(parents=True,exist_ok=True)
    protocol=dict(data=identity,diagnostics_sha256=digest(savedpath),adapter_sha256={p:digest(Path(__file__).with_name(p)) for p in ['pid.py','run_pid.py','common.py']},A=A.tolist(),B=B.tolist(),starts=starts,bounds=GAIN_BOUNDS,reference_bound=REFERENCE_BOUND,initial_training_episode_indices=initial_ids.tolist(),design_steps=200,design_cost='shared physical Q,R,adjacent input and terminal costs, averaged 16 training initial states',optimizer='L-BFGS-B, six starts, maxiter=100, no simulator',integral='normalized DT; clip +/-10; conditional anti-windup; reset each episode',selection='lowest finite design cost among fitted augmented spectral radius <1; candidate-order ties')
    path=root/'protocol.json'
    # JSON represents tuples as lists.
    protocol=json.loads(json.dumps(protocol))
    if path.exists():assert json.loads(path.read_text())==protocol
    else:dump(path,protocol)
    scores=[]
    for index,start in enumerate(starts):
        p=root/f'candidate{index}.json'
        if p.exists():scores.append(json.loads(p.read_text()));continue
        calls=[0]
        def fun(g):
            calls[0]+=1
            if calls[0]%10==0:budget.checkpoint()
            return design_cost(g,A,B,initial)
        result=minimize(fun,start,method='L-BFGS-B',bounds=GAIN_BOUNDS,options=dict(maxiter=100,ftol=1e-10))
        radius=float(np.max(np.abs(np.linalg.eigvals(local_matrix(A,B,result.x)))))
        row=dict(index=index,gains=result.x.tolist(),design_cost=float(result.fun),stable_local=radius<1.,local_augmented_spectral_radius=radius,optimizer_success=bool(result.success),termination=str(result.message),iterations=int(result.nit),evaluations=calls[0]);dump(p,row);scores.append(row);budget.checkpoint();print(json.dumps(dict(seed=seed,candidate=row)),flush=True)
    valid=[r for r in scores if r['stable_local'] and np.isfinite(r['design_cost'])]
    if not valid:
        dump(root/'tuning_failure.json',dict(state='unsuccessful tuning',candidates=scores));return None,protocol
    chosen=min(valid,key=lambda r:(r['design_cost'],r['index']));selected=root/'selection.json'
    if selected.exists():assert json.loads(selected.read_text())==chosen
    else:dump(selected,chosen)
    return chosen,protocol


def checks():
    gains=np.array([-.1,-.4,-60.,-30.,-2.]);x=np.array([.2,-.1,.03,.1]);I=.04
    u,updated,ref,raw=command(x,I,gains);literal=-60*(np.clip(-.1*.2-.4*(-.1),-.25,.25)-.03)-(-30)*.1-2*.04
    assert abs(raw-literal)<1e-12 and abs(updated-(I+DT*(ref-x[2])))<1e-12
    z=np.zeros(4);assert command(z,0.,gains)[0]==0.
    a=PID(gains);a.solve(x);a.reset();assert a.integral==0.
    # Positive derivative disturbance must produce -kw*omega, exactly.
    assert command(np.array([0.,0.,0.,.1]),0.,gains)[0]==3.
    _,i,_,_=command(np.array([0.,0.,1.,0.]),1.,np.array([0.,0.,-100.,0.,-2.]));assert i==1.
    dump(OUT/'pid/checks.json',dict(passed=True,literal_controller=True,derivative_on_measurement=True,saturation_antiwindup_reset=True,units='same normalized states/input as other methods; integral normalized time DT'))


def evaluate(seed,chosen,protocol,case,budget):
    key=f"seed{seed}_DD-PID_{case['set']}_{case['index']:04d}_{case['seconds']:g}s";p=OUT/'pid/cases'/f'{key}.json'
    identity=dict(protocol=protocol,selected=chosen,case=case,simulator_sha256=digest(ROOT/'src/inverted_pendulum.py'),endpoint_sha256=digest(__file__))
    if p.exists():r=json.loads(p.read_text());assert r['identity']==identity;return r
    steps=round(case['seconds']/.02);X=np.empty((steps+1,4));U=np.empty((steps,1));I=[];references=[];X[0]=case['state'];law=PID(chosen['gains']);step=plant_interface();start=time.monotonic();cpu=cpu_seconds();latencies=[]
    for k in range(steps):
        if k%25==0:budget.checkpoint()
        before=time.monotonic();plan,info=law.solve(X[k]);latencies.append(time.monotonic()-before);U[k]=plan[0];I.append(info['integral']);references.append(info['reference'])
        with np.errstate(over='ignore',invalid='ignore'):X[k+1]=step(X[k],U[k,0])
        if not np.isfinite(X[k+1]).all():X=X[:k+2];U=U[:k+1];break
    t=np.arange(len(X))*.02;tail=t>=case['seconds']-3.;finite=bool(np.isfinite(X).all());complete=len(U)==steps;balanced=bool(complete and finite and np.all(np.abs(X[tail,2])<=.02) and np.all(np.abs(X[tail,3])*np.sqrt(9.81)<=.05));no_fall=bool(finite and np.all(np.abs(X[:,2])<np.pi/2));raw=OUT/'pid/raw'/f'{key}.npz';raw.parent.mkdir(exist_ok=True);np.savez_compressed(raw,X=X,U=U,t=t,integral=I,angle_reference=references)
    row=dict(identity=identity,case=key,seed=seed,arm='DD-PID',set=case['set'],index=case['index'],seconds=case['seconds'],complete=complete,primary_success=balanced and no_fall,balanced=balanced,no_fall=no_fall,nonfinite_state=not finite,raw_path=str(raw.relative_to(ROOT)),cpu_seconds=cpu_seconds()-cpu,wall_seconds=time.monotonic()-start,effort=float(np.sum(U**2)*.02),saturation_seconds=float(np.sum(np.abs(U)>=10-1e-6)*.02),max_abs_integral=float(np.max(np.abs(I))),solve_latency_quantiles=np.quantile(latencies,[.5,.95,1.]).tolist());dump(p,row);budget.checkpoint();return row


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['pilot','matrix','expansion'],required=True);args=parser.parse_args();isolation('pid');budget=Budget('pid',1);budget.checkpoint();checks()
    original=json.loads((ROOT/'validation/results_v1/initial_conditions.json').read_text())['sets']['C'];fresh=json.loads((OUT/'inventory/fresh_200_states.json').read_text())['states']
    if args.phase!='pilot':
        projection=json.loads((OUT/'pid/pilot_projection.json').read_text())
        requested_cost=projection['minimum_matrix_core_hours'] if args.phase=='matrix' else projection['additional_C200_F200_core_hours']
        assert requested_cost<=1-budget.checkpoint()['cpu_seconds']/3600,'PID requested matrix exceeds reserve suballocation'
    for seed in ([0] if args.phase=='pilot' else range(3)):
        chosen,protocol=checked_fit(seed,budget)
        if chosen is None:continue
        requested=cases()[:16] if args.phase=='pilot' else cases()
        if args.phase=='expansion':requested=[dict(set=name,index=i,state=x,seconds=14.) for name,states in [('C',original),('F',fresh)] for i,x in enumerate(states)]
        for case in requested:evaluate(seed,chosen,protocol,case,budget)
    if args.phase=='pilot':
        rows=[json.loads(p.read_text()) for p in (OUT/'pid/cases').glob('*.json')];mean=float(np.mean([r['cpu_seconds'] for r in rows])) if rows else 0.;remaining=1-budget.checkpoint()['cpu_seconds']/3600
        dump(OUT/'pid/pilot_projection.json',dict(minimum_matrix_core_hours=mean*(168+9*3000/700)/3600,additional_C200_F200_core_hours=mean*1200/3600,remaining_core_hours=remaining))
    dump(OUT/'pid'/f'{args.phase}_completion.json',dict(state='complete',accounting=budget.checkpoint()))


if __name__=='__main__':main()
