"""E4-only RoKDeePC stress; frozen models, causal applied-force context."""
import json,time,pickle
import numpy as np
from common import *
from run_rokdeepc import fit,data,score
from rokdeepc import RoKDeePC
from e4_noisy_data import collect
from casebook import cases
from stress_protocol import CONDITIONS,REQUESTED,disturbances,mismatch_step


class PreservingBudget(Budget):
    def checkpoint(self):
        value=dict(cpu_seconds=self.previous['cpu_seconds']+cpu_seconds()-self.start_cpu,wall_seconds=self.previous['wall_seconds']+time.monotonic()-self.start_wall,gpu_seconds=self.previous.get('gpu_seconds',0.))
        dump(self.path,value)
        if value['cpu_seconds']>=self.ceiling or sum(json.loads(p.read_text())['cpu_seconds'] for p in OUT.glob('*/accounting.json'))>=84*3600:raise RuntimeError('CPU allocation exhausted; cap-limited partial result')
        usage_gate();return value


def noisy_fit(seed,budget):
    source,acquisition=collect(seed,budget);saved=np.load(source);X,U=saved['X'].astype(float),saved['U'].astype(float)
    order=np.random.default_rng(seed).permutation(len(X));n=round(.15*len(X));train=(X[order[2*n:]],U[order[2*n:]]);validation=(X[order[n:2*n]],U[order[n:2*n]])
    scale=train[0][:,:-1].reshape(-1,4).std(0).clip(1e-8);root=source.parent.parent
    grid=[dict(bandwidth_factor=b,gamma=r) for b in [.5,1.,2.] for r in [.02,.2]]
    identity=dict(acquisition=acquisition,data_sha256=digest(source),split_indices=dict(train=order[2*n:].tolist(),validation=order[n:2*n].tolist(),test=order[:n].tolist()),grid=grid,selection='same training-standardized 30-step validation rollout MSE',adapter_sha256=digest(__file__),kernel_sha256=digest(Path(__file__).with_name('rokdeepc.py')))
    p=root/'identity.json'
    if p.exists():assert json.loads(p.read_text())==identity
    else:dump(p,identity)
    rows=[]
    for index,config in enumerate(grid):
        p=root/f'candidate{index}.json';modelpath=root/f'candidate{index}.pkl'
        if p.exists() and modelpath.exists():rows.append(json.loads(p.read_text()));continue
        budget.checkpoint();model=RoKDeePC(**config).fit(*train);value=score(model,validation,scale)
        modelpath.write_bytes(pickle.dumps(model));row=dict(index=index,config=config,validation_rollout_mse=value);dump(p,row);rows.append(row)
    selected=min(rows,key=lambda r:(r['validation_rollout_mse'],r['index']));dump(root/'selection.json',selected)
    path=root/f"candidate{selected['index']}.pkl";kernel=pickle.loads(path.read_bytes());linear=RoKDeePC(**selected['config'],linear=True).fit(*train)
    identity['selected_model_sha256']=digest(path);identity['selected']=selected
    return kernel,linear,identity


def evaluate(seed,arm,case,condition,model,identity,scale,gain,budget):
    dest=OUT/'rokdeepc/stress';key=f"seed{seed}_{arm}_{condition}_{case['set']}_{case['index']:04d}_14s";p=dest/'cases'/f'{key}.json';noise=disturbances(seed,condition,case,scale);expected=dict(fit=identity,case=case,condition=condition,seed=seed,arm=arm,gain=gain.tolist(),source_sha256=digest(__file__),protocol_sha256=digest(Path(__file__).with_name('stress_protocol.py')),simulator_sha256=digest(ROOT/'src/inverted_pendulum.py'))
    if p.exists():row=json.loads(p.read_text());assert row['identity']==expected;return row
    solver=model;solver.reset();step=mismatch_step();X=np.empty((701,4));U=np.empty((700,1));applied=np.empty_like(U);X[0]=case['state'];local=[];diagnostics=[];latencies=[];previous=0.;start=time.monotonic();cpu=cpu_seconds()
    for k in range(700):
        if k%25==0:budget.checkpoint()
        observed=X[k]+noise[k] if CONDITIONS[condition]['mode']=='measurement' else X[k];catch=False;local.append(catch);solver.previous_input=0. if k==0 else float(applied[k-1,0])
        if catch:u=np.clip(-gain@observed,-10,10)
        else:before=time.monotonic();plan,info=solver.solve(observed);latencies.append(time.monotonic()-before);diagnostics.append(info);u=plan[0]
        U[k]=u;applied[k]=previous if condition=='input_delay' else u;previous=float(U[k,0])
        with np.errstate(over='ignore',invalid='ignore'):X[k+1]=step(X[k],applied[k,0],condition)+(noise[k] if CONDITIONS[condition]['mode']=='process' else 0.)
        if not np.isfinite(X[k+1]).all():X=X[:k+2];U=U[:k+1];applied=applied[:k+1];break
    t=np.arange(len(X))*.02;tail=t>=11.;finite=bool(np.isfinite(X).all());complete=len(U)==700;no_fall=bool(finite and np.all(np.abs(X[:,2])<np.pi/2));balanced=bool(complete and finite and np.all(np.abs(X[tail,2])<=.02) and np.all(np.abs(X[tail,3])*np.sqrt(9.81)<=.05));raw=dest/'raw'/f'{key}.npz';raw.parent.mkdir(exist_ok=True);np.savez_compressed(raw,t=t,X=X,U=U,U_applied=applied,noise=noise,local=local)
    def number(v):return float(v) if np.isfinite(v) else None
    row=dict(case=key,identity=expected,seed=seed,arm=arm,set=case['set'],index=case['index'],condition=condition,seconds=14.,complete=complete,primary_success=balanced and no_fall,balanced=balanced,no_fall=no_fall,nonfinite_state=not finite,tail_angle_rms=number(np.sqrt(np.mean(X[tail,2]**2))),tail_angular_velocity_rms_physical=number(np.sqrt(np.mean((X[tail,3]*np.sqrt(9.81))**2))),noise_std_normalized=(scale*CONDITIONS[condition].get('level',0.)).tolist(),noise_std_physical=(scale*CONDITIONS[condition].get('level',0.)*np.array([1,np.sqrt(9.81),1,np.sqrt(9.81)])).tolist(),solver_failures=sum(not d['accepted'] for d in diagnostics),solve_latency_quantiles=np.quantile(latencies,[.5,.95,1.]).tolist() if latencies else [],cpu_seconds=cpu_seconds()-cpu,wall_seconds=time.monotonic()-start,raw_path=str(raw.relative_to(ROOT)));dump(dest/'solver_logs'/f'{key}.json',diagnostics);dump(p,row);budget.checkpoint();return row


def main():
    isolation('rokdeepc');budget=PreservingBudget('rokdeepc',12);budget.checkpoint();root=OUT/'rokdeepc/stress';p=root/'plan.json'
    if p.exists():selection=json.loads(p.read_text())
    else:
        timings=json.loads((OUT/'rokdeepc/pilot_projection.json').read_text())['mean_case_cpu_seconds'];cost=sum(timings.values())*168/3600;remaining=12-budget.checkpoint()['cpu_seconds']/3600;chosen=[]
        for group in [['input_delay'],['mass_minus25','mass_plus25'],['length_minus25','length_plus25'],['measurement_0.01'],['process_0.01']]:
            fitting_reserve=.25 if 'process_0.01' in group else 0.
            if cost*len(group)*1.1+fitting_reserve<=remaining:chosen+=group;remaining-=cost*len(group)*1.1+fitting_reserve
        selection=dict(conditions=chosen,requested=REQUESTED,untested=[c for c in REQUESTED if c not in chosen],projected_core_hours_per_condition=cost,process_acquisition_fit_reservation_core_hours=.25,rule='ordered whole three-seed/56-state conditions, paired mismatches together, 10% timing margin; no outcome selection',process_noise='new E4 recollection from saved E4 initial states/force sequences; opaque plant acquisition with matched E4-scaled noise, same six-candidate validation selection',cohort='E4 only; same disturbance RNG indices, E4 training standard deviations; original E3 H/L counts are not matched references',source_sha256=digest(__file__));dump(p,selection)
    for seed in range(3):
        kernel,linear,identity=fit(seed,budget);split,_=data(seed);scale=split['train'][0][:,:-1].astype(np.float32).reshape(-1,4).std(0).clip(1e-6);selected=json.loads((OUT/'rokdeepc'/f'seed{seed}/selection.json').read_text())['index'];identity['selected_model_sha256']=digest(OUT/'rokdeepc'/f"seed{seed}/candidate{selected}.pkl")
        for condition in selection['conditions']:
            actual_kernel,actual_linear,actual_identity=noisy_fit(seed,budget) if condition=='process_0.01' else (kernel,linear,identity)
            for arm,model in [('RoKDeePC',actual_kernel),('linear_kernel',actual_linear)]:
                for case in cases()[:56]:evaluate(seed,arm,case,condition,model,actual_identity,scale,np.zeros(4),budget)
    dump(root/'completion.json',dict(state='complete' if not selection['untested'] else 'cap_limited',completed_conditions=selection['conditions'],untested_conditions=selection['untested'],accounting=budget.checkpoint()))


if __name__=='__main__':main()
