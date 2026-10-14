"""Timing-bounded PETS-style stress; frozen TS-infinity and matched E3 data."""
import json,time
import numpy as np
import run_pets
from common import *
from neural_common import GPUBudget
from pets import TSInfinityCEM
from casebook import cases
from stress_protocol import CONDITIONS,REQUESTED,disturbances,mismatch_step
from neural_noise_data import load as load_noisy


def evaluate(seed,arm,case,condition,model,identity,scale,gain,budget):
    dest=OUT/'pets/stress';key=f"seed{seed}_{arm}_{condition}_{case['set']}_{case['index']:04d}_14s";p=dest/'cases'/f'{key}.json';noise=disturbances(seed,condition,case,scale);expected=dict(fit=identity,case=case,condition=condition,seed=seed,arm=arm,gain=gain.tolist(),source_sha256=digest(__file__),protocol_sha256=digest(Path(__file__).with_name('stress_protocol.py')),simulator_sha256=digest(ROOT/'src/inverted_pendulum.py'))
    if p.exists():row=json.loads(p.read_text());assert row['identity']==expected;return row
    solver=TSInfinityCEM(model,seed,identity['planner']['particles'],identity['planner']['population']);step=mismatch_step();X=np.empty((701,4));U=np.empty((700,1));applied=np.empty_like(U);X[0]=case['state'];local=[];diagnostics=[];latencies=[];previous=0.;start=time.monotonic();cpu=cpu_seconds()
    for k in range(700):
        if k%25==0:budget.checkpoint()
        observed=X[k]+noise[k] if CONDITIONS[condition]['mode']=='measurement' else X[k];catch=arm.endswith('+L') and observed@Q@observed<.75;local.append(catch)
        if catch:u=np.clip(-gain@observed,-10,10)
        else:before=time.monotonic();plan,info=solver.solve(observed);latencies.append(time.monotonic()-before);diagnostics.append(info);u=plan[0]
        U[k]=u;applied[k]=previous if condition=='input_delay' else u;previous=float(U[k,0])
        with np.errstate(over='ignore',invalid='ignore'):X[k+1]=step(X[k],applied[k,0],condition)+(noise[k] if CONDITIONS[condition]['mode']=='process' else 0.)
        if not np.isfinite(X[k+1]).all():X=X[:k+2];U=U[:k+1];applied=applied[:k+1];break
    t=np.arange(len(X))*.02;tail=t>=11.;finite=bool(np.isfinite(X).all());complete=len(U)==700;no_fall=bool(finite and np.all(np.abs(X[:,2])<np.pi/2));balanced=bool(complete and finite and np.all(np.abs(X[tail,2])<=.02) and np.all(np.abs(X[tail,3])*np.sqrt(9.81)<=.05));raw=dest/'raw'/f'{key}.npz';raw.parent.mkdir(exist_ok=True);np.savez_compressed(raw,t=t,X=X,U=U,U_applied=applied,noise=noise,local=local)
    def number(v):return float(v) if np.isfinite(v) else None
    row=dict(case=key,identity=expected,seed=seed,arm=arm,set=case['set'],index=case['index'],condition=condition,seconds=14.,complete=complete,primary_success=balanced and no_fall,balanced=balanced,no_fall=no_fall,nonfinite_state=not finite,tail_angle_rms=number(np.sqrt(np.mean(X[tail,2]**2))),tail_angular_velocity_rms_physical=number(np.sqrt(np.mean((X[tail,3]*np.sqrt(9.81))**2))),noise_std_normalized=(scale*CONDITIONS[condition].get('level',0.)).tolist(),noise_std_physical=(scale*CONDITIONS[condition].get('level',0.)*np.array([1,np.sqrt(9.81),1,np.sqrt(9.81)])).tolist(),solver_failures=sum(not d['accepted'] for d in diagnostics),solve_latency_quantiles=np.quantile(latencies,[.5,.95,1.]).tolist() if latencies else [],cpu_seconds=cpu_seconds()-cpu,wall_seconds=time.monotonic()-start,raw_path=str(raw.relative_to(ROOT)));dump(dest/'solver_logs'/f'{key}.json',diagnostics);dump(p,row);budget.checkpoint();return row


def main():
    isolation('pets');budget=GPUBudget('pets',8,8);budget.checkpoint();root=OUT/'pets/stress';p=root/'plan.json';planner=json.loads((OUT/'pets/planner_selection.json').read_text())
    if p.exists():selection=json.loads(p.read_text())
    else:
        timing=json.loads((OUT/'pets/fresh_expansion_plan.json').read_text())['mean_case_seconds'];cost=timing*168/3600;remaining=8-budget.checkpoint()['gpu_seconds']/3600;chosen=[]
        for group in [['input_delay'],['mass_minus25','mass_plus25'],['length_minus25','length_plus25'],['measurement_0.01'],['process_0.01']]:
            fit_reserve=.1 if 'process_0.01' in group else 0.
            if cost*len(group)*1.1+fit_reserve<=remaining:chosen+=group;remaining-=cost*len(group)*1.1+fit_reserve
        selection=dict(conditions=chosen,requested=REQUESTED,untested=[c for c in REQUESTED if c not in chosen],projected_gpu_hours_per_condition=cost,process_fit_gpu_hours_reservation=.1,rule='ordered whole three-seed/56-state conditions with paired mismatches, 10% timing margin; no outcome selection',source_sha256=digest(__file__),process_training='same fixed architecture/validation NLL stopping on exact E3 process_0.01 collection; original weights preserved');dump(p,selection)
    for condition in selection['conditions']:
        if condition=='process_0.01':run_pets.OUT=OUT/'stress_process_0.01';run_pets.load_data=load_noisy
        for seed in range(3):
            model=run_pets.fit(seed,budget);dest=run_pets.OUT/'pets'/f'seed{seed}';identity=dict(fit=json.loads((dest/'identity.json').read_text()),model_sha256=digest(dest/'best.pt'),planner=planner,semantics='TS-infinity conditional sampled costs, native first-action MPC; unchanged port limitations');saved=np.load(ROOT/f'validation/results_v1/models/episodes3500_seed{seed}/data.npz');order=np.random.default_rng(seed).permutation(3500);scale=saved['X'][order[1050:],:-1].reshape(-1,4).std(0).clip(1e-6)
            for case in cases()[:56]:evaluate(seed,'PETS',case,condition,model,identity,scale,np.zeros(4),budget)
    dump(root/'completion.json',dict(state='complete' if not selection['untested'] else 'cap_limited',completed_conditions=selection['conditions'],untested_conditions=selection['untested'],accounting=budget.checkpoint()))


if __name__=='__main__':main()
