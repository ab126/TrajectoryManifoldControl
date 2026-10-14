"""Timing-bounded full-state stress cells; exact E3 scales/indexing and applied u."""
import argparse,json,time
import numpy as np
import torch
import run_neural,run_neural_fast,control
from common import *
from neural_common import GPUBudget
from run_output_safe import CPUBudget
from stress_protocol import REQUESTED,CONDITIONS,disturbances,mismatch_step
from casebook import cases
from neural_noise_data import load as load_noisy


def plan(method,budget):
    stage=method+'_control';path=OUT/stage/'stress/plan.json'
    if path.exists():return json.loads(path.read_text())
    arms=['MPC','MPC+L']+(['filtered_L'] if method=='dfine' else []);pilot=json.loads((OUT/stage/'pilot_projection.json').read_text())['mean_case_seconds'];means={a:pilot[a] for a in arms};gpu_cap=4 if method=='deepkoopman' else 6;remaining=gpu_cap-budget.checkpoint().get('gpu_seconds',0.)/3600;per_condition=sum(means.values())*168/3600;chosen=[]
    for group in [['input_delay'],['mass_minus25','mass_plus25'],['length_minus25','length_plus25'],['measurement_0.01'],['process_0.01']]:
        if per_condition*len(group)*1.1<=remaining:chosen+=group;remaining-=per_condition*len(group)*1.1
    result=dict(requested=REQUESTED,conditions=chosen,untested=[c for c in REQUESTED if c not in chosen],arms=arms,mean_case_wall_seconds=means,projected_gpu_hours_per_condition=per_condition,rule='ordered full three-seed/56-state conditions, paired mismatches kept together, 10% margin; timings only',delay='uncompensated frozen planner; causal filter receives previous actually applied force',source_sha256=digest(__file__),protocol_sha256=digest(Path(__file__).with_name('stress_protocol.py')));dump(path,result);return result


def configure_noisy():
    run_neural.OUT=OUT/'stress_process_0.01';run_neural.load_data=load_noisy


def evaluate(method,seed,arm,case,condition,model,identity,gain,scale,budget):
    root=OUT/(method+'_control')/'stress';key=f"seed{seed}_{arm}_{condition}_{case['set']}_{case['index']:04d}_14s";p=root/'cases'/f'{key}.json';expected=dict(model=identity,seed=seed,arm=arm,case=case,condition=condition,gain=gain.tolist(),source_sha256=digest(__file__),simulator_sha256=digest(ROOT/'src/inverted_pendulum.py'));noise=disturbances(seed,condition,case,scale)
    if p.exists():row=json.loads(p.read_text());assert row['identity']==expected;return row
    filtered=method=='dfine'
    if filtered:model.reset()
    def forecast(x,u):
        z=model.current_mean.expand(len(u),-1);out=[model.decode_dynamic(z)]
        for k in range(30):z=z@model.source.ldm.A.T+u[:,k]@model.B.T;out.append(model.decode_dynamic(z))
        return torch.stack(out,1)
    solver=run_neural_fast.VectorizedCEM(forecast if filtered else model.rollout,model.scale,seed);step=mismatch_step();X=np.empty((701,4));U=np.empty((700,1));applied=np.empty_like(U);X[0]=case['state'];rec=[];local=[];logs=[];latencies=[];start=time.monotonic();cpu=cpu_seconds();previous=0.
    for k in range(700):
        if k%25==0:budget.checkpoint()
        observed=X[k]+noise[k] if CONDITIONS[condition]['mode']=='measurement' else X[k];estimate=model.observe(observed,None if k==0 else applied[k-1]) if filtered else observed;rec.append(estimate);catch=arm=='filtered_L' or (arm.endswith('+L') and estimate@Q@estimate<.75);local.append(catch)
        if not np.isfinite(estimate).all():u=np.zeros(1);logs.append(dict(accepted=False,status='nonfinite causal state estimate'))
        elif catch:u=np.clip(-gain@estimate,-10,10)
        else:before=time.monotonic();proposal,info=solver.solve(estimate);latencies.append(time.monotonic()-before);logs.append(info);u=proposal[0]
        U[k]=np.clip(u,-10,10);applied[k]=previous if condition=='input_delay' else U[k];previous=float(U[k,0])
        with np.errstate(over='ignore',invalid='ignore'):X[k+1]=step(X[k],applied[k,0],condition)+(noise[k] if CONDITIONS[condition]['mode']=='process' else 0.)
        if not np.isfinite(X[k+1]).all():X=X[:k+2];U=U[:k+1];applied=applied[:k+1];break
    t=np.arange(len(X))*.02;tail=t>=11.;complete=len(U)==700;finite=bool(np.isfinite(X).all() and np.isfinite(U).all());no_fall=bool(finite and np.all(np.abs(X[:,2])<np.pi/2));balanced=bool(complete and finite and np.all(np.abs(X[tail,2])<=.02) and np.all(np.abs(X[tail,3])*np.sqrt(9.81)<=.05));raw=root/'raw'/f'{key}.npz';raw.parent.mkdir(exist_ok=True);np.savez_compressed(raw,X=X,U=U,U_applied=applied,t=t,noise=noise,reconstructed=rec,local=local)
    def number(v):return float(v) if np.isfinite(v) else None
    row=dict(case=key,identity=expected,seed=seed,arm=arm,set=case['set'],index=case['index'],condition=condition,seconds=14.,primary_success=balanced and no_fall,balanced=balanced,no_fall=no_fall,complete=complete,nonfinite_state=not finite,tail_angle_rms=number(np.sqrt(np.mean(X[tail,2]**2))),tail_angular_velocity_rms_physical=number(np.sqrt(np.mean((X[tail,3]*np.sqrt(9.81))**2))),noise_std_normalized=(scale*CONDITIONS[condition].get('level',0.)).tolist(),noise_std_physical=(scale*CONDITIONS[condition].get('level',0.)*np.array([1,np.sqrt(9.81),1,np.sqrt(9.81)])).tolist(),solver_failures=sum(not r['accepted'] for r in logs),local_fraction=float(np.mean(local)),solve_latency_quantiles=np.quantile(latencies,[.5,.95,1.]).tolist() if latencies else [],cpu_seconds=cpu_seconds()-cpu,wall_seconds=time.monotonic()-start,raw_path=str(raw.relative_to(ROOT)));dump(root/'solver_logs'/f'{key}.json',logs);dump(p,row);budget.checkpoint();return row


def main():
    p=argparse.ArgumentParser();p.add_argument('method',choices=['deepkoopman','dfine']);p.add_argument('--phase',choices=['fit','matrix'],required=True);args=p.parse_args();method=args.method;stage=method+'_control';isolation(method);cpu_cap=8 if method=='deepkoopman' else 12;gpu_cap=4 if method=='deepkoopman' else 6
    if args.phase=='fit':torch.set_num_threads(1);torch.set_num_interop_threads(1);budget=CPUBudget(stage,cpu_cap)
    else:budget=GPUBudget(stage,cpu_cap,gpu_cap)
    budget.checkpoint();selection=plan(method,budget)
    if args.phase=='fit':
        if 'process_0.01' in selection['conditions']:
            configure_noisy()
            for seed in range(3):run_neural.fit_seed(method,'control',seed,budget,'cpu')
        dump(OUT/stage/'stress/fit_completion.json',dict(state='complete',process_noise_models_fitted='process_0.01' in selection['conditions'],accounting=budget.checkpoint()));return
    for condition in selection['conditions']:
        if condition=='process_0.01':configure_noisy()
        for seed in range(3):
            model,identity=run_neural_fast.fit(method,'control',seed,budget,'cuda');arrays=np.load(ROOT/f'validation/results_v1/models/episodes3500_seed{seed}/data.npz');order=np.random.default_rng(seed).permutation(3500);scale=arrays['X'][order[1050:],:-1].reshape(-1,4).std(0).clip(1e-6);gainpath=ROOT/(f'validation/results_v1/E3/models/process_0.01/episodes3500_seed{seed}/diagnostics.json' if condition=='process_0.01' else f'validation/results_v1/models/episodes3500_seed{seed}/diagnostics.json');gain=np.asarray(json.loads(gainpath.read_text())['local']['gain']);identity['gain_sha256']=digest(gainpath)
            for arm in selection['arms']:
                for case in cases()[:56]:evaluate(method,seed,arm,case,condition,model,identity,gain,scale,budget)
    dump(OUT/stage/'stress/completion.json',dict(state='complete' if not selection['untested'] else 'cap_limited',completed_conditions=selection['conditions'],untested_conditions=selection['untested'],accounting=budget.checkpoint()))


if __name__=='__main__':main()
