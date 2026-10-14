"""Full timing-feasible SINDYc stress groups; unchanged learned-map gradients."""
import json,time,pickle
import numpy as np
import sindy,run_classical,run_sindy_compiled
from common import *
from sindy_gradient import ExactGradientNMPC,cost_gradient
from stress_protocol import CONDITIONS,REQUESTED,disturbances,mismatch_step
from casebook import cases
from neural_noise_data import load as load_noisy


def noisy_fit(seed,budget):
    data,identity=load_noisy(seed);root=OUT/'sindy/stress/process_0.01'/f'seed{seed}';identity.update(grid=run_classical.model_grid('sindy'),source_sha256=digest(__file__),selection='same training-standardized 30-step validation rollout MSE');p=root/'identity.json'
    if p.exists():assert json.loads(p.read_text())==identity
    else:dump(p,identity)
    X,U=data['train'];scale=X[:,:-1].reshape(-1,4).std(0).clip(1e-8);rows=[]
    for index,config in enumerate(run_classical.model_grid('sindy')):
        p=root/f'candidate{index}.json';path=root/f'candidate{index}.pkl'
        if p.exists() and path.exists():rows.append(json.loads(p.read_text()));continue
        budget.checkpoint();model=sindy.SINDYc(**config).fit(X,U);value=forecast_score(model,data['validation'],scale);path.write_bytes(pickle.dumps(model));row=dict(index=index,config=config,validation_rollout_mse=value);dump(p,row);rows.append(row)
    selected=min([r for r in rows if r['validation_rollout_mse'] is not None],key=lambda r:(r['validation_rollout_mse'],r['index']));dump(root/'selection.json',selected);path=root/f"candidate{selected['index']}.pkl";identity['model_sha256']=digest(path);identity['selected']=selected;return pickle.loads(path.read_bytes()),identity


def evaluate(seed,arm,case,condition,model,identity,scale,gain,budget):
    root=OUT/'sindy/stress';key=f"seed{seed}_{arm}_{condition}_{case['set']}_{case['index']:04d}_14s";p=root/'cases'/f'{key}.json';expected=dict(fit=identity,case=case,seed=seed,arm=arm,condition=condition,source_sha256=digest(__file__),gain=gain.tolist(),simulator_sha256=digest(ROOT/'src/inverted_pendulum.py'));noise=disturbances(seed,condition,case,scale)
    if p.exists():r=json.loads(p.read_text());assert r['identity']==expected;return r
    solver=ExactGradientNMPC(model) if arm.startswith('NMPC') else sindy.CEM(model,seed);step=mismatch_step();X=np.empty((701,4));U=np.empty((700,1));applied=np.empty_like(U);X[0]=case['state'];previous=0.;local=[];logs=[];latencies=[];start=time.monotonic();cpu=cpu_seconds()
    for k in range(700):
        if k%25==0:budget.checkpoint()
        observed=X[k]+noise[k] if CONDITIONS[condition]['mode']=='measurement' else X[k];catch=arm.endswith('+L') and observed@Q@observed<.75;local.append(catch)
        if catch:u=np.clip(-gain@observed,-10,10)
        else:before=time.monotonic();proposal,info=solver.solve(observed);latencies.append(time.monotonic()-before);logs.append(info);u=proposal[0]
        U[k]=np.clip(u,-10,10);applied[k]=previous if condition=='input_delay' else U[k];previous=float(U[k,0])
        with np.errstate(over='ignore',invalid='ignore'):X[k+1]=step(X[k],applied[k,0],condition)+(noise[k] if CONDITIONS[condition]['mode']=='process' else 0.)
        if not np.isfinite(X[k+1]).all():X=X[:k+2];U=U[:k+1];applied=applied[:k+1];break
    t=np.arange(len(X))*.02;tail=t>=11.;finite=bool(np.isfinite(X).all());complete=len(U)==700;no_fall=bool(finite and np.all(np.abs(X[:,2])<np.pi/2));balanced=bool(complete and finite and np.all(np.abs(X[tail,2])<=.02) and np.all(np.abs(X[tail,3])*np.sqrt(9.81)<=.05));raw=root/'raw'/f'{key}.npz';raw.parent.mkdir(exist_ok=True);np.savez_compressed(raw,X=X,U=U,U_applied=applied,t=t,noise=noise,local=local)
    def number(v):return float(v) if np.isfinite(v) else None
    row=dict(case=key,identity=expected,seed=seed,arm=arm,set=case['set'],index=case['index'],condition=condition,seconds=14.,complete=complete,primary_success=balanced and no_fall,balanced=balanced,no_fall=no_fall,nonfinite_state=not finite,solver_failures=sum(not x['accepted'] for x in logs),tail_angle_rms=number(np.sqrt(np.mean(X[tail,2]**2))),tail_angular_velocity_rms_physical=number(np.sqrt(np.mean((X[tail,3]*np.sqrt(9.81))**2))),noise_std_normalized=(scale*CONDITIONS[condition].get("level",0.)).tolist(),noise_std_physical=(scale*CONDITIONS[condition].get("level",0.)*np.array([1,np.sqrt(9.81),1,np.sqrt(9.81)])).tolist(),local_fraction=float(np.mean(local)),input_rms=number(np.sqrt(np.mean(U**2))),input_saturation_fraction=float(np.mean(np.abs(U)>=9.999)),solve_latency_quantiles=np.quantile(latencies,[.5,.95,1.]).tolist() if latencies else [],cpu_seconds=cpu_seconds()-cpu,wall_seconds=time.monotonic()-start,raw_path=str(raw.relative_to(ROOT)));dump(root/'solver_logs'/f'{key}.json',logs);dump(p,row);budget.checkpoint();return row


def main():
    isolation('sindy');budget=Budget('sindy',12);budget.checkpoint();sindy.SINDYc.predict_inputs=run_sindy_compiled.predict_inputs;run_classical.code_identity=run_sindy_compiled.identity;root=OUT/'sindy/stress';p=root/'plan.json';arms=['CEM','CEM+L','NMPC_exact','NMPC_exact+L']
    if p.exists():plan=json.loads(p.read_text())
    else:
        timings=json.loads((OUT/'sindy/exact_pilot_projection.json').read_text())['mean_case_cpu_seconds'];cost=sum(timings.values())*168/3600;remaining=12-budget.checkpoint()['cpu_seconds']/3600;conditions=[]
        for group in [['input_delay'],['mass_minus25','mass_plus25'],['length_minus25','length_plus25'],['measurement_0.01'],['process_0.01']]:
            if cost*len(group)*1.1<=remaining:conditions+=group;remaining-=cost*len(group)*1.1
        plan=dict(conditions=conditions,requested=REQUESTED,untested=[c for c in REQUESTED if c not in conditions],projected_core_hours_per_condition=cost,rule='ordered whole three-seed/56-state conditions, paired mismatches together, 10% timing margin; no outcome selection',process_noise='same twelve-candidate noisy-data fit on exact E3 collection, not nominal-model substitution',source_sha256=digest(__file__));dump(p,plan)
    for seed in range(3):
        model,identity,selected=run_classical.select('sindy',seed,budget);cost_gradient(np.zeros(4),np.zeros(30),model.library.powers_,model.coef,model.trig);saved=np.load(ROOT/f'validation/results_v1/models/episodes3500_seed{seed}/data.npz');order=np.random.default_rng(seed).permutation(3500);scale=saved['X'][order[1050:],:-1].reshape(-1,4).std(0).clip(1e-6);gain=np.asarray(json.loads((ROOT/f'validation/results_v1/models/episodes3500_seed{seed}/diagnostics.json').read_text())['local']['gain']);identity['selected_model_sha256']=digest(OUT/'sindy'/f"seed{seed}/candidate{selected['index']}.pkl")
        for condition in plan['conditions']:
            actual,actual_identity,actual_gain=model,identity,gain
            if condition=='process_0.01':
                actual,actual_identity=noisy_fit(seed,budget);actual_gain=np.asarray(json.loads((ROOT/f'validation/results_v1/E3/models/process_0.01/episodes3500_seed{seed}/diagnostics.json').read_text())['local']['gain']);cost_gradient(np.zeros(4),np.zeros(30),actual.library.powers_,actual.coef,actual.trig)
            for arm in arms:
                for case in cases()[:56]:evaluate(seed,arm,case,condition,actual,actual_identity,scale,actual_gain,budget)
    dump(root/'completion.json',dict(state='complete' if not plan['untested'] else 'partial',completed_conditions=plan['conditions'],untested_conditions=plan['untested'],accounting=budget.checkpoint()))


if __name__=='__main__':main()
