"""EDMD stress and affordable fixed operating-region sets after fresh holdout."""
import argparse,json,pickle
import numpy as np
from common import *
from edmd import EDMD,QP
from run_classical import select,model_grid,evaluate
from casebook import cases
from stress_protocol import CONDITIONS,REQUESTED,disturbances,mismatch_step


def noisy_fit(seed,budget):
    root=OUT/'edmd/stress'/f'seed{seed}/process_0.01';source=ROOT/f'validation/results_v1/E3/models/process_0.01/episodes3500_seed{seed}/data.npz';arrays=np.load(source);X,U=arrays['X'].astype(float),arrays['U'].astype(float);assert X.shape==(3500,31,4) and U.shape==(3500,30,1);order=np.random.default_rng(seed).permutation(3500);train=(X[order[1050:]],U[order[1050:]]);validation=(X[order[525:1050]],U[order[525:1050]]);scale=train[0][:,:-1].reshape(-1,4).std(0).clip(1e-8);identity=dict(data_sha256=digest(source),split_seed=seed,source=str(source.relative_to(ROOT)),candidate_grid=model_grid('edmd'),selection='same six-candidate train-standardized 30-step validation rollout MSE',training_noise='reuse exact E3 process-noise collection; held-out initial-state outcomes never used')
    path=root/'identity.json'
    if path.exists():assert json.loads(path.read_text())==identity
    else:dump(path,identity)
    scores=[]
    for index,config in enumerate(model_grid('edmd')):
        p=root/f'candidate{index}.json';m=root/f'candidate{index}.pkl'
        if p.exists() and m.exists():scores.append(json.loads(p.read_text()));continue
        budget.checkpoint();model=EDMD(**config).fit(*train);score=forecast_score(model,validation,scale);m.write_bytes(pickle.dumps(model));row=dict(index=index,config=config,validation_rollout_mse=score);dump(p,row);scores.append(row)
    selected=min([r for r in scores if r['validation_rollout_mse'] is not None],key=lambda r:(r['validation_rollout_mse'],r['index']));dump(root/'selection.json',selected);model=pickle.loads((root/f"candidate{selected['index']}.pkl").read_bytes());identity['selected']=selected;identity['model_sha256']=digest(root/f"candidate{selected['index']}.pkl");return model,identity


def stress(seed,arm,case,condition,model,identity,scale,gain,budget):
    dest=OUT/'edmd/stress';key=f"seed{seed}_{arm}_{condition}_{case['set']}_{case['index']:04d}_14s";p=dest/'cases'/f'{key}.json';noise=disturbances(seed,condition,case,scale);expected=dict(fit=identity,case=case,condition=condition,seed=seed,arm=arm,gain=gain.tolist(),source_sha256=digest(__file__),protocol_sha256=digest(Path(__file__).with_name('stress_protocol.py')),simulator_sha256=digest(ROOT/'src/inverted_pendulum.py'))
    if p.exists():row=json.loads(p.read_text());assert row['identity']==expected;return row
    solver=QP(model);step=mismatch_step();X=np.empty((701,4));U=np.empty((700,1));applied=np.empty_like(U);X[0]=case['state'];local=[];diagnostics=[];latencies=[];previous=0.;start=time.monotonic();cpu=cpu_seconds()
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
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['stress','maps'],required=True);args=parser.parse_args();isolation('edmd');budget=Budget('edmd',8);budget.checkpoint();assert (OUT/'preparation/audit_completion.json').exists()
    if args.phase=='stress':
        dump(OUT/'edmd/stress/protocol.json',dict(conditions=REQUESTED,source_e3_sha256=digest(ROOT/'validation/experiments/e3_robustness.py'),noise_key='exact E3 SeedSequence[20261009,seed,condition_index,case_index+1000 for C]',process_training='reuse E3 process_0.01 collection and same whole-episode split; EDMD six-candidate validation selection',initial_delay_force=0.,physical_delay_seconds=.02,full_reference_cohort='E3 H/L',simulator_for_evaluation_only=True))
    else:
        original=json.loads((ROOT/'validation/results_v1/initial_conditions.json').read_text())['sets'];requested=[dict(set='B',index=i,state=x,seconds=14.) for i,x in enumerate(original['B'])]+[dict(set='D',index=i*41+j,state=original['D'][i*41+j],seconds=14.) for i in range(0,41,2) for j in range(0,41,2)]
        dump(OUT/'edmd/maps_protocol.json',dict(B_count=171,D_count=441,D_indices=[c['index'] for c in requested if c['set']=='D'],initial_state_sha256=digest(ROOT/'validation/results_v1/initial_conditions.json'),no_interpolation=True))
    for seed in range(3):
        data,_=load_data(seed);original_arrays=np.load(ROOT/f'validation/results_v1/models/episodes3500_seed{seed}/data.npz');order=np.random.default_rng(seed).permutation(3500);scale=np.std(original_arrays['X'][order[1050:],:-1].reshape(-1,4),axis=0).clip(1e-6);model,identity,selected=select('edmd',seed,budget);gain=np.asarray(json.loads((ROOT/f'validation/results_v1/models/episodes3500_seed{seed}/diagnostics.json').read_text())['local']['gain'])
        if args.phase=='stress':
            noisy,noisy_identity=noisy_fit(seed,budget)
            # Match E3's independently fitted noisy-data L gain, not nominal L.
            ng=np.asarray(json.loads((ROOT/f'validation/results_v1/E3/models/process_0.01/episodes3500_seed{seed}/diagnostics.json').read_text())['local']['gain'])
            for condition in REQUESTED:
                m,ident,g=(noisy,noisy_identity,ng) if condition=='process_0.01' else (model,identity,gain)
                for arm in ['QP','QP+L']:
                    pilot=[]
                    for index,case in enumerate(cases()[:56]):
                        row=stress(seed,arm,case,condition,m,ident,scale,g,budget);pilot.append(row['cpu_seconds'])
                        if seed==0 and arm=='QP' and index==15:
                            projection=float(np.mean(pilot))*56*3*2*len(REQUESTED)/3600;remaining=8-budget.checkpoint()['cpu_seconds']/3600;dump(OUT/'edmd/stress'/f'pilot_projection_{condition}.json',dict(projected_core_hours=projection,remaining_core_hours=remaining));assert projection<=remaining,'EDMD stress cap-limited'
        else:
            for arm in ['QP','QP+L']:
                for case in requested:evaluate('edmd',seed,arm,model,identity,selected,case,budget)
    dump(OUT/'edmd'/f'{args.phase}_completion.json',dict(state='complete',accounting=budget.checkpoint()))


if __name__=='__main__':main()
