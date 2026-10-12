"""One prescribed SMC design on saved 200 states; original E6 artifacts read-only."""
import importlib.util
import json
import time
import numpy as np
from common import *


def main():
    isolation('smc');budget=Budget('smc_C200',1);budget.checkpoint()
    source=ROOT/'validation/experiments/smc_baseline.py'
    spec=importlib.util.spec_from_file_location('saved_smc',source);smc=importlib.util.module_from_spec(spec);spec.loader.exec_module(smc)
    states=json.loads((ROOT/'validation/results_v1/initial_conditions.json').read_text())['sets']['C']
    dest=OUT/'smc_C200';dest.mkdir(exist_ok=True);step=plant_interface();manifest=[]
    all_rows=[]
    for seed in range(3):
        data,identity=load_data(seed)
        path=ROOT/f'validation/results_v1/models/episodes3500_seed{seed}/diagnostics.json'
        saved=json.loads(path.read_text());A,B=np.array(saved['local']['A']),np.array(saved['local']['B'])
        C,P,equivalent=smc.surface(A,B,Q,np.array([[.03]]))
        calibration=smc.calibrate(A,B,C,data['train'],data['validation'])
        reference_model=ROOT/f'validation/results_v1/E6_SMC/models/nominal_seed{seed}.json'
        original=json.loads(reference_model.read_text())
        assert np.array_equal(original['A'],A) and np.array_equal(original['B'],B)
        assert np.allclose(original['C'],C,rtol=1e-8,atol=1e-8)
        assert np.isclose(original['calibration']['phi'],calibration['phi'],rtol=1e-8,atol=1e-8)
        # Use the exact saved surface/calibration; do not refit this reference.
        C=np.array(original['C']);calibration=original['calibration']
        law=smc.SlidingFeedback(A,C,calibration['phi'],correction=True,rho=.9,bound=10.)
        model_identity=dict(data_sha256=identity['sha256'],diagnostics_sha256=digest(path),source_sha256=digest(source),
                            A=A.tolist(),B=B.tolist(),C=C.tolist(),phi=calibration['phi'],rho=.9,beta=.45*calibration['phi'],
                            input_bound=10.,physical_dt=.02,normalized_dt=DT,endpoint='last 3 s: |theta|<=.02, physical |omega|<=.05, no |theta|>=pi/2',
                            observation='full measured state',nominal=True,seed=seed)
        manifest.append(model_identity)
        for index,state in enumerate(states):
            budget.checkpoint();key=f'seed{seed}_SMC_C_{index:04d}_14s';p=dest/'cases'/f'{key}.json'
            if p.exists():
                row=json.loads(p.read_text());assert row['identity']==model_identity and row['initial_state']==state;all_rows.append(row);continue
            prior=ROOT/f'validation/results_v1/E6_SMC/cases/seed{seed}_SMC_nominal_C50_{index:04d}_14s.json'
            start=time.monotonic();cpu=cpu_seconds()
            if prior.exists():
                original_row=json.loads(prior.read_text());assert original_row['initial_state']==state and original_row['seed']==seed and original_row['arm']=='SMC' and original_row['seconds']==14.
                assert original_row['smc_recorded'] and abs(original_row['phi']-calibration['phi'])<1e-12 and abs(original_row['beta']-.45*calibration['phi'])<1e-12
                rawpath=ROOT/original_row['raw_path'];arrays=np.load(rawpath);X,U,t=arrays['X'],arrays['U'],arrays['t']
                assert X.shape==(701,4) and U.shape==(700,1) and np.array_equal(X[0],state)
                assert np.allclose(t,np.arange(701)*.02,rtol=0,atol=1e-12)
                # Verify the saved prescribed law on every measured state and replay samples.
                commands=np.array([law.command(x)[0] for x in X[:-1]])
                assert np.allclose(commands,U,rtol=0,atol=1e-10,equal_nan=False)
                for k in [0,349,699]:assert np.allclose(step(X[k],U[k,0]),X[k+1],rtol=0,atol=1e-9)
                reused=True;raw=str(rawpath.relative_to(ROOT))
            else:
                X=np.empty((701,4));U=np.empty((700,1));X[0]=state
                for k in range(700):
                    U[k]=law.command(X[k])[0]
                    with np.errstate(over='ignore',invalid='ignore'):X[k+1]=step(X[k],U[k,0])
                t=np.arange(701)*.02;rawpath=dest/'raw'/f'{key}.npz';rawpath.parent.mkdir(exist_ok=True);np.savez_compressed(rawpath,t=t,X=X,U=U)
                raw=str(rawpath.relative_to(ROOT));reused=False
            finite=bool(np.isfinite(X).all() and np.isfinite(U).all());tail=t>=11.
            balanced=bool(finite and np.all(np.abs(X[tail,2])<=.02) and np.all(np.abs(X[tail,3])*np.sqrt(9.81)<=.05));no_fall=bool(finite and np.all(np.abs(X[:,2])<np.pi/2))
            row=dict(case=key,seed=seed,index=index,arm='SMC',seconds=14.,initial_state=state,identity=model_identity,
                     balanced=balanced,no_fall=no_fall,primary_success=balanced and no_fall,fell=not no_fall,
                     raw_path=raw,reused_reference=reused,source_case=str(prior.relative_to(ROOT)) if reused else None,
                     wall_seconds=time.monotonic()-start,cpu_seconds=cpu_seconds()-cpu)
            if reused:assert row['primary_success']==original_row['primary_success']
            dump(p,row);all_rows.append(row)
            if index==5 and seed==0:
                mean=np.mean([r['cpu_seconds'] for r in all_rows]);projection=mean*600/3600
                dump(dest/'pilot_projection.json',dict(projected_core_hours=projection,mean_case_cpu_seconds=float(mean),ceiling_hours=1))
                if projection>1:raise RuntimeError('SMC projection exceeds 1 core-hour; preserve partial')
            if index%25==0:print(json.dumps(dict(seed=seed,index=index,success=row['primary_success'],reused=reused)),flush=True)
    dump(dest/'manifest.json',dict(models=manifest,state_sha256=digest(ROOT/'validation/results_v1/initial_conditions.json'),count_per_seed=200,source='fixed E6 SMC, correction enabled',core_reserve_hours_ceiling=1,adapter_sha256=digest(__file__)))
    summaries=[]
    for seed in range(3):
        rows=[r for r in all_rows if r['seed']==seed];assert len(rows)==200 and len({r['index'] for r in rows})==200
        item=dict(seed=seed,n=200,successes=sum(r['primary_success'] for r in rows),falls=sum(r['fell'] for r in rows),paired={})
        for arm in ['H','L']:
            refs=[]
            for index,state in enumerate(states):
                p=ROOT/f'validation/results_v1/E1/cases/seed{seed}_{arm}_C_{index:04d}_14s.json';r=json.loads(p.read_text());assert r['initial_state']==state and r['seed']==seed and r['seconds']==14.;refs.append(r)
            item['paired'][arm]=dict(n=200,reference_successes=sum(r['primary_success'] for r in refs),reference_only=sum(a['primary_success'] and not b['primary_success'] for a,b in zip(refs,rows)),smc_only=sum(b['primary_success'] and not a['primary_success'] for a,b in zip(refs,rows)))
        summaries.append(item)
    dump(dest/'summary.json',summaries);dump(dest/'completion.json',dict(state='complete',cases=600,accounting=budget.checkpoint()))


if __name__=='__main__':main()
