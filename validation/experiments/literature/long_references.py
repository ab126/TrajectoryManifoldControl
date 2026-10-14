"""Regenerate only missing exact-state 60-second original H/L references."""
import json,time
from types import SimpleNamespace
import numpy as np
from common import *
from neural_common import GPUBudget
from reference_transition import saved_decoder
from fresh_references import factory


def main():
    isolation('deepkoopman');budget=GPUBudget('core_reserve',4,2);budget.checkpoint()
    root=OUT/'long_references';original=json.loads((ROOT/'validation/results_v1/initial_conditions.json').read_text())['sets'];step=plant_interface()
    pilot=json.loads((OUT/'fresh_references/pilot_projection.json').read_text());projected=sum(pilot['mean_case_seconds'].values())*3*(3000/700)/3600
    remaining=2-budget.checkpoint()['gpu_seconds']/3600
    dump(root/'pilot_projection.json',dict(projected_gpu_hours=projected,remaining_gpu_hours=remaining,origin='matched original H/L 16-case fresh-reference pilot, conservatively scaled to six missing 60-second cases'))
    assert projected<=remaining,'Missing long references project beyond stage cap; cap-limited'
    for seed in range(3):
        decoder=saved_decoder(seed);saved=ROOT/f'validation/results_v1/models/episodes3500_seed{seed}';diag=json.loads((saved/'diagnostics.json').read_text());config=SimpleNamespace(**{**diag['config'],'device':'cuda'});gain=np.asarray(diag['local']['gain'])
        identity=dict(data_sha256=digest(saved/'data.npz'),checkpoint_sha256=digest(saved/'transition.pt'),gain_sha256=digest(saved/'diagnostics.json'),controller_sha256=digest(ROOT/'scripts/run_inverted_pendulum_mc.py'),class_sha256=digest(ROOT/'src/manifold_control.py'),adapter_sha256=digest(__file__),factory_sha256=digest(Path(__file__).with_name('fresh_references.py')),simulator_sha256=digest(ROOT/'src/inverted_pendulum.py'),interpreter='/tmp/trajectory-literature-envs/deepkoopman/bin/python')
        for i in [0,2,5]:
            state=original['A'][i];indices=[j for j,x in enumerate(original['B']) if x==state]
            for arm in ['H','L']:
                existing=[ROOT/f'validation/results_v1/E1/cases/seed{seed}_{arm}_B_{j:04d}_60s.json' for j in indices]
                if any(p.exists() for p in existing):
                    for p in existing:
                        if p.exists():
                            row=json.loads(p.read_text());assert row['initial_state']==state and row['seconds']==60. and row['seed']==seed and row['episodes']==3500 and row['arm']==arm
                    continue
                key=f'seed{seed}_{arm}_A_{i:04d}_60s';p=root/'cases'/f'{key}.json';expected=dict(model=identity,seed=seed,arm=arm,index=i,set='A',seconds=60.,initial_state=state)
                if p.exists():assert json.loads(p.read_text())['identity']==expected;continue
                law=factory(decoder,config,gain) if arm=='H' else None;X=np.empty((3001,4));U=np.empty((3000,1));X[0]=state;start=time.monotonic();cpu=cpu_seconds()
                for k in range(3000):
                    if k%25==0:budget.checkpoint()
                    U[k]=law(k,X[k]) if law is not None else np.clip(-gain@X[k],-10,10)
                    with np.errstate(over='ignore',invalid='ignore'):X[k+1]=step(X[k],U[k,0])
                    if not np.isfinite(X[k+1]).all():X=X[:k+2];U=U[:k+1];break
                t=np.arange(len(X))*.02;tail=t>=57.;finite=bool(np.isfinite(X).all() and np.isfinite(U).all());complete=len(U)==3000;no_fall=bool(finite and np.all(np.abs(X[:,2])<np.pi/2));balanced=bool(complete and finite and np.all(np.abs(X[tail,2])<=.02) and np.all(np.abs(X[tail,3])*np.sqrt(9.81)<=.05));raw=root/'raw'/f'{key}.npz';raw.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(raw,X=X,U=U,t=t)
                row=dict(case=key,identity=expected,seed=seed,arm=arm,index=i,set='A',seconds=60.,initial_state=state,balanced=balanced,no_fall=no_fall,primary_success=balanced and no_fall,complete=complete,nonfinite_state=not finite,cpu_seconds=cpu_seconds()-cpu,wall_seconds=time.monotonic()-start,raw_path=str(raw.relative_to(ROOT)));dump(p,row);budget.checkpoint()
    dump(root/'completion.json',dict(state='complete',policy='Exact-state existing B-grid references preserved; only missing A0/A2/A5 60-second H/L records regenerated',accounting=budget.checkpoint()))


if __name__=='__main__':main()
