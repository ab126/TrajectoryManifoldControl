"""E4 H/L matched original/fresh states; E4-only weights, separate results cohort."""
import argparse,json,time
from types import SimpleNamespace
import numpy as np
import torch
from common import *
from neural_common import GPUBudget
from reference_transition import classes
from fresh_references import factory
from casebook import cases


def evaluate(seed,arm,case,decoder,config,gain,identity,budget):
    root=OUT/'rokdeepc/references';key=f"seed{seed}_{arm}_{case['set']}_{case['index']:04d}_{case['seconds']:g}s";p=root/'cases'/f'{key}.json';expected=dict(model=identity,arm=arm,seed=seed,case=case,adapter_sha256=digest(__file__))
    if p.exists():row=json.loads(p.read_text());assert row['identity']==expected;return row
    law=factory(decoder,config,gain) if arm=='H' else None;step=plant_interface();steps=round(case['seconds']/.02);X=np.empty((steps+1,4));U=np.empty((steps,1));X[0]=case['state'];start=time.monotonic();cpu=cpu_seconds()
    for k in range(steps):
        if k%25==0:budget.checkpoint()
        U[k]=law(k,X[k]) if law is not None else np.clip(-gain@X[k],-10,10)
        with np.errstate(over='ignore',invalid='ignore'):X[k+1]=step(X[k],U[k,0])
        if not np.isfinite(X[k+1]).all():X=X[:k+2];U=U[:k+1];break
    t=np.arange(len(X))*.02;tail=t>=case['seconds']-3.;finite=bool(np.isfinite(X).all());complete=len(U)==steps;no_fall=bool(finite and np.all(np.abs(X[:,2])<np.pi/2));balanced=bool(complete and finite and np.all(np.abs(X[tail,2])<=.02) and np.all(np.abs(X[tail,3])*np.sqrt(9.81)<=.05));raw=root/'raw'/f'{key}.npz';raw.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(raw,X=X,U=U,t=t)
    row=dict(case=key,identity=expected,seed=seed,arm=arm,index=case['index'],set=case['set'],seconds=case['seconds'],initial_state=case['state'],primary_success=balanced and no_fall,balanced=balanced,no_fall=no_fall,complete=complete,raw_path=str(raw.relative_to(ROOT)),cpu_seconds=cpu_seconds()-cpu,wall_seconds=time.monotonic()-start);dump(p,row);budget.checkpoint();return row


def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['pilot','matrix'],required=True);args=p.parse_args();isolation('deepkoopman');budget=GPUBudget('rokdeepc',12,2);budget.checkpoint();root=OUT/'rokdeepc/references';original=json.loads((ROOT/'validation/results_v1/initial_conditions.json').read_text())['sets']['C'];fresh=json.loads((OUT/'inventory/fresh_200_states.json').read_text())['states'];Transition,Decoder=classes()
    if args.phase=='matrix':
        projection=json.loads((root/'pilot_projection.json').read_text());assert projection['projected_gpu_hours']<=2-budget.checkpoint()['gpu_seconds']/3600,'E4 matched reference expansion cap-limited'
    for seed in ([0] if args.phase=='pilot' else range(3)):
        saved=ROOT/f'validation/results_v1/E4/models/seed{seed}';payload=torch.load(saved/'transition.pt',map_location='cuda',weights_only=False);model=Transition(4,1,payload['hidden_dims']).cuda();model.load_state_dict(payload['transition_state_dict']);model.eval();decoder=Decoder(model,30).cuda();diag=json.loads((saved/'diagnostics.json').read_text());config=SimpleNamespace(**{**diag['config'],'horizon':30,'device':'cuda'});gain=np.asarray(diag['local']['gain']);identity=dict(data_sha256=digest(saved/'data.npz'),checkpoint_sha256=digest(saved/'transition.pt'),gain_sha256=digest(saved/'diagnostics.json'),cohort='E4, 3281 x 32, exact saved E4 weights/gain',controller='unmodified source H five-step pending queue/common L; or L alone',controller_sha256=digest(ROOT/'scripts/run_inverted_pendulum_mc.py'),source_class_sha256=digest(ROOT/'src/manifold_control.py'),factory_adapter_sha256=digest(Path(__file__).with_name('fresh_references.py')),initial_state_sha256=digest(ROOT/'validation/results_v1/initial_conditions.json'),fresh_states_sha256=digest(OUT/'inventory/fresh_200_states.json'))
        requested=cases()[:16] if args.phase=='pilot' else cases()+[dict(set=name,index=i,state=x,seconds=14.) for name,states in [('C',original),('F',fresh)] for i,x in enumerate(states)]
        for arm in ['H','L']:
            for case in requested:evaluate(seed,arm,case,decoder,config,gain,identity,budget)
    if args.phase=='pilot':
        rows=[json.loads(p.read_text()) for p in (root/'cases').glob('*.json')];means={a:float(np.mean([r['wall_seconds'] for r in rows if r['arm']==a])) for a in ['H','L']};dump(root/'pilot_projection.json',dict(mean_case_seconds=means,projected_gpu_hours=sum(means.values())*(3*(6+400)+9*3000/700)/3600,remaining_gpu_hours=2-budget.checkpoint()['gpu_seconds']/3600))
    dump(root/f'{args.phase}_completion.json',dict(state='complete',accounting=budget.checkpoint()))


if __name__=='__main__':main()
