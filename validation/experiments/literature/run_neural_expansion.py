"""Timing-only bounded selection of full fresh/original 200-state latent sets."""
import argparse,json,math
import numpy as np
import torch
import control
import run_neural_fast
from common import *
from neural_common import GPUBudget


def main():
    parser=argparse.ArgumentParser();parser.add_argument('method',choices=['deepkoopman','dfine']);parser.add_argument('--set',choices=['fresh','original'],default='fresh');args=parser.parse_args();method=args.method;stage=method+'_control';isolation(method);gpu_hours=4 if method=='deepkoopman' else 6;budget=GPUBudget(stage,8 if method=='deepkoopman' else 12,gpu_hours);budget.checkpoint();assert (OUT/stage/'matrix_completion.json').exists();control.TorchCEM=run_neural_fast.VectorizedCEM
    arms=['MPC','MPC+L']+(['filtered_L'] if method=='dfine' else []);pilot=json.loads((OUT/stage/'pilot_projection.json').read_text())['mean_case_seconds'];rows=[json.loads(p.read_text()) for p in (OUT/stage/'cases').glob('*.json')];means={arm:max(pilot[arm],float(np.mean([r['wall_seconds'] for r in rows if r['arm']==arm and r['seconds']==14.]))) for arm in arms};sets=[('F',json.loads((OUT/'inventory/fresh_200_states.json').read_text())['states']),('C',json.loads((ROOT/'validation/results_v1/initial_conditions.json').read_text())['sets']['C'])]
    sets=[item for item in sets if item[0]==('F' if args.set=='fresh' else 'C')]
    plan=OUT/stage/f'{args.set}_expansion_plan.json'
    if plan.exists():selected=json.loads(plan.read_text())
    else:
        remaining=gpu_hours-budget.checkpoint()['gpu_seconds']/3600;alloc=[]
        for name,states in sets:
            for seed in range(3):
                # Conservative full-set projection includes any previously tested C50.
                cost=sum(means.values())*200/3600
                if cost*1.1<=remaining:alloc.append(dict(set=name,seed=seed,projected_gpu_hours=cost));remaining-=cost*1.1
        selected=dict(sets=alloc,priority='fresh, then stress, then original; seed order 0,1,2',timing_seconds=means,rule='largest ordered full seed/set cells fitting remaining stage allocation with 10% timing margin; no outcome-based state selection',requested_states_per_cell=200,adapter_sha256=digest(__file__));dump(plan,selected)
    for cell in selected['sets']:
        seed=cell['seed'];name=cell['set'];states=dict(sets)[name];model,identity=run_neural_fast.fit(method,'control',seed,budget,'cuda')
        def factory():
            if method=='deepkoopman':forecast=model.rollout
            else:
                def forecast(x,u):
                    z=model.current_mean.expand(len(u),-1);out=[model.decode_dynamic(z)]
                    for k in range(30):z=z@model.source.ldm.A.T+u[:,k]@model.B.T;out.append(model.decode_dynamic(z))
                    return torch.stack(out,1)
            return control.TorchCEM(forecast,model.scale,seed)
        for arm in arms:
            for index,state in enumerate(states):control.evaluate(stage,seed,arm,dict(set=name,index=index,state=state,seconds=14.),identity,factory,budget,model if method=='dfine' else None)
    dump(OUT/stage/f'{args.set}_expansion_completion.json',dict(state='complete' if len(selected['sets'])==3 else 'cap_limited',completed_seed_sets=selected['sets'],untested_seed_sets=[dict(set=n,seed=s) for n,_ in sets for s in range(3) if not any(c['set']==n and c['seed']==s for c in selected['sets'])],accounting=budget.checkpoint()))


if __name__=='__main__':main()
