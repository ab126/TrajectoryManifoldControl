"""PETS frozen planner; ordered full seed/set cells only where timing fits."""
import argparse,json
import numpy as np
from common import *
from neural_common import GPUBudget
from run_pets import fit
from pets import TSInfinityCEM
from control import evaluate


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--set',choices=['fresh','original'],default='fresh');args=parser.parse_args()
    isolation('pets');budget=GPUBudget('pets',8,8);budget.checkpoint();assert (OUT/'pets/deterministic_reference/matrix_completion.json').exists();planner=json.loads((OUT/'pets/planner_selection.json').read_text());rows=[json.loads(p.read_text()) for p in (OUT/'pets/cases').glob('*.json')];timing=max(json.loads((OUT/'pets/pilot_projection.json').read_text())['mean_short_case_seconds'],float(np.mean([r['wall_seconds'] for r in rows if r['seconds']==14.])));sets=[('F',json.loads((OUT/'inventory/fresh_200_states.json').read_text())['states']),('C',json.loads((ROOT/'validation/results_v1/initial_conditions.json').read_text())['sets']['C'])];sets=[sets[0] if args.set=='fresh' else sets[1]];prefix='fresh' if args.set=='fresh' else 'original';path=OUT/'pets'/f'{prefix}_expansion_plan.json'
    if path.exists():plan=json.loads(path.read_text())
    else:
        remaining=8-budget.checkpoint()['gpu_seconds']/3600;cells=[]
        for name,_ in sets:
            for seed in range(3):
                cost=timing*200/3600
                if cost*1.1<=remaining:cells.append(dict(set=name,seed=seed,projected_gpu_hours=cost));remaining-=cost*1.1
        plan=dict(cells=cells,mean_case_seconds=timing,rule='ordered full seed/set cells with 10% margin using timings only; no planner/model changes; fresh, then stress, then original; this phase selects one set',source_sha256=digest(__file__));dump(path,plan)
    for cell in plan['cells']:
        seed=cell['seed'];name=cell['set'];model=fit(seed,budget);identity=dict(fit=json.loads((OUT/'pets'/f'seed{seed}/identity.json').read_text()),model_sha256=digest(OUT/'pets'/f'seed{seed}/best.pt'),planner=planner,semantics='TS-infinity, conditional particle sampled costs, native first-action MPC');factory=lambda:TSInfinityCEM(model,seed,planner['particles'],planner['population'])
        for index,state in enumerate(dict(sets)[name]):evaluate('pets',seed,'PETS',dict(set=name,index=index,state=state,seconds=14.),identity,factory,budget)
    dump(OUT/'pets'/f'{prefix}_expansion_completion.json',dict(state='complete' if len(plan['cells'])==3 else 'cap_limited',completed_seed_sets=plan['cells'],accounting=budget.checkpoint()))


if __name__=='__main__':main()
