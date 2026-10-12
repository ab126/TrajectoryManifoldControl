"""Approved original/fresh 200-state EDMD sets; same frozen fits and controllers."""
import json
import numpy as np
from common import *
from run_classical import select,evaluate


def main():
    isolation('edmd');budget=Budget('edmd',8);budget.checkpoint()
    assert json.loads((OUT/'edmd/matrix_completion.json').read_text())['state']=='complete'
    original=json.loads((ROOT/'validation/results_v1/initial_conditions.json').read_text())['sets']['C']
    fresh=np.random.default_rng(20261012).uniform([-1.5,-2.5,-1.3,-3.5],[1.5,2.5,1.3,3.5],(200,4)).tolist()
    frozen=OUT/'inventory/fresh_200_states.json'
    value=dict(seed=20261012,states=fresh,low=[-1.5,-2.5,-1.3,-3.5],high=[1.5,2.5,1.3,3.5])
    if frozen.exists():assert json.loads(frozen.read_text())==value
    else:dump(frozen,value)
    protocol=dict(source_sha256=digest(__file__),sets=dict(C=dict(n=200,source='results_v1/initial_conditions.json:sets.C'),F=dict(n=200,source='inventory/fresh_200_states.json')),fitting_and_selection_unchanged=True)
    dump(OUT/'edmd/expansion_manifest.json',protocol)
    for seed in range(3):
        model,identity,chosen=select('edmd',seed,budget)
        for arm in ['QP','QP+L']:
            for name,states in [('C',original),('F',fresh)]:
                for index,state in enumerate(states):evaluate('edmd',seed,arm,model,identity,chosen,dict(set=name,index=index,state=state,seconds=14.),budget)
    dump(OUT/'edmd/expansion_completion.json',dict(state='complete',accounting=budget.checkpoint()))


if __name__=='__main__':main()
