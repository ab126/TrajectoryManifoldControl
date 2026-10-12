"""Frozen SINDYc model/search with independently checked exact learned-map gradients."""
import argparse
import copy
import json
import numpy as np
import sindy
import run_classical
import run_sindy_compiled
from sindy_gradient import ExactGradientNMPC,cost_gradient
from common import *


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['pilot','matrix'],required=True);args=parser.parse_args()
    isolation('sindy');budget=Budget('sindy',12);budget.checkpoint()
    assert json.loads((OUT/'sindy/gradient_parity.json').read_text())['passed']
    sindy.SINDYc.predict_inputs=run_sindy_compiled.predict_inputs;sindy.NMPC=ExactGradientNMPC
    run_classical.code_identity=run_sindy_compiled.identity
    arms=['CEM','CEM+L','NMPC_exact','NMPC_exact+L']
    for seed in ([0] if args.phase=='pilot' else range(3)):
        model,identity,selected=run_classical.select('sindy',seed,budget)
        cost_gradient(np.zeros(4),np.zeros(30),model.library.powers_,model.coef,model.trig)
        if args.phase=='matrix':
            projection=json.loads((OUT/'sindy/exact_pilot_projection.json').read_text())
            if projection['projected_core_hours']>projection['remaining_stage_core_hours']:raise RuntimeError('Exact-gradient minimum matrix exceeds stage cap; retain pilot partial')
        for arm in arms:
            actual=copy.deepcopy(identity)
            if arm.startswith('NMPC'):
                actual['optimizer_backend']=dict(gradient='exact sensitivities of fitted coefficients',source_sha256=digest(Path(__file__).with_name('sindy_gradient.py')),driver_sha256=digest(__file__),maximum_iterations=50,per_solve_wall_seconds=2.)
            for case in (run_classical.cases()[:16] if args.phase=='pilot' else run_classical.cases()):run_classical.evaluate('sindy',seed,arm,model,actual,selected,case,budget)
    if args.phase=='pilot':
        rows=[json.loads(p.read_text()) for p in (OUT/'sindy/cases').glob('*.json')];times={arm:float(np.mean([r['cpu_seconds'] for r in rows if r['arm']==arm])) for arm in arms}
        projection=sum(times.values())*(168+9*3000/700)/3600
        dump(OUT/'sindy/exact_pilot_projection.json',dict(mean_case_cpu_seconds=times,projected_core_hours=projection,remaining_stage_core_hours=12-json.loads((OUT/'sindy/accounting.json').read_text())['cpu_seconds']/3600,arms=arms,minimum_runs_per_arm=177))
    dump(OUT/'sindy'/f'exact_{args.phase}_completion.json',dict(state='complete',accounting=budget.checkpoint()))


if __name__=='__main__':main()
