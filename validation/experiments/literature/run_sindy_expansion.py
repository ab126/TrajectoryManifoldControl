"""Same frozen exact/compiled SINDYc four arms on fresh/original 200 states."""
import copy,json
import numpy as np
import sindy,run_classical,run_sindy_compiled
from sindy_gradient import ExactGradientNMPC,cost_gradient
from common import *


def main():
    isolation('sindy');budget=Budget('sindy',12);budget.checkpoint()
    assert json.loads((OUT/'preparation/audit_completion.json').read_text())['state']=='complete'
    bindings=json.loads((OUT/'inventory/classical_reuse_bindings.json').read_text())['bindings']
    sindy.SINDYc.predict_inputs=run_sindy_compiled.predict_inputs;sindy.NMPC=ExactGradientNMPC;run_classical.code_identity=run_sindy_compiled.identity
    original=json.loads((ROOT/'validation/results_v1/initial_conditions.json').read_text())['sets']['C'];fresh=json.loads((OUT/'inventory/fresh_200_states.json').read_text())['states']
    arms=['CEM','CEM+L','NMPC_exact','NMPC_exact+L'];timing=json.loads((OUT/'sindy/exact_pilot_projection.json').read_text())['mean_case_cpu_seconds'];projection=sum(timing.values())*1200/3600
    dump(OUT/'sindy/expansion_projection.json',dict(projected_core_hours=projection,remaining_core_hours=12-budget.checkpoint()['cpu_seconds']/3600,priority=['fresh200','originalC200']))
    assert projection<=12-budget.checkpoint()['cpu_seconds']/3600,'SINDy expansion cap-limited'
    for name,states in [('F',fresh),('C',original)]:
        for seed in range(3):
            binding=next(b for b in bindings if b['method']=='sindy' and b['seed']==seed)
            model,identity,selected=run_classical.select('sindy',seed,budget);assert selected['config']==binding['selected_configuration']
            assert digest(ROOT/f'validation/results_v1/models/episodes3500_seed{seed}/diagnostics.json')==binding['gain_artifact_sha256']
            assert digest(ROOT/'src/inverted_pendulum.py')==binding['simulator_sha256']
            cost_gradient(np.zeros(4),np.zeros(30),model.library.powers_,model.coef,model.trig)
            for arm in arms:
                actual=copy.deepcopy(identity)
                if arm.startswith('NMPC'):actual['optimizer_backend']=dict(gradient='exact sensitivities of fitted coefficients',source_sha256=digest(Path(__file__).with_name('sindy_gradient.py')),driver_sha256=digest(Path(__file__).with_name('run_sindy_exact.py')),maximum_iterations=50,per_solve_wall_seconds=2.)
                for index,state in enumerate(states):run_classical.evaluate('sindy',seed,arm,model,actual,selected,dict(set=name,index=index,state=state,seconds=14.),budget)
    dump(OUT/'sindy/expansion_completion.json',dict(state='complete',accounting=budget.checkpoint()))


if __name__=='__main__':main()
