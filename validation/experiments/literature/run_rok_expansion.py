"""Affordable fresh/original 200-state expansion, E4 fits remain a separate cohort."""
import json
from common import *
from run_rokdeepc import fit,evaluate


def main():
    isolation('rokdeepc');budget=Budget('rokdeepc',12);budget.checkpoint();assert (OUT/'rokdeepc/matrix_completion.json').exists();timing=json.loads((OUT/'rokdeepc/pilot_projection.json').read_text());projection=sum(timing['mean_case_cpu_seconds'].values())*1200/3600;remaining=12-budget.checkpoint()['cpu_seconds']/3600;dump(OUT/'rokdeepc/expansion_projection.json',dict(projected_core_hours=projection,remaining_core_hours=remaining,priority=['fresh200','originalC200']));assert projection<=remaining
    original=json.loads((ROOT/'validation/results_v1/initial_conditions.json').read_text())['sets']['C'];fresh=json.loads((OUT/'inventory/fresh_200_states.json').read_text())['states']
    for name,states in [('F',fresh),('C',original)]:
        for seed in range(3):
            kernel,linear,identity=fit(seed,budget)
            selection=json.loads((OUT/'rokdeepc'/f'seed{seed}/selection.json').read_text());path=OUT/'rokdeepc'/f"seed{seed}/candidate{selection['index']}.pkl";binding=OUT/'rokdeepc'/f'seed{seed}/model_binding.json';record=dict(data_sha256=identity['sha256'],selection_sha256=digest(OUT/'rokdeepc'/f'seed{seed}/selection.json'),selected_kernel_sha256=digest(path),simulator_sha256=digest(ROOT/'src/inverted_pendulum.py'),endpoint_sha256=digest(Path(__file__).with_name('run_rokdeepc.py')),initial_state_sha256=digest(ROOT/'validation/results_v1/initial_conditions.json'),fresh_state_sha256=digest(OUT/'inventory/fresh_200_states.json'))
            if binding.exists():assert json.loads(binding.read_text())==record
            else:dump(binding,record)
            for arm,model in [('RoKDeePC',kernel),('linear_kernel',linear)]:
                for index,state in enumerate(states):evaluate(seed,arm,model,identity,dict(set=name,index=index,state=state,seconds=14.),budget)
    dump(OUT/'rokdeepc/expansion_completion.json',dict(state='complete',accounting=budget.checkpoint()))


if __name__=='__main__':main()
