"""Bounded CPU wiring checks before original/E4 reference controller pilots."""
import json
from types import SimpleNamespace
import numpy as np
import torch
from common import ROOT,OUT,Budget,dump,digest,isolation
from reference_transition import classes
from fresh_references import factory


def main():
    isolation('deepkoopman');torch.set_num_threads(1);torch.set_num_interop_threads(1)
    budget=Budget('preparation',2);budget.checkpoint();Transition,Decoder=classes();rows=[]
    for cohort in ['original','E4']:
        for seed in range(3):
            saved=ROOT/(f'validation/results_v1/models/episodes3500_seed{seed}' if cohort=='original' else f'validation/results_v1/E4/models/seed{seed}')
            payload=torch.load(saved/'transition.pt',map_location='cpu',weights_only=False)
            model=Transition(4,1,hidden_dims=payload['hidden_dims']);model.load_state_dict(payload['transition_state_dict']);model.eval()
            decoder=Decoder(model,30).eval();diag=json.loads((saved/'diagnostics.json').read_text())
            config=SimpleNamespace(**{**diag['config'],'horizon':30,'device':'cpu'});gain=np.asarray(diag['local']['gain'])
            law=factory(decoder,config,gain);x=np.array([0.,0.,.01,0.]);u=law(0,x)
            assert np.allclose(u,np.clip(-gain@x,-10,10))
            assert law.diagnostics['terminal_policy_used']==[True]
            # One nonlocal solve and its four queued actions verify source H routing.
            law=factory(decoder,config,gain);actions=[law(k,np.array([0.,0.,1.,0.])) for k in range(5)]
            assert np.isfinite(actions).all() and np.max(np.abs(actions))<=10+1e-6
            assert len(law.diagnostics['loss'])==1
            assert law.diagnostics['terminal_policy_used']==[False]*5
            rows.append(dict(cohort=cohort,seed=seed,checkpoint_sha256=digest(saved/'transition.pt'),local_feedback=True,one_solve_five_actions=True,finite_bounded=True))
            budget.checkpoint()
    dump(OUT/'reference_preflight/completion.json',dict(passed=True,rows=rows,source_sha256=digest(__file__),factory_sha256=digest(__file__.replace('reference_preflight.py','fresh_references.py')),accounting=budget.checkpoint()))


if __name__=='__main__':main()
