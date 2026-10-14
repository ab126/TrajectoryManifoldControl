"""Bounded noise/index/opaque replay checks before E4 stress acquisition."""
import ast,json
import numpy as np
from common import ROOT,OUT,dump,digest,isolation,plant_interface
from rok_stress import PreservingBudget
from stress_protocol import CONDITIONS,disturbances


def main():
    isolation('rokdeepc');budget=PreservingBudget('rokdeepc',12);budget.checkpoint()
    source=ROOT/'validation/experiments/e3_robustness.py';tree=ast.parse(source.read_text())
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='normal_sequence']
    namespace=dict(np=np,CONDITIONS=CONDITIONS);exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),namespace)
    scale=np.array([.4,.3,.2,.1]);maximum=0.
    for seed in range(3):
        for condition in CONDITIONS:
            for name,index in [('A',2),('C',7)]:
                case=dict(set=name,index=index);a=disturbances(seed,condition,case,scale)
                b=namespace['normal_sequence'](seed,condition,index+(1000 if name=='C' else 0),700,scale*CONDITIONS[condition].get('level',0.))
                maximum=max(maximum,float(np.max(np.abs(a-b))));assert np.array_equal(a,b)
    saved=np.load(ROOT/'validation/results_v1/E4/models/seed0/data.npz');step=plant_interface();errors=[]
    for e in range(5):
        for k in range(4):errors.append(float(np.max(np.abs(step(saved['X'][e,k],float(saved['U'][e,k,0]))-saved['X'][e,k+1]))))
    assert max(errors)<2e-6
    dump(OUT/'rokdeepc/stress/checks.json',dict(passed=True,noise_rng_parity_max_error=maximum,opaque_saved_replay_max_error=max(errors),data_privilege='plant only for transition acquisition and evaluation; model/control fit receives measured X/U only',acquisition_adapter_sha256=digest(ROOT/'validation/experiments/literature/e4_noisy_data.py'),stress_adapter_sha256=digest(ROOT/'validation/experiments/literature/rok_stress.py'),accounting=budget.checkpoint()))


if __name__=='__main__':main()
