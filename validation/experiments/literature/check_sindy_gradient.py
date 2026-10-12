"""Independently finite-difference the complete objective at bounded model states."""
import json
import pickle
import numpy as np
from common import *
from sindy_gradient import cost_gradient


def main():
    isolation('sindy');budget=Budget('preparation',2);budget.checkpoint();rows=[]
    for seed in range(3):
        p=OUT/'sindy'/f'seed{seed}';selected=json.loads((p/'selection.json').read_text());model=pickle.loads((p/f"candidate{selected['index']}.pkl").read_bytes())
        for trig in [model.trig]:
            x=np.array([.01,.02,.03,.01]);u=np.linspace(-.1,.1,30)
            value,grad=cost_gradient(x,u,model.library.powers_,model.coef,trig)
            literal=float(objective(model.predict_inputs(x,u[:,None]),u[:,None]));assert abs(value-literal)<1e-8
            finite=np.empty(30)
            for k in range(30):
                plus=u.copy();minus=u.copy();plus[k]+=1e-6;minus[k]-=1e-6
                finite[k]=(objective(model.predict_inputs(x,plus[:,None]),plus[:,None])-objective(model.predict_inputs(x,minus[:,None]),minus[:,None]))/2e-6
            error=float(np.max(np.abs(grad-finite)));assert error<2e-5,error
            rows.append(dict(seed=seed,objective_error=abs(value-literal),gradient_max_error=error))
    dump(OUT/'sindy/gradient_parity.json',dict(passed=True,results=rows,derivatives='learned coefficient map only; no simulator equations'))
    budget.checkpoint();print(rows,flush=True)


if __name__=='__main__':main()
