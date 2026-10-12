"""Check compiled evaluator parity before replacing scientific rollout execution."""
import pickle
import time
import numpy as np
from common import *
from sindy_kernel import rollout


def main():
    isolation('sindy');budget=Budget('preparation',2);budget.checkpoint()
    results=[]
    for seed in range(3):
        dest=OUT/'sindy'/f'seed{seed}';choice=__import__('json').loads((dest/'selection.json').read_text())
        model=pickle.loads((dest/f"candidate{choice['index']}.pkl").read_bytes())
        data,_=load_data(seed);X,U=data['validation'];X=X[:100];U=U[:100]
        before=time.monotonic();reference=model.predict_inputs(X[:,0],U);reference_time=time.monotonic()-before
        prediction=rollout(X[:,0].copy(),U.copy(),model.library.powers_,model.coef,model.trig)
        error=float(np.max(np.abs(reference-prediction)))
        assert error<1e-9,error
        before=time.monotonic();rollout(X[:,0].copy(),U.copy(),model.library.powers_,model.coef,model.trig);compiled_time=time.monotonic()-before
        results.append(dict(seed=seed,maximum_absolute_error=error,source_library_seconds=reference_time,compiled_seconds=compiled_time))
    dump(OUT/'sindy/kernel_parity.json',dict(passed=True,parallel=False,refit=False,results=results))
    budget.checkpoint();print(results,flush=True)


if __name__=='__main__':main()
