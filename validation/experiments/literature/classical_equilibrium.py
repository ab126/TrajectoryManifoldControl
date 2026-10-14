"""Frozen selected classical zero-state/input diagnostics; no fit repairs."""
import argparse,json
import numpy as np
from common import *
import run_classical


def main():
    parser=argparse.ArgumentParser();parser.add_argument('method',choices=['edmd','sindy']);args=parser.parse_args();method=args.method;isolation(method);budget=Budget(method,8 if method=='edmd' else 12);budget.checkpoint()
    if method=='sindy':
        import sindy,run_sindy_compiled
        sindy.SINDYc.predict_inputs=run_sindy_compiled.predict_inputs;run_classical.code_identity=run_sindy_compiled.identity
    rows=[]
    for seed in range(3):
        model,identity,selected=run_classical.select(method,seed,budget);x=np.zeros((1,4));u=np.zeros((1,30,1));zero=model.predict_inputs(x,u);forced=model.predict_inputs(x,np.full_like(u,.1));finite=bool(np.isfinite(zero).all() and np.isfinite(forced).all());path=OUT/method/f"seed{seed}/candidate{selected['index']}.pkl";rows.append(dict(method=method,seed=seed,model_sha256=digest(path),selected_config=selected['config'],finite=finite,zero_input_initial_state=[0.,0.,0.,0.],zero_input_final=zero[0,-1].tolist() if np.isfinite(zero[0,-1]).all() else None,equilibrium_max_abs_error=float(np.max(np.abs(zero))) if np.isfinite(zero).all() else None,force_sequence=.1,force_effect_max_abs=float(np.max(np.abs(forced-zero))) if finite else None,diagnostic_only=True,no_model_feedback_or_selection_changes=True));budget.checkpoint()
    dump(OUT/method/'selected_equilibrium.json',rows);dump(OUT/method/'diagnostics_completion.json',dict(state='complete',accounting=budget.checkpoint()))


if __name__=='__main__':main()
