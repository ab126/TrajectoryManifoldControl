"""Frozen selected-model equilibrium/input diagnostics and compact forecast tables."""
import json
import numpy as np
import torch
from common import *
from deepkoopman import DeepKoopman
from dfine import DFINEAdapter


def main():
    isolation('dfine');torch.set_num_threads(1);torch.set_num_interop_threads(1);budget=Budget('preparation',2);budget.checkpoint();rows=[];diagnostics=[]
    for stage in ['deepkoopman_passive','dfine_passive','deepkoopman_control','dfine_control']:
        assert (OUT/stage/'fit_completion.json').exists()
        for seed in range(3):
            root=OUT/stage/f'seed{seed}';selection=json.loads((root/'selection.json').read_text());config=selection['config'];controlled=stage.endswith('control');path=root/f"candidate{selection['index']}/best.pt"
            model=DeepKoopman(config['latent'],controlled=controlled) if stage.startswith('deepkoopman') else DFINEAdapter(config['latent'],config['regularization'],seed,controlled=controlled)
            model.load_state_dict(torch.load(path,map_location='cpu',weights_only=False)['model']);model.eval()
            x=np.zeros((1,4));zero=np.zeros((1,30,1));forced=np.full((1,30,1),.1)
            with np.errstate(over='ignore',invalid='ignore'):free=model.predict_inputs(x,zero);driven=model.predict_inputs(x,forced)
            finite=bool(np.isfinite(free).all() and np.isfinite(driven).all())
            diagnostics.append(dict(method=stage,seed=seed,model_sha256=digest(path),zero_input_initial_state=[0.,0.,0.,0.],horizon=30,finite=finite,zero_input_final=free[0,-1].tolist() if np.isfinite(free[0,-1]).all() else None,equilibrium_max_abs_error=float(np.max(np.abs(free))) if np.isfinite(free).all() else None,force_sequence_value=.1,force_effect_max_abs=float(np.max(np.abs(driven-free))) if finite else None,interpretation='Frozen-model diagnostic only; no coefficient, feedback or selection repair'))
            rows.append(dict(method=stage,seed=seed,selected_config=config,**json.loads((root/'test_forecast.json').read_text())));budget.checkpoint()
    for seed in range(3):
        for method in ['linear','edmd','neural']:
            rows.append(dict(method='passive_'+method,seed=seed,**json.loads((OUT/'passive_baselines'/f'seed{seed}/{method}_metrics.json').read_text())))
    dump(OUT/'prediction_tables.json',rows);dump(OUT/'inventory/selected_latent_equilibrium.json',diagnostics);dump(OUT/'reviews/latent_forecast_packet.json',dict(question='Review source-port and selected-model equilibrium/forecast interpretation; control conclusions remain separate',source='experiments/literature/latent_closure.py',forecasts='prediction_tables.json',diagnostics='inventory/selected_latent_equilibrium.json',control_training='original forced data only; no passive weights reused',results=rows));dump(OUT/'latent_closure/completion.json',dict(state='complete',accounting=budget.checkpoint()))


if __name__=='__main__':main()
