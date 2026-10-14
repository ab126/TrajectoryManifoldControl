"""CPU compilation/parity check of frozen controlled forecast acceleration."""
import json
import numpy as np
import torch
from common import *
from deepkoopman import DeepKoopman
from dfine import DFINEAdapter
from fast_deep_forecast import FastDeepForecast
from fast_dfine_forecast import FastDFINEForecast
from output_control import LatentPhysicalForecast
from run_output_history import latent_forecast


def main():
    isolation('dfine');torch.set_num_threads(1);torch.set_num_interop_threads(1);budget=Budget('preparation',2);budget.checkpoint();rows=[]
    for method in ['deepkoopman','dfine']:
        for seed in range(3):
            root=OUT/(method+'_control')/f'seed{seed}';selected=json.loads((root/'selection.json').read_text());config=selected['config'];model=DeepKoopman(config['latent'],True) if method=='deepkoopman' else DFINEAdapter(config['latent'],config['regularization'],seed,True);p=root/f"candidate{selected['index']}/best.pt";model.load_state_dict(torch.load(p,map_location='cpu',weights_only=False)['model']);model.eval();u=torch.zeros(8,30,1);u[:,::2]=.1
            with torch.no_grad():
                if method=='deepkoopman':
                    module=torch.jit.script(FastDeepForecast(model).eval());x=model.mean[None].expand(8,-1);source=model.rollout(x,u)[:,1:];actual=module(x,u)[:,1:]
                else:
                    module=torch.jit.script(FastDFINEForecast(model).eval());initial=model.source.ldm.mu_0;source=latent_forecast(model,initial.expand(8,-1),u);actual=module(initial,u)
                assert torch.isfinite(source).all() and torch.isfinite(actual).all();error=float(((source-actual)/model.scale).abs().max());assert error<1e-4
            rows.append(dict(method=method,seed=seed,model_sha256=digest(p),torchscript_compiles=True,max_standardized_forecast_parity=error,weights_changed=False));budget.checkpoint()
    for seed in range(3):
        root=OUT/'dfine_control/output_only'/f'seed{seed}';config=json.loads((root/'protocol.json').read_text())['config'];model=DFINEAdapter(config['latent'],config['regularization'],seed,True,2);p=root/'best.pt';model.load_state_dict(torch.load(p,map_location='cpu',weights_only=False)['model']);model.eval();weights=np.load(root/'readouts.npz')['latent'];module=torch.jit.script(LatentPhysicalForecast(model,weights));u=torch.full((4,30,1),.1)
        with torch.no_grad():
            initial=model.source.ldm.mu_0;source=latent_forecast(model,initial.expand(4,-1),u);actual=module(initial,u)[...,[0,2]];assert torch.isfinite(source).all();error=float(((source-actual)/model.scale).abs().max());assert error<1e-4
        rows.append(dict(method='dfine_output_only',seed=seed,model_sha256=digest(p),torchscript_compiles=True,max_standardized_observed_forecast_parity=error,weights_changed=False));budget.checkpoint()
    dump(OUT/'forecast_preflight/checks.json',dict(passed=True,device='cpu',models=rows,GPU_runtime_parity_still_required=True));dump(OUT/'forecast_preflight/completion.json',dict(state='complete',accounting=budget.checkpoint()))


if __name__=='__main__':main()
