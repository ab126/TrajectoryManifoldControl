"""Controller-only vectorization; source fitting/loss code and weights unchanged."""
from pathlib import Path
import torch
import control
import run_neural
from common import OUT,digest,dump
from fast_deep_forecast import FastDeepForecast
from fast_dfine_forecast import FastDFINEForecast


original_cem=control.TorchCEM
original_fit=run_neural.fit_seed
cache={}


class VectorizedCEM(original_cem):
    def __init__(self,forecast,scale,seed,device='cuda'):
        owner=getattr(forecast,'__self__',None)
        if owner is None:
            owners=[cell.cell_contents for cell in (getattr(forecast,'__closure__',None) or []) if cell.cell_contents.__class__.__name__=='DFINEAdapter']
            if owners:owner=owners[0]
        if owner is not None and owner.__class__.__name__=='DeepKoopman':
            if owner not in cache:
                module=FastDeepForecast(owner).eval()
                cache[owner]=torch.jit.script(module)
                # Small matched-input parity check on bounded measured states.
                with torch.no_grad():
                    x=owner.mean[None].expand(8,-1).clone();u=torch.zeros(8,30,1,device=x.device)
                    u[:,::2]=.1;reference=owner.rollout(x,u)[:,1:];actual=cache[owner](x,u)[:,1:]
                    finite=torch.isfinite(reference).all() and torch.isfinite(actual).all()
                    if not finite:raise RuntimeError('Nonfinite bounded vectorization parity example; do not launch controller')
                    error=float(torch.max(torch.abs(reference-actual)));scale_error=float(torch.max(torch.abs((reference-actual)/owner.scale)))
                    if scale_error>1e-4:raise RuntimeError(f'Vectorized source forecast parity error {scale_error}')
                dump(OUT/'deepkoopman_control'/f'forecast_parity_seed{seed}.json',dict(passed=True,maximum_absolute_error=error,maximum_standardized_error=scale_error,source_weight_changes=False,initial_state='measured x0 kept explicitly; reconstruction remains separately scored',source_sha256=digest(Path(__file__).with_name('fast_deep_forecast.py'))))
            forecast=cache[owner]
        elif owner is not None and owner.__class__.__name__=='DFINEAdapter':
            if owner not in cache:
                cache[owner]=torch.jit.script(FastDFINEForecast(owner).eval())
                with torch.no_grad():
                    initial=owner.source.ldm.mu_0;u=torch.zeros(8,30,1,device=initial.device);u[:,::2]=.1
                    z=initial.expand(8,-1);source=[owner.decode_dynamic(z)]
                    for k in range(30):z=z@owner.source.ldm.A.T+u[:,k]@owner.B.T;source.append(owner.decode_dynamic(z))
                    source=torch.stack(source,dim=1);actual=cache[owner](initial,u)
                    if not torch.isfinite(source).all() or not torch.isfinite(actual).all():raise RuntimeError('Nonfinite bounded condensed DFINE parity example')
                    error=float((source-actual).abs().max());scaled=float(((source-actual)/owner.scale).abs().max())
                    if scaled>1e-4:raise RuntimeError(f'Condensed DFINE forecast parity error {scaled}')
                dump(OUT/'dfine_control'/f'forecast_parity_seed{seed}.json',dict(passed=True,maximum_absolute_error=error,maximum_standardized_error=scaled,source_weight_changes=False,scope='linear controlled latent conditional means, same source nonlinear decoder',source_sha256=digest(Path(__file__).with_name('fast_dfine_forecast.py'))))
            module=cache[owner]
            forecast=lambda x,u:module(owner.current_mean,u)
        super().__init__(forecast,scale,seed,device)


def fit(*args,**kwargs):
    model,identity=original_fit(*args,**kwargs)
    if args[0]=='deepkoopman' and args[1]=='control':identity['controller_forecast_backend']=dict(source='vectorized source frequency blocks + TorchScript',sha256=digest(Path(__file__).with_name('fast_deep_forecast.py')),driver_sha256=digest(__file__),fitting_unchanged=True)
    if args[0]=='dfine' and args[1]=='control':identity['controller_forecast_backend']=dict(source='condensed linear latent means + batched nonlinear decoding',sha256=digest(Path(__file__).with_name('fast_dfine_forecast.py')),driver_sha256=digest(__file__),fitting_unchanged=True,conditional_mean_cost_approximation=True)
    return model,identity


if __name__=='__main__':
    control.TorchCEM=VectorizedCEM;run_neural.fit_seed=fit;run_neural.main()
