"""Keep frozen Deep/DFINE backend identities; separately accelerate NN reference."""
from pathlib import Path
import torch
import control,run_neural,run_neural_fast
from common import OUT,digest,dump
from fast_single_forecast import FastSingleForecast

cache={};original_evaluate=control.evaluate


class ReferenceCEM(run_neural_fast.VectorizedCEM):
    def __init__(self,forecast,scale,seed,device='cuda'):
        owner=getattr(forecast,'__self__',None)
        if owner is not None and owner.__class__.__name__=='CanonicalBehaviorDecoder':
            if owner not in cache:
                module=torch.jit.script(FastSingleForecast(owner.transition).eval())
                with torch.no_grad():
                    x=owner.transition.xu_mean[:owner.x_dim][None].expand(8,-1);u=torch.zeros(8,30,1,device=x.device);u[:,::2]=.1;source=owner.rollout(x,u);actual=module(x,u);assert torch.isfinite(source).all() and torch.isfinite(actual).all();error=float(((source-actual)/owner.transition.xu_std[:owner.x_dim]).abs().max());assert error<1e-4
                cache[owner]=module;dump(OUT/'neural_first_action'/f'forecast_parity_seed{seed}.json',dict(passed=True,maximum_standardized_error=error,source_sha256=digest(Path(__file__).with_name('fast_single_forecast.py')),model_weights_changed=False,optimizer_allowances_changed=False,feedback_changed=False))
            forecast=cache[owner]
        super().__init__(forecast,scale,seed,device)


def evaluate(stage,seed,arm,case,identity,factory,budget,filter_model=None):
    if stage=='neural_first_action':identity={**identity,'forecast_backend':dict(source='unchanged submitted transition network, TorchScript evaluation',source_sha256=digest(Path(__file__).with_name('fast_single_forecast.py')),driver_sha256=digest(__file__))}
    return original_evaluate(stage,seed,arm,case,identity,factory,budget,filter_model)


if __name__=='__main__':
    control.TorchCEM=ReferenceCEM;control.evaluate=evaluate;run_neural.fit_seed=run_neural_fast.fit;run_neural.main()
