"""CPU parity of unchanged ten-feature history networks before GPU pilot."""
import torch
from common import ROOT,OUT,Budget,dump,digest,isolation
from reference_transition import classes
from fast_single_forecast import FastSingleForecast


def main():
    isolation('dfine');torch.set_num_threads(1);torch.set_num_interop_threads(1)
    budget=Budget('preparation',2);budget.checkpoint();Transition,Decoder=classes();rows=[]
    for seed in range(3):
        path=OUT/f'dfine_control/output_only/seed{seed}/neural/best.pt'
        saved=torch.load(path,map_location='cpu',weights_only=False)
        model=Transition(10,1,[192]*3);model.load_state_dict(saved['model']);model.eval()
        decoder=Decoder(model,30).eval();fast=torch.jit.script(FastSingleForecast(model).eval())
        generator=torch.Generator().manual_seed(20261013+seed)
        with torch.no_grad():
            x=model.xu_mean[:10]+.1*model.xu_std[:10]*torch.randn(8,10,generator=generator)
            u=.1*torch.randn(8,30,1,generator=generator);expected=decoder.rollout(x,u);actual=fast(x,u)
            assert torch.isfinite(expected).all() and torch.isfinite(actual).all()
            error=float(((expected-actual)/model.xu_std[:10]).abs().max());assert error<1e-4
        rows.append(dict(seed=seed,checkpoint_sha256=digest(path),maximum_standardized_error=error));budget.checkpoint()
    dump(OUT/'forecast_preflight/output_history_completion.json',dict(passed=True,rows=rows,backend_sha256=digest(ROOT/'validation/experiments/literature/fast_single_forecast.py'),source_sha256=digest(__file__),model_weights_changed=False,accounting=budget.checkpoint()))


if __name__=='__main__':main()
