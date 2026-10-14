"""Output-only controlled DFINE and matched history adapters, separate cohort table."""
import argparse,json,time
import numpy as np
import torch
from torch import nn
from common import *
from neural_common import GPUBudget
from run_output_history import latent_forecast
from run_output_safe import fit
from output_history import HistoryObserver,apply_readout
from control import TorchCEM
from casebook import cases
from fast_single_forecast import FastSingleForecast


class LatentPhysicalForecast(nn.Module):
    def __init__(self,model,readout):
        super().__init__();self.decoder=model.source.decoder;d=model.dynamic_dim
        A=model.source.ldm.A.detach().cpu();B=model.B.detach().cpu();F=[torch.eye(d)];G=[torch.zeros(d,30)]
        for k in range(30):
            F.append(A@F[-1]);g=A@G[-1];g[:,k]=B[:,0];G.append(g)
        self.register_buffer('F',torch.stack(F).reshape(-1,d));self.register_buffer('G',torch.stack(G).reshape(-1,30));self.register_buffer('C',model.source.ldm.C.detach().cpu());self.register_buffer('mean',model.mean.detach().cpu());self.register_buffer('scale',model.scale.detach().cpu());self.register_buffer('readout',torch.tensor(readout,dtype=torch.float32));self.dimension=d
    def forward(self,mu,u):
        z=(u[:,:,0]@self.G.T+self.F@mu).reshape(-1,31,self.dimension);observed=self.decoder(z@self.C.T)*self.scale+self.mean;hidden=z@self.readout[:-1]+self.readout[-1]
        return torch.stack((observed[:,:,0],hidden[:,:,0],observed[:,:,1],hidden[:,:,1]),-1)


class LatentObserver:
    def __init__(self,model,weights):self.model=model;self.weights=weights;self.reset()
    def reset(self):self.model.reset()
    def observe(self,y,u=None):
        assert np.asarray(y).shape==(2,),'Only two observed channels may enter filter'
        observed=self.model.observe(y,u);hidden=apply_readout(self.model.current_mean.detach().cpu().numpy(),self.weights)
        return np.array([observed[0],hidden[0],observed[1],hidden[1]])


class NeuralObserver:
    def __init__(self,kf,weights,decoder):
        self.filter=HistoryObserver(kf);self.weights=weights;self.decoder=decoder;self.current=None
        self.fast=torch.jit.script(FastSingleForecast(decoder.transition).eval())
    def reset(self):self.filter.reset();self.current=None
    def observe(self,y,u=None):
        assert np.asarray(y).shape==(2,)
        self.current=self.filter.observe(y,u);hidden=apply_readout(self.current,self.weights)
        return np.array([self.current[0],hidden[0],self.current[1],hidden[1]])
    def forecast(self,x,u):
        h=torch.tensor(self.current,dtype=torch.float32,device=u.device).expand(len(u),-1);z=self.fast(h,u);weights=torch.tensor(self.weights,dtype=torch.float32,device=u.device);hidden=z@weights[:-1]+weights[-1]
        return torch.stack((z[:,:,0],hidden[:,:,0],z[:,:,1],hidden[:,:,1]),-1)


def evaluate(seed,arm,case,identity,observer,factory,budget):
    root=OUT/'dfine_control/output_only';key=f"seed{seed}_{arm}_{case['set']}_{case['index']:04d}_{case['seconds']:g}s";p=root/'cases'/f'{key}.json';expected=dict(model=identity,seed=seed,arm=arm,case=case,code_sha256=digest(__file__),simulator_sha256=digest(ROOT/'src/inverted_pendulum.py'))
    if p.exists():row=json.loads(p.read_text());assert row['identity']==expected;return row
    observer.reset();solver=factory();gain=np.asarray(json.loads((ROOT/f'validation/results_v1/models/episodes3500_seed{seed}/diagnostics.json').read_text())['local']['gain']);steps=round(case['seconds']/.02);X=np.empty((steps+1,4));U=np.empty((steps,1));X[0]=case['state'];rec=[];local=[];logs=[];latency=[];step=plant_interface();start=time.monotonic();cpu=cpu_seconds()
    for k in range(steps):
        if k%25==0:budget.checkpoint()
        estimate=observer.observe(X[k,[0,2]],None if k==0 else U[k-1]);rec.append(estimate)
        catch=arm in ['DFINE_filtered_L','history_L'] or (arm.endswith('+L') and estimate@Q@estimate<.75);local.append(catch)
        if not np.isfinite(estimate).all():u=np.zeros(1);logs.append(dict(accepted=False,status='nonfinite causal estimate'))
        elif catch:u=np.clip(-gain@estimate,-10,10)
        else:
            before=time.monotonic();plan,info=solver.solve(estimate);latency.append(time.monotonic()-before);logs.append(info);u=plan[0]
        U[k]=np.clip(u,-10,10)
        with np.errstate(over='ignore',invalid='ignore'):X[k+1]=step(X[k],U[k,0])
        if not np.isfinite(X[k+1]).all():X=X[:k+2];U=U[:k+1];break
    t=np.arange(len(X))*.02;tail=t>=case['seconds']-3.;finite=bool(np.isfinite(X).all() and np.isfinite(U).all());complete=len(U)==steps;no_fall=bool(finite and np.all(np.abs(X[:,2])<np.pi/2));balanced=bool(complete and finite and np.all(np.abs(X[tail,2])<=.02) and np.all(np.abs(X[tail,3])*np.sqrt(9.81)<=.05));raw=root/'raw'/f'{key}.npz';raw.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(raw,X=X,U=U,t=t,reconstructed=rec,local=local)
    error=np.asarray(rec)-X[:len(rec)]
    def number(x):return float(x) if np.isfinite(x) else None
    row=dict(identity=expected,case=key,seed=seed,arm=arm,set=case['set'],seconds=case['seconds'],primary_success=balanced and no_fall,balanced=balanced,no_fall=no_fall,complete=complete,nonfinite_state=not finite,observed_channels=['position','angle'],no_hidden_observations=True,reconstruction_bias=[number(v) for v in error.mean(0)],reconstruction_rmse=[number(v) for v in np.sqrt((error**2).mean(0))],local_fraction=float(np.mean(local)),switch_toggles=int(np.sum(np.diff(local)!=0)),solver_failures=sum(not r['accepted'] for r in logs),solve_count=len(logs),solve_latency_quantiles=np.quantile(latency,[.5,.95,1.]).tolist() if latency else [],effort=number(np.sum(U**2)*.02),input_limit_seconds=float(np.sum(np.abs(U)>=10-1e-6)*.02),wall_seconds=time.monotonic()-start,cpu_seconds=cpu_seconds()-cpu,raw_path=str(raw.relative_to(ROOT)));dump(root/'solver_logs'/f'{key}.json',logs);dump(p,row);budget.checkpoint();print(json.dumps(dict(case=key,success=row['primary_success'],wall_seconds=row['wall_seconds'])),flush=True);return row


def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['pilot','matrix'],required=True);args=p.parse_args();isolation('dfine');budget=GPUBudget('dfine_control',12,6);budget.checkpoint();root=OUT/'dfine_control/output_only';assert (root/'fit_completion.json').exists()
    arms=['DFINE_MPC','DFINE_MPC+L','DFINE_filtered_L','NN_history_MPC','NN_history_MPC+L','history_L']
    if args.phase=='matrix':
        projection=json.loads((root/'pilot_projection.json').read_text());assert projection['projected_gpu_hours']<=6-budget.checkpoint()['gpu_seconds']/3600,'Output-only requested matrix exceeds DFINE cap'
    for seed in ([0] if args.phase=='pilot' else range(3)):
        model,decoder,kf,weights,protocol,data=fit(seed,'control',budget);model.cuda();decoder.cuda();module=torch.jit.script(LatentPhysicalForecast(model,weights['latent']).cuda().eval());observer=LatentObserver(model,weights['latent']);other=NeuralObserver(kf,weights['history'],decoder);scale=data['train'][0][:,:-1].reshape(-1,4).std(0).clip(1e-8)
        with torch.no_grad():
            initial=model.source.ldm.mu_0;u=torch.full((4,30,1),.1,device='cuda');source=latent_forecast(model,initial.expand(4,-1),u);actual=module(initial,u);error=float(((source-actual[:,:,[0,2]])/model.scale).abs().max());assert error<1e-4
            # Hidden channels are never passed to either observation adapter.
            observer.observe(np.array([.01,.02]));before=model.current_mean.clone();observer.reset();observer.observe(np.array([.01,.02]));assert torch.equal(before,model.current_mean)
            h=decoder.transition.xu_mean[:10][None].expand(4,-1);neural_source=decoder.rollout(h,u);neural_actual=other.fast(h,u);neural_error=float(((neural_source-neural_actual)/decoder.transition.xu_std[:10]).abs().max());assert neural_error<1e-4
        dump(root/f'checks_seed{seed}.json',dict(passed=True,source_condensed_observed_parity=error,neural_history_forecast_parity=neural_error,neural_backend_sha256=digest(Path(__file__).with_name('fast_single_forecast.py')),reset_deterministic=True,observation_shape_assertion=2,hidden_labels_training_only=True))
        path=root/f'seed{seed}';identity=dict(protocol=protocol,latent_sha256=digest(path/'best.pt'),neural_sha256=digest(path/'neural/best.pt'),readout_sha256=digest(path/'readouts.npz'),gain_sha256=digest(ROOT/f'validation/results_v1/models/episodes3500_seed{seed}/diagnostics.json'),neural_backend_sha256=digest(Path(__file__).with_name('fast_single_forecast.py')),conditional_mean_cost_approximation=True,controller='256 candidates, 32 elites, 3 CEM iterations; native or declared common L only')
        for arm in arms:
            obs=observer if arm.startswith('DFINE') else other;factory=(lambda:TorchCEM(lambda x,u:module(model.current_mean,u),scale,seed)) if arm.startswith('DFINE') else (lambda:TorchCEM(other.forecast,scale,seed))
            for case in (cases()[:16] if args.phase=='pilot' else cases()):evaluate(seed,arm,case,identity,obs,factory,budget)
    if args.phase=='pilot':
        rows=[json.loads(p.read_text()) for p in (root/'cases').glob('*.json')];means={arm:float(np.mean([r['wall_seconds'] for r in rows if r['arm']==arm])) for arm in arms};projection=sum(means.values())*(168+9*3000/700)/3600;dump(root/'pilot_projection.json',dict(mean_case_wall_seconds=means,projected_gpu_hours=projection,remaining_gpu_hours=6-budget.checkpoint()['gpu_seconds']/3600,minimum_runs_per_arm=177))
    dump(root/f'{args.phase}_completion.json',dict(state='complete',accounting=budget.checkpoint()))


if __name__=='__main__':main()
