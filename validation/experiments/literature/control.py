"""Shared costs/endpoint, explicit fallback, causal applied-input controller wrapper."""
import time
import json
import numpy as np
import torch
from common import *


class TorchCEM:
    def __init__(self,forecast,scale,seed,device='cuda'):
        self.forecast=forecast;self.scale=torch.as_tensor(scale,dtype=torch.float32,device=device)
        self.device=device;self.rng=torch.Generator(device=device).manual_seed(seed)
        self.previous=torch.zeros(HORIZON,1,device=device);self.Q=torch.as_tensor(Q,dtype=torch.float32,device=device)

    def solve(self,x):
        with torch.no_grad():
            x=torch.as_tensor(x,dtype=torch.float32,device=self.device)
            mean=torch.cat((self.previous[1:],self.previous[-1:]));fraction=torch.clamp(2*torch.sqrt(torch.mean((x/self.scale)**2)),0,1)
            std=torch.ones_like(mean)*((.05+.45*fraction)*20);best_cost=float('inf');best=torch.zeros_like(mean)
            for _ in range(3):
                candidates=(mean+std*torch.randn((256,HORIZON,1),device=self.device,generator=self.rng)).clamp(-10,10);candidates[0]=mean.clamp(-10,10)
                states=self.forecast(x.expand(256,-1),candidates)
                costs=(torch.einsum('bki,ij,bkj->b',states[:,:-1],self.Q,states[:,:-1])
                       +torch.einsum('bi,ij,bj->b',states[:,-1],12*self.Q,states[:,-1])
                       +.03*candidates.square().sum(dim=(1,2))+.002*(candidates[:,1:]-candidates[:,:-1]).square().sum(dim=(1,2)))
                costs=torch.where(torch.isfinite(costs),costs,torch.inf);elite=candidates[torch.topk(costs,32,largest=False).indices]
                mean=.15*mean+.85*elite.mean(0);std=.15*std+.85*elite.std(0).clamp_min(.02)
                index=int(costs.argmin());value=float(costs[index])
                if value<best_cost:best_cost=value;best=candidates[index].clone()
            self.previous=best
            return best.cpu().numpy(),dict(accepted=bool(np.isfinite(best_cost)),objective=best_cost if np.isfinite(best_cost) else None,status='finite' if np.isfinite(best_cost) else 'nonfinite')


def evaluate(stage,seed,arm,case,identity,solver_factory,budget,filter_model=None):
    dest=OUT/stage;key=f"seed{seed}_{arm}_{case['set']}_{case['index']:04d}_{case['seconds']:g}s";path=dest/'cases'/f'{key}.json'
    expected=dict(model=identity,case=case,arm=arm,seed=seed,controller_code=digest(__file__))
    if path.exists():
        row=json.loads(path.read_text());assert row['identity']==expected,'Control resume identity mismatch';return row
    if filter_model is not None:filter_model.reset()
    solver=solver_factory();step=plant_interface();steps=round(case['seconds']/.02)
    gain=np.array(json.loads((ROOT/f'validation/results_v1/models/episodes3500_seed{seed}/diagnostics.json').read_text())['local']['gain'])
    X=np.empty((steps+1,4));U=np.empty((steps,1));X[0]=case['state'];reconstructed=[];local=[];info=[];inputs=[];solves=[];latencies=[];start=time.monotonic();cpu=cpu_seconds()
    for k in range(steps):
        if k%25==0:budget.checkpoint()
        observed=X[k] if filter_model is None else filter_model.observe(X[k],None if k==0 else U[k-1]);reconstructed.append(observed)
        catch=bool(arm.endswith('+L') and observed@Q@observed<.75) or arm=='filtered_L'
        local.append(catch)
        if catch:u=np.clip(-gain@observed,-10,10)
        else:
            before=time.monotonic();proposal,diagnostics=solver.solve(observed);latencies.append(time.monotonic()-before)
            info.append(diagnostics);inputs.append(proposal);solves.append(k);u=proposal[0]
        U[k]=np.clip(u,-10,10)
        with np.errstate(over='ignore',invalid='ignore'):X[k+1]=step(X[k],U[k,0])
        if not np.isfinite(X[k+1]).all():X=X[:k+2];U=U[:k+1];break
    t=np.arange(len(X))*.02;complete=len(U)==steps;tail=t>=case['seconds']-3.;finite=bool(np.isfinite(X).all() and np.isfinite(U).all())
    no_fall=bool(finite and np.all(np.abs(X[:,2])<np.pi/2));balanced=bool(complete and finite and np.all(np.abs(X[tail,2])<=.02) and np.all(np.abs(X[tail,3])*np.sqrt(9.81)<=.05))
    def number(v):return float(v) if np.isfinite(v) else None
    rec=np.array(reconstructed);error=rec-X[:len(rec)]
    row=dict(case=key,identity=expected,seed=seed,arm=arm,set=case['set'],seconds=case['seconds'],initial_state=case['state'],
             complete=complete,balanced=balanced,no_fall=no_fall,primary_success=balanced and no_fall,nonfinite_state=not finite,
             cart_rest=bool(complete and finite and np.all(np.abs(X[tail,1])*np.sqrt(9.81)<=.05)),
             max_abs_p=number(np.max(np.abs(X[:,0]))),final_theta=number(X[-1,2]),max_abs_u=number(np.max(np.abs(U))),
             effort=number(np.sum(U*U)*.02),input_limit_seconds=float(np.sum(np.abs(U)>=10-1e-6)*.02),
             local_fraction=float(np.mean(local)),switch_toggles=int(np.sum(np.diff(local)!=0)),
             reconstruction_bias=[number(v) for v in np.mean(error,axis=0)],reconstruction_rmse=[number(v) for v in np.sqrt(np.mean(error**2,axis=0))],
             solver_failures=sum(not v['accepted'] for v in info),solve_count=len(info),
             solve_latency_quantiles=np.quantile(latencies,[.5,.95,1.]).tolist() if latencies else [],
             wall_seconds=time.monotonic()-start,cpu_seconds=cpu_seconds()-cpu)
    raw=dest/'raw'/f'{key}.npz';raw.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(raw,t=t,X=X,U=U,reconstructed=rec,local=local,plan_inputs=inputs,solve_steps=solves,solve_seconds=latencies)
    row['raw_path']=str(raw.relative_to(ROOT));dump(dest/'solver_logs'/f'{key}.json',info);dump(path,row);budget.checkpoint()
    print(json.dumps({k:row[k] for k in ['case','primary_success','solver_failures','wall_seconds']}),flush=True);return row
