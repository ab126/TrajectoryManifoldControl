"""Matched original-data five-member deterministic reference; mean trajectory cost."""
import argparse,json,time
import numpy as np
import torch
from torch import nn
from common import *
from neural_common import GPUBudget
from transition_training import fit_transition
from control import TorchCEM,evaluate
from casebook import cases


class EnsembleMap(nn.Module):
    def __init__(self,members):
        super().__init__()
        for i,index in enumerate([0,2,4,6]):
            self.register_buffer('W'+str(i),torch.stack([m.net[index].weight.T.detach() for m in members]));self.register_buffer('b'+str(i),torch.stack([m.net[index].bias.detach() for m in members])[:,None])
        for name in ['xu_mean','xu_std','dx_mean','dx_std']:self.register_buffer(name,torch.stack([getattr(m,name).detach() for m in members])[:,None])
    def advance(self,x,u):
        z=(torch.cat((x,u.expand(5,-1,-1)),dim=-1)-self.xu_mean)/self.xu_std
        z=torch.nn.functional.silu(torch.bmm(z,self.W0)+self.b0);z=torch.nn.functional.silu(torch.bmm(z,self.W1)+self.b1);z=torch.nn.functional.silu(torch.bmm(z,self.W2)+self.b2)
        return x+(torch.bmm(z,self.W3)+self.b3)*self.dx_std+self.dx_mean
    def forward(self,x,u):
        current=x.unsqueeze(0).expand(5,-1,-1);out=[current]
        for k in range(30):current=self.advance(current,u[:,k]);out.append(current)
        return torch.stack(out,dim=2)


class EnsembleCEM(TorchCEM):
    def solve(self,x):
        with torch.no_grad():
            x=torch.as_tensor(x,dtype=torch.float32,device=self.device);mean=torch.cat((self.previous[1:],self.previous[-1:]));fraction=torch.clamp(2*torch.sqrt(torch.mean((x/self.scale)**2)),0,1);std=torch.ones_like(mean)*((.05+.45*fraction)*20);best_cost=float('inf');best=torch.zeros_like(mean)
            for _ in range(3):
                u=(mean+std*torch.randn((256,30,1),device=self.device,generator=self.rng)).clamp(-10,10);u[0]=mean.clamp(-10,10);states=self.forecast(x.expand(256,-1),u)
                costs=(torch.einsum('mbki,ij,mbkj->mb',states[:,:,:-1],self.Q,states[:,:,:-1])+torch.einsum('mbi,ij,mbj->mb',states[:,:,-1],12*self.Q,states[:,:,-1])).mean(0)+.03*u.square().sum((1,2))+.002*(u[:,1:]-u[:,:-1]).square().sum((1,2));costs=torch.where(torch.isfinite(costs),costs,torch.inf);elite=u[torch.topk(costs,32,largest=False).indices];mean=.15*mean+.85*elite.mean(0);std=.15*std+.85*elite.std(0).clamp_min(.02);index=int(costs.argmin());value=float(costs[index])
                if value<best_cost:best_cost=value;best=u[index].clone()
            self.previous=best;return best.cpu().numpy(),dict(accepted=bool(np.isfinite(best_cost)),objective=best_cost if np.isfinite(best_cost) else None,status='finite' if np.isfinite(best_cost) else 'nonfinite')


def fit(seed,budget):
    data,identity=load_data(seed);X,U=data['train'];root=OUT/'pets/deterministic_reference'/f'seed{seed}';members=[];records=[]
    for member in range(5):
        ids=np.random.default_rng(np.random.SeedSequence([20261009,seed,member,13579])).integers(len(X),size=len(X));member_seed=10000+5*seed+member
        decoder=fit_transition((X[ids],U[ids]),data['validation'],member_seed,root/f'member{member}',budget,device='cuda');members.append(decoder.transition);records.append(dict(member=member,member_seed=member_seed,bootstrap_sha256=__import__('hashlib').sha256(ids.tobytes()).hexdigest(),model_sha256=digest(root/f'member{member}/best.pt')))
    module=torch.jit.script(EnsembleMap(members).eval())
    with torch.no_grad():
        x=torch.zeros(8,4,device='cuda');u=torch.full((8,30,1),.1,device='cuda');actual=module(x,u);error=0.
        for m,member in enumerate(members):
            current=x;out=[current]
            for k in range(30):current=member(current,u[:,k]);out.append(current)
            expected=torch.stack(out,1);error=max(error,float((actual[m]-expected).abs().max()))
        assert error<1e-4
        literal=sum(objective(actual[m].cpu().numpy(),u.cpu().numpy()) for m in range(5))/5
        states=actual;Qtorch=torch.tensor(Q,dtype=torch.float32,device='cuda');cost=(torch.einsum('mbki,ij,mbkj->mb',states[:,:,:-1],Qtorch,states[:,:,:-1])+torch.einsum('mbi,ij,mbj->mb',states[:,:,-1],12*Qtorch,states[:,:,-1])).mean(0)+.03*u.square().sum((1,2))+.002*(u[:,1:]-u[:,:-1]).square().sum((1,2));relative=float(np.max(np.abs(cost.cpu().numpy()-literal)/(1+np.abs(literal))));assert relative<1e-5
    record=dict(data_identity=identity,members=records,source_sha256=digest(ROOT/'validation/experiments/ensemble_baseline.py'),adapter_sha256=digest(__file__),recipe_sha256=digest(Path(__file__).with_name('transition_training.py')),semantics='whole-episode bootstrap; deterministic five-member mean trajectory costs; no aleatoric sampling; native every-step first action; no common L',planner=dict(candidates=256,elite=32,iterations=3))
    dump(root/'checks.json',dict(passed=True,vectorized_prediction_parity=error,literal_mean_cost_relative_error=relative));dump(root/'manifest.json',record)
    return module,members[0].xu_std[:4],record


def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['fit','pilot','matrix'],required=True);args=p.parse_args();isolation('pets');budget=GPUBudget('pets',8,8);budget.checkpoint();dest=OUT/'pets/deterministic_reference'
    if args.phase=='matrix':
        projection=json.loads((dest/'pilot_projection.json').read_text());assert projection['projected_gpu_hours']<=8-budget.checkpoint()['gpu_seconds']/3600,'Deterministic reference minimum exceeds PETS allocation'
    for seed in ([0] if args.phase=='pilot' else range(3)):
        module,scale,record=fit(seed,budget)
        if args.phase=='fit':continue
        factory=lambda:EnsembleCEM(module,scale,seed)
        for case in (cases()[:16] if args.phase=='pilot' else cases()):evaluate('pets/deterministic_reference',seed,'deterministic_ensemble',case,record,factory,budget)
    if args.phase=='pilot':
        rows=[json.loads(p.read_text()) for p in (dest/'cases').glob('*.json')];mean=float(np.mean([r['wall_seconds'] for r in rows]));dump(dest/'pilot_projection.json',dict(mean_case_seconds=mean,projected_gpu_hours=mean*(168+9*3000/700)/3600,remaining_gpu_hours=8-budget.checkpoint()['gpu_seconds']/3600))
    dump(dest/f'{args.phase}_completion.json',dict(state='complete',accounting=budget.checkpoint()))


if __name__=='__main__':main()
