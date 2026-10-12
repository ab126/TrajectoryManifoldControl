"""Audited PyTorch port of PETS probabilistic ensemble and TS-infinity semantics.

Reference: kchua/handful-of-trials, MIT license (saved in results_literature/sources).
Offline fixed-data adaptation; no online data aggregation or catch feedback.
"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from common import Q, HORIZON


class EnsembleLayer(nn.Module):
    def __init__(self,inputs,outputs):
        super().__init__()
        self.weight=nn.Parameter(torch.empty(5,inputs,outputs))
        self.bias=nn.Parameter(torch.zeros(5,1,outputs))
        nn.init.trunc_normal_(self.weight,std=1/(2*np.sqrt(inputs)))

    def forward(self,x):return torch.bmm(x,self.weight)+self.bias


class PETS(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers=nn.ModuleList([EnsembleLayer(a,b) for a,b in zip([5,200,200,200,200],[200,200,200,200,8])])
        self.max_logvar=nn.Parameter(torch.full((1,1,4),.5))
        self.min_logvar=nn.Parameter(torch.full((1,1,4),-10.))
        self.register_buffer('input_mean',torch.zeros(5));self.register_buffer('input_std',torch.ones(5))
        self.register_buffer('target_mean',torch.zeros(4));self.register_buffer('target_std',torch.ones(4))

    def forward(self,x):
        x=(x-self.input_mean)/self.input_std
        for layer in self.layers[:-1]:x=F.silu(layer(x))
        output=self.layers[-1](x);mean,raw=output[...,:4],output[...,4:]
        logvar=self.max_logvar-F.softplus(self.max_logvar-raw)
        logvar=self.min_logvar+F.softplus(logvar-self.min_logvar)
        return mean,logvar

    def moments(self,x):
        mean,lv=self(x)
        return mean*self.target_std+self.target_mean,torch.exp(lv)*self.target_std**2

    def normalized_loss(self,inputs,targets):
        mean,lv=self(inputs)
        target=(targets-self.target_mean)/self.target_std
        nll=((mean-target)**2*torch.exp(-lv)+lv).mean(dim=(-1,-2)).sum()
        # Source's learned log-variance bound regularizer and layer weight decay.
        decay=sum(rate*(layer.weight**2).sum()/2 for rate,layer in zip([2.5e-5,5e-5,7.5e-5,7.5e-5,1e-4],self.layers))
        return nll+.01*self.max_logvar.sum()-.01*self.min_logvar.sum()+decay

    def predict_inputs(self,x,U):
        """Deterministic ensemble-mean forecasts for shared rollout selection only."""
        device=self.input_mean.device
        with torch.no_grad():
            current=torch.as_tensor(x,dtype=torch.float32,device=device)
            inputs=torch.as_tensor(U,dtype=torch.float32,device=device)
            out=[current]
            for k in range(inputs.shape[-2]):
                xu=torch.cat((current,inputs[...,k,:]),dim=-1).reshape(-1,5)
                mean,_=self.moments(xu.unsqueeze(0).expand(5,-1,-1))
                current=current+mean.mean(0).reshape(current.shape);out.append(current)
            return torch.stack(out,dim=-2).cpu().numpy()


class TSInfinityCEM:
    """Each particle keeps one member throughout a rollout, with aleatoric draws."""
    def __init__(self,model,seed,particles=20,population=256):
        assert particles%5==0
        self.model=model;self.particles=particles;self.population=population
        self.device=model.input_mean.device;self.generator=torch.Generator(device=self.device).manual_seed(seed)
        self.previous=torch.zeros(HORIZON,1,device=self.device)
        self.Q=torch.as_tensor(Q,dtype=torch.float32,device=self.device)

    def costs(self,x,U):
        n=len(U);per_member=self.particles//5
        # Axes: member, candidate, particle belonging to member, state.
        states=x.expand(5,n,per_member,4).clone()
        values=torch.zeros(5,n,per_member,device=self.device)
        for k in range(HORIZON):
            values+=torch.einsum('enpi,ij,enpj->enp',states,self.Q,states)
            applied=U[:,k].unsqueeze(0).unsqueeze(2).expand(5,n,per_member,1)
            xu=torch.cat((states,applied),dim=-1).reshape(5,-1,5)
            mean,var=self.model.moments(xu)
            noise=torch.randn(mean.shape,generator=self.generator,device=self.device)
            states=states+(mean+var.sqrt()*noise).reshape(5,n,per_member,4)
        values+=torch.einsum('enpi,ij,enpj->enp',states,12*self.Q,states)
        cost=values.mean(dim=(0,2))+.03*(U**2).sum(dim=(1,2))+.002*(U[:,1:]-U[:,:-1]).square().sum(dim=(1,2))
        return torch.where(torch.isfinite(cost),cost,torch.inf)

    def solve(self,x):
        with torch.no_grad():
            x=torch.as_tensor(x,dtype=torch.float32,device=self.device)
            mean=torch.cat((self.previous[1:],self.previous[-1:]))
            fraction=torch.clamp(2*torch.sqrt(torch.mean((x/self.model.input_std[:4])**2)),0,1)
            std=torch.ones_like(mean)*((.05+.45*fraction)*20)
            best_cost=float('inf');best=torch.zeros_like(mean)
            for _ in range(3):
                noise=torch.randn((self.population,HORIZON,1),generator=self.generator,device=self.device)
                candidates=(mean+std*noise).clamp(-10,10);candidates[0]=mean.clamp(-10,10)
                costs=self.costs(x,candidates);elite=candidates[torch.topk(costs,32,largest=False).indices]
                mean=.15*mean+.85*elite.mean(0);std=.15*std+.85*elite.std(0).clamp_min(.02)
                index=int(costs.argmin());value=float(costs[index])
                if value<best_cost:best_cost=value;best=candidates[index].clone()
            accepted=bool(np.isfinite(best_cost));self.previous=best
            return best.cpu().numpy(),dict(accepted=accepted,status='finite' if accepted else 'nonfinite',objective=best_cost if accepted else None,particles=self.particles,population=self.population)
