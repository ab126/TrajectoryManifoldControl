"""Separate ordinary shooting implementation; imports no submitted solver/decoder.

Shared dependencies: NumPy/PyTorch, transition object/weights, Q/R numerical
specification, local gain, indexed externally supplied standard normal tensors.
Rollout, objective, CEM, queue, switch and warm start are implemented here.
Limitation: written by the orchestrator after source inspection; no fresh-context
Terra task was available. This is implementation separation, not author independence.
"""
import math
import numpy as np
import torch

class Shooting:
    def __init__(self, transition, horizon=30, gain=None, device='cpu', Q=None, R=None,
                 population=256, elites=32, iterations=3):
        self.f=transition; self.N=horizon; self.gain=gain; self.device=device
        self.Q=torch.diag(torch.tensor([.02,30.,100.,10.],device=device)) if Q is None else Q
        self.R=torch.tensor([[.03]],device=device) if R is None else R
        self.Qf=12*self.Q; self.P=population; self.K=elites; self.I=iterations
        self.previous=torch.zeros(horizon,1,device=device)
        self.queue=torch.empty(0,1,device=device)
        self.last=None; self.branch=None

    def rollout(self,state,actions):
        rows=[state]
        for action in actions.unbind(-2):
            rows.append(self.f(rows[-1],action))
        return torch.stack(rows,dim=-2)

    def objective(self,states,actions):
        # Preserve float32 reduction order to permit the declared 1e-5 tolerance.
        running=torch.einsum('bki,ij,bkj->b',states[:,:-1],self.Q,states[:,:-1])
        effort=torch.einsum('bki,ij,bkj->b',actions,self.R,actions)
        end=torch.einsum('bi,ij,bj->b',states[:,-1],self.Qf,states[:,-1])
        rate=.002*((actions[:,1:]-actions[:,:-1])**2).sum((1,2))
        return running+effort+end+rate

    @torch.no_grad()
    def solve(self,state,initial,draws):
        mean=initial.clone()
        normalized=((state/self.f.xu_std[:4])**2).mean().sqrt()
        fraction=(2*normalized).clamp(0,1)
        std=torch.full_like(mean,(.05+.45*float(fraction))*20)
        best_value=math.inf; best=mean; trace=[]
        for eps in draws:
            trial=(mean+std*eps).clamp(-10,10)
            trial[0]=mean.clamp(-10,10)
            trajectories=self.rollout(state.expand(self.P,-1),trial)
            values=self.objective(trajectories,trial); trace.append(values.clone())
            chosen=trial[torch.topk(values,self.K,largest=False).indices]
            mean=.15*mean+.85*chosen.mean(0)
            std=.15*std+.85*chosen.std(0).clamp_min(.02)
            index=int(values.argmin())
            value=float(values[index])
            if value<best_value: best_value,best=value,trial[index].clone()
        return {'u':best,'x':self.rollout(state,best),'loss':best_value,
                'finite':math.isfinite(best_value),'costs':trace}

    def control(self,state,draws):
        if self.gain is not None and float(np.asarray(state)@self.Q.cpu().numpy()@np.asarray(state))<.75:
            self.queue=torch.empty(0,1,device=self.device)
            self.branch='local'
            return np.clip(-np.asarray(self.gain)@state,-10,10).reshape(1)
        if len(self.queue):
            value=self.queue[0].clone(); self.queue=self.queue[1:]
            self.branch='queue'; return value.cpu().numpy()
        state=torch.as_tensor(state,dtype=torch.float32,device=self.device)
        initial=torch.cat((self.previous[5:],self.previous[-5:]))
        self.last=self.solve(state,initial,draws)
        if not self.last['finite']:
            self.previous=torch.zeros_like(self.previous); self.branch='nonfinite'
            return np.zeros(1)
        self.previous=self.last['u'].detach()
        self.queue=self.previous[:5].clone()
        value=self.queue[0].clone(); self.queue=self.queue[1:]
        self.branch='plan'; return value.cpu().numpy()
