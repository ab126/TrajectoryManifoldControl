"""Opt-in native manifold MPC variants; original implementations stay unchanged.

Only a learned decoder, measured state, previous applied input and bounded input
coordinates enter these optimizers. No linear policy or plant access is present.
"""
from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
import torch

from .manifold_control import DataDrivenCEMSolver, DataDrivenMPCSolution


@dataclass(frozen=True)
class NativeConfig:
    solver: str = 'safe_cem'
    replan_interval: int = 1
    periodic: bool = False
    terminal_window: int = 1
    include_previous_input: bool = False
    iterations: int = 40
    learning_rate: float = .15
    starts: int = 3
    population: int = 256
    elite: int = 32
    cem_iterations: int = 3
    seed: int = 0


class LearnedObjective:
    def __init__(self, Q, R, *, periodic=False, terminal_window=1,
                 input_rate_weight=.002, include_previous_input=False):
        if terminal_window<1:raise ValueError('terminal_window must be positive')
        if periodic and not torch.equal(Q,torch.diag(torch.diagonal(Q))):
            raise ValueError('periodic objective requires diagonal state weights')
        self.Q,self.R=Q,R
        self.periodic,self.terminal_window=periodic,terminal_window
        self.rate=input_rate_weight
        self.include_previous_input=include_previous_input

    def state_cost(self,x):
        if self.periodic:
            # Smooth periodic replacement has the same local quadratic curvature.
            other=x.clone();other[...,2]=0
            return torch.einsum('...i,ij,...j->...',other,self.Q,other)+2*self.Q[2,2]*(1-torch.cos(x[...,2]))
        return torch.einsum('...i,ij,...j->...',x,self.Q,x)

    def __call__(self,x,u,previous_input=None):
        value=self.state_cost(x[:,:-1]).sum(1)
        value=value+torch.einsum('bki,ij,bkj->b',u,self.R,u)
        window=min(self.terminal_window,u.shape[1])
        value=value+12*self.state_cost(x[:,-window:]).mean(1)
        value=value+self.rate*(u[:,1:]-u[:,:-1]).square().sum((1,2))
        if self.include_previous_input and previous_input is not None:
            value=value+self.rate*(u[:,0]-previous_input).square().sum(1)
        return value


class NativeSolver:
    def __init__(self,decoder,Q,R,config,*,bounds=(-10.,10.)):
        if config.solver not in ('original_cem','safe_cem','adam'):
            raise ValueError('unknown solver')
        if config.starts<1 or config.iterations<1 or not 0<config.elite<=config.population:
            raise ValueError('invalid optimizer sizes')
        if not bounds[0]<bounds[1]:raise ValueError('bounds must increase')
        self.decoder,self.config=decoder,config
        self.lower,self.upper=bounds
        for parameter in decoder.parameters():parameter.requires_grad_(False)
        self.objective=LearnedObjective(Q,R,periodic=config.periodic,
            terminal_window=config.terminal_window,include_previous_input=config.include_previous_input)
        self.generator=torch.Generator(device=Q.device).manual_seed(config.seed)
        self.original=DataDrivenCEMSolver(decoder,Q,R,Q_terminal=12*Q,
            u_bounds=bounds,population=config.population,elite=config.elite,
            iterations=config.cem_iterations,input_rate_weight=.002,seed=config.seed)
        self.diagnostics={}

    def solve(self,state,initial=None,previous_input=None):
        cfg=self.config;shape=(self.decoder.horizon,self.decoder.u_dim)
        initial=torch.zeros(shape,device=state.device,dtype=state.dtype) if initial is None else initial.detach().clone()
        initial=torch.nan_to_num(initial).clamp(self.lower,self.upper)
        self.diagnostics=dict(evaluations=0,nonfinite_candidates=0,accepted_updates=0)
        if cfg.solver=='original_cem':
            # Identical original solver, shared objective changes only if opted in.
            self.original._cost=lambda x,u:self.objective(x,u,previous_input)
            result=self.original.solve(state,initial)
            self.diagnostics['evaluations']=cfg.population*cfg.cem_iterations
            return result
        best_loss=float('inf');best_u=initial.clone();best_x=None;completed=0

        def evaluate(inputs):
            nonlocal best_loss,best_u,best_x
            states=self.decoder.rollout(state.expand(len(inputs),-1),inputs)
            losses=self.objective(states,inputs,previous_input)
            valid=torch.isfinite(losses)&torch.isfinite(states).all(dim=(1,2))&torch.isfinite(inputs).all(dim=(1,2))
            self.diagnostics['evaluations']+=len(inputs)
            self.diagnostics['nonfinite_candidates']+=int((~valid).sum())
            safe=losses.masked_fill(~valid,float('inf'))
            index=int(safe.argmin())
            if float(safe[index].detach())<best_loss:
                best_loss=float(safe[index].detach());best_u=inputs[index].detach().clone()
                best_x=states[index].detach().clone()
                self.diagnostics['accepted_updates']+=1
            return losses,safe,valid

        if cfg.solver=='safe_cem':
            mean=initial
            scale=self.decoder.transition.xu_std[:self.decoder.x_dim].clamp_min(1e-6)
            radius=torch.sqrt(((state/scale)**2).mean())
            exploration=(.05+.45*float((2*radius).clamp(0,1)))*(self.upper-self.lower)
            std=torch.full_like(mean,exploration)
            with torch.no_grad():
                for iteration in range(cfg.cem_iterations):
                    noise=torch.randn((cfg.population,*shape),device=state.device,dtype=state.dtype,generator=self.generator)
                    inputs=(mean+std*noise).clamp(self.lower,self.upper);inputs[0]=mean
                    _,safe,valid=evaluate(inputs);completed=iteration+1
                    count=min(cfg.elite,int(valid.sum()))
                    if count==0:break
                    elites=inputs[torch.topk(safe,count,largest=False).indices]
                    # population variance also handles a single finite candidate.
                    new_std=elites.std(dim=0,correction=1 if count>1 else 0).clamp_min(.02)
                    mean=.15*mean+.85*elites.mean(0);std=.15*std+.85*new_std
        else:
            starts=initial.expand(cfg.starts,*shape).clone()
            if cfg.starts>1:starts[1]=0
            if cfg.starts>2:
                starts[2:]=self.lower+(self.upper-self.lower)*torch.rand(
                    (cfg.starts-2,*shape),device=state.device,dtype=state.dtype,generator=self.generator)
            inputs=torch.nn.Parameter(starts)
            optimizer=torch.optim.Adam([inputs],lr=cfg.learning_rate)
            for iteration in range(cfg.iterations+1):
                optimizer.zero_grad(set_to_none=True)
                losses,_,valid=evaluate(inputs)
                if iteration==cfg.iterations or not bool(valid.any()):break
                loss=losses[valid].sum()
                loss.backward()
                if inputs.grad is None:break
                inputs.grad.nan_to_num_(nan=0.,posinf=0.,neginf=0.)
                inputs.grad[~valid]=0
                torch.nn.utils.clip_grad_norm_([inputs],100.)
                optimizer.step()
                with torch.no_grad():inputs.clamp_(self.lower,self.upper)
                completed=iteration+1
        if best_x is None:
            with torch.no_grad():best_x=self.decoder.rollout(state,best_u)
        return DataDrivenMPCSolution(x=best_x,u=best_u,loss=best_loss,
            iterations=completed,finite=np.isfinite(best_loss))


def native_controller(decoder,config,*,Q=None,R=None,bounds=(-10.,10.)):
    """Standalone policy with explicit failures, no linear feedback branch."""
    if not 1<=config.replan_interval<=decoder.horizon:
        raise ValueError('feedback interval must lie within the horizon')
    parameter=next(decoder.parameters())
    Q=torch.diag(parameter.new_tensor([.02,30.,100.,10.])) if Q is None else Q
    R=parameter.new_tensor([[.03]]) if R is None else R
    solver=NativeSolver(decoder,Q,R,config,bounds=bounds)
    previous=None;plan_step=None;last_step=None;previous_input=parameter.new_zeros(decoder.u_dim)
    diagnostics=dict(loss=[],iterations=[],solve_seconds=[],predictions=[],terminal_policy_used=[],
        fallback_events=[],solve_steps=[],warm_start_age=[],solver=[])

    def control(k,state):
        nonlocal previous,plan_step,last_step,previous_input
        if last_step is not None and k<=last_step:raise ValueError('sample indices must increase')
        consecutive=last_step is None or k==last_step+1
        age=None if plan_step is None else k-plan_step
        if previous is not None and consecutive and age<config.replan_interval:
            action=previous[age]
        else:
            if previous is not None and age<decoder.horizon and consecutive:
                initial=torch.cat((previous[age:],previous[-1:].expand(age,-1)))
            else:
                initial=parameter.new_zeros((decoder.horizon,decoder.u_dim))
            start=time.perf_counter()
            result=solver.solve(parameter.new_tensor(np.asarray(state)),initial,previous_input)
            if parameter.is_cuda:torch.cuda.synchronize(parameter.device)
            diagnostics['solve_seconds'].append(time.perf_counter()-start)
            diagnostics['solve_steps'].append(k);diagnostics['loss'].append(result.loss)
            diagnostics['iterations'].append(result.iterations);diagnostics['warm_start_age'].append(age)
            diagnostics['solver'].append(solver.diagnostics.copy())
            diagnostics['predictions'].append(result.x.detach().cpu().numpy())
            accepted=result.finite and bool(torch.isfinite(result.u).all()) and bool(torch.isfinite(result.x).all())
            accepted=accepted and bool(((result.u>=bounds[0])&(result.u<=bounds[1])).all())
            if accepted:
                previous=result.u.detach().clone();plan_step=k;action=previous[0]
            else:
                diagnostics['fallback_events'].append(k)
                action=initial[0];previous=None;plan_step=None
        previous_input=action.detach().clone();last_step=k
        diagnostics['terminal_policy_used'].append(False)
        return action.detach().cpu().numpy().copy()
    control.diagnostics=diagnostics
    control.solver=solver
    return control
