import numpy as np
import pytest
import torch

from src.manifold_control import CanonicalBehaviorDecoder, DataDrivenTransition, DataDrivenMPCSolution
from src.manifold_experiments import NativeConfig, NativeSolver, LearnedObjective, native_controller


class Additive(DataDrivenTransition):
    def __init__(self):super().__init__(1,1,())
    def forward(self,x,u):return x+u


def test_adam_accepts_matching_snapshot_and_improves_over_initial():
    decoder=CanonicalBehaviorDecoder(Additive(),4)
    solver=NativeSolver(decoder,torch.ones(1,1),torch.ones(1,1)*.03,
                        NativeConfig(solver='adam',iterations=30,starts=3),bounds=(-.3,.3))
    state=torch.ones(1);initial=torch.zeros(4,1)
    first=float(solver.objective(decoder.rollout(state,initial)[None],initial[None])[0])
    result=solver.solve(state,initial)
    actual=float(solver.objective(result.x[None],result.u[None])[0])
    assert result.finite and result.loss==pytest.approx(actual)
    assert result.loss<first
    assert torch.all(result.u.abs()<=.3)
    assert all(p.grad is None for p in decoder.parameters())


def test_safe_cem_masks_nonfinite_candidates():
    class Partial(Additive):
        def forward(self,x,u):
            return torch.where(u.abs()>.5,torch.full_like(x,float('nan')),x+u)
    decoder=CanonicalBehaviorDecoder(Partial(),3)
    solver=NativeSolver(decoder,torch.ones(1,1),torch.ones(1,1),
        NativeConfig(population=32,elite=8,cem_iterations=2),bounds=(-1.,1.))
    result=solver.solve(torch.ones(1))
    assert result.finite and torch.isfinite(result.x).all()
    assert solver.diagnostics['nonfinite_candidates']>0


def test_all_invalid_is_explicit_failure():
    class Invalid(Additive):
        def forward(self,x,u):return torch.full_like(x,float('nan'))
    solver=NativeSolver(CanonicalBehaviorDecoder(Invalid(),3),torch.ones(1,1),torch.ones(1,1),
                        NativeConfig(population=4,elite=2))
    result=solver.solve(torch.ones(1))
    assert not result.finite


def test_periodic_objective_and_previous_applied_input():
    q=torch.eye(4);r=torch.ones(1,1)
    objective=LearnedObjective(q,r,periodic=True,include_previous_input=True)
    x=torch.zeros(1,4,4);u=torch.zeros(1,3,1)
    value=objective(x,u,torch.tensor([2.]))
    x[:,:,2]+=2*torch.pi
    torch.testing.assert_close(objective(x,u,torch.tensor([2.])),value)
    assert float(value)==pytest.approx(.008)


def test_warm_start_shifts_by_applied_actions_and_resets_after_gap():
    decoder=CanonicalBehaviorDecoder(Additive(),4)
    ctl=native_controller(decoder,NativeConfig(replan_interval=2),Q=torch.ones(1,1),R=torch.ones(1,1))
    seen=[]
    def solve(state,initial,previous_input):
        seen.append((initial.clone(),previous_input.clone()))
        u=torch.arange(4,dtype=torch.float32)[:,None]
        return DataDrivenMPCSolution(decoder.rollout(state,u),u,0.,1,True)
    ctl.solver.solve=solve
    ctl(0,np.zeros(1));ctl(1,np.zeros(1));ctl(2,np.zeros(1));ctl(8,np.zeros(1))
    torch.testing.assert_close(seen[1][0],torch.tensor([[2.],[3.],[3.],[3.]]))
    torch.testing.assert_close(seen[1][1],torch.tensor([1.]))
    assert torch.equal(seen[2][0],torch.zeros(4,1))
    assert not any(ctl.diagnostics['terminal_policy_used'])


def test_failed_solver_records_fallback_without_linear_switch():
    decoder=CanonicalBehaviorDecoder(Additive(),4)
    ctl=native_controller(decoder,NativeConfig(),Q=torch.ones(1,1),R=torch.ones(1,1))
    def fail(state,initial,previous_input):
        return DataDrivenMPCSolution(decoder.rollout(state,initial),initial,float('inf'),0,False)
    ctl.solver.solve=fail
    assert ctl(0,np.ones(1))==np.array([0.])
    assert ctl.diagnostics['fallback_events']==[0]
    assert ctl.diagnostics['terminal_policy_used']==[False]
