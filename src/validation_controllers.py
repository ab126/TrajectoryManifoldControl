"""Reference-aware controllers for new experiments; frozen validation is untouched."""
from __future__ import annotations

import time
import numpy as np
import torch
from .manifold_experiments import LearnedObjective, NativeSolver


class ControllerFailure(RuntimeError):
    """No valid manifold action exists; standalone evaluation must stop."""


def validation_lqr(gain, *, x_ref=None, u_ref=None, umax=10.):
    """Validation L law: clip(u_ref - gain @ (x - x_ref)).

    References use normalized plant coordinates. Arbitrary references are accepted;
    exact stationary tracking requires a physically attainable equilibrium.
    """
    gain = np.asarray(gain, dtype=float)
    if gain.ndim != 2 or not np.isfinite(gain).all():
        raise ValueError('gain must be a finite matrix')
    nu, nx = gain.shape
    xr = np.zeros(nx) if x_ref is None else np.asarray(x_ref, dtype=float).reshape(nx).copy()
    ur = np.zeros(nu) if u_ref is None else np.asarray(u_ref, dtype=float).reshape(nu).copy()
    if not np.isfinite(xr).all() or not np.isfinite(ur).all() or not np.isfinite(umax) or umax <= 0:
        raise ValueError('references and positive input bound must be finite')

    def control(k, state):
        state = np.asarray(state, dtype=float).reshape(nx)
        if not np.isfinite(state).all():
            raise ControllerFailure('nonfinite measurement')
        return np.clip(ur - gain @ (state - xr), -umax, umax).reshape(nu)
    return control


class TrackingObjective(LearnedObjective):
    def __init__(self, Q, R, x_ref, u_ref, **kwargs):
        super().__init__(Q, R, **kwargs)
        self.x_ref, self.u_ref = x_ref, u_ref

    def __call__(self, x, u, previous_input=None):
        previous = None if previous_input is None else previous_input - self.u_ref
        return super().__call__(x - self.x_ref, u - self.u_ref, previous)


def trajectory_controller(decoder, config, *, x_ref=None, u_ref=None,
                          Q=None, R=None, umax=10., linear=None, switch_cost=.75):
    """Standalone TMX (linear=None), or hybrid sharing an injected L callable.

    Hybrid uses L inside the reference-centered switch region and on failed
    solves. Standalone raises ControllerFailure, never silently substitutes input.
    """
    if not 1 <= config.replan_interval <= decoder.horizon:
        raise ValueError('replan interval must lie within horizon')
    if not np.isfinite(switch_cost) or switch_cost < 0:
        raise ValueError('switch cost must be finite and nonnegative')
    p = next(decoder.parameters())
    Q = torch.diag(p.new_tensor([.02, 30., 100., 10.])) if Q is None else Q
    R = p.new_tensor([[.03]]) if R is None else R
    xr = p.new_zeros(decoder.x_dim) if x_ref is None else p.new_tensor(x_ref).reshape(decoder.x_dim)
    ur = p.new_zeros(decoder.u_dim) if u_ref is None else p.new_tensor(u_ref).reshape(decoder.u_dim)
    if not torch.isfinite(xr).all() or not torch.isfinite(ur).all() or not np.isfinite(umax) or umax <= 0:
        raise ValueError('references and positive input bound must be finite')
    solver = NativeSolver(decoder, Q, R, config, bounds=(-umax, umax))
    solver.objective = TrackingObjective(Q, R, xr, ur, periodic=config.periodic,
        terminal_window=config.terminal_window, include_previous_input=config.include_previous_input)
    previous = None
    plan_step = last_step = None
    previous_input = ur.clamp(-umax, umax)
    d = dict(solve_seconds=[], loss=[], solve_steps=[], linear_steps=[],
             fallback_events=[], failure_steps=[])

    def control(k, state):
        nonlocal previous, plan_step, last_step, previous_input
        if last_step is not None and k <= last_step:
            raise ValueError('sample indices must increase')
        current = p.new_tensor(np.asarray(state)).reshape(decoder.x_dim)
        if not torch.isfinite(current).all():
            raise ControllerFailure('nonfinite measurement')
        consecutive = last_step is None or k == last_step + 1
        age = None if plan_step is None else k - plan_step
        error = current - xr
        if linear is not None and float(error @ Q @ error) < switch_cost:
            action = np.asarray(linear(k, state), dtype=float)
            d['linear_steps'].append(k)
            previous = None
            plan_step = None
        elif previous is not None and consecutive and age < config.replan_interval:
            action = previous[age]
        else:
            initial = ur.expand(decoder.horizon, -1).clone().clamp(-umax, umax)
            if previous is not None and consecutive and age < decoder.horizon:
                initial = torch.cat((previous[age:], previous[-1:].expand(age, -1)))
            start = time.perf_counter()
            result = solver.solve(current, initial, previous_input)
            if p.is_cuda:
                torch.cuda.synchronize(p.device)
            d['solve_seconds'].append(time.perf_counter() - start)
            d['solve_steps'].append(k)
            d['loss'].append(result.loss)
            accepted = result.finite and np.isfinite(result.loss)
            accepted = accepted and bool(torch.isfinite(result.x).all() and torch.isfinite(result.u).all())
            accepted = accepted and bool((result.u.abs() <= umax).all())
            if accepted:
                previous = result.u.detach().clone()
                plan_step = k
                action = previous[0]
            else:
                d['failure_steps'].append(k)
                previous = None
                plan_step = None
                if linear is None:
                    raise ControllerFailure(f'no valid manifold plan at sample {k}')
                action = np.asarray(linear(k, state), dtype=float)
                d['fallback_events'].append(k)
                d['linear_steps'].append(k)
        previous_input = torch.as_tensor(action, device=p.device, dtype=p.dtype).detach().clone()
        last_step = k
        return action.detach().cpu().numpy().copy() if torch.is_tensor(action) else action.copy()

    control.solver, control.diagnostics = solver, d
    return control
