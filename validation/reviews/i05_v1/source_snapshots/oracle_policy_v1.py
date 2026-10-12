"""I03 policy support. Import is inert; no plant, RNG draws or CUDA setup.

Production objects require CUDA prediction. Explicit fixture constructors use
CPU/synthetic transitions and mark all evidence synthetic. No launcher here.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import time

import oracle_config as oc
import oracle_math_v1 as om


class PolicyError(ValueError):
    def __init__(self, code, detail, evidence=None):
        self.code, self.evidence = code, evidence
        super().__init__(f'{code}: {detail}')


def _index(value, name):
    if type(value) is not int or not 0 <= value < 2**63:
        raise PolicyError('index', f'{name} must be nonnegative int64')


def noise_seed(config, case, sample):
    _index(case, 'case'); _index(sample, 'sample')
    root = config.to_dict()['cases']['root_seed']
    payload = json.dumps([root, 'cem', ['oracle_v1', case, 0], sample], separators=(',', ':')).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], 'little') % (2**63-1)


def indexed_noise(config, case, sample):
    """Full CPU float32 draw; no global seed or device change."""
    import torch
    seed = noise_seed(config, case, sample)
    generator = torch.Generator(device='cpu').manual_seed(seed)
    draws = torch.randn(3, 256, 60, 1, generator=generator, dtype=torch.float32, device='cpu')
    return draws, {'key': ['oracle_v1', case, 0], 'seed': seed,
                   'draw_sha256': hashlib.sha256(draws.numpy().tobytes(order='C')).hexdigest()}


def shift_warm(controls, elapsed):
    import numpy as np
    _index(elapsed, 'elapsed')
    if not isinstance(controls, np.ndarray) or controls.dtype != np.float64 or controls.shape != (60, 1):
        raise PolicyError('warm_shape', 'float64 (60,1) required')
    if not np.isfinite(controls).all() or (abs(controls) > 10).any():
        raise PolicyError('warm_input', 'finite bounded controls required')
    if elapsed == 0:
        return controls.copy()
    if elapsed >= 60:
        return np.repeat(controls[-1:], 60, axis=0)
    return np.concatenate((controls[elapsed:], np.repeat(controls[-1:], elapsed, axis=0)))


def local_rollout(local, step_cpu, state, steps):
    import numpy as np
    x = np.array(state, dtype=np.float64, copy=True)
    if x.shape != (4,) or not np.isfinite(x).all():
        raise PolicyError('local_nonfinite', 'finite shape-(4,) state required')
    states, controls = [x.copy()], []
    for _ in range(steps):
        u, _ = local.input(x)
        nxt = step_cpu(x.copy(), u)
        if not isinstance(nxt, np.ndarray) or nxt.dtype != np.float64 or nxt.shape != (4,) or not np.isfinite(nxt).all():
            raise PolicyError('local_nonfinite', 'malformed/nonfinite local prediction')
        controls.append([u]); states.append(nxt.copy()); x = nxt
    return np.array(states, dtype=np.float64), np.array(controls, dtype=np.float64)


def _host(value):
    return value.detach().cpu().numpy().copy()


@dataclass(frozen=True)
class PlanResult:
    controls: object
    states: object
    feasible: bool
    rank: tuple
    trace: dict


class CEMPlanner:
    def __init__(self, config, gain, step_cpu, step_prediction, *, _fixture=False, noise_provider=None):
        import torch
        self.config, self.c = config, config.to_dict()
        self.local = om.LocalArithmetic(config, gain)
        self.step_cpu, self.step_prediction = step_cpu, step_prediction
        self.fixture = _fixture
        self.device = torch.device('cpu' if _fixture else self.c['execution']['prediction_device'])
        if noise_provider is not None and not _fixture:
            raise PolicyError('noise_override', 'custom draws are fixture-only')
        self.noise_provider = noise_provider or indexed_noise

    @classmethod
    def for_fixture(cls, config, gain, step_cpu, step_prediction, *, noise_provider=None):
        return cls(config, gain, step_cpu, step_prediction, _fixture=True, noise_provider=noise_provider)

    def _rollout(self, state, controls):
        import torch
        z = state.expand(controls.shape[0], 4)
        states = [z]
        for j in range(60):
            z = self.step_prediction(z, controls[:, j, :])
            if (not isinstance(z, torch.Tensor) or z.dtype != torch.float64
                    or z.device != state.device or z.shape != (controls.shape[0], 4)):
                raise PolicyError('prediction_shape', 'matching float64 batched prediction required')
            states.append(z)
        return torch.stack(states, dim=1)

    def plan(self, state, *, case, sample, warm):
        import numpy as np
        import torch
        _index(case, 'case'); _index(sample, 'sample')
        if not isinstance(state, np.ndarray) or state.dtype != np.float64 or state.shape != (4,) or not np.isfinite(state).all():
            raise PolicyError('state', 'finite CPU float64 observation required')
        warm = shift_warm(warm, 0)
        trace = dict(scientific_sha256=oc.scientific_hash(self.config),
                     evidence_kind='synthetic_fixture' if self.fixture else 'scientific',
                     case=case, sample=sample, iterations=[], selected=None)
        try:
            draws, rng = self.noise_provider(self.config, case, sample)
            if (not isinstance(draws, torch.Tensor) or draws.dtype != torch.float32
                    or draws.device.type != 'cpu' or tuple(draws.shape) != (3, 256, 60, 1)
                    or not torch.isfinite(draws).all()):
                raise PolicyError('noise_contract', 'full finite float32 CPU draws required')
            actual_hash = hashlib.sha256(draws.numpy().tobytes(order='C')).hexdigest()
            if (rng.get('key') != ['oracle_v1', case, 0] or rng.get('seed') != noise_seed(self.config, case, sample)
                    or rng.get('draw_sha256') != actual_hash):
                raise PolicyError('noise_contract', 'indexed seed/key/hash mismatch')
            trace.update(rng)
            local_states, local_controls = local_rollout(self.local, self.step_cpu, state, 60)
            trace.update(warm_controls=warm.copy(), local_controls=local_controls, local_states=local_states)
            with torch.no_grad():
                x = torch.as_tensor(state, dtype=torch.float64, device=self.device)
                original_warm = torch.as_tensor(warm, dtype=torch.float64, device=self.device)
                local_plan = torch.as_tensor(local_controls, dtype=torch.float64, device=self.device)
                noise = draws.to(device=self.device, dtype=torch.float64)
                mean = original_warm.clone()
                rms = math.sqrt(float(np.mean(state**2)))
                fraction = min(1., max(0., 2*rms))
                std = torch.full_like(mean, 20*(.05+.45*fraction))
                incumbent = None
                for iteration in range(3):
                    before_mean, before_std = mean.clone(), std.clone()
                    trial = (mean+std*noise[iteration]).clamp(-10, 10)
                    trial[0] = mean.clamp(-10, 10); trial[1] = 0
                    trial[2] = local_plan; trial[3] = original_warm
                    states = self._rollout(x, trial)
                    if iteration == 0:
                        trace['warm_states'] = _host(states[3])
                    scores = om.score_candidates(self.config, states, trial, first_seen_offset=256*iteration)
                    order = om.rank_candidates(scores)
                    row = dict(mean_before=_host(before_mean), std_before=_host(before_std),
                               moments_updated=False, elites=None, mean_after=None, std_after=None)
                    for name in ('first_seen', 'arrays_finite', 'input_bounded', 'arithmetic_finite',
                                 'eligible', 'feasible', 'D', 'W'):
                        row[name] = _host(getattr(scores, name))
                    row['J'] = _host(scores.components.total)
                    row['components'] = {name: _host(getattr(scores.components, name))
                                         for name in ('stage_state', 'stage_input', 'terminal', 'slew')}
                    row['eligible_count'] = int(scores.eligible.sum().item())
                    row['feasible_count'] = int(scores.feasible.sum().item())
                    trace['iterations'].append(row)
                    moments = om.elite_moments(self.config, scores, trial)
                    best = int(order[0].item())
                    key = (not bool(scores.feasible[best].item()), float(scores.D[best].item()),
                           float(scores.W[best].item()), float(scores.components.total[best].item()),
                           int(scores.first_seen[best].item()))
                    if incumbent is None or key < incumbent[0]:
                        incumbent = (key, trial[best].clone(), states[best].clone(),
                                     {name: float(getattr(scores.components, name)[best].item())
                                      for name in ('stage_state', 'stage_input', 'terminal', 'slew')})
                    mean = .15*mean+.85*moments.mean
                    std = .15*std+.85*moments.std
                    row.update(moments_updated=True, elites=_host(moments.indices),
                               mean_after=_host(mean), std_after=_host(std))
                key, u, z, components = incumbent
                trace['selected'] = dict(controls=_host(u), states=_host(z), feasible=not key[0],
                                         eligible=True, rank=key, first_seen=key[-1], components=components)
                return PlanResult(trace['selected']['controls'].copy(), trace['selected']['states'].copy(),
                                  not key[0], key, trace)
        except Exception as error:
            if isinstance(error, (KeyboardInterrupt, SystemExit)):
                raise
            raise PolicyError(getattr(error, 'code', type(error).__name__), str(error), trace) from error


@dataclass(frozen=True)
class Action:
    value: object  # Python float, or None for nonfinite stop
    record: dict


class OraclePolicy:
    """Clarified OA-1 reference state machine, not the pending revised Gate A.

    Dependency injection makes CPU synthetic tests explicit. A production
    planner uses CUDA; the later launcher must check its reviewed release.
    """
    def __init__(self, config, planner, *, case, clock=time.perf_counter):
        _index(case, 'case')
        self.config, self.c, self.case = config, config.to_dict(), case
        self.planner, self.local, self.step_cpu = planner, planner.local, planner.step_cpu
        self.clock = clock
        self.mode, self.previous, self.plan_sample, self.queue_end = 'planning', None, None, None
        self.last_sample, self.first_violation = None, None
        self.safety_latched = self.exception_latched = self.interrupted = False

    def _queue_length(self, sample):
        return 0 if self.queue_end is None else max(0, self.queue_end-sample)

    def _clear(self, record, reason):
        self.previous = self.plan_sample = self.queue_end = None
        record['invalidations'].append(reason)

    def interrupt(self):
        self.previous = self.plan_sample = self.queue_end = None
        self.mode, self.interrupted = 'planning', True

    def _predict(self, x, u):
        import numpy as np
        z = self.step_cpu(x.copy(), u)
        if not isinstance(z, np.ndarray) or z.dtype != np.float64 or z.shape != (4,) or not np.isfinite(z).all():
            raise PolicyError('prediction_nonfinite', 'malformed/nonfinite one-step prediction')
        return z.copy()

    def _local_guard(self, x, mode):
        import math
        cost, safe = self.local.state_cost(x), self.local.safe(x)
        if not math.isfinite(cost):
            raise PolicyError('local_nonfinite', 'nonfinite local state cost')
        threshold = self.c['local']['entry_max' if mode == 'entry' else 'stay_max']
        if not safe or cost > threshold:
            return dict(mode=mode, evaluated=False, passed=False, state_cost=cost,
                        threshold=threshold, current_safe=safe, reason='safety' if not safe else 'threshold')
        steps = self.c['local']['entry_steps' if mode == 'entry' else 'stay_steps']
        states, controls = local_rollout(self.local, self.step_cpu, x, steps)
        guard = self.local.predicate(x, states, mode=mode)
        guard.update(evaluated=True, predicted_states=states, predicted_controls=controls)
        return guard

    def _queue_guard(self, x, sample):
        import numpy as np
        age = sample-self.plan_sample
        expected = self.previous.states[age]
        u = float(self.previous.controls[age, 0])
        match = bool(np.allclose(x, expected, atol=1e-10, rtol=1e-10))
        nxt = self._predict(x, u)
        return dict(age=age, expected_node=expected.copy(), consistent=match,
                    next_node=nxt, next_safe=self.local.safe(nxt), passed=match and self.local.safe(nxt), input=u)

    def act(self, sample, state):
        import numpy as np
        _index(sample, 'sample')
        if self.interrupted:
            raise PolicyError('interrupted', 'case cannot continue after interruption')
        if self.last_sample is not None and sample <= self.last_sample:
            raise PolicyError('sample_order', 'physical sample indices must increase')
        if not self.planner.fixture and sample != (0 if self.last_sample is None else self.last_sample+1):
            raise PolicyError('sample_order', 'scientific cases require consecutive physical samples')
        if not isinstance(state, np.ndarray) or state.dtype != np.float64 or state.shape != (4,):
            raise PolicyError('state', 'CPU float64 shape-(4,) observation required')
        self.last_sample = sample
        start, x = self.clock(), state.copy()
        record = dict(case=self.case, sample=sample, scientific_sha256=oc.scientific_hash(self.config),
                      evidence_kind='synthetic_fixture' if self.planner.fixture else 'scientific', observation=x.copy(),
                      mode_before=self.mode, mode_after=None, branch=None, input=None, local_guard=None,
                      queue_before=self._queue_length(sample), queue_after=None,
                      previous_plan_sample=self.plan_sample, warm_shift=None, invalidations=[], queue_guard=None,
                      planning=None, planning_calls=0, zero_prediction=None, error=None, solve_seconds=0., control_seconds=None)

        def finish(value, branch):
            record.update(input=value, branch=branch, mode_after=self.mode,
                          queue_after=self._queue_length(sample+1), safety_latched=self.safety_latched,
                          exception_latched=self.exception_latched, first_violation=self.first_violation,
                          control_seconds=max(0., self.clock()-start))
            return Action(value, record)

        def zero(reason, error=None):
            self.mode = 'planning'; self._clear(record, reason)
            if error is not None:
                self.exception_latched = True
                record['error'] = dict(code=getattr(error, 'code', type(error).__name__), cause=str(error))
                partial = getattr(error, 'evidence', None)
                if partial is not None:
                    record['planning'] = partial
            try:
                nxt = self._predict(x, 0.)
                record['zero_prediction'] = dict(state=nxt, safe=self.local.safe(nxt))
            except Exception as prediction_error:
                self.exception_latched = True
                record['zero_prediction'] = dict(state=None, safe=False, error=str(prediction_error))
            return finish(0., 'exception_zero' if self.exception_latched and error is not None else reason)

        if not np.isfinite(x).all():
            self.safety_latched = True
            if self.first_violation is None:
                self.first_violation = sample
            self._clear(record, 'nonfinite_stop'); self.mode = 'planning'
            return finish(None, 'nonfinite_stop')
        if not self.local.safe(x):
            self.safety_latched = True
            if self.first_violation is None:
                self.first_violation = sample
        try:
            mode = 'stay' if self.mode == 'local' else 'entry'
            guard = self._local_guard(x, mode)
            record['local_guard'] = guard
            if guard['passed']:
                branch = 'local_stay' if self.mode == 'local' else 'local_enter'
                if branch == 'local_enter':
                    self._clear(record, 'local_enter')
                self.mode = 'local'
                return finish(float(guard['predicted_controls'][0, 0]), branch)
            if self.mode == 'local':
                self._clear(record, 'local_exit'); self.mode = 'planning'
            if self._queue_length(sample):
                queue = self._queue_guard(x, sample); record['queue_guard'] = queue
                if queue['passed']:
                    return finish(queue['input'], 'queued')
                # User clarification: both memories invalidated, one solve from zero.
                self._clear(record, 'queue_guard_failed')
            if self.queue_end is not None and sample >= self.queue_end:
                self.queue_end = None
                record['invalidations'].append('queue_expired')
            elapsed = None if self.previous is None else sample-self.plan_sample
            record['warm_shift'] = elapsed
            warm = np.zeros((60, 1), dtype=np.float64) if self.previous is None else shift_warm(self.previous.controls, elapsed)
            record['planning_calls'] = 1
            solve_start = self.clock()
            try:
                result = self.planner.plan(x, case=self.case, sample=sample, warm=warm)
            finally:
                record['solve_seconds'] = max(0., self.clock()-solve_start)
            record['planning'] = result.trace
            if (result.controls.shape != (60, 1) or result.states.shape != (61, 4)
                    or result.controls.dtype != np.float64 or result.states.dtype != np.float64
                    or not np.isfinite(result.controls).all() or not np.isfinite(result.states).all()
                    or (abs(result.controls) > 10).any()):
                raise PolicyError('plan_malformed', 'malformed selected plan')
            if not result.feasible:
                return zero('no_feasible_plan')
            # Independently verify all-node feasibility before trusting solver flag.
            if any(not self.local.safe(z) for z in result.states):
                raise PolicyError('plan_feasibility', 'selected feasible flag contradicts nodes')
            self.previous, self.plan_sample, self.queue_end = result, sample, sample+5
            queue = self._queue_guard(x, sample); record['queue_guard'] = queue
            if not queue['passed']:
                raise PolicyError('new_plan_guard_failed', 'no second solve allowed on this sample')
            return finish(queue['input'], 'planned')
        except Exception as error:
            return zero('exception_zero', error)
