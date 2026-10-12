"""I04 independent numerical replay. Import is inert; no I02/I03 or SUT imports.

Scientific replay uses the pinned CUDA device; fixture callbacks are CPU-only.
No missing experimental evidence is regenerated. This is independent code,
authored by the same reviewer, not independent-author certification.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

import oracle_config as oc


class AuditIssue(ValueError):
    def __init__(self, code, field, detail='', status='fail'):
        self.code, self.field, self.status = code, field, status
        super().__init__(f'{code} at {field}: {detail}')


def _require(ok, code, field):
    if not ok:
        raise AuditIssue(code, field)


def _same(want, got, field):
    """Strict shapes/types/keys, tolerant finite arithmetic; equal NaN flags."""
    import numpy as np
    if isinstance(want, np.ndarray):
        _require(isinstance(got, np.ndarray) and got.dtype == want.dtype and got.shape == want.shape,
                 'array_contract', field)
        ok = (np.allclose(want, got, atol=1e-10, rtol=1e-10, equal_nan=True)
              if want.dtype.kind == 'f' else np.array_equal(want, got))
        _require(bool(ok), 'numeric_mismatch', field)
    elif isinstance(want, dict):
        _require(isinstance(got, dict) and set(want) == set(got), 'record_fields', field)
        for k in want:
            _same(want[k], got[k], field+'.'+k)
    elif isinstance(want, (list, tuple)):
        _require(type(want) is type(got) and len(want) == len(got), 'sequence_contract', field)
        for j, (a, b) in enumerate(zip(want, got)):
            _same(a, b, f'{field}[{j}]')
    elif type(want) is float:
        _require(type(got) is float and math.isclose(want, got, abs_tol=1e-10, rel_tol=1e-10),
                 'numeric_mismatch', field)
    else:
        _require(type(want) is type(got) and want == got, 'value_mismatch', field)


def check(operation):
    """Unavailable and malformed evidence are separate, nonpassing outcomes."""
    try:
        operation()
        return oc.Report((oc.Check('pass', 'replay', 'trace'),))
    except AuditIssue as error:
        return oc.Report((oc.Check(error.status, error.code, error.field, observed=str(error)),))
    except RuntimeError as error:
        return oc.Report((oc.Check('unavailable','backend_runtime','trace',observed=str(error)),))
    except (ValueError, TypeError, KeyError, IndexError, OverflowError) as error:
        return oc.Report((oc.Check(getattr(error,'status','fail'), getattr(error,'code','malformed_trace'),
                                   'trace', observed=str(error)),))


def read_pinned(path, expected_sha256):
    try:
        raw = Path(path).read_bytes()
    except FileNotFoundError as error:
        raise AuditIssue('archive_missing', str(path), status='unavailable') from error
    _require(hashlib.sha256(raw).hexdigest() == expected_sha256, 'artifact_hash', str(path))
    return raw


def uniform_rod_step(config, x, u):
    """Independent held-input RK4; accepts NumPy or Torch float64 batches.

    The 2x2 mass matrix is inverted analytically. Never imports the original
    simulator; its numerical agreement remains a separately reviewed check.
    """
    torch_backend = type(x).__module__.split('.')[0] == 'torch'
    if torch_backend:
        import torch as xp
        stack = lambda a: xp.stack(a, dim=-1)
    else:
        import numpy as xp
        stack = lambda a: xp.stack(a, axis=-1)
    control = u[..., 0] if hasattr(u, 'shape') and u.shape == x.shape[:-1]+(1,) else u
    def rhs(z):
        c, s = xp.cos(z[..., 2]), xp.sin(z[..., 2])
        cross = c/2
        force = control+s*z[..., 3]**2/2
        gravity = s/2
        det = 1-cross**2  # (M+1)/3 for M=2
        return stack((z[..., 1], (force/3-cross*gravity)/det,
                      z[..., 3], (3*gravity-cross*force)/det))
    h = config.to_dict()['plant']['dt_normalized']
    a = rhs(x); b = rhs(x+h*a/2); c = rhs(x+h*b/2); d = rhs(x+h*c)
    return x+h*(a+2*b+2*c+d)/6


def verify_gain(config, record):
    """Independent augmented-system RK4 polynomial and DARE reconstruction."""
    import numpy as np
    from scipy.linalg import solve_discrete_are
    c = config.to_dict()
    _require(isinstance(record,dict) and set(record)=={'artifact_version','scientific_sha256',
        'specification_sha256','method','A','B','P','K','spectral_radius','dare_relative_residual','reference_verified'},
        'gain_fields','gain')
    _require(type(record['artifact_version']) is int,'gain_type','gain.artifact_version')
    for name,shape in [('A',(4,4)),('B',(4,1)),('P',(4,4)),('K',(1,4))]:
        values=record[name]
        _require(type(values) is list and len(values)==shape[0] and all(type(row) is list
            and len(row)==shape[1] and all(type(v) in (int,float) and math.isfinite(v) for v in row)
            for row in values),'gain_type','gain.'+name)
    L = np.zeros((5, 5), dtype=np.float64)
    L[0, 1] = L[2, 3] = 1.
    L[1, 2], L[3, 2], L[1, 4], L[3, 4] = -1/3, 2., 4/9, -2/3
    Z = np.eye(5); term = np.eye(5)
    for j in range(1, 5):
        term = term @ (c['plant']['dt_normalized']*L)/j
        Z = Z+term
    A, B = Z[:4, :4], Z[:4, 4:5]
    Q = np.diag(c['objective']['Q']); R = np.array([[c['objective']['R']]])
    P = solve_discrete_are(A, B, Q, R)
    K = np.linalg.solve(R+B.T@P@B, B.T@P@A)
    for name, expected in [('A', A), ('B', B), ('P', P), ('K', K)]:
        actual = np.asarray(record[name], dtype=np.float64)
        _same(expected, actual, 'gain.'+name)
    _require(record['scientific_sha256'] == oc.scientific_hash(config)
             and record['specification_sha256'] == oc.SPEC_SHA256
             and record['method'] == c['local']['gain_construction']
             and record['artifact_version'] == 1 and record['reference_verified'] is True,
             'gain_binding', 'gain')
    # Recompute diagnostics using the artifact matrices, independently of I02.
    p, k = np.asarray(record['P']), np.asarray(record['K'])
    residual = Q+A.T@p@A-p-A.T@p@B@k
    relative = float(np.linalg.norm(residual, np.inf)/max(1., np.linalg.norm(p, np.inf)))
    radius = float(np.max(np.abs(np.linalg.eigvals(A-B@k))))
    _require(np.linalg.eigvalsh(p).min() > 0 and relative <= 2e-10 and radius < 1,
             'gain_stability', 'gain')
    _same(relative, record['dare_relative_residual'], 'gain.residual')
    _same(radius, record['spectral_radius'], 'gain.radius')
    return p, k


def scores(config, X, U, offset=0):
    """Independent batch scoring; preserve prediction backend for reductions."""
    import numpy as np
    if type(X).__module__.split('.')[0] == 'torch':
        import torch
        _require(X.dtype == U.dtype == torch.float64 and X.device == U.device, 'dtype', 'candidates')
        q = X.new_tensor(config.to_dict()['objective']['Q'])
        qf = X.new_tensor(config.to_dict()['objective']['Qf'])
        sum_ = lambda a, dims: a.sum(dim=dims)
        all_ = lambda a, dims: a.all(dim=dims)
        maximum = lambda a: a.amax(dim=-1)
        pos = lambda a: a.clamp_min(0)
        finite = torch.isfinite
        host = lambda a: a.detach().cpu().numpy().copy()
        pairmax = torch.maximum
    else:
        _require(isinstance(X, np.ndarray) and isinstance(U, np.ndarray)
                 and X.dtype == U.dtype == np.float64, 'dtype', 'candidates')
        q = np.array(config.to_dict()['objective']['Q']); qf = 12*q
        sum_ = lambda a, dims: a.sum(axis=dims)
        all_ = lambda a, dims: a.all(axis=dims)
        maximum = lambda a: a.max(axis=-1)
        pos = lambda a: np.maximum(a, 0)
        finite, host, pairmax = np.isfinite, lambda a: a.copy(), np.maximum
    _require(X.ndim == U.ndim == 3 and X.shape == (U.shape[0], 61, 4)
             and U.shape[1:] == (60, 1) and U.shape[0] > 0, 'shape', 'candidates')
    parts = dict(stage_state=sum_(X[:, :-1]**2*q, (-2, -1)),
                 stage_input=.03*sum_(U**2, (-2, -1)),
                 terminal=sum_(X[:, -1]**2*qf, -1),
                 slew=.002*sum_((U[:, 1:]-U[:, :-1])**2, (-2, -1)))
    J = parts['stage_state']+parts['stage_input']+parts['terminal']+parts['slew']
    c = config.to_dict()
    rail = pos(abs(X[:, :, 0])/c['endpoint']['rail_m']-1)
    angle = pos(abs(X[:, :, 2])/c['endpoint']['angle_rad']-1)
    D = pairmax(maximum(rail), maximum(angle)); W = sum_(rail**2+angle**2, -1)
    arrays_finite = all_(finite(X), (-2, -1)) & all_(finite(U), (-2, -1))
    input_bounded = all_((U >= -10) & (U <= 10), (-2, -1))
    arithmetic_finite = finite(J) & finite(D) & finite(W)
    eligible = arrays_finite & input_bounded & arithmetic_finite
    result = {name: host(value) for name, value in dict(J=J, D=D, W=W,
        arrays_finite=arrays_finite, input_bounded=input_bounded,
        arithmetic_finite=arithmetic_finite, eligible=eligible, feasible=eligible & (D == 0)).items()}
    result['components'] = {name: host(value) for name, value in parts.items()}
    result['first_seen'] = np.arange(offset, offset+len(X), dtype=np.int64)
    return result


def rank(row):
    def key(i):
        if not row['eligible'][i]:
            return (True, True, math.inf, math.inf, math.inf, int(row['first_seen'][i]))
        return (False, not bool(row['feasible'][i]), float(row['D'][i]),
                float(row['W'][i]), float(row['J'][i]), int(row['first_seen'][i]))
    return sorted(range(len(row['J'])), key=key)


@dataclass(frozen=True)
class CaseResult:
    case: int
    angle: float
    complete: bool
    success: bool
    components: dict
    evidence_kind: str


class Auditor:
    def __init__(self, config, gain_path, *, gain_sha256, fixture=False,
                 step_cpu=None, step_prediction=None, noise_provider=None):
        self.config, self.c, self.fixture = config, config.to_dict(), fixture
        _require(fixture or all(x is None for x in (step_cpu, step_prediction, noise_provider)),
                 'fixture_override', 'auditor')
        self.gain = oc._decode(read_pinned(gain_path, gain_sha256))
        self.P, self.K = verify_gain(config, self.gain)
        self.cpu = step_cpu or (lambda x, u: uniform_rod_step(config, x, u))
        self.prediction = step_prediction or (lambda x, u: uniform_rod_step(config, x, u))
        self.noise_provider = noise_provider

    @property
    def kind(self):
        return 'synthetic_fixture' if self.fixture else 'scientific'

    def safe(self, x):
        import numpy as np
        return bool(np.isfinite(x).all() and abs(x[0]) <= self.c['endpoint']['rail_m']
                    and abs(x[2]) <= self.c['endpoint']['angle_rad'])

    def local(self, x, mode):
        import numpy as np
        q = np.array(self.c['objective']['Q'])
        cost = float(np.dot(x*x, q)); safe = self.safe(x)
        _require(math.isfinite(cost), 'local_nonfinite', 'local.cost')
        threshold = self.c['local']['entry_max' if mode == 'entry' else 'stay_max']
        if not safe or cost > threshold:
            return dict(mode=mode, evaluated=False, passed=False, state_cost=cost,
                        threshold=threshold, current_safe=safe, reason='safety' if not safe else 'threshold')
        count = self.c['local']['entry_steps' if mode == 'entry' else 'stay_steps']
        X, U = self.local_rollout(x, count)
        transitions = []
        for z, u, nxt in zip(X[:-1], U[:, 0], X[1:]):
            raw = float(-(self.K@z)[0]); V = float(z@self.P@z); Vn = float(nxt@self.P@nxt)
            stage = float(np.dot(z*z, q)+.03*u*u)
            left, right = Vn-V, -.5*stage+1e-10*max(1., V)
            _require(all(math.isfinite(a) for a in (V,Vn,stage,left,right,raw)), 'local_nonfinite', 'local')
            transitions.append(dict(V=V, V_next=Vn, state_cost=float(np.dot(z*z,q)), lhs=left, rhs=right,
                raw_input=raw, input=float(u), saturation=abs(raw-float(u)), finite=True,
                current_safe=self.safe(z), next_safe=self.safe(nxt), decrease=left <= right,
                passed=self.safe(z) and self.safe(nxt) and left <= right))
        rm = float(np.min(2.4-abs(X[:, 0]))); am = float(np.min(math.pi/2-abs(X[:, 2])))
        return dict(mode=mode, passed=safe and all(t['passed'] for t in transitions),
            state_cost=cost, threshold=threshold, current_safe=safe, transitions=transitions,
            max_saturation=max(t['saturation'] for t in transitions),
            minimum_safety_margin=min(rm/2.4, am/(math.pi/2)), minimum_rail_margin_m=rm,
            minimum_angle_margin_rad=am, evaluated=True, predicted_states=X, predicted_controls=U)

    def local_rollout(self, x, count):
        import numpy as np
        nodes, controls = [x.copy()], []
        for _ in range(count):
            u = float(np.clip(-(self.K@nodes[-1])[0], -10, 10))
            nxt = self.cpu(nodes[-1].copy(), u)
            _require(isinstance(nxt, np.ndarray) and nxt.dtype == np.float64 and nxt.shape == (4,)
                     and np.isfinite(nxt).all(), 'local_nonfinite', 'local.prediction')
            nodes.append(nxt.copy()); controls.append([u])
        return np.array(nodes, dtype=np.float64), np.array(controls, dtype=np.float64)

    def plan(self, trace, x, warm, case, sample):
        """Replay every candidate from its indexed draw; no new optimization."""
        import numpy as np
        import torch
        fields = {'scientific_sha256','evidence_kind','case','sample','iterations','selected',
                  'key','seed','draw_sha256','warm_controls','warm_states','local_controls','local_states'}
        _require(isinstance(trace, dict) and set(trace) == fields, 'record_fields', 'plan')
        payload = json.dumps([self.c['cases']['root_seed'], 'cem', ['oracle_v1', case, 0], sample], separators=(',', ':')).encode()
        seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], 'little') % (2**63-1)
        draws = (self.noise_provider(case, sample) if self.noise_provider is not None else
                 torch.randn((3, 256, 60, 1), generator=torch.Generator(device='cpu').manual_seed(seed), dtype=torch.float32))
        _require(isinstance(draws, torch.Tensor) and draws.device.type == 'cpu'
                 and draws.dtype == torch.float32 and tuple(draws.shape) == (3,256,60,1), 'rng', 'draws')
        for field, value in dict(scientific_sha256=oc.scientific_hash(self.config), evidence_kind=self.kind,
             case=case, sample=sample, key=['oracle_v1',case,0], seed=seed,
             draw_sha256=hashlib.sha256(draws.numpy().tobytes()).hexdigest(), warm_controls=warm).items():
            _same(value, trace[field], 'plan.'+field)
        local_X, local_U = self.local_rollout(x, 60)
        _same(local_X, trace['local_states'], 'plan.local_states'); _same(local_U, trace['local_controls'], 'plan.local_controls')
        device = 'cpu' if self.fixture else self.c['execution']['prediction_device']
        if not self.fixture and not torch.cuda.is_available():
            raise AuditIssue('cuda_unavailable','runtime',status='unavailable')
        mean = torch.tensor(warm, dtype=torch.float64, device=device)
        std = torch.full_like(mean, 20*(.05+.45*min(1., max(0., 2*math.sqrt(float(np.mean(x*x)))))))
        noise = draws.to(device=device, dtype=torch.float64)
        warm_t, local_t = mean.clone(), torch.tensor(local_U, dtype=torch.float64, device=device)
        z0 = torch.tensor(x, dtype=torch.float64, device=device)
        incumbent = None
        _require(type(trace['iterations']) is list and 1 <= len(trace['iterations']) <= 3, 'iterations', 'plan')
        with torch.no_grad():
            for j, observed in enumerate(trace['iterations']):
                controls = torch.clamp(mean+std*noise[j], -10, 10)
                controls[0] = mean.clamp(-10, 10); controls[1] = 0.; controls[2] = local_t; controls[3] = warm_t
                states = [z0.expand(256, 4)]
                for k in range(60):
                    nxt = self.prediction(states[-1], controls[:, k])
                    _require(isinstance(nxt, torch.Tensor) and nxt.dtype == torch.float64
                             and nxt.device == mean.device and nxt.shape == (256,4), 'prediction_contract', 'plan')
                    states.append(nxt)
                X = torch.stack(states, dim=1)
                host = lambda t: t.detach().cpu().numpy().copy()
                if j == 0: _same(host(X[3]), trace['warm_states'], 'plan.warm_states')
                row = scores(self.config, X, controls, j*256)
                ordered = rank(row); eligible_count = int(row['eligible'].sum())
                row.update(mean_before=host(mean), std_before=host(std), eligible_count=eligible_count,
                           feasible_count=int(row['feasible'].sum()), moments_updated=False, elites=None,
                           mean_after=None, std_after=None)
                if eligible_count < 32:
                    _same(row, observed, f'plan.iterations[{j}]')
                    _require(len(trace['iterations']) == j+1 and trace['selected'] is None, 'partial_trace', 'plan')
                    return {'exception': 'insufficient_finite_candidates'}
                elites = np.array(ordered[:32], dtype=np.int64)
                mean = .15*mean+.85*controls[elites.tolist()].mean(dim=0)
                std = .15*std+.85*controls[elites.tolist()].std(dim=0, correction=1).clamp_min(.02)
                row.update(elites=elites, moments_updated=True, mean_after=host(mean), std_after=host(std))
                _same(row, observed, f'plan.iterations[{j}]')
                i = ordered[0]
                key = (not bool(row['feasible'][i]), float(row['D'][i]), float(row['W'][i]),
                       float(row['J'][i]), int(row['first_seen'][i]))
                if incumbent is None or key < incumbent['rank']:
                    incumbent = dict(controls=host(controls[i]), states=host(X[i]), feasible=not key[0],
                        eligible=True, rank=key, first_seen=key[-1],
                        components={name:float(values[i]) for name,values in row['components'].items()})
        _require(len(trace['iterations']) == 3, 'iterations', 'plan')
        _same(incumbent, trace['selected'], 'plan.selected')
        return incumbent

    def actions(self, records):
        """Reconstruct queue/local state from observations and audited plans."""
        import numpy as np
        mode, previous, plan_sample, end = 'planning', None, None, None
        safety = exception = False; first = last = None
        for rec in records:
            case, sample, x = rec['case'], rec['sample'], rec['observation']
            _require(case==records[0]['case'],'case_identity','action.case')
            _require(type(case) is int and case >= 0 and type(sample) is int and sample >= 0,
                     'index', 'action')
            _require((last is None or sample > last) and (self.fixture or sample == (0 if last is None else last+1)),
                     'sample_order', 'action.sample')
            last = sample
            _require(isinstance(x, np.ndarray) and x.dtype == np.float64 and x.shape == (4,), 'state', 'action.observation')
            want = dict(case=case,sample=sample,scientific_sha256=oc.scientific_hash(self.config),evidence_kind=self.kind,
                observation=x, mode_before=mode, mode_after=None,branch=None,input=None,local_guard=None,
                queue_before=0 if end is None else max(0,end-sample),queue_after=None,previous_plan_sample=plan_sample,
                warm_shift=None,invalidations=[],queue_guard=None,planning=None,planning_calls=0,zero_prediction=None,error=None)
            def clear(reason):
                nonlocal previous,plan_sample,end
                previous=plan_sample=end=None; want['invalidations'].append(reason)
            def queue_guard():
                age=sample-plan_sample; u=float(previous['controls'][age,0]); nxt=self.cpu(x.copy(),u)
                _require(np.isfinite(nxt).all(), 'prediction_nonfinite', 'queue')
                match=bool(np.allclose(x,previous['states'][age],atol=1e-10,rtol=1e-10))
                return dict(age=age,expected_node=previous['states'][age].copy(),consistent=match,
                            next_node=nxt,next_safe=self.safe(nxt),passed=match and self.safe(nxt),input=u)
            finite=bool(np.isfinite(x).all())
            if not self.safe(x):
                safety=True
                if first is None:first=sample
            if not finite:
                clear('nonfinite_stop');mode='planning';want['branch']='nonfinite_stop'
            else:
                guard=self.local(x,'stay' if mode=='local' else 'entry');want['local_guard']=guard
                if guard['passed']:
                    want['branch']='local_stay' if mode=='local' else 'local_enter'
                    if mode!='local':clear('local_enter')
                    mode='local';want['input']=float(guard['predicted_controls'][0,0])
                else:
                    if mode=='local':clear('local_exit');mode='planning'
                    if end is not None and end>sample:
                        q=queue_guard();want['queue_guard']=q
                        if q['passed']:want.update(branch='queued',input=q['input'])
                        else:clear('queue_guard_failed')
                    if want['branch'] is None:
                        if end is not None and sample>=end:end=None;want['invalidations'].append('queue_expired')
                        shift=None if previous is None else sample-plan_sample
                        warm=(np.zeros((60,1),dtype=np.float64) if previous is None else
                              previous['controls'][np.minimum(np.arange(60)+shift,59)].copy())
                        want.update(warm_shift=shift,planning_calls=1,planning=rec['planning'])
                        if rec['planning'] is None:
                            raise AuditIssue('exception_not_reproducible','action.planning',status='unavailable')
                        selected=self.plan(rec['planning'],x,warm,case,sample)
                        if 'exception' in selected:
                            exception=True; mode='planning';clear('exception_zero')
                            want.update(branch='exception_zero',input=0.,error=rec['error'])
                            _require(isinstance(rec['error'],dict) and set(rec['error'])=={'code','cause'}
                                     and rec['error']['code']==selected['exception'] and type(rec['error']['cause']) is str,
                                     'exception_code','action.error')
                        elif not selected['feasible']:
                            mode='planning';clear('no_feasible_plan');want.update(branch='no_feasible_plan',input=0.)
                        else:
                            previous,plan_sample,end=selected,sample,sample+5
                            q=queue_guard();want['queue_guard']=q
                            _require(q['passed'],'new_plan_guard','action.queue_guard')
                            want.update(branch='planned',input=q['input'])
                        if want['branch'] in ('exception_zero','no_feasible_plan'):
                            nxt=self.cpu(x.copy(),0.)
                            want['zero_prediction']=dict(state=nxt,safe=self.safe(nxt))
            want.update(mode_after=mode,queue_after=0 if end is None else max(0,end-sample-1),
                        safety_latched=safety,exception_latched=exception,first_violation=first)
            for name in ('solve_seconds','control_seconds'):
                value=rec[name]
                _require(type(value) is float and math.isfinite(value) and value>=0,'timing','action.'+name)
            _require(rec['solve_seconds']<=rec['control_seconds']+1e-10,'timing','action')
            _require(want['planning_calls']!=0 or rec['solve_seconds']==0.,'timing_without_solve','action')
            observed={k:v for k,v in rec.items() if k not in ('solve_seconds','control_seconds')}
            _same(want,observed,f'action[{sample}]')
            if not finite:
                _require(rec is records[-1],'nonfinite_continuation','action')

    def case(self, data, expected_initial):
        import numpy as np
        _require(set(data)=={'case','scientific_sha256','evidence_kind','status','X','U','actions'},'record_fields','case')
        case=data['case'];X,U,actions=data['X'],data['U'],data['actions']
        _require(type(case) is int and 0<=case<200,'case_id','case')
        _same(oc.scientific_hash(self.config),data['scientific_sha256'],'case.config')
        _same(self.kind,data['evidence_kind'],'case.kind')
        _require(isinstance(X,np.ndarray) and isinstance(U,np.ndarray) and X.dtype==U.dtype==np.float64
                 and U.ndim==2 and U.shape[1:]==(1,) and X.shape==(len(U)+1,4) and len(U)<=1000,'case_shape','case')
        _same(expected_initial,X[0],'case.initial')
        _require(type(actions) is list and len(actions) in (len(U),len(U)+1),'action_count','case')
        for k,rec in enumerate(actions):
            _same(case,rec['case'],'action.case');_same(k,rec['sample'],'action.sample');_same(X[k],rec['observation'],'action.observation')
            if k<len(U):_same(float(U[k,0]),rec['input'],'action.input')
            else:_require(rec['branch']=='nonfinite_stop' and rec['input'] is None,'stop_record','case')
        _require(all(np.isfinite(X[k]).all() and np.isfinite(U[k]).all() for k in range(len(U))), 'nonfinite_prefix','case')
        for k in range(len(U)):_same(self.cpu(X[k].copy(),float(U[k,0])),X[k+1],f'plant[{k}]')
        self.actions(actions)
        _require(data['status'] in ('completed','interrupted','nonfinite_stop'),'case_status','case')
        complete=data['status']=='completed' and len(U)==1000 and len(actions)==1000
        _require(data['status']!='completed' or complete,'incomplete_completed','case')
        _require(data['status']!='nonfinite_stop' or not np.isfinite(X[-1]).all(),'false_nonfinite_stop','case')
        components=endpoint_components(self.config,X,U,complete=complete,
                    exception=any(r['exception_latched'] for r in actions))
        return CaseResult(case,float(X[0,2]),complete,components['strict_reference_success'],components,self.kind)


def endpoint_components(config,X,U,*,complete,exception):
    import numpy as np
    c=config.to_dict(); ep=c['endpoint'];finite=bool(np.isfinite(X).all() and np.isfinite(U).all())
    complete=bool(complete and X.shape==(1001,4) and U.shape==(1000,1))
    tail=X[850:1001]*np.array([1.,math.sqrt(9.81),1.,math.sqrt(9.81)])
    valid=complete and finite and len(tail)==151
    limits=np.array(ep['final_tolerances'])
    settled=np.all(abs(tail)<=limits,axis=0) if valid else np.zeros(4,dtype=bool)
    safe=bool(finite and np.all(abs(X[:,0])<=ep['rail_m']) and np.all(abs(X[:,2])<=ep['angle_rad']))
    bounded=bool(np.isfinite(U).all() and np.all(abs(U)<=10))
    return dict(complete=complete,finite=finite,balanced=bool(settled[2] and settled[3]),
                cart_rest=bool(settled[1]),recentered=bool(settled[0]),safe_all=safe,input_bounded=bounded,
                exception=bool(exception),strict_reference_success=bool(valid and safe and bounded and settled.all() and not exception))


def summarize(config,results):
    """Aggregate only auditor-produced CaseResults; no missing-case exclusion."""
    _require(len(results)==200 and {r.case for r in results}==set(range(200)), 'case_coverage','run')
    halves={name:[r for r in results if (r.angle<0 if name=='negative' else r.angle>=0)]
            for name in ('negative','positive')}
    counts={name:dict(n=len(rows),successes=sum(r.success for r in rows)) for name,rows in halves.items()}
    success=sum(r.success for r in results); ep=config.to_dict()['endpoint']
    passed=all(r.complete for r in results) and success>=ep['minimum_successes']
    numerator,denominator=ep['half_fraction']
    passed=passed and all(v['n']>0 and v['successes']*denominator>=numerator*v['n'] for v in counts.values())
    _require(len({r.evidence_kind for r in results})==1,'mixed_evidence','run')
    return dict(reference_id='OA-1',endpoint='historical-strict-reference-v1',evidence_kind=results[0].evidence_kind,
                revised_balancing_endpoint='pending',cases=200,successes=success,halves=counts,passed=bool(passed))


def audit_run(auditor,cases,*,case_callback=None):
    import numpy as np
    c=auditor.c
    expected=np.random.default_rng(c['cases']['case_seed']).uniform(-1,1,(200,4))*np.array(c['cases']['half_widths'])
    results=[]
    for data in cases:
        _require(type(data.get('case')) is int and 0<=data['case']<200,'case_id','run')
        result=auditor.case(data,expected[data['case']])
        if case_callback is not None:case_callback(result)
        results.append(result)
    return summarize(auditor.config,results)
