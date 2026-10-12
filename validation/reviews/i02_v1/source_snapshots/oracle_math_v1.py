"""I02 OA-1 arithmetic; no plant, controller, RNG, launcher or device setup.

Scoring supports batched float64 NumPy/Torch arrays without device transfers.
Scientific prediction/device validation and independent audit remain separate.
NumPy/SciPy/Torch are imported only by the operations that need them.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import oracle_config as oc


class MathError(ValueError):
    def __init__(self, code: str, detail: str):
        self.code = code
        super().__init__(f"{code}: {detail}")


class SearchError(MathError):
    """I03 must clear memory, latch exception failure and use zero one sample."""
    def __init__(self, finite_count: int, required: int):
        self.finite_count, self.required = finite_count, required
        super().__init__("insufficient_finite_candidates", f"{finite_count} < {required}")


class _Backend:
    def __init__(self, value):
        if type(value).__module__.split('.')[0] == 'torch':
            import torch
            self.xp, self.torch = torch, True
            if not isinstance(value, torch.Tensor) or value.dtype != torch.float64:
                raise MathError("dtype", "float64 tensor required")
        else:
            import numpy as np
            self.xp, self.torch = np, False
            if not isinstance(value, np.ndarray) or value.dtype != np.float64:
                raise MathError("dtype", "float64 ndarray required")
        self.like = value

    def require(self, other):
        if self.torch:
            valid = (isinstance(other, self.xp.Tensor) and other.dtype == self.like.dtype
                     and other.device == self.like.device)
        else:
            valid = isinstance(other, self.xp.ndarray) and other.dtype == self.like.dtype
        if not valid:
            raise MathError("backend", "matching backend, float64 dtype and device required")

    def array(self, values):
        return self.like.new_tensor(values) if self.torch else self.xp.asarray(values, dtype=self.xp.float64)

    def sum(self, value, axes):
        return value.sum(dim=axes) if self.torch else value.sum(axis=axes)

    def all(self, value, axes):
        return value.all(dim=axes) if self.torch else value.all(axis=axes)

    def maximum(self, value, axes):
        return value.amax(dim=axes) if self.torch else value.max(axis=axes)

    def positive(self, value):
        return value.clamp_min(0) if self.torch else self.xp.maximum(value, 0)


def _config(config):
    if not isinstance(config, oc.ScientificConfig):
        raise TypeError("I01 ScientificConfig required")
    return config.to_dict()


def _plans(states, controls):
    b = _Backend(states)
    b.require(controls)
    if (states.ndim < 2 or controls.ndim != states.ndim or states.shape[-1] != 4
            or controls.shape[-1] != 1 or states.shape[:-2] != controls.shape[:-2]
            or controls.shape[-2] < 1 or states.shape[-2] != controls.shape[-2]+1):
        raise MathError("shape", "states (...,T+1,4), controls (...,T,1), T>=1 required")
    return b


@dataclass(frozen=True)
class ObjectiveComponents:
    stage_state: Any
    stage_input: Any
    terminal: Any
    slew: Any

    @property
    def total(self):
        return self.stage_state+self.stage_input+self.terminal+self.slew


def objective_components(config, states, controls) -> ObjectiveComponents:
    """Any positive T for arithmetic fixtures; production scoring requires T=60.

    No dt multiplier, first-input slew, angle wrapping or common-cost change.
    Nonfinite/overflow results are excluded by score_candidates, not repaired.
    """
    c, b = _config(config), _plans(states, controls)
    q, qf = b.array(c['objective']['Q']), b.array(c['objective']['Qf'])
    return ObjectiveComponents(
        b.sum(states[..., :-1, :]**2*q, (-2, -1)),
        c['objective']['R']*b.sum(controls**2, (-2, -1)),
        b.sum(states[..., -1, :]**2*qf, -1),
        c['objective']['rate_weight']*b.sum((controls[..., 1:, :]-controls[..., :-1, :])**2, (-2, -1)),
    )


@dataclass(frozen=True)
class CandidateScores:
    components: ObjectiveComponents
    arrays_finite: Any
    input_bounded: Any
    arithmetic_finite: Any
    eligible: Any
    feasible: Any
    D: Any
    W: Any
    first_seen: Any


def score_candidates(config, states, controls, *, first_seen_offset=0) -> CandidateScores:
    """Production shapes (M,61,4)/(M,60,1); M may vary for fixtures/incumbents.

    Indices are globally increasing across all iterations of one planning call.
    Overflow in a score is ineligible even if the raw arrays are finite; the
    separate flags preserve that diagnostic distinction.
    """
    c, b = _config(config), _plans(states, controls)
    if states.ndim != 3 or controls.shape[1] != c['search']['horizon'] or states.shape[0] < 1:
        raise MathError("shape", "nonempty candidate batch with locked horizon required")
    if (type(first_seen_offset) is not int or first_seen_offset < 0
            or first_seen_offset+states.shape[0] >= 2**63):
        raise MathError("candidate_index", "nonnegative int64 first-seen range required")
    parts = objective_components(config, states, controls)
    rail = b.positive(abs(states[..., 0])/c['endpoint']['rail_m']-1)
    angle = b.positive(abs(states[..., 2])/c['endpoint']['angle_rad']-1)
    D = b.xp.maximum(b.maximum(rail, -1), b.maximum(angle, -1))
    W = b.sum(rail**2+angle**2, -1)
    finite = b.all(b.xp.isfinite(states), (-2, -1)) & b.all(b.xp.isfinite(controls), (-2, -1))
    lo, hi = c['plant']['input_bounds']
    bounded = b.all((controls >= lo) & (controls <= hi), (-2, -1))
    arithmetic = b.xp.isfinite(parts.total) & b.xp.isfinite(D) & b.xp.isfinite(W)
    eligible = finite & bounded & arithmetic
    kwargs = {'device': states.device, 'dtype': b.xp.int64} if b.torch else {'dtype': b.xp.int64}
    indices = b.xp.arange(first_seen_offset, first_seen_offset+states.shape[0], **kwargs)
    return CandidateScores(parts, finite, bounded, arithmetic, eligible, eligible & (D == 0), D, W, indices)


def rank_candidates(scores: CandidateScores):
    """Stable lexicographic order; same function for elites and incumbent.

    All invalid candidates follow every finite bounded candidate. Their NaNs
    never enter sort keys or moments. No Torch-to-CPU conversion occurs here.
    """
    b = _Backend(scores.D)
    xp = b.xp
    inf = xp.full_like(scores.D, float('inf'))
    # Least significant first; stable passes preserve first-seen tie breaking.
    keys = (scores.first_seen, xp.where(scores.eligible, scores.components.total, inf),
            xp.where(scores.eligible, scores.W, inf), xp.where(scores.eligible, scores.D, inf),
            ~scores.feasible, ~scores.eligible)
    kwargs = {'device': scores.D.device, 'dtype': xp.int64} if b.torch else {'dtype': xp.int64}
    order = xp.arange(scores.D.shape[0], **kwargs)
    for key in keys:
        permutation = xp.argsort(key[order], stable=True) if b.torch else xp.argsort(key[order], kind='stable')
        order = order[permutation]
    return order


@dataclass(frozen=True)
class EliteMoments:
    indices: Any
    mean: Any
    std: Any


def elite_moments(config, scores: CandidateScores, controls) -> EliteMoments:
    """No update on insufficient candidates; I03 owns exception/fallback state.

    One scalar finite-count read is required on Torch. Population scoring,
    ordering and reductions remain on device. Smoothing is an I03 operation.
    """
    c, b = _config(config), _Backend(controls)
    b.require(scores.D)
    if controls.shape != (scores.D.shape[0], c['search']['horizon'], 1):
        raise MathError("shape", "controls must match scored population/horizon")
    count = int(scores.eligible.sum().item())
    required = c['search']['elites']
    if count < required:
        raise SearchError(count, required)
    indices = rank_candidates(scores)[:required]
    chosen = controls[indices]
    mean = chosen.mean(dim=0) if b.torch else chosen.mean(axis=0)
    std = chosen.std(dim=0, correction=1) if b.torch else chosen.std(axis=0, ddof=1)
    std = std.clamp_min(c['search']['std_floor']) if b.torch else b.xp.maximum(std, c['search']['std_floor'])
    return EliteMoments(indices, mean, std)


def upright_rk4_matrices(config):
    """Exact derivative at upright for the submitted normalized uniform rod."""
    import numpy as np
    c = _config(config)
    h = c['plant']['dt_normalized']
    ac = np.array([[0, 1, 0, 0], [0, 0, -1/3, 0], [0, 0, 0, 1], [0, 0, 2, 0]], dtype=np.float64)
    bc = np.array([[0], [4/9], [0], [-2/3]], dtype=np.float64)
    powers = [np.linalg.matrix_power(ac, j) for j in range(5)]
    A = sum(h**j/math.factorial(j)*powers[j] for j in range(5))
    B = sum(h**(j+1)/math.factorial(j+1)*powers[j] for j in range(4)) @ bc
    return A, B


def _canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(',', ':'), allow_nan=False).encode('utf-8')


def _reference(config):
    # Reuse the public exact-proposal allowlist. No old diagnosis is imported/run.
    path = oc.HERE/'oa1_config_proposed.json'
    original, _, _ = oc.convert_proposal(path)
    if oc.scientific_hash(original) != oc.scientific_hash(config):
        raise MathError("reference_config", "reference belongs to another scientific profile")
    return json.loads(path.read_bytes())['gain']


@dataclass(frozen=True)
class GainArtifact:
    """Immutable serialized artifact; construct_gain/load_gain validate contents."""
    _bytes: bytes

    def to_dict(self):
        return json.loads(self._bytes)

    @property
    def sha256(self):
        return hashlib.sha256(self._bytes).hexdigest()

    def canonical_bytes(self):
        return self._bytes


def _validate_gain(config, record):
    import numpy as np
    c = _config(config)
    expected_fields = {'artifact_version', 'scientific_sha256', 'specification_sha256', 'method',
                       'A', 'B', 'P', 'K', 'spectral_radius', 'dare_relative_residual', 'reference_verified'}
    if not isinstance(record, dict) or set(record) != expected_fields:
        raise MathError("gain_fields", "exact gain artifact fields required")
    if (type(record['artifact_version']) is not int or record['artifact_version'] != 1
            or record['scientific_sha256'] != oc.scientific_hash(config)
            or record['specification_sha256'] != oc.SPEC_SHA256
            or record['method'] != c['local']['gain_construction']
            or record['reference_verified'] is not True):
        raise MathError("gain_binding", "version, config, specification, method and verification must match")
    matrices = {}
    for name, shape in (('A', (4, 4)), ('B', (4, 1)), ('P', (4, 4)), ('K', (1, 4))):
        rows = record[name]
        if (not isinstance(rows, list) or len(rows) != shape[0]
                or any(not isinstance(row, list) or len(row) != shape[1] for row in rows)
                or any(type(v) not in (int, float) for row in rows for v in row)):
            raise MathError("gain_shape", name)
        try:
            matrix = np.array(rows, dtype=np.float64)
        except (ValueError, OverflowError) as error:
            raise MathError("gain_numeric", name) from error
        if not np.isfinite(matrix).all():
            raise MathError("gain_numeric", name)
        matrices[name] = matrix
    A, B, P, K = (matrices[name] for name in ('A', 'B', 'P', 'K'))
    atol, rtol = c['local']['gain_atol'], c['local']['gain_rtol']
    close = lambda x, y: np.allclose(x, y, atol=atol, rtol=rtol)
    expected_A, expected_B = upright_rk4_matrices(config)
    ref = _reference(config)
    if not (close(A, expected_A) and close(B, expected_B)
            and close(P, ref['P_reference']) and close(K, ref['K_reference'])):
        raise MathError("gain_reference", "A/B/P/K differ from fixed construction/reference")
    if not close(P, P.T) or np.linalg.eigvalsh(P).min() <= 0:
        raise MathError("gain_positive_definite", "symmetric positive definite P required")
    Q, R = np.diag(c['objective']['Q']), np.array([[c['objective']['R']]])
    calculated_K = np.linalg.solve(R+B.T@P@B, B.T@P@A)
    residual = A.T@P@A-P-A.T@P@B@calculated_K+Q
    relative = float(np.linalg.norm(residual, ord=np.inf)/max(1., np.linalg.norm(P, ord=np.inf)))
    radius = float(abs(np.linalg.eigvals(A-B@K)).max())
    if not close(K, calculated_K) or relative > atol+rtol or radius >= 1:
        raise MathError("gain_stability", "DARE consistency/residual and stable A-BK required")
    for name, value in (('spectral_radius', radius), ('dare_relative_residual', relative)):
        reported = record[name]
        if type(reported) not in (int, float) or not math.isfinite(reported) or not math.isclose(reported, value, rel_tol=rtol, abs_tol=atol):
            raise MathError("gain_diagnostic", name)
    return matrices


def construct_gain(config) -> GainArtifact:
    """Fixed linear algebra only; no nonlinear rollout, tuning or plant import."""
    import numpy as np
    from scipy.linalg import solve_discrete_are
    c = _config(config)
    A, B = upright_rk4_matrices(config)
    Q, R = np.diag(c['objective']['Q']), np.array([[c['objective']['R']]], dtype=np.float64)
    P = solve_discrete_are(A, B, Q, R)
    K = np.linalg.solve(R+B.T@P@B, B.T@P@A)
    residual = A.T@P@A-P-A.T@P@B@K+Q
    record = dict(artifact_version=1, scientific_sha256=oc.scientific_hash(config),
                  specification_sha256=oc.SPEC_SHA256, method=c['local']['gain_construction'],
                  A=A.tolist(), B=B.tolist(), P=P.tolist(), K=K.tolist(),
                  spectral_radius=float(abs(np.linalg.eigvals(A-B@K)).max()),
                  dare_relative_residual=float(np.linalg.norm(residual, ord=np.inf)/max(1., np.linalg.norm(P, ord=np.inf))),
                  reference_verified=True)
    _validate_gain(config, record)
    return GainArtifact(_canonical(record))


def load_gain(config, path: Path, *, expected_sha256: str) -> GainArtifact:
    """Require external byte pin before numerical validation; never adopt output."""
    try:
        raw = Path(path).read_bytes()
    except FileNotFoundError as error:
        raise MathError("unavailable", str(path)) from error
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise MathError("gain_hash", "external reviewed artifact hash mismatch")
    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise MathError("gain_json", "duplicate key")
            value[key] = item
        return value
    try:
        record = json.loads(raw.decode('utf-8'), object_pairs_hook=unique,
                            parse_constant=lambda _: (_ for _ in ()).throw(MathError('gain_json', 'nonfinite constant')))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise MathError("gain_json", "UTF-8 JSON required") from error
    _validate_gain(config, record)
    if raw != _canonical(record):
        raise MathError("gain_canonical", "canonical bytes required")
    return GainArtifact(raw)


class LocalArithmetic:
    """CPU float64 local operations; cache verified gain/config once, not per step.

    I03 must obtain its GainArtifact from construct_gain/load_gain and bind its
    external artifact hash. No rollout or control-policy decision occurs here.
    """
    def __init__(self, config, gain: GainArtifact):
        import numpy as np
        self._np, self._c = np, _config(config)
        if not isinstance(gain, GainArtifact):
            raise TypeError("verified GainArtifact required")
        matrices = _validate_gain(config, gain.to_dict())
        self._P, self._K = matrices['P'], matrices['K']
        self._Q = np.array(self._c['objective']['Q'], dtype=np.float64)

    def _state(self, state):
        if not isinstance(state, self._np.ndarray) or state.dtype != self._np.float64 or state.shape != (4,):
            raise MathError("local_state", "CPU float64 ndarray shape (4,) required")
        return state

    def safe(self, state):
        x = self._state(state)
        return bool(self._np.isfinite(x).all() and abs(x[0]) <= self._c['endpoint']['rail_m']
                    and abs(x[2]) <= self._c['endpoint']['angle_rad'])

    def state_cost(self, state):
        x = self._state(state)
        return float((x*x*self._Q).sum())

    def V(self, state):
        x = self._state(state)
        return float(x @ self._P @ x)

    def input(self, state):
        x = self._state(state)
        if not self._np.isfinite(x).all():
            raise MathError("local_nonfinite", "cannot form local input from nonfinite state")
        raw = -float((self._K @ x).item())
        if not math.isfinite(raw):
            raise MathError("local_nonfinite", "nonfinite feedback input")
        return float(self._np.clip(raw, *self._c['plant']['input_bounds'])), raw

    def transition(self, state, control, next_state):
        """Caller supplies its known-model transition; records both inequality sides."""
        x, nxt = self._state(state), self._state(next_state)
        if not self._np.isfinite(x).all() or not self._np.isfinite(nxt).all():
            raise MathError("local_nonfinite", "nonfinite transition node; exception fallback required")
        if type(control) not in (int, float) or not math.isfinite(control):
            raise MathError("local_input", "finite real input required")
        applied, raw = self.input(x)
        if control != applied:
            raise MathError("local_input", "decrease must use the actual clipped local input")
        v, vn, cost = self.V(x), self.V(nxt), self.state_cost(x)
        left = vn-v
        right = (-self._c['local']['decrease_factor']*(cost+self._c['objective']['R']*control**2)
                 +self._c['local']['decrease_tolerance']*max(1., v))
        finite = all(math.isfinite(item) for item in (v, vn, cost, left, right))
        if not finite:
            raise MathError("local_nonfinite", "nonfinite local arithmetic; exception fallback required")
        return dict(current_safe=self.safe(x), next_safe=self.safe(nxt), V=v, V_next=vn,
                    state_cost=cost, input=applied, raw_input=raw, saturation=abs(raw-applied),
                    lhs=left, rhs=right, finite=finite,
                    decrease=finite and left <= right,
                    passed=finite and self.safe(x) and self.safe(nxt) and left <= right)

    def predicate(self, state, predicted_states, *, mode):
        """Entry/stay arithmetic on caller-supplied local-feedback predictions.

        Does not generate or certify those predictions. I03 must supply all
        nodes from its clipped-law rollout; I04 independently checks the trace.
        """
        if mode not in ('entry', 'stay'):
            raise MathError("local_mode", "entry or stay required")
        x, c, np = self._state(state), self._c, self._np
        steps = c['local']['entry_steps'] if mode == 'entry' else c['local']['stay_steps']
        if (not isinstance(predicted_states, np.ndarray) or predicted_states.dtype != np.float64
                or predicted_states.shape != (steps+1, 4)):
            raise MathError("local_prediction", "locked step count and CPU float64 required")
        if not np.array_equal(x, predicted_states[0], equal_nan=True):
            raise MathError("local_prediction", "first predicted node must equal observation")
        threshold = c['local']['entry_max'] if mode == 'entry' else c['local']['stay_max']
        cost = self.state_cost(x)
        safe = self.safe(x)
        if not np.isfinite(predicted_states).all():
            raise MathError("local_nonfinite", "nonfinite predicted node; exception fallback required")
        transitions = []
        for current, nxt in zip(predicted_states[:-1], predicted_states[1:]):
            u, _ = self.input(current)
            transitions.append(self.transition(current, u, nxt))
        rail_margin = float(np.min(c['endpoint']['rail_m']-abs(predicted_states[:, 0])))
        angle_margin = float(np.min(c['endpoint']['angle_rad']-abs(predicted_states[:, 2])))
        margin = min(rail_margin/c['endpoint']['rail_m'], angle_margin/c['endpoint']['angle_rad'])
        passed = safe and math.isfinite(cost) and cost <= threshold and all(t['passed'] for t in transitions)
        return dict(mode=mode, passed=passed, state_cost=cost, threshold=threshold,
                    current_safe=safe, transitions=transitions,
                    max_saturation=max(t['saturation'] for t in transitions), minimum_safety_margin=margin,
                    minimum_rail_margin_m=rail_margin, minimum_angle_margin_rad=angle_margin)
