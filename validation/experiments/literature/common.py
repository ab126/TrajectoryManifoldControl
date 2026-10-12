"""Literature-only artifacts, immutable input identities, and bounded job accounting."""
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import site
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'validation/results_literature'
Q = np.diag([.02, 30., 100., 10.])
HORIZON = 30
DT = .02 * np.sqrt(9.81)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def isolation(method):
    prefix = Path('/tmp/trajectory-literature-envs') / method
    assert Path(sys.prefix).resolve() == prefix.resolve()
    assert not site.ENABLE_USER_SITE
    assert set(os.sched_getaffinity(0)) <= set(range(80, 128))
    for key in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
        assert os.environ.get(key) == '1', key
    assert not any('site-packages' in p and not Path(p).is_relative_to(prefix) for p in sys.path)


def usage_gate():
    """Reuse the original guard, refusing missing telemetry; no paid-credit use."""
    spec = importlib.util.spec_from_file_location('literature_usage_guard', ROOT/'validation/codex_supervisor.py')
    guard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guard)
    limits = guard.latest_limits()
    if limits is None or guard.balance(limits) is None:
        raise RuntimeError('Missing usage telemetry; preserve checkpoint and refuse numerical launch')
    manifest = json.loads((OUT/'manifest.json').read_text())
    if guard.balance(limits) < manifest['credit_balance_baseline'] - 1e-6:
        raise RuntimeError('Credit guard: preserve checkpoint and stop')
    used = [guard.window_state(limits.get(k))[0] for k in ['primary', 'secondary']]
    if used[0] >= 85 or used[1] >= 90:
        raise RuntimeError(f'Usage pause: {used}; preserve checkpoint and resume under supervisor')
    if (OUT/'STATUS').read_text().splitlines()[0] != 'RUNNING':
        raise RuntimeError('Task not RUNNING')
    return dict(five_hour_percent=used[0], weekly_percent=used[1], balance=guard.balance(limits))


def cpu_seconds():
    r = resource.getrusage(resource.RUSAGE_SELF)
    return r.ru_utime + r.ru_stime


class Budget:
    """Persist process CPU time at every checkpoint; charge prior resumptions too."""
    def __init__(self, stage, ceiling_hours):
        self.path = OUT/stage/'accounting.json'
        self.previous = json.loads(self.path.read_text()) if self.path.exists() else dict(cpu_seconds=0., wall_seconds=0., gpu_seconds=0.)
        self.start_cpu, self.start_wall = cpu_seconds(), time.monotonic()
        self.ceiling = ceiling_hours * 3600

    def checkpoint(self):
        value = dict(cpu_seconds=self.previous['cpu_seconds']+cpu_seconds()-self.start_cpu,
                     wall_seconds=self.previous['wall_seconds']+time.monotonic()-self.start_wall, gpu_seconds=0.)
        dump(self.path, value)
        total = sum(json.loads(p.read_text())['cpu_seconds'] for p in OUT.glob('*/accounting.json'))
        if value['cpu_seconds'] >= self.ceiling or total >= 84*3600:
            raise RuntimeError('CPU allocation exhausted; cap-limited partial result')
        usage_gate()
        return value


def load_data(seed):
    path = ROOT/f'validation/results_v1/models/episodes3500_seed{seed}/data.npz'
    data = np.load(path)
    X, U = data['X'].astype(float), data['U'].astype(float)
    assert X.shape == (3500,31,4) and U.shape == (3500,30,1)
    assert np.isfinite(X).all() and np.isfinite(U).all() and np.max(np.abs(U)) <= 10
    order = np.random.default_rng(seed).permutation(len(X))
    idx = dict(test=order[:525], validation=order[525:1050], train=order[1050:])
    identity = dict(path=str(path.relative_to(ROOT)), sha256=digest(path), seed=seed,
                    split_indices={k:v.tolist() for k,v in idx.items()},
                    units=['position/l', 'velocity/sqrt(gl)', 'angle radians', 'angular velocity*sqrt(l/g)'],
                    input_units='F/(mg)', normalized_dt=DT, physical_dt=.02)
    return {k:(X[v],U[v]) for k,v in idx.items()}, identity


def plant_interface():
    """Load only the submitted opaque evaluation functions, avoiding batch imports.

    These functions are never passed to identification or planning.
    """
    path = ROOT/'src/inverted_pendulum.py'
    tree = ast.parse(path.read_text())
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ('rk4_step','dynamics_open_loop')]
    assert len(functions) == 2
    namespace = dict(np=np)
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), 'exec'), namespace)
    return lambda x,u: namespace['rk4_step'](x,float(u),DT,2.)


def objective(X, U):
    return (np.einsum('...ki,ij,...kj->...',X[...,:-1,:],Q,X[...,:-1,:])
            + .03*np.sum(U*U,axis=(-2,-1))
            + np.einsum('...i,ij,...j->...',X[...,-1,:],12*Q,X[...,-1,:])
            + .002*np.sum(np.diff(U,axis=-2)**2,axis=(-2,-1)))


def forecast_score(model, data, scale):
    X,U = data
    with np.errstate(over='ignore', invalid='ignore'):
        pred = model.predict_inputs(X[:,0],U)
        e = (pred[:,1:]-X[:,1:])/scale
        value = np.mean(e*e)
    return float(value) if np.isfinite(value) else None
