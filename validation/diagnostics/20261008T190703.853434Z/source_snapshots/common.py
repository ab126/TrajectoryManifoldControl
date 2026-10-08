"""Determinism, indexed randomness, and conservative worker-time accounting."""
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ['MPLCONFIGDIR'] = str(__import__('pathlib').Path(__file__).parent / 'artifacts' / 'mpl')
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import hashlib
import json
import random
import resource
import sys
import time
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'validation' / 'artifacts'
sys.path.insert(0, str(ROOT))
from compat import install
CVXPY_IMPORT_WORKAROUND = install()
SEED = 20261006

def configure(seed=SEED):
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_default_dtype(torch.float32)

def indexed_seed(stream, *indices):
    payload = json.dumps([SEED, stream, *indices], separators=(',', ':')).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], 'little') % (2**63 - 1)

def normals(case, sample, horizon=30, population=256, iterations=3, device='cpu', stream='cem'):
    """Method-independent tensors indexed by case and physical sample, not call order."""
    gen = torch.Generator(device='cpu').manual_seed(indexed_seed(stream, case, sample))
    return torch.randn(iterations, population, horizon, 1, generator=gen).to(device)

def disturbances(case, sample, size=4, stream='observation_noise'):
    return np.random.default_rng(indexed_seed(stream, case, sample)).standard_normal(size)

def save(name, value):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')

def sync(device):
    if str(device).startswith('cuda'):
        torch.cuda.synchronize(device)

def summary(values):
    a = np.asarray(values)
    return {'n': len(a), 'median_s': float(np.median(a)), 'p95_s': float(np.quantile(a, .95)),
            'p99_s': float(np.quantile(a, .99)), 'mean_s': float(a.mean()), 'max_s': float(a.max())}

class Meter:
    """Charge wall seconds per active experimental worker (not CPU core-seconds)."""
    def __init__(self, label, reservation=600):
        self.label, self.reservation = label, reservation
    def __enter__(self):
        self.path = OUT / 'usage.jsonl'
        rows = [json.loads(s) for s in self.path.read_text().splitlines()] if self.path.exists() else []
        used = sum(r['worker_seconds'] for r in rows)
        if used + self.reservation > 3600:
            raise RuntimeError(f'Worker-hour cap: {used=}, {self.reservation=}')
        self.start = time.perf_counter()
        self.cpu = time.process_time()
        return self
    def __exit__(self, exc_type, exc, tb):
        row = {'label': self.label, 'worker_seconds': time.perf_counter()-self.start,
               'process_cpu_seconds': time.process_time()-self.cpu,
               'peak_rss_MiB': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
               'passed': exc_type is None}
        with self.path.open('a') as f:
            f.write(json.dumps(row)+'\n')

def determinism_manifest():
    return {'seed': SEED, 'python': sys.version, 'torch': torch.__version__,
            'cvxpy_unused_import_workaround': CVXPY_IMPORT_WORKAROUND,
            'torch_cuda_build': torch.version.cuda, 'cuda_available': torch.cuda.is_available(),
            'dtype': 'float32 learned; float64 plant/reference/oracle',
            'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
            'TF32_matmul': torch.backends.cuda.matmul.allow_tf32,
            'TF32_cudnn': torch.backends.cudnn.allow_tf32,
            'torch_threads': torch.get_num_threads(), 'interop_threads': torch.get_num_interop_threads(),
            'environment': {k: os.environ.get(k) for k in ('PYTHONHASHSEED','OMP_NUM_THREADS',
                'MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','CUBLAS_WORKSPACE_CONFIG')},
            'rng': 'SHA256(root seed, stream, case, physical sample); CPU-generated CEM tensors',
            'model_selection': {'requested': 'gpt-5.6-terra', 'available': False,
                'configured_model': 'inherited session; no model override; precise identifier not exposed',
                'configured_reasoning_effort': 'inherited session; not exposed',
                'delegated_tasks': 0, 'runtime_model_effort': 'not exposed; local orchestrator only',
                'LLM_usage': 'token/cost telemetry not exposed'}}
