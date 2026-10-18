"""Import the submitted implementation with the existing unused-CVXPY workaround."""
import os
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(name, '1')
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'validation/results_v1/.mpl'))
sys.path[:0] = [str(ROOT), str(ROOT / 'validation')]
from compat import install
CVXPY_WORKAROUND = install()
import numpy as np
import torch
from scripts import run_inverted_pendulum_mc as runner
from src.manifold_control import CanonicalBehaviorDecoder, DataDrivenCEMSolver
