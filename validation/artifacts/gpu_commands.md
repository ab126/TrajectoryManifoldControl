GPU follow-up commands (from workspace root):

```bash
TrajectoryManifoldControl/validation/.venv/bin/python TrajectoryManifoldControl/validation/setup_gpu.py
# CUDA environment copy only; no GPU experiment.
env LD_LIBRARY_PATH=/home/ap60/github/data-driven-control/TrajectoryManifoldControl/validation/.venv-gpu/lib LD_PRELOAD=/home/ap60/github/data-driven-control/TrajectoryManifoldControl/validation/.venv-gpu/lib/libstdc++.so.6 PYTHONDONTWRITEBYTECODE=1 TrajectoryManifoldControl/validation/.venv-gpu/bin/python TrajectoryManifoldControl/validation/gpu_manifest.py
bash TrajectoryManifoldControl/validation/run_gpu.sh readiness
# Observed: probe exit 2; coordinator stopped before equivalence/smoke/benchmarks.
bash TrajectoryManifoldControl/validation/run.sh gate_a --cases 200 --starts 1 --max-worker-hours 7.7
# CPU preflight only, execute=false; retained reservation 6.377 worker-h.
TrajectoryManifoldControl/validation/.venv/bin/python TrajectoryManifoldControl/validation/estimate_stages.py --device cpu
```

Syntax and follow-up assertions passed; private CUDA pip check passed.
No scientific calibration or sweep ran. The intended next command is the GPU
readiness launcher in the host terminal where nvidia-smi succeeds.

User host readiness completed successfully; command arguments are in commands.log.
CUDA Gate A preflight (no --execute) reviewed: 11.8996 planning worker-h versus cap 7; allocation fails with 20% overhead.
GPU experimental work was not repeated by the agent.
