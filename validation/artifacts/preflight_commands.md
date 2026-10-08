Read-only preflight commands used before the metered experiments (run from the
parent workspace except Git commands, which target the nested checkout):

```bash
rg --files -g AGENTS.md -g '*plan.md' -g '*requirements*' -g environment.yml -g '*slurm*' -g '*pbs*'
cat review/stress_test_plan.md review/critique_plan.md
git -C TrajectoryManifoldControl rev-parse HEAD
git -C TrajectoryManifoldControl status --porcelain=v1
git -C TrajectoryManifoldControl log -6 --format=fuller
git -C TrajectoryManifoldControl diff
lscpu
free -b
nvidia-smi
cat /proc/driver/nvidia/version
cat /proc/driver/nvidia/gpus/0000:41:00.0/information
/usr/local/cuda/bin/nvcc --version
command -v sbatch srun squeue qsub nvcc
cat /proc/self/cgroup
cat /proc/self/status
uptime
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/ap60/venvs/mtbi-torch-cu12/bin/python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'
```

Outputs are in the matching artifacts files. The CUDA-build probe returned
`2.5.1+cu121 12.1 False`. NVIDIA kernel metadata says L4 / 610.43.02; nvcc is
13.3. Scheduler executable/environment/documentation searches found no policy.
Initial Git calls from the parent failed because its .git is not a repository;
the nested TrajectoryManifoldControl checkout supplies the recorded HEAD.

Environment preparation (not experimental compute):

```bash
/home/ap60/miniconda3/envs/general/bin/python -m venv TrajectoryManifoldControl/validation/.venv
TrajectoryManifoldControl/validation/.venv/bin/python -m pip install --disable-pip-version-check 'numpy>=2,<3' 'scipy>=1.13' 'matplotlib>=3.8' 'tqdm>=4.66' 'cvxpy>=1.5' 'pytest>=8' 'torch>=2.2' --extra-index-url https://download.pytorch.org/whl/cpu
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/ap60/miniconda3/envs/mtbi_detection0/bin/python TrajectoryManifoldControl/validation/setup_offline.py
/home/ap60/miniconda3/envs/general/bin/python TrajectoryManifoldControl/validation/complete_native.py
TrajectoryManifoldControl/validation/.venv/bin/python -m pip freeze
TrajectoryManifoldControl/validation/.venv/bin/python -m pip check
```

The online install failed DNS resolution and was interrupted; it installed no
packages. Offline copying omitted development headers, copied package metadata
and native runtime libraries; initial missing libtorch_python and C++ ABI errors
were repaired by copying that library and preloading the copied libstdc++.
The first smoke attempt reached plotting, then lacked libjpeg; the copied
graphics closure repaired it. CVXPY remains absent: compat.py is explicitly used.
Do not interpret pip check as certifying the complete environment.yml.

All experimental invocations are timestamped in commands.log; actual/per-worker
accounting is in usage.jsonl. Post-timing artifact capture uses provenance.py
and estimate_stages.py. No calibration --execute command appears in the log.
