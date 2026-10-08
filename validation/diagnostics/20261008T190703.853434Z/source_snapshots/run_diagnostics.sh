#!/usr/bin/env bash
set -euo pipefail
validation_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
diagnostic_environment=.venv-gpu
if [[ "${1:-}" == local ]]; then diagnostic_environment=.venv; fi
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONHASHSEED=20261006 PYTHONDONTWRITEBYTECODE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
export MPLCONFIGDIR="$validation_dir/artifacts/mpl"
export LD_LIBRARY_PATH="$validation_dir/$diagnostic_environment/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export LD_PRELOAD="$validation_dir/$diagnostic_environment/lib/libstdc++.so.6"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-0}"
exec "$validation_dir/$diagnostic_environment/bin/python" "$validation_dir/diagnostic_supervisor.py" "$@"
