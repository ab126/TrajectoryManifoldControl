#!/usr/bin/env bash
set -euo pipefail
validation_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONHASHSEED=20261006 PYTHONDONTWRITEBYTECODE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
export MPLCONFIGDIR="$validation_dir/artifacts/mpl"
export LD_LIBRARY_PATH="$validation_dir/.venv-gpu/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export LD_PRELOAD="$validation_dir/.venv-gpu/lib/libstdc++.so.6"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-0}"
printf '%s overnight ' "$(date -u +%FT%TZ)" >> "$validation_dir/artifacts/commands.log"
printf '%s ' "$@" >> "$validation_dir/artifacts/commands.log"
printf '\n' >> "$validation_dir/artifacts/commands.log"
exec "$validation_dir/.venv-gpu/bin/python" "$validation_dir/overnight_queue.py" "$@"
