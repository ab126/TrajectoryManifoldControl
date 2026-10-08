#!/usr/bin/env bash
set -euo pipefail
validation_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONHASHSEED=20261006 PYTHONDONTWRITEBYTECODE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
export MPLCONFIGDIR="$validation_dir/artifacts/mpl"
export LD_LIBRARY_PATH="$validation_dir/.venv-gpu/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export LD_PRELOAD="$validation_dir/.venv-gpu/lib/libstdc++.so.6"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-0}"
if [[ ! -x "$validation_dir/.venv-gpu/bin/python" ]]; then
  echo 'Run validation/setup_gpu.py with the private CPU Python first.' >&2
  exit 2
fi
printf '%s gpu ' "$(date -u +%FT%TZ)" >> "$validation_dir/artifacts/commands.log"
printf '%s ' "$@" >> "$validation_dir/artifacts/commands.log"
printf '\n' >> "$validation_dir/artifacts/commands.log"
case "${1:-}" in
  probe) exec "$validation_dir/.venv-gpu/bin/python" "$validation_dir/launch.py" gpu_probe ;;
  readiness) exec "$validation_dir/.venv-gpu/bin/python" "$validation_dir/gpu_readiness.py" ;;
  gate_a|equivalence|smoke|benchmark|concurrency)
    exec "$validation_dir/.venv-gpu/bin/python" "$validation_dir/launch.py" "$@" --device cuda ;;
  *) echo 'Usage: run_gpu.sh {probe|readiness|equivalence|smoke|benchmark|concurrency|gate_a} [arguments]' >&2; exit 2 ;;
esac
