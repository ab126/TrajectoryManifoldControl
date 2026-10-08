#!/usr/bin/env bash
set -euo pipefail
validation_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export JUPYTER_PATH="$validation_dir/notebooks${JUPYTER_PATH:+:$JUPYTER_PATH}"
export MPLCONFIGDIR="$validation_dir/artifacts/mpl"
export JUPYTER_DATA_DIR="$validation_dir/notebooks/.jupyter-data"
export JUPYTER_RUNTIME_DIR="$validation_dir/notebooks/.jupyter-runtime"
export IPYTHONDIR="$validation_dir/notebooks/.ipython"
mkdir -p "$JUPYTER_DATA_DIR" "$JUPYTER_RUNTIME_DIR"
/home/ap60/miniconda3/envs/overseeg/bin/python -m jupyter trust "$validation_dir/notebooks/gate_a_examples.ipynb"
exec /home/ap60/miniconda3/envs/overseeg/bin/python -m jupyter notebook "$validation_dir/notebooks/gate_a_examples.ipynb" "$@"
