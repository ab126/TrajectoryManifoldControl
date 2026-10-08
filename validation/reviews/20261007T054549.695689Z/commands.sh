validation_dir="$PWD/TrajectoryManifoldControl/validation"
LD_LIBRARY_PATH="$validation_dir/.venv/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
LD_PRELOAD="$validation_dir/.venv/lib/libstdc++.so.6" \
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
"$validation_dir/.venv/bin/python" "$validation_dir/review_gate_a.py" \
  "$validation_dir/overnight_runs/20261007T054549.695689Z"
