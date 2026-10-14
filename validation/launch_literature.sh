#!/usr/bin/env bash
set -euo pipefail
VALIDATION_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BOOTSTRAP_PYTHON="$VALIDATION_DIR/.venv/bin/python"
SUPERVISOR_ENV="/tmp/trajectory-literature-envs/supervisor"
PYTHON_BIN="$SUPERVISOR_ENV/bin/python"
RESULTS_DIR="$VALIDATION_DIR/results_literature"
# Do not inherit another project's Python packages or active conda environment.
unset PYTHONHOME PYTHONPATH PYTHONUSERBASE CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_PROMPT_MODIFIER CONDA_SHLVL
export PYTHONNOUSERSITE=1
if [[ ! -x "$PYTHON_BIN" ]]; then
    taskset -c 80-127 "$BOOTSTRAP_PYTHON" -I -m venv --without-pip "$SUPERVISOR_ENV"
fi
taskset -c 80-127 "$PYTHON_BIN" -I - "$SUPERVISOR_ENV" <<'PY'
import pathlib, site, sys
expected = pathlib.Path(sys.argv[1]).resolve()
if pathlib.Path(sys.prefix).resolve() != expected or sys.prefix == sys.base_prefix:
    raise SystemExit('Dedicated literature supervisor venv required.')
if site.ENABLE_USER_SITE or any('site-packages' in p and not pathlib.Path(p).is_relative_to(expected) for p in sys.path):
    raise SystemExit('External site-packages detected; refusing mixed environments.')
PY
export VIRTUAL_ENV="$SUPERVISOR_ENV"
export PATH="$SUPERVISOR_ENV/bin:$PATH"
if [[ "${1:-}" == "--check" ]]; then
    exec taskset -c 80-127 "$PYTHON_BIN" -I "$VALIDATION_DIR/literature_supervisor.py" --check
fi
RESUME_APPROVED=0
if [[ $# -eq 1 && "$1" == '--resume-approved' ]]; then
    RESUME_APPROVED=1
elif [[ $# -ne 0 ]]; then
    echo 'Usage: bash launch_literature.sh [--check|--resume-approved]' >&2
    exit 2
fi
if [[ ! -w "$HOME/.codex" ]]; then
    echo 'Run this script in the host terminal: this session cannot write ~/.codex to start the supervised model.' >&2
    exit 2
fi
command -v codex >/dev/null
command -v taskset >/dev/null
taskset -c 80-127 "$PYTHON_BIN" -I "$VALIDATION_DIR/literature_supervisor.py" --check
mkdir -p "$RESULTS_DIR"
exec 9>"$RESULTS_DIR/.launch.lock"
flock -n 9 || { echo 'Another literature launch is already in progress.' >&2; exit 2; }
if [[ -f "$RESULTS_DIR/supervisor.pid" ]]; then
    PREVIOUS_PID="$(cat "$RESULTS_DIR/supervisor.pid")"
    if kill -0 "$PREVIOUS_PID" 2>/dev/null && "$PYTHON_BIN" -I - "$PREVIOUS_PID" <<'PY'
import pathlib, sys
try:
    command = (pathlib.Path('/proc') / sys.argv[1] / 'cmdline').read_bytes()
except OSError:
    raise SystemExit(1)
raise SystemExit(0 if b'literature_supervisor.py' in command else 1)
PY
    then
        echo "Literature supervisor already active: PID $PREVIOUS_PID"
        exit 0
    fi
fi
if [[ "$RESUME_APPROVED" == 1 ]]; then
    taskset -c 80-127 "$PYTHON_BIN" -I "$VALIDATION_DIR/prepare_literature_resume.py"
fi
if [[ -f "$RESULTS_DIR/CREDIT_ALERT" ]]; then
    echo 'Credit alert remains active. Only the explicitly authorized resumption can clear the historical alert.' >&2
    exit 2
fi
if [[ "$(head -n 1 "$RESULTS_DIR/STATUS")" == 'DONE' ]]; then
    echo 'Literature task is DONE; refusing an accidental restart.' >&2
    exit 2
fi
if command -v nvidia-smi >/dev/null; then
    nvidia-smi --query-gpu=name --format=csv,noheader || echo 'GPU probe failed; the worker must diagnose before GPU stages.' >&2
fi
printf 'RUNNING\nApproved literature continuation; supervisor launch requested on CPUs80–127.\nNext: verify checkpoints, resume incomplete jobs within remaining allocations, and refresh the report.\n' > "$RESULTS_DIR/STATUS.tmp"
mv -- "$RESULTS_DIR/STATUS.tmp" "$RESULTS_DIR/STATUS"
# All model/tools/children inherit this affinity, including detached numerical jobs.
# workspace-write is explicit; no sandbox/approval bypass is used.
setsid nohup taskset -c 80-127 "$PYTHON_BIN" -I "$VALIDATION_DIR/literature_supervisor.py" -- -m gpt-6.1-sol -s workspace-write > "$RESULTS_DIR/launcher.log" 2>&1 < /dev/null 9>&- &
LAUNCHED_PID=$!
printf '%s\n' "$LAUNCHED_PID" > "$RESULTS_DIR/supervisor.pid"
echo "Started supervised literature task: PID $LAUNCHED_PID, CPUs80–127, model gpt-6.1-sol, sandbox workspace-write."
echo "Logs: $RESULTS_DIR/supervisor.log and $RESULTS_DIR/codex_exec.log"
