#!/usr/bin/env bash
set -euo pipefail
VALIDATION_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="$VALIDATION_DIR/.venv/bin/python"
RESULTS_DIR="$VALIDATION_DIR/results_literature"
if [[ "${1:-}" == "--check" ]]; then
    exec taskset -c 80-127 "$PYTHON_BIN" "$VALIDATION_DIR/literature_supervisor.py" --check
fi
if [[ $# -ne 0 ]]; then
    echo 'Usage: bash launch_literature.sh [--check]' >&2
    exit 2
fi
if [[ ! -w "$HOME/.codex" ]]; then
    echo 'Run this script in the host terminal: this session cannot write ~/.codex to start the supervised model.' >&2
    exit 2
fi
command -v codex >/dev/null
command -v taskset >/dev/null
taskset -c 80-127 "$PYTHON_BIN" "$VALIDATION_DIR/literature_supervisor.py" --check
mkdir -p "$RESULTS_DIR"
exec 9>"$RESULTS_DIR/.launch.lock"
flock -n 9 || { echo 'Another literature launch is already in progress.' >&2; exit 2; }
if [[ -f "$RESULTS_DIR/supervisor.pid" ]]; then
    PREVIOUS_PID="$(cat "$RESULTS_DIR/supervisor.pid")"
    if kill -0 "$PREVIOUS_PID" 2>/dev/null; then
        echo "Literature supervisor already active: PID $PREVIOUS_PID"
        exit 0
    fi
fi
if [[ "$(head -n 1 "$RESULTS_DIR/STATUS")" == 'DONE' ]]; then
    echo 'Literature task is DONE; refusing an accidental restart.' >&2
    exit 2
fi
if command -v nvidia-smi >/dev/null; then
    nvidia-smi --query-gpu=name --format=csv,noheader || echo 'GPU probe failed; the worker must diagnose before GPU stages.' >&2
fi
printf 'RUNNING\nApproved literature stages1–8; supervisor launch requested on CPUs80–127.\nNext: inventory, isolated setup, EDMD/SINDy implementation and bounded pilots.\n' > "$RESULTS_DIR/STATUS"
# All model/tools/children inherit this affinity, including detached numerical jobs.
# workspace-write is explicit; no sandbox/approval bypass is used.
setsid nohup taskset -c 80-127 "$PYTHON_BIN" "$VALIDATION_DIR/literature_supervisor.py" -- -m gpt-6.1-sol -s workspace-write > "$RESULTS_DIR/launcher.log" 2>&1 < /dev/null 9>&- &
LAUNCHED_PID=$!
printf '%s\n' "$LAUNCHED_PID" > "$RESULTS_DIR/supervisor.pid"
echo "Started supervised literature task: PID $LAUNCHED_PID, CPUs80–127, model gpt-6.1-sol, sandbox workspace-write."
echo "Logs: $RESULTS_DIR/supervisor.log and $RESULTS_DIR/codex_exec.log"
