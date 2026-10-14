#!/usr/bin/env python3
"""Run the approved literature extension with the unchanged original usage guard."""
from __future__ import annotations

import fcntl
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

VALIDATION = Path(__file__).resolve().parent
RESULTS = VALIDATION / 'results_literature'
GUARD = VALIDATION / 'codex_supervisor.py'
GUARD_SHA256 = 'c134dba7cfe47a38d96f3026f9a1b24143560fd69281a6a87cd8e30757bd08cc'
CPUS = set(range(80, 128))


def configured_guard():
    if hashlib.sha256(GUARD.read_bytes()).hexdigest() != GUARD_SHA256:
        raise RuntimeError('Original supervisor changed; review before using its guard.')
    available = os.sched_getaffinity(0)
    if not CPUS.issubset(available):
        raise RuntimeError(f'Required CPUs 80–127 unavailable; allowed CPUs: {sorted(available)}')
    os.sched_setaffinity(0, CPUS)
    for key in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
        os.environ[key] = '1'
    spec = importlib.util.spec_from_file_location('literature_original_guard', GUARD)
    guard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guard)
    guard.RESULTS = RESULTS
    guard.INITIAL_PROMPT = (
        'Execute the user-approved validation/LITERATURE_COMPARISON_PLAN.md, including '
        'stages 7–8 controlled Deep Koopman and controlled DFINE. First read '
        'validation/results_literature/TASK.md, FOLLOWUP_COMPARISON_MATRIX.md and '
        'manifest.json. Use only '
        'validation/results_literature for status/logs/state; never resume the completed '
        'results_v1 campaign. Every process must stay on CPUs 80–127; numerical commands '
        'must explicitly use taskset -c 80-127, with single-thread libraries and at most '
        'four CPU workers. Preserve submitted code and shared environments. Use public '
        'code in dedicated project venvs under /tmp/trajectory-literature-envs and '
        'isolated /tmp checkouts. Never borrow overseeg or another project environment; '
        'use explicit method interpreter paths. You are under the unchanged usage '
        'supervisor. Implement and execute the approved stages resumably within 30 GPU-hours '
        'and 84 aggregate CPU core-hours. Read the independent-review routing note in TASK.md '
        'and send concise uncertainty/results packets to overseer when reachable. Begin with '
        'inventory, isolated setup, bounded correctness checks and EDMD/SINDy. Report their '
        'checkpoint, then continue the approved stages without per-stage approval. '
        'Detach jobs longer than two minutes, record JOB files, and maintain RESULTS.md '
        'and STATUS. A correct unsuccessful controller is a scientific result; never hide '
        'it or add undeclared feedback. Preserve work on usage pauses. Commit only reviewed '
        'new task files, preserving all pre-existing edits; if Git write permissions prevent '
        'commits, save a reviewable diff and document it rather than stopping scientific work.'
    )
    guard.RESUME_PROMPT = (
        'Continue the approved literature extension. Read validation/results_literature/'
        'TASK.md, CONTINUATION_20261014.md when present, FOLLOWUP_COMPARISON_MATRIX.md, '
        'STATUS, manifest.json, RESULTS.md and '
        'JOB files. Check existing detached '
        'jobs before launching anything. Preserve CPU affinity80–127, unchanged usage '
        'guard, 30 GPU-hour/84 CPU core-hour ceilings and dedicated project venvs; '
        'never borrow overseeg or another project environment. Follow '
        'validation/LITERATURE_COMPARISON_PLAN.md through stages1–8; do not restart '
        'results_v1. Route concise uncertainty/result checkpoints to overseer as specified '
        'in TASK.md, then continue permitted work.'
    )
    return guard


def main():
    # The inherited thresholds, polling and credit guard cannot be overridden here.
    forbidden = ('--max-5h', '--max-week', '--poll', '--reset-slack', '--session')
    if any(arg.split('=', 1)[0] in forbidden for arg in sys.argv[1:]):
        raise SystemExit('Threshold/guard/session overrides are disabled for this task.')
    guard = configured_guard()
    RESULTS.mkdir(parents=True, exist_ok=True)
    if '--check' in sys.argv[1:]:
        return guard.main()
    limits = guard.latest_limits()
    if limits is None or guard.balance(limits) is None:
        raise SystemExit('Usage or credit telemetry missing; refusing an unguarded start.')
    manifest = json.loads((RESULTS / 'manifest.json').read_text())
    if guard.balance(limits) < manifest['credit_balance_baseline'] - 1e-6:
        raise SystemExit('Credits below task baseline; refusing to start.')
    if guard.status() != 'RUNNING':
        raise SystemExit('Use launch_literature.sh after host preflight; task is not RUNNING.')
    # Held for the entire run; a duplicate launcher cannot start another model session.
    with open(RESULTS / '.supervisor.lock', 'a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('Literature supervisor already active.')
        (RESULTS / 'supervisor.pid').write_text(f'{os.getpid()}\n')
        guard.log('literature task: CPUs80–127; separate session/state; original usage guard unchanged')
        return guard.main()


if __name__ == '__main__':
    raise SystemExit(main())
