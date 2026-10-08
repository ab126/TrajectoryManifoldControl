"""Bound all readiness executions and charge import/process overhead."""
import json
import os
import subprocess
import signal
import sys
import time
from pathlib import Path
HERE=Path(__file__).resolve().parent
usage=HERE/'artifacts/usage.jsonl'
def used():
    return sum(json.loads(s)['worker_seconds'] for s in usage.read_text().splitlines()) if usage.exists() else 0.

task=sys.argv[1]
allowed={'plant_checks','equivalence','existing_tests','smoke','benchmark','concurrency','gate_a','harness_checks','gpu_probe','overnight_tests'}
if task not in allowed: raise SystemExit('Unknown task')
# Gate A is outside the one-hour readiness campaign; its own conservative
# reservation/cap applies. No --execute command was run during readiness.
if task=='gate_a':
    raise SystemExit(subprocess.call([sys.executable,str(HERE/f'{task}.py'),*sys.argv[2:]],env=os.environ.copy()))
before=used(); remaining=3600-before
if remaining<120: raise SystemExit('Readiness worker-hour budget exhausted')
start=time.perf_counter()
try:
    process=subprocess.Popen([sys.executable,str(HERE/f'{task}.py'),*sys.argv[2:]],env=os.environ.copy(),start_new_session=True)
    code=process.wait(timeout=min(remaining-30,900))
except subprocess.TimeoutExpired:
    # Terminate the whole job group, including concurrency children. Its handler
    # writes occupancy in finally before exit; force-kill after bounded cleanup.
    os.killpg(process.pid,signal.SIGTERM)
    try: process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid,signal.SIGKILL); process.wait()
    code=124
elapsed=time.perf_counter()-start
delta=used()-before
# Concurrency records complete worker process occupancy itself. For other jobs
# the inner Meter records useful work; charge the remainder (imports/cleanup).
overhead=elapsed if task=='concurrency' else max(0,elapsed-delta)
with usage.open('a') as f:
    f.write(json.dumps({'label':f'{task} supervisor/import overhead','worker_seconds':overhead,
                        'wall_seconds':elapsed,'exit_code':code})+'\n')
raise SystemExit(code)
