"""One fixed diagnostic campaign, <=0.25 worker-hours including startup.

Default preflight. Shared occupancy persists across local and CUDA invocations.
"""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
RUN = HERE/'overnight_runs/20261007T054549.695689Z'
ROOT = HERE/'diagnostics'


def main(args):
    ROOT.mkdir(exist_ok=True)
    lock = (ROOT/'campaign.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    used = sum(json.loads(line)['worker_seconds'] for line in (ROOT/'usage.jsonl').read_text().splitlines()) if (ROOT/'usage.jsonl').exists() else 0.
    if (ROOT/'active.json').exists():
        raise RuntimeError('Unclosed diagnostic occupancy record; review before restarting')
    baseline = json.loads((RUN/'run.json').read_text())
    readiness = sum(json.loads(line)['worker_seconds'] for line in (HERE/'artifacts/usage.jsonl').read_text().splitlines())
    other_rows = [json.loads(path.read_text()) for path in (HERE/'overnight_runs').glob('*/run.json')]
    if any(row['status']=='running' for row in other_rows):
        raise RuntimeError('Unclosed calibration occupancy; review before diagnosis')
    other_runs = sum(row['worker_seconds'] for row in other_rows)
    stage0_used = readiness+other_runs+used
    reservation = {'local': 120., 'cem': 240., 'all': 360.}[args.mode]
    remaining = min(900-used, baseline['stage0_cap_hours']*3600-stage0_used)
    proposal = {'mode': args.mode, 'diagnostic_campaign_cap_s': 900, 'diagnostic_used_s': used,
                'remaining_s': remaining, 'reservation_s': reservation, 'fits': reservation<=remaining,
                'execute': args.execute, 'scope': 'fixed diagnostics only; no calibration or dependent stages'}
    print(json.dumps(proposal, indent=2), flush=True)
    if not args.execute:
        return 0 if proposal['fits'] else 2
    if not proposal['fits']:
        raise RuntimeError('Diagnostic reservation exceeds shared allocation')
    for name, expected in baseline['submitted_source_hashes'].items():
        if hashlib.sha256((HERE.parent.parent/name).read_bytes()).hexdigest()!=expected:
            raise RuntimeError(f'Submitted source changed: {name}')
    previous = [json.loads(path.read_text()) for path in ROOT.glob('*/job.json')]
    done = {item['mode'] for item in previous if item['returncode']==0}
    if args.mode in done or 'all' in done or args.mode=='all' and done:
        raise RuntimeError('Avoid duplicate diagnostic work; use cem after a completed local job')
    destination = ROOT/datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    destination.mkdir()
    snapshots = destination/'source_snapshots'; snapshots.mkdir()
    for name in ('oracle_diagnostics.py', 'diagnostic_supervisor.py', 'run_diagnostics.sh',
                 'common.py', 'models.py', 'compat.py', 'gate_a.py', 'overnight_runtime.py', 'oracle_diagnostic_spec.md'):
        shutil.copyfile(HERE/name, snapshots/name)
    command = [sys.executable, str(HERE/'oracle_diagnostics.py'), '--mode', args.mode,
               '--device', 'cpu' if args.mode=='local' else 'cuda', '--run-dir', str(RUN), '--output-dir', str(destination)]
    state = {'mode': args.mode, 'command': command, 'status': 'running', 'stage0_prior_seconds': stage0_used,
             'source_SHA256': {name: hashlib.sha256((HERE/name).read_bytes()).hexdigest()
                               for name in ('oracle_diagnostics.py', 'diagnostic_supervisor.py', 'run_diagnostics.sh')},
             'scope': 'development diagnosis; no H modifications or Gate A pass inference'}
    (ROOT/'active.json').write_text(json.dumps(state, indent=2))
    start = time.perf_counter()
    process = None
    def interrupted(signum, frame):
        raise InterruptedError(f'Supervisor interrupted by signal {signum}')
    old_signals = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        with (destination/'process.log').open('w') as log:
            process = subprocess.Popen(command, env=os.environ.copy(), stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            code = process.wait(timeout=min(reservation, remaining)-30)
    except subprocess.TimeoutExpired:
        code = 124
    except (InterruptedError, KeyboardInterrupt) as error:
        code = 130; state['interruption'] = str(error)
    except Exception as error:
        code = 127; state['launch_error'] = str(error)
    finally:
        for sig, handler in old_signals.items():
            signal.signal(sig, handler)
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try: process.wait(timeout=30)
            except subprocess.TimeoutExpired: os.killpg(process.pid, signal.SIGKILL); process.wait()
    elapsed = time.perf_counter()-start
    state.update({'status': 'finished', 'returncode': code, 'worker_seconds': elapsed})
    (destination/'job.json').write_text(json.dumps(state, indent=2)+'\n')
    with (ROOT/'usage.jsonl').open('a') as file:
        file.write(json.dumps({'label': f'oracle_diagnostic_{args.mode}', 'worker_seconds': elapsed, 'returncode': code,
                               'directory': str(destination)})+'\n')
    (ROOT/'active.json').unlink()
    (ROOT/'latest.json').write_text(json.dumps({'directory': str(destination)}, indent=2)+'\n')
    print(json.dumps(state, indent=2))
    return code


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['local', 'cem', 'all'])
    parser.add_argument('--execute', action='store_true')
    raise SystemExit(main(parser.parse_args()))
