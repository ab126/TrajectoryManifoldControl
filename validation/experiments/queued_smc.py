"""Launch the user-requested SMC extension only after required core measurements."""
import fcntl
import json
import os
from pathlib import Path
import shlex
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'validation/results_v1'
TARGET = OUT/'E6_SMC'
PREREQUISITES = ('E2', 'E2_distribution', 'E3', 'E3_linear', 'E4')


def prerequisites_ready():
    return all((OUT/name/'manifest.json').exists()
               and json.loads((OUT/name/'manifest.json').read_text()).get('status') == 'DONE'
               for name in PREREQUISITES)


def launch_after_core(dry_run=False):
    request_path = TARGET/'REQUEST.json'
    if not request_path.exists():
        return dict(status='NOT_REQUESTED')
    if not prerequisites_ready():
        return dict(status='QUEUED', after=list(PREREQUISITES))
    manifest_path = TARGET/'manifest.json'
    if manifest_path.exists():
        value = json.loads(manifest_path.read_text())
        if value.get('status') in ('DONE', 'PARTIAL', 'SKIPPED'):
            return dict(status=value['status'])
    if dry_run:
        return dict(status='READY', after=list(PREREQUISITES))
    with (TARGET/'.launch.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        job = TARGET/'JOB'
        if job.exists():
            lines = job.read_text().splitlines()
            pid = int(next(line.split()[1] for line in reversed(lines) if line.startswith('PID ')))
            command_path = Path(f'/proc/{pid}/cmdline')
            if command_path.exists() and b'e6_smc.py' in command_path.read_bytes():
                return dict(status='RUNNING', pid=pid)
        request = json.loads(request_path.read_text())
        elapsed = sum(json.loads(path.read_text()).get('wall_seconds', 0.) for path in OUT.glob('*/manifest.json'))
        elapsed += json.loads((OUT/'E0/authored_manifest.json').read_text()).get('wall_seconds', 0.)
        old = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        # The experiment's limits are cumulative across resumes. Its already
        # charged wall time is added back when computing that cumulative limit.
        limit = min(float(request['max_wall_seconds']), 40*3600-elapsed+old.get('wall_seconds', 0.))
        available = limit-old.get('wall_seconds', 0.)
        if available <= 0 or old.get('cpu_seconds', 0.) >= request['max_cpu_seconds']:
            value = dict(old, experiment='E6_SMC', status='PARTIAL' if old.get('completed_new_cases') else 'SKIPPED', device='cpu',
                         wall_seconds=old.get('wall_seconds', 0.), reason='Existing compute or SMC allowance exhausted')
            manifest_path.write_text(json.dumps(value, indent=2)+'\n')
            return dict(status=value['status'], reason=value['reason'])
        python = ROOT/'validation/.venv-gpu/bin/python'
        # Only numerical CPU feedback is run. This launches no Codex process or
        # model request and changes no usage-supervisor settings.
        command = ['timeout', str(max(1, int(available))), str(python),
                   str(ROOT/'validation/experiments/e6_smc.py'),
                   '--max-wall-seconds', str(limit), '--max-cpu-seconds', str(request['max_cpu_seconds'])]
        work = Path(tempfile.gettempdir())/'trajectory-manifold-smc'
        work.mkdir(exist_ok=True)
        env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                   PYTHONUNBUFFERED='1', CUDA_VISIBLE_DEVICES='',
                   LD_LIBRARY_PATH=str(python.parent.parent/'lib'),
                   LD_PRELOAD=str(python.parent.parent/'lib/libstdc++.so.6'))
        with (TARGET/'run.log').open('a') as output:
            child = subprocess.Popen(command, cwd=work, env=env, stdout=output, stderr=subprocess.STDOUT,
                                     stdin=subprocess.DEVNULL, start_new_session=True)
        job.write_text(f'PID {child.pid}\nCOMMAND '+' '.join(shlex.quote(arg) for arg in command)
                       +f'\nWORKDIR {work}\nMAX_WALL_SECONDS {limit}\nMAX_CPU_SECONDS {request["max_cpu_seconds"]}\n'
                       +'NOTE User-requested post-E4 CPU-only feedback comparison; saved-case resume.\n')
        return dict(status='RUNNING', pid=child.pid)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    print(json.dumps(launch_after_core(dry_run=args.check), indent=2))
