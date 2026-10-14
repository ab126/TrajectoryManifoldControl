"""Isolated PID environment reconstruction; no inherited third-party packages."""
import os,sys,json,subprocess,time,importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];OUT=ROOT/'validation/results_literature';prefix=Path('/tmp/trajectory-literature-envs/pid')
assert set(os.sched_getaffinity(0))<=set(range(80,128))
spec=importlib.util.spec_from_file_location('guard',ROOT/'validation/codex_supervisor.py');guard=importlib.util.module_from_spec(spec);spec.loader.exec_module(guard)
limits=guard.latest_limits();assert limits is not None and guard.balance(limits)>=json.loads((OUT/'manifest.json').read_text())['credit_balance_baseline']-1e-6
assert guard.window_state(limits.get('primary'))[0]<85 and guard.window_state(limits.get('secondary'))[0]<90
env=os.environ.copy()
for k in list(env):
    if k.startswith('CONDA_') or k.startswith('_CE_') or k in ['PYTHONPATH','PYTHONHOME','PYTHONUSERBASE','VIRTUAL_ENV','_CONDA_EXE']:env.pop(k,None)
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:env[k]='1'
env['PYTHONNOUSERSITE']='1'
commands=[['taskset','-c','80-127',sys.executable,'-I','-m','venv',str(prefix)],['taskset','-c','80-127',str(prefix/'bin/python'),'-I','-m','pip','install','numpy==2.2.6','scipy==1.15.3']]
dest=OUT/'environments';job=dict(pid=os.getpid(),state='running',commands=commands,started_at=time.time());p=dest/'JOB_setup_pid.json';p.write_text(json.dumps(job,indent=2))
try:
    for c in commands:subprocess.run(c,env=env,check=True)
    freeze=subprocess.check_output(['taskset','-c','80-127',str(prefix/'bin/python'),'-I','-m','pip','freeze'],env=env,text=True);(dest/'pid.freeze.txt').write_text(freeze)
    (dest/'pid.json').write_text(json.dumps(dict(interpreter=str(prefix/'bin/python'),prefix=str(prefix),system_site_packages=False,user_site_packages=False,reconstruction_commands=commands),indent=2))
    job.update(state='complete',finished_at=time.time());p.write_text(json.dumps(job,indent=2))
except Exception:
    job.update(state='error',finished_at=time.time());p.write_text(json.dumps(job,indent=2));raise
