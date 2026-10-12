"""Detached literature worker watchdog: original usage guard, one worker, JOB identity.

Run with the dedicated stdlib-only supervisor interpreter, explicit taskset, and
nohup/setsid. A usage interruption preserves completed atomic case checkpoints.
"""
import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import resource

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'validation/results_literature'


def write(path,obj):
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(obj,indent=2)+'\n');temp.replace(path)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('method',choices=['edmd','sindy','sindy_audit','pets','smc','rokdeepc','deepkoopman','dfine','deepkoopman_passive','deepkoopman_control','dfine_passive','dfine_control']);parser.add_argument('phase',choices=['checks','fit','pilot','matrix','timing','expansion','collect','fit_cpu','exact_pilot','exact_matrix']);args=parser.parse_args()
    method=args.method;stage=OUT/('smc_C200' if method=='smc' else method);stage.mkdir(parents=True,exist_ok=True)
    assert set(os.sched_getaffinity(0))<=set(range(80,128))
    manifest=json.loads((OUT/'manifest.json').read_text())
    guardpath=ROOT/'validation/codex_supervisor.py'
    assert hashlib.sha256(guardpath.read_bytes()).hexdigest()==manifest['protected_source_hashes']['validation/codex_supervisor.py']
    spec=importlib.util.spec_from_file_location('guard',guardpath);guard=importlib.util.module_from_spec(spec);spec.loader.exec_module(guard)
    def permitted():
        limits=guard.latest_limits()
        if limits is None or guard.balance(limits) is None:return 'missing usage telemetry'
        if guard.balance(limits)<manifest['credit_balance_baseline']-1e-6:return 'credit decrease'
        if guard.window_state(limits.get('primary'))[0]>=85 or guard.window_state(limits.get('secondary'))[0]>=90:return 'usage pause'
        if (OUT/'STATUS').read_text().splitlines()[0]!='RUNNING':return 'task not RUNNING'
        return None
    with open(stage/'.worker.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if method in ('edmd','sindy'):
            if args.phase.startswith('exact_'):
                command=['taskset','-c','80-127','/tmp/trajectory-literature-envs/sindy/bin/python',str(Path(__file__).with_name('run_sindy_exact.py')),'--phase',args.phase.split('_',1)[1]]
            elif args.phase=='expansion':
                command=['taskset','-c','80-127',f'/tmp/trajectory-literature-envs/{method}/bin/python',str(Path(__file__).with_name('run_edmd_expansion.py'))]
            else:
                runner='run_sindy_compiled.py' if method=='sindy' else 'run_classical.py'
                command=['taskset','-c','80-127',f'/tmp/trajectory-literature-envs/{method}/bin/python',str(Path(__file__).with_name(runner)),method,'--phase',args.phase]
        elif method=='pets':
            command=['taskset','-c','80-127',f'/tmp/trajectory-literature-envs/{method}/bin/python',str(Path(__file__).with_name('run_pets.py')),'--phase',args.phase]
        elif method=='smc':
            command=['taskset','-c','80-127','/tmp/trajectory-literature-envs/smc/bin/python',str(Path(__file__).with_name('run_smc.py'))]
        elif method=='sindy_audit':
            command=['taskset','-c','80-127','/tmp/trajectory-literature-envs/sindy/bin/python',str(Path(__file__).with_name('check_sindy_gradient.py'))]
        elif method=='rokdeepc':
            command=['taskset','-c','80-127','/tmp/trajectory-literature-envs/rokdeepc/bin/python',str(Path(__file__).with_name('run_rokdeepc.py')),'--phase',args.phase]
        elif method in ['deepkoopman_passive','deepkoopman_control','dfine_passive','dfine_control']:
            base,mode=method.rsplit('_',1);phase='fit' if args.phase=='fit_cpu' else args.phase;device='cpu' if args.phase in ['fit_cpu','collect'] else 'cuda'
            runner='run_neural_fast.py' if mode=='control' and phase in ['pilot','matrix'] else 'run_neural.py'
            command=['taskset','-c','80-127',f'/tmp/trajectory-literature-envs/{base}/bin/python',str(Path(__file__).with_name(runner)),base,'--mode',mode,'--phase',phase,'--device',device]
        else:
            command=['taskset','-c','80-127',f'/tmp/trajectory-literature-envs/{method}/bin/python',str(Path(__file__).with_name('check_neural.py')),method]
        job=dict(watchdog_pid=os.getpid(),method=method,phase=args.phase,command=command,state='starting',started_at=time.time(),cpu_affinity=list(range(80,128)),max_workers=1,library_threads=1)
        path=stage/f'JOB_{args.phase}.json';write(path,job)
        reason=permitted()
        if reason:job.update(state='paused',reason=reason);write(path,job);return
        env=os.environ.copy()
        for k in list(env):
            if k.startswith('CONDA_') or k.startswith('_CE_') or k in ['PYTHONPATH','PYTHONHOME','PYTHONUSERBASE','VIRTUAL_ENV','_CONDA_EXE']:env.pop(k,None)
        for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:env[k]='1'
        env['PYTHONNOUSERSITE']='1'
        logpath=stage/f'{args.phase}_attempt_{time.time_ns()}.log'
        output=logpath.open('w')
        accounting_stage='preparation' if method in ['deepkoopman','dfine','sindy_audit'] else ('smc_C200' if method=='smc' else method)
        accounting_path=OUT/accounting_stage/'accounting.json'
        before_account=json.loads(accounting_path.read_text()) if accounting_path.exists() else dict(cpu_seconds=0.,wall_seconds=0.,gpu_seconds=0.)
        worker_started=time.time()
        process=subprocess.Popen(command,env=env,start_new_session=True,stdout=output,stderr=subprocess.STDOUT)
        job.update(worker_pid=process.pid,state='running',log_path=str(logpath.relative_to(ROOT)));write(path,job)
        def stop(signum=None,frame=None):
            if process.poll() is None:
                os.killpg(process.pid,signal.SIGTERM)
                try:process.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
        signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
        reason=None
        while process.poll() is None:
            try:process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                reason=permitted()
                if reason:stop();break
        output.close()
        # Child rusage includes all worker threads, startup/imports and a killed
        # partial interval. Preserve live checkpoint accounting and add only gaps.
        measured=resource.getrusage(resource.RUSAGE_CHILDREN)
        child_cpu=measured.ru_utime+measured.ru_stime
        own=resource.getrusage(resource.RUSAGE_SELF);watchdog_cpu=own.ru_utime+own.ru_stime
        measured_wall=time.time()-worker_started
        accounting=json.loads(accounting_path.read_text()) if accounting_path.exists() else before_account.copy()
        accounted_cpu=max(0.,accounting['cpu_seconds']-before_account['cpu_seconds'])
        accounted_wall=max(0.,accounting['wall_seconds']-before_account['wall_seconds'])
        accounting['cpu_seconds']+=max(0.,child_cpu-accounted_cpu)+watchdog_cpu
        accounting['wall_seconds']+=max(0.,measured_wall-accounted_wall)
        gpu_increment=accounting.get('gpu_seconds',0.)-before_account.get('gpu_seconds',0.)
        if gpu_increment>0:accounting['gpu_seconds']+=max(0.,measured_wall-gpu_increment)
        accounting_path.parent.mkdir(parents=True,exist_ok=True);write(accounting_path,accounting)
        job.update(worker_cpu_seconds=child_cpu,watchdog_cpu_seconds=watchdog_cpu,worker_elapsed_seconds=measured_wall,accounting_stage=accounting_stage)
        if process.returncode and not reason:
            tail=logpath.read_text()[-16000:]
            if 'RuntimeError: Usage pause:' in tail:reason='usage pause'
            elif 'RuntimeError: Missing usage telemetry' in tail:reason='missing usage telemetry'
        job.update(state='paused' if reason else ('complete' if process.returncode==0 else 'error'),reason=reason,exit_code=process.returncode,finished_at=time.time())
        write(path,job)


if __name__=='__main__':main()
