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


def projection_stop(method,phase):
    """Compare completed-case cost to the same requested phase's pilot projection."""
    if phase in ['expansion','old_expansion'] and method in ['pets','deepkoopman_control','dfine_control']:
        name='original' if phase=='old_expansion' else 'fresh';p=OUT/method/f'{name}_expansion_plan.json'
        if not p.exists():return None
        selection=json.loads(p.read_text());cells=selection.get('cells',selection.get('sets',[]));predicted=sum(c['projected_gpu_hours'] for c in cells)
        actual=0.
        for p in (OUT/method/'cases').glob('*.json'):
            row=json.loads(p.read_text())
            if row.get('seconds')==14. and any(c['seed']==row['seed'] and c['set']==row['set'] for c in cells):actual+=row.get('wall_seconds',0.)
        if predicted and actual>3*predicted*3600:return 'three-times pilot projection reached; cap-limited expansion partial retained'
        return None
    if phase in ['stress','stress_matrix'] and method in ['pets','sindy','rokdeepc','deepkoopman_control','dfine_control']:
        plan=OUT/method/'stress/plan.json'
        if not plan.exists():return None
        selection=json.loads(plan.read_text())
        gpu=phase=='stress_matrix' or method=='pets'
        predicted=selection.get('projected_gpu_hours_per_condition' if gpu else 'projected_core_hours_per_condition',0.)*len(selection['conditions'])
        if not predicted:return None
        actual=sum(json.loads(p.read_text()).get('wall_seconds' if gpu else 'cpu_seconds',0.) for p in (OUT/method/'stress/cases').glob('*.json'))
        if actual>3*predicted*3600:return 'three-times pilot projection reached; cap-limited stress partial retained'
        return None
    if method=='long_references' and phase=='matrix':
        p=OUT/'long_references/pilot_projection.json'
        if p.exists():
            predicted=json.loads(p.read_text())['projected_gpu_hours'];actual=sum(json.loads(p.read_text()).get('wall_seconds',0.) for p in (OUT/'long_references/cases').glob('*.json'))
            if actual>3*predicted*3600:return 'three-times pilot projection reached; cap-limited reference partial retained'
        return None
    specifications={
        ('pets','matrix'):('pets/pilot_projection.json','projected_gpu_hours',['pets/cases'],'wall_seconds'),
        ('pets','reference_matrix'):('pets/deterministic_reference/pilot_projection.json','projected_gpu_hours',['pets/deterministic_reference/cases'],'wall_seconds'),
        ('deepkoopman_control','matrix'):('deepkoopman_control/pilot_projection.json','projected_hours',['deepkoopman_control/cases','neural_first_action/cases'],'wall_seconds'),
        ('dfine_control','matrix'):('dfine_control/pilot_projection.json','projected_hours',['dfine_control/cases'],'wall_seconds'),
        ('dfine_control','output_matrix'):('dfine_control/output_only/pilot_projection.json','projected_gpu_hours',['dfine_control/output_only/cases'],'wall_seconds'),
        ('sindy','exact_matrix'):('sindy/exact_pilot_projection.json','projected_core_hours',['sindy/cases'],'cpu_seconds'),
        ('sindy','expansion'):('sindy/expansion_projection.json','projected_core_hours',['sindy/cases'],'cpu_seconds'),
        ('rokdeepc','matrix'):('rokdeepc/pilot_projection.json','projected_core_hours',['rokdeepc/cases'],'cpu_seconds'),
        ('rokdeepc','expansion'):('rokdeepc/expansion_projection.json','projected_core_hours',['rokdeepc/cases'],'cpu_seconds'),
    }
    spec=specifications.get((method,phase))
    if spec is None:return None
    projection_path,key,paths,cost_key=spec
    p=OUT/projection_path
    if not p.exists():return None
    projected=json.loads(p.read_text()).get(key)
    if not projected or projected<=0:return None
    actual=0.
    for folder in paths:
        for p in (OUT/folder).glob('*.json'):
            row=json.loads(p.read_text())
            if method=='sindy' and row['arm'] not in ['CEM','CEM+L','NMPC_exact','NMPC_exact+L']:continue
            if phase=='expansion':
                if row.get('set') not in ['C','F'] or row.get('seconds')!=14.:continue
            else:
                case=row.get('identity',{}).get('case',{})
                index=case.get('index',row.get('index'))
                if index is None:index=int(row['case'].rsplit('_',2)[1])
                if row.get('set')=='F' or (row.get('set')=='C' and index>=50):continue
                if row.get('set') not in ['A','C']:continue
            actual+=row.get(cost_key,0.)
    if actual>3*projected*3600:return 'three-times pilot projection reached; cap-limited partial retained'
    return None


def main():
    parser=argparse.ArgumentParser();parser.add_argument('method',choices=['long_references','replay_inventory','reference_preflight','forecast_preflight','figures','report','latent_closure','classical_audit','pid','edmd','sindy','sindy_audit','pets','smc','rokdeepc','deepkoopman','dfine','deepkoopman_passive','deepkoopman_control','dfine_passive','dfine_control']);parser.add_argument('phase',choices=['stress_fit','stress_matrix','old_expansion','diagnostics','checks','fit','pilot','matrix','timing','expansion','collect','fit_cpu','exact_pilot','exact_matrix','baselines','inference','output_fit','output_inference','output_pilot','output_matrix','validation','reference_fit','reference_pilot','reference_matrix','stress','maps','references_pilot','references_matrix']);args=parser.parse_args()
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
            if args.phase=='diagnostics':
                command=['taskset','-c','80-127',f'/tmp/trajectory-literature-envs/{method}/bin/python',str(Path(__file__).with_name('classical_equilibrium.py')),method]
            elif method=='sindy' and args.phase=='stress':
                command=['taskset','-c','80-127','/tmp/trajectory-literature-envs/sindy/bin/python',str(Path(__file__).with_name('sindy_stress.py'))]
            elif method=='edmd' and args.phase in ['stress','maps']:
                command=['taskset','-c','80-127','/tmp/trajectory-literature-envs/edmd/bin/python',str(Path(__file__).with_name('run_edmd_details.py')),'--phase',args.phase]
            elif args.phase.startswith('exact_'):
                command=['taskset','-c','80-127','/tmp/trajectory-literature-envs/sindy/bin/python',str(Path(__file__).with_name('run_sindy_exact.py')),'--phase',args.phase.split('_',1)[1]]
            elif args.phase=='expansion':
                runner='run_sindy_expansion.py' if method=='sindy' else 'run_edmd_expansion.py'
                command=['taskset','-c','80-127',f'/tmp/trajectory-literature-envs/{method}/bin/python',str(Path(__file__).with_name(runner))]
            else:
                runner='run_sindy_compiled.py' if method=='sindy' else 'run_classical.py'
                command=['taskset','-c','80-127',f'/tmp/trajectory-literature-envs/{method}/bin/python',str(Path(__file__).with_name(runner)),method,'--phase',args.phase]
        elif method=='long_references':
            command=['taskset','-c','80-127','/tmp/trajectory-literature-envs/deepkoopman/bin/python',str(Path(__file__).with_name('long_references.py'))]
        elif method=='replay_inventory':
            command=['taskset','-c','80-127','/tmp/trajectory-literature-envs/dfine/bin/python',str(Path(__file__).with_name('replay_inventory.py'))]
        elif method=='reference_preflight':
            command=['taskset','-c','80-127','/tmp/trajectory-literature-envs/deepkoopman/bin/python',str(Path(__file__).with_name('reference_preflight.py'))]
        elif method=='forecast_preflight':
            command=['taskset','-c','80-127','/tmp/trajectory-literature-envs/dfine/bin/python',str(Path(__file__).with_name('output_forecast_preflight.py' if args.phase=='timing' else 'forecast_preflight.py'))]
        elif method=='figures':
            command=['taskset','-c','80-127','/tmp/trajectory-literature-envs/dfine/bin/python',str(Path(__file__).with_name('figures_literature.py'))]
        elif method=='report':
            command=['taskset','-c','80-127','/tmp/trajectory-literature-envs/supervisor/bin/python','-I',str(Path(__file__).with_name('stress_report.py' if args.phase=='collect' else 'collect_report.py'))]
        elif method=='latent_closure':
            command=['taskset','-c','80-127','/tmp/trajectory-literature-envs/dfine/bin/python',str(Path(__file__).with_name('latent_closure.py'))]
        elif method=='classical_audit':
            command=['taskset','-c','80-127','/tmp/trajectory-literature-envs/edmd/bin/python',str(Path(__file__).with_name('classical_audit.py'))]
        elif method=='pid':
            command=['taskset','-c','80-127','/tmp/trajectory-literature-envs/pid/bin/python',str(Path(__file__).with_name('run_pid.py')),'--phase',args.phase]
        elif method=='pets':
            if args.phase in ['expansion','old_expansion']:command=['taskset','-c','80-127','/tmp/trajectory-literature-envs/pets/bin/python',str(Path(__file__).with_name('run_pets_expansion.py')),'--set','original' if args.phase=='old_expansion' else 'fresh']
            elif args.phase=='stress':command=['taskset','-c','80-127','/tmp/trajectory-literature-envs/pets/bin/python',str(Path(__file__).with_name('pets_stress.py'))]
            elif args.phase=='validation':command=['taskset','-c','80-127','/tmp/trajectory-literature-envs/pets/bin/python',str(Path(__file__).with_name('pets_validation.py'))]
            elif args.phase.startswith('reference_'):command=['taskset','-c','80-127','/tmp/trajectory-literature-envs/pets/bin/python',str(Path(__file__).with_name('deterministic_ensemble.py')),'--phase',args.phase.split('_',1)[1]]
            else:command=['taskset','-c','80-127',f'/tmp/trajectory-literature-envs/{method}/bin/python',str(Path(__file__).with_name('run_pets.py')),'--phase',args.phase]
        elif method=='smc':
            runner='run_smc_fresh.py' if args.phase=='expansion' else 'run_smc.py'
            command=['taskset','-c','80-127','/tmp/trajectory-literature-envs/smc/bin/python',str(Path(__file__).with_name(runner))]
        elif method=='sindy_audit':
            command=['taskset','-c','80-127','/tmp/trajectory-literature-envs/sindy/bin/python',str(Path(__file__).with_name('check_sindy_gradient.py'))]
        elif method=='rokdeepc':
            if args.phase=='diagnostics':command=['taskset','-c','80-127','/tmp/trajectory-literature-envs/rokdeepc/bin/python',str(Path(__file__).with_name('e4_stress_checks.py'))]
            elif args.phase=='validation':command=['taskset','-c','80-127','/tmp/trajectory-literature-envs/rokdeepc/bin/python',str(Path(__file__).with_name('rok_plan_replay.py'))]
            elif args.phase=='stress':command=['taskset','-c','80-127','/tmp/trajectory-literature-envs/rokdeepc/bin/python',str(Path(__file__).with_name('rok_stress.py'))]
            elif args.phase.startswith('references_'):command=['taskset','-c','80-127','/tmp/trajectory-literature-envs/deepkoopman/bin/python',str(Path(__file__).with_name('e4_references.py')),'--phase',args.phase.split('_',1)[1]]
            elif args.phase=='expansion':command=['taskset','-c','80-127','/tmp/trajectory-literature-envs/rokdeepc/bin/python',str(Path(__file__).with_name('run_rok_expansion.py'))]
            else:command=['taskset','-c','80-127','/tmp/trajectory-literature-envs/rokdeepc/bin/python',str(Path(__file__).with_name('run_rokdeepc.py')),'--phase',args.phase]
        elif method in ['deepkoopman_passive','deepkoopman_control','dfine_passive','dfine_control']:
            base,mode=method.rsplit('_',1)
            if args.phase in ['stress_fit','stress_matrix']:
                command=['taskset','-c','80-127',f'/tmp/trajectory-literature-envs/{base}/bin/python',str(Path(__file__).with_name('neural_stress.py')),base,'--phase','fit' if args.phase=='stress_fit' else 'matrix']
            elif args.phase in ['expansion','old_expansion'] and mode=='control':
                command=['taskset','-c','80-127',f'/tmp/trajectory-literature-envs/{base}/bin/python',str(Path(__file__).with_name('run_neural_expansion.py')),base,'--set','original' if args.phase=='old_expansion' else 'fresh']
            elif args.phase.startswith('references_'):
                command=['taskset','-c','80-127','/tmp/trajectory-literature-envs/deepkoopman/bin/python',str(Path(__file__).with_name('fresh_references.py')),'--phase',args.phase.split('_',1)[1]]
            elif args.phase in ['output_pilot','output_matrix']:
                command=['taskset','-c','80-127',f'/tmp/trajectory-literature-envs/{base}/bin/python',str(Path(__file__).with_name('output_control.py')),'--phase',args.phase.split('_',1)[1]]
            elif args.phase.startswith('output_'):
                command=['taskset','-c','80-127',f'/tmp/trajectory-literature-envs/{base}/bin/python',str(Path(__file__).with_name('run_output_safe.py')),'--mode',mode,'--phase',args.phase.split('_',1)[1]]
            elif args.phase in ['baselines','inference']:
                runner='passive_baselines.py' if args.phase=='baselines' else 'dfine_inference.py'
                command=['taskset','-c','80-127',f'/tmp/trajectory-literature-envs/{base}/bin/python',str(Path(__file__).with_name(runner))]
            else:
                phase='fit' if args.phase=='fit_cpu' else args.phase;device='cpu' if args.phase in ['fit_cpu','collect'] else 'cuda'
                runner='run_neural_reference_fast.py' if mode=='control' and phase in ['pilot','matrix'] else 'run_neural.py'
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
        env['PYTHONDONTWRITEBYTECODE']='1'
        cache_root=Path('/tmp/trajectory-literature-work');cache_root.mkdir(exist_ok=True)
        for key,folder in [('NUMBA_CACHE_DIR','numba-cache'),('CUDA_CACHE_PATH','cuda-cache'),('MPLCONFIGDIR','matplotlib-cache'),('TORCH_HOME','torch-cache')]:
            cache=cache_root/folder;cache.mkdir(exist_ok=True);env[key]=str(cache)
        logpath=stage/f'{args.phase}_attempt_{time.time_ns()}.log'
        output=logpath.open('w')
        accounting_stage='preparation' if method in ['replay_inventory','reference_preflight','forecast_preflight','figures','report','latent_closure','classical_audit','deepkoopman','dfine','sindy_audit'] else ('smc_C200' if method=='smc' else method)
        if (args.phase.startswith('references_') and method=='deepkoopman_control') or method=='long_references':accounting_stage='core_reserve'
        # Multiple method names can share the preparation/reserve ledger. Bind
        # accounting ownership before loading Budget.previous in the child.
        account_lock=open(OUT/f'.accounting_{accounting_stage}.lock','a')
        fcntl.flock(account_lock,fcntl.LOCK_EX)
        reason=permitted()
        if reason:job.update(state='paused',reason=reason);write(path,job);output.close();return
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
                if reason is None:reason=projection_stop(method,args.phase)
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
            elif 'RuntimeError: Credit guard:' in tail:reason='credit decrease'
            elif 'RuntimeError: Task not RUNNING' in tail:reason='task not RUNNING'
            elif any(message in tail for message in ['CPU allocation exhausted','CPU ceiling reached','GPU ceiling reached','cap-limited','exceeds stage cap','beyond stage cap','exceeds measured allocation','exceeds PETS allocation','exceeds DFINE cap','exceeds core reserve GPU cap','exceeds reserve suballocation']):reason='allocation/projection cap; preserve partial results'
        limited=reason is not None and ('cap' in reason or 'projection reached' in reason)
        job.update(state='cap_limited' if limited else ('paused' if reason else ('complete' if process.returncode==0 else 'error')),reason=reason,exit_code=process.returncode,finished_at=time.time())
        write(path,job)


if __name__=='__main__':main()
