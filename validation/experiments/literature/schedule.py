"""Guarded dependency queue for already implemented/authorized literature phases.

No models/agents are spawned. Only dedicated numerical job.py workers are queued.
The queue stops on usage/credit pauses; restart it after supervised resumption.
"""
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[3];OUT=ROOT/'validation/results_literature'
SPECS=[
    ('replay_inventory','validation',['deepkoopman_control/matrix_completion.json'],False),
    ('report','collect',['edmd/stress_completion.json'],False),
    ('report','validation',['edmd/maps_completion.json'],False),
    ('figures','validation',['edmd/maps_completion.json'],False),
    ('reference_preflight','checks',[],False),
    ('forecast_preflight','timing',['dfine_control/output_only/fit_completion.json'],False),
    ('smc','expansion',['smc_C200/completion.json'],False),
    ('forecast_preflight','checks',['dfine_control/output_only/fit_completion.json'],False),
    ('pets','matrix',['pets/pilot_completion.json'],True),
    ('deepkoopman_control','fit_cpu',[],False),
    ('dfine_control','fit_cpu',[],False),
    ('rokdeepc','matrix',['rokdeepc/pilot_completion.json'],False),
    ('dfine_passive','inference',['dfine_passive/fit_completion.json','passive_baselines/completion.json'],False),
    ('classical_audit','checks',[],False),
    ('sindy','expansion',['preparation/audit_completion.json','sindy/exact_matrix_completion.json'],False),
    ('latent_closure','checks',['deepkoopman_control/fit_completion.json','dfine_control/fit_completion.json','passive_baselines/completion.json'],False),
    ('report','checks',['latent_closure/completion.json'],False),
    ('figures','checks',['prediction_tables.json'],False),
    ('pets','validation',['pets/matrix_completion.json'],True),
    ('pets','reference_fit',['pets/validation_completion.json'],True),
    ('pets','reference_pilot',['pets/deterministic_reference/fit_completion.json'],True),
    ('pets','reference_matrix',['pets/deterministic_reference/pilot_completion.json'],True),
    ('deepkoopman_control','pilot',['deepkoopman_control/fit_completion.json','forecast_preflight/completion.json'],True),
    ('deepkoopman_control','matrix',['deepkoopman_control/pilot_completion.json'],True),
    ('dfine_control','pilot',['dfine_control/fit_completion.json','forecast_preflight/completion.json'],True),
    ('dfine_control','matrix',['dfine_control/pilot_completion.json'],True),
    ('dfine_passive','output_fit',['dfine_passive/inference/completion.json'],False),
    ('dfine_passive','output_inference',['dfine_passive/output_only/fit_completion.json'],False),
    ('dfine_control','output_fit',['dfine_control/fit_completion.json'],False),
    ('dfine_control','output_pilot',['dfine_control/output_only/fit_completion.json','dfine_control/matrix_completion.json','forecast_preflight/output_history_completion.json'],True),
    ('dfine_control','output_matrix',['dfine_control/output_only/pilot_completion.json'],True),
    ('edmd','stress',['edmd/expansion_completion.json','preparation/audit_completion.json'],False),
    ('edmd','maps',['edmd/stress_completion.json'],False),
    ('deepkoopman_control','references_pilot',['deepkoopman_control/matrix_completion.json','reference_preflight/completion.json'],True),
    ('deepkoopman_control','references_matrix',['fresh_references/pilot_completion.json'],True),
    ('long_references','matrix',['fresh_references/pilot_completion.json'],True),
    ('rokdeepc','expansion',['rokdeepc/matrix_completion.json'],False),
    ('rokdeepc','references_pilot',['rokdeepc/expansion_completion.json','reference_preflight/completion.json'],True),
    ('rokdeepc','references_matrix',['rokdeepc/references/pilot_completion.json'],True),
    ('rokdeepc','diagnostics',['rokdeepc/expansion_completion.json'],False),
    ('rokdeepc','stress',['rokdeepc/expansion_completion.json','rokdeepc/stress/checks.json'],False),
    ('rokdeepc','validation',['rokdeepc/stress/completion.json'],False),
    ('deepkoopman_control','expansion',['deepkoopman_control/matrix_completion.json','dfine_control/output_only/matrix_completion.json'],True),
    ('dfine_control','expansion',['dfine_control/output_only/matrix_completion.json'],True),
    ('deepkoopman_control','stress_fit',['deepkoopman_control/fresh_expansion_completion.json'],False),
    ('deepkoopman_control','stress_matrix',['deepkoopman_control/stress/fit_completion.json'],True),
    ('dfine_control','stress_fit',['dfine_control/fresh_expansion_completion.json'],False),
    ('dfine_control','stress_matrix',['dfine_control/stress/fit_completion.json'],True),
    ('deepkoopman_control','old_expansion',['deepkoopman_control/stress/completion.json'],True),
    ('dfine_control','old_expansion',['dfine_control/stress/completion.json'],True),
    ('pets','expansion',['pets/deterministic_reference/matrix_completion.json','dfine_control/output_only/matrix_completion.json'],True),
    ('pets','stress',['pets/fresh_expansion_completion.json'],True),
    ('pets','old_expansion',['pets/stress/completion.json'],True),
    ('edmd','diagnostics',['edmd/maps_completion.json'],False),
    ('sindy','diagnostics',['sindy/expansion_completion.json'],False),
    ('sindy','stress',['sindy/expansion_completion.json'],False),
    ('report','diagnostics',['dfine_control/output_only/matrix_completion.json'],False),
    ('figures','diagnostics',['dfine_control/output_only/matrix_completion.json'],False),
]


def write(path,value):
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)


def alive(pid):
    try:
        path=Path(f'/proc/{pid}');state=(path/'stat').read_text().split(') ',1)[1].split()[0]
        if state=='Z':return False
        return b'validation/experiments/literature/job.py' in (path/'cmdline').read_bytes()
    except (OSError,ValueError):return False


def active():
    jobs=[]
    for path in OUT.glob('*/JOB_*.json'):
        try:
            j=json.loads(path.read_text())
            if j.get('state') in ['starting','running'] and alive(j.get('watchdog_pid',0)):jobs.append(j)
        except (OSError,ValueError):continue
    # Each job has one single-thread numerical worker. Source audits count too.
    unique={j['watchdog_pid']:j for j in jobs};return list(unique.values())


def gpu_job(job):
    command=job.get('command',[])
    return job.get('method') in ['pets','long_references'] or job.get('phase') in ['output_pilot','output_matrix','references_pilot','references_matrix','stress_matrix'] or (job.get('phase') in ['expansion','old_expansion'] and job.get('method') in ['deepkoopman_control','dfine_control']) or ('--device' in command and command[command.index('--device')+1]=='cuda')


def main():
    assert set(os.sched_getaffinity(0))<=set(range(80,128))
    spec=importlib.util.spec_from_file_location('original_guard',ROOT/'validation/codex_supervisor.py');guard=importlib.util.module_from_spec(spec);spec.loader.exec_module(guard)
    with open(OUT/'.queue.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        statepath=OUT/'QUEUE.json';state=dict(pid=os.getpid(),state='running',specifications=[dict(method=m,phase=p,requires=r,gpu=g) for m,p,r,g in SPECS],started_at=time.time())
        write(statepath,state)
        while True:
            limits=guard.latest_limits();manifest=json.loads((OUT/'manifest.json').read_text())
            reason=None
            if limits is None or guard.balance(limits) is None:reason='missing usage telemetry'
            elif guard.balance(limits)<manifest['credit_balance_baseline']-1e-6:reason='credit decrease'
            elif guard.window_state(limits.get('primary'))[0]>=85 or guard.window_state(limits.get('secondary'))[0]>=90:reason='usage pause'
            elif (OUT/'STATUS').read_text().splitlines()[0]!='RUNNING':reason='task not RUNNING'
            if reason:state.update(state='paused',reason=reason);write(statepath,state);return
            accounts=[json.loads(p.read_text()) for p in OUT.glob('*/accounting.json')]
            if sum(a['cpu_seconds'] for a in accounts)>=84*3600 or sum(a.get('gpu_seconds',0.) for a in accounts)>=30*3600:
                state.update(state='cap_limited',reason='aggregate allocation reached; no new launches');write(statepath,state);return
            jobs=active();pending=[];launched=False
            for method,phase,requirements,gpu in SPECS:
                directory='smc_C200' if method=='smc' else method
                jobpath=OUT/directory/f'JOB_{phase}.json'
                j=json.loads(jobpath.read_text()) if jobpath.exists() else None
                if j and j.get('state')=='complete':continue
                if j and j.get('state')=='cap_limited':pending.append(dict(method=method,phase=phase,state='closed cap-limited; no automatic retry',job=str(jobpath.relative_to(ROOT))));continue
                if j and j.get('state')=='error':pending.append(dict(method=method,phase=phase,state='needs bounded diagnosis',job=str(jobpath.relative_to(ROOT))));continue
                if j and j.get('state') in ['starting','running'] and alive(j.get('watchdog_pid',0)):continue
                pending.append(dict(method=method,phase=phase,state='pending'))
                if not all((OUT/p).exists() for p in requirements):continue
                if any(j.get('method')==method for j in jobs):continue
                if len(jobs)>=4 or (gpu and any(gpu_job(j) for j in jobs)):continue
                stage=OUT/directory;stage.mkdir(exist_ok=True)
                logfile=stage/'queue_launcher.log'
                command=['taskset','-c','80-127',sys.executable,'-I',str(Path(__file__).with_name('job.py')),method,phase]
                with logfile.open('a') as output:process=subprocess.Popen(command,cwd=ROOT,start_new_session=True,stdin=subprocess.DEVNULL,stdout=output,stderr=subprocess.STDOUT)
                state['last_launch']=dict(method=method,phase=phase,pid=process.pid,command=command,time=time.time());write(statepath,state)
                # Wait for the JOB registration before considering any other launch.
                launched=True;break
            state.update(pending=pending,active_jobs=len(jobs),active_gpu_jobs=sum(gpu_job(j) for j in jobs),updated_at=time.time());write(statepath,state)
            if not launched and not pending and not any(j.get('method') in [m for m,_,_,_ in SPECS] for j in jobs):state.update(state='queue_complete');write(statepath,state);return
            time.sleep(20)


if __name__=='__main__':main()
