"""Serial, capped Gate A queue. --execute plus explicit stage allocation required.

Only Gate A calibration is experimental. Postprocessing replays saved traces;
Stage 1, tuning, confirmation, box fallback and transfer are never launched.
"""
import datetime
import fcntl
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
import argparse

HERE=Path(__file__).resolve().parent
OUT=HERE/'artifacts'
ROOT=HERE.parent
STOP=False

def atomic(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)

def digest(path):
    value=hashlib.sha256()
    with path.open('rb') as file:
        for block in iter(lambda:file.read(1024*1024),b''):value.update(block)
    return value.hexdigest()

def prior_seconds():
    result=sum(json.loads(line)['worker_seconds'] for line in (OUT/'usage.jsonl').read_text().splitlines())
    diagnostic_usage=HERE/'diagnostics/usage.jsonl'
    if (HERE/'diagnostics/active.json').exists():raise RuntimeError('Unclosed diagnostic occupancy; review before calibration')
    if diagnostic_usage.exists():result+=sum(json.loads(line)['worker_seconds'] for line in diagnostic_usage.read_text().splitlines())
    for path in (HERE/'overnight_runs').glob('*/run.json'):
        row=json.loads(path.read_text())
        if row.get('status')=='running':raise RuntimeError(f'Unclosed previous run: {path}; review occupancy before relaunch')
        result+=row.get('worker_seconds',0.)
    for directory in ('gate_a_artifacts','gate_a_artifacts_cuda'):
        # Legacy, manually launched calibration is outside overnight_runs.
        path=HERE/directory/'summary.json'
        if path.exists():result+=json.loads(path.read_text()).get('worker_seconds',0.)
    return result

def verify_sources():
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    expected_head=json.loads((OUT/'config_manifest.json').read_text())['HEAD']
    if head!=expected_head:raise RuntimeError('HEAD differs from reviewed readiness')
    if subprocess.run(['git','diff','--quiet','HEAD'],cwd=ROOT).returncode:raise RuntimeError('Tracked working-tree changes since reviewed readiness')
    initial=json.loads((OUT/'source_hashes_initial.json').read_text())
    amendments=json.loads((OUT/'approved_plan_amendments.json').read_text())
    hashes={}
    for name,old in initial.items():
        actual=digest(ROOT.parent/name)
        expected=amendments.get('changes',{}).get(name,{}).get('after_SHA256',old)
        if actual!=expected:raise RuntimeError(f'Source changed since reviewed readiness: {name}')
        hashes[name]=actual
    for name in ('plant_checks.json','equivalence.json','equivalence_cuda.json','gpu_probe.json'):
        if not json.loads((OUT/name).read_text())['passed']:raise RuntimeError(f'Correctness prerequisite failed: {name}')
    return hashes

def run_job(name,command,slot,run_dir,deadline):
    remaining=deadline-time.perf_counter()
    if remaining<min(slot,20):return {'job':name,'returncode':124,'skipped':'queue allocation exhausted','wall_s':0.}
    allowance=min(slot,remaining-2)
    if allowance<=0:return {'job':name,'returncode':124,'skipped':'queue allocation exhausted','wall_s':0.}
    started=time.perf_counter();exit_code=None;timed_out=False;last_heartbeat=0.
    print(f'Starting {name}; maximum {allowance/3600:.3f} h',flush=True)
    with (run_dir/f'{name}.log').open('w') as log:
        process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=os.environ.copy(),start_new_session=True)
        try:
            while process.poll() is None:
                elapsed=time.perf_counter()-started
                if STOP or elapsed>=allowance-30:
                    timed_out=not STOP
                    os.killpg(process.pid,signal.SIGTERM)
                    try:process.wait(timeout=max(.1,min(30,allowance-elapsed)))
                    except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
                    break
                if elapsed-last_heartbeat>=30 or last_heartbeat==0:
                    atomic(run_dir/'heartbeat.json',{'job':name,'elapsed_s':elapsed,'remaining_slot_s':allowance-elapsed,
                                                      'updated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
                    last_heartbeat=elapsed+.001
                time.sleep(.5)
            exit_code=process.returncode
        finally:
            if process.poll() is None:
                os.killpg(process.pid,signal.SIGTERM)
                try:process.wait(timeout=5)
                except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
    row={'job':name,'argv':command,'returncode':exit_code,'wall_s':time.perf_counter()-started,
         'timeout':timed_out,'cancelled':STOP,'log':str(run_dir/f'{name}.log')}
    print(f'Finished {name}: code={exit_code}, wall={row["wall_s"]:.1f}s',flush=True)
    return row

def review(run_dir,state):
    calibration=run_dir/'calibration';audit_path=calibration/'independent_audit.json'
    audit=json.loads(audit_path.read_text()) if audit_path.exists() else {}
    summary_path=calibration/'summary.json'
    summary=json.loads(summary_path.read_text()) if summary_path.exists() else {}
    calibration_job=next((job for job in state['jobs'] if job['job']=='gate_a'),{})
    passed=bool(summary.get('passed') and audit.get('Gate_A_passed') and audit.get('audit_passed')
                and calibration_job.get('returncode')==0 and not calibration_job.get('timeout') and not calibration_job.get('cancelled'))
    text=f'''# Overnight morning review

Status: {state['status']}. Gate A passed: {passed}.

- Completed cases: {summary.get('cases',0)}/200.
- Physical success rate among completed cases: {audit.get('success_rate')} (descriptive only if incomplete).
- Angle-sign half rates: {audit.get('angle_half_rates')}.
- Independent endpoint/plant replay audit: {audit.get('audit_passed')}.
- Incomplete: {summary.get('incomplete',True)}.
- Failure categories: {audit.get('failure_counts',{})}.
- Replay maximum absolute error: {audit.get('plant_replay_max_abs_error')}.
- Overnight charged worker-hours: {state['worker_seconds']/3600:.5f}.
- Prior Stage-0 worker-hours: {state['prior_stage0_seconds']/3600:.5f}.
- Total Stage-0 worker-hours: {(state['worker_seconds']+state['prior_stage0_seconds'])/3600:.5f}.
- Explicit Stage-0 allocation: {state['stage0_cap_hours']} h.

Only development oracle calibration was run. No Stage 1, tuning, confirmation,
transfer, extra starts, or automatic box reduction. Timing is exempt for Gate A.
If incomplete or failed, review the cause before any task/optimizer amendment.
If passed, freeze the box and review the still-unresolved Stage-1 recipe,
deadline endpoint and worst-case allocation before authorizing dependent work.

Logs and traces are local to this directory. `run.json` has commands, budgets,
exit codes, wall times and hashes. `calibration/independent_audit.json` has per-case
diagnostics; `calibration/progress.json` records completed-case checkpoints.
'''
    (run_dir/'morning_review.md').write_text(text)
    files={}
    for path in run_dir.rglob('*'):
        if path.is_file() and path.name not in ('checksums.json','run.json','heartbeat.json'):
            files[str(path.relative_to(run_dir))]=digest(path)
    atomic(run_dir/'checksums.json',files)
    return passed

def main(args):
    hashes=verify_sources();prior=prior_seconds()
    tau=json.loads((OUT/'benchmark_oracle_cuda_1.json').read_text())['single_start']['mean_s']
    reserve=200*1000*tau/3600*1.2
    # Slots include process imports, shutdown/checkpoint grace, and postprocessing.
    slots={'runtime_check':90,'gate_a_preflight':60,'gate_a':14.42*3600,'audit':120,'review':60}
    proposed={'execute':args.execute,'stage0_cap_hours':args.stage0_cap_hours,'prior_stage0_hours':prior/3600,
        'available_worker_hours':args.stage0_cap_hours-prior/3600,'oracle_cases':200,'oracle_starts':1,
        'box_scale':1.,'calibration_internal_cap_hours':14.4,'reservation_with_20pct_overhead_hours':reserve,
        'queue_reserved_hours':sum(slots.values())/3600,'queue_jobs':list(slots),
        'fits':reserve<=14.4 and sum(slots.values())+prior<=args.stage0_cap_hours*3600,
        'scope':'Gate A only + diagnostics; no downstream experimental jobs'}
    print(json.dumps(proposed,indent=2),flush=True)
    if not args.execute:return 0 if proposed['fits'] else 2
    if not proposed['fits']:raise RuntimeError('Queue does not fit explicit allocation; no calibration launched')
    (HERE/'overnight_runs').mkdir(exist_ok=True)
    lock=(HERE/'overnight_runs/queue.lock').open('a')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise RuntimeError('Another overnight queue is active; no duplicate calibration launched')
    run_dir=HERE/'overnight_runs'/datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    run_dir.mkdir()
    source_snapshot={name:digest(path) for path in HERE.iterdir() if path.is_file() and path.suffix in ('.py','.sh','.md','.lock') for name in [path.name]}
    state={'status':'running','stage0_cap_hours':args.stage0_cap_hours,'prior_stage0_seconds':prior,
           'worker_seconds':0.,'proposal':proposed,'submitted_source_hashes':hashes,
           'validation_source_hashes':source_snapshot,'GPU_environment_hashes':json.loads((OUT/'gpu_native_fingerprints.json').read_text()),
           'allocation_authorization':'explicit user invocation with --execute and --stage0-cap-hours',
           'jobs':[],'run_directory':str(run_dir),'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    atomic(run_dir/'run.json',state);atomic(HERE/'overnight_runs/latest.json',{'run_directory':str(run_dir)})
    start=time.perf_counter();deadline=start+args.stage0_cap_hours*3600-prior
    calibration=run_dir/'calibration'
    common=[sys.executable,str(HERE/'gate_a.py'),'--device','cuda','--cases','200','--starts','1',
            '--box-scale','1','--max-worker-hours','14.4','--output-dir',str(calibration)]
    try:
        check=run_job('runtime_check',[sys.executable,str(HERE/'overnight_runtime.py'),str(run_dir/'runtime_check.json')],slots['runtime_check'],run_dir,deadline)
        state['jobs'].append(check)
        if check['returncode']!=0:raise RuntimeError('Runtime check failed; no calibration launched')
        preflight=run_job('gate_a_preflight',common,slots['gate_a_preflight'],run_dir,deadline)
        state['jobs'].append(preflight)
        if preflight['returncode']!=0:raise RuntimeError('Preflight failed; no calibration launched')
        job=run_job('gate_a',common+['--execute'],slots['gate_a'],run_dir,deadline)
        state['jobs'].append(job)
        state['status']='calibration_finished' if job['returncode'] in (0,3) else 'calibration_interrupted_or_failed'
        # Always audit existing output, including failure, without another experiment.
        if calibration.exists() and not STOP:
            state['jobs'].append(run_job('audit',[sys.executable,str(HERE/'overnight_audit.py'),str(calibration)],slots['audit'],run_dir,deadline))
    except Exception as error:
        state['status']='stopped';state['reason']=str(error)
        print(f'Stopped: {error}',flush=True)
    finally:
        state['worker_seconds']=time.perf_counter()-start
        try:passed=review(run_dir,state)
        except Exception as error:
            passed=False;state['postprocessing_error']=str(error)
            (run_dir/'morning_review.md').write_text(f'Gate A is not cleared. Postprocessing failed: {error}. Inspect run.json and job logs; no downstream work ran.\n')
        state['Gate_A_passed']=passed
        state['status']='Gate_A_passed_awaiting_review' if passed else 'stopped_awaiting_review'
        state['worker_seconds']=time.perf_counter()-start
        state['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
        atomic(run_dir/'run.json',state)
        print(f'Morning review: {run_dir}/morning_review.md',flush=True)
        print('No downstream experiment launched. Review artifacts in the morning.',flush=True)
        fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
    return 0 if state.get('Gate_A_passed') else 3

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--execute',action='store_true')
    parser.add_argument('--stage0-cap-hours',type=float,default=8.)
    args=parser.parse_args()
    if not 0<args.stage0_cap_hours<=15:parser.error('Explicit Stage-0 allocation must be >0 and <=15 hours')
    def stop(_signum,_frame):
        global STOP
        STOP=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    raise SystemExit(main(args))
