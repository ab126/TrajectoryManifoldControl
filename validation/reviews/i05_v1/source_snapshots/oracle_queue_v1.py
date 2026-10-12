"""I05 read-only preflight and one-worker capped supervisor; inert import.

No authority comes from this module. Current proposed metadata cannot execute.
Trusted external decisions, full independent audits and fresh timing are needed.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import oracle_config as oc
import oracle_trace_io_v1 as trace_io

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
ACTIVE={'running','active','starting','auditing'}
CLOSURE={f'validation/{name}' for name in ('oracle_config.py','oracle_math_v1.py','oracle_policy_v1.py',
    'oracle_audit_v1.py','oracle_queue_v1.py','gate_a_v1.py','oracle_trace_io_v1.py','compat.py',
    'oa1_config_proposed.json','oracle_launch_contract_v1.md','oracle_audit_contract_v1.md',
    'trace_contract_v1.md')}|{'src/__init__.py','src/utils.py'}


class LaunchError(ValueError):
    def __init__(self,code,detail,status='fail'):
        self.code,self.status=code,status
        super().__init__(f'{code}: {detail}')


def require(ok,code,detail=''):
    if not ok:raise LaunchError(code,detail)


def finite_number(value,*,positive=False):
    return type(value) in (int,float) and math.isfinite(value) and (value>0 if positive else value>=0)


def relative_file(root,rel):
    oc._path(rel,'path')
    p=root/rel
    require(p.resolve().is_relative_to(root.resolve()),'path_escape',rel)
    return p


def read_json(path,sha=None):
    if not Path(path).is_file():raise LaunchError('unavailable',str(path),'unavailable')
    raw=Path(path).read_bytes()
    if sha is not None:require(hashlib.sha256(raw).hexdigest()==sha,'artifact_hash',str(path))
    return oc._decode(raw)


def usage(root,*,ignore_run=None):
    """One authoritative source per occupancy; never count diagnostic jobs twice."""
    paths=[root/'validation/artifacts/usage.jsonl',root/'validation/diagnostics/usage.jsonl']
    paths+=sorted((root/'validation/overnight_runs').glob('*/run.json'))
    paths+=sorted((root/'validation/oracle_runs').glob('*/run.json'))
    identities=[];total=0.
    for p in paths:
        if ignore_run is not None and p.parent.resolve()==Path(ignore_run).resolve():continue
        require(p.resolve().is_relative_to(root.resolve()),'ledger_path',str(p))
        if not p.is_file():raise LaunchError('ledger_missing',str(p),'unavailable')
        raw=p.read_bytes();identities.append([str(p.relative_to(root)),hashlib.sha256(raw).hexdigest()])
        rows=[oc._decode(line) for line in raw.splitlines() if line.strip()] if p.suffix=='.jsonl' else [oc._decode(raw)]
        for row in rows:
            require(row.get('status') not in ACTIVE,'active_or_stale_job',str(p))
            if p.suffix=='.json':require(row.get('status') in
                ('stopped_awaiting_review','passed_awaiting_review','Gate_A_passed_awaiting_review'),
                'active_or_stale_job',str(p))
            value=row.get('worker_seconds');require(finite_number(value),'ledger_seconds',str(p));total+=value
    for p in (root/'validation/diagnostics').glob('*/job.json'):
        require(read_json(p).get('status')=='finished','active_or_stale_job',str(p))
    return total,hashlib.sha256(oc._canonical(identities)).hexdigest()


def observe_runtime(config):
    """Read-only hardware/settings observation; no CUDA tensor or test solve."""
    import torch
    visible=os.environ.get('CUDA_VISIBLE_DEVICES','0')
    require(',' not in visible and visible not in ('','-1'),'gpu_visibility',visible)
    try:
        line=subprocess.check_output(['nvidia-smi','--id='+visible,
            '--query-gpu=uuid,name,driver_version','--format=csv,noheader'],text=True,timeout=10).strip()
    except (OSError,subprocess.SubprocessError) as error:
        raise LaunchError('runtime_unavailable',str(error),'unavailable') from error
    fields=[v.strip() for v in line.split(',')]
    require(len(fields)==3,'gpu_observation',line)
    return dict(prediction_device='cuda',gpu_uuid=fields[0],hardware=fields[1],
        python_version='.'.join(map(str,sys.version_info[:3])),driver=fields[2],
        torch_version=str(torch.__version__),cuda_runtime=torch.version.cuda,
        oracle_dtype='float64',plant_dtype='float64',local_dtype='float64',
        cpu_threads=torch.get_num_threads(),workers=1,
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        TF32=bool(torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32))


def configure_settings(config):
    """Explicit adopted-worker settings only; no tensor allocation or solve."""
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        os.environ[key]='1'
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8';os.environ['PYTHONDONTWRITEBYTECODE']='1'
    import torch
    torch.set_num_threads(1)
    if torch.get_num_interop_threads()!=1:torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True


@dataclass(frozen=True)
class Preflight:
    report: oc.Report
    config: object
    metadata: object
    root: Path
    request_path: object
    request_sha256: object
    used_seconds: float=0.
    fixture: bool=False

    @property
    def passed(self):return self.report.passed


def preflight(root,config_path,metadata_path,*,request_path=None,request_sha256=None,
              fixture=False,runtime_probe=None,disk_probe=None,head_probe=None,ignore_run=None):
    """No directories/files/jobs created. Injection is explicitly fixture-only."""
    root=Path(root).resolve();config=oc.load_scientific(config_path);metadata=oc.load_release_metadata(metadata_path)
    m=metadata.to_dict();checks=[];used=0.
    def note(status,code,field,observed=None):checks.append(oc.Check(status,code,field,observed=observed))
    try:
        require(fixture or all(p is None for p in (runtime_probe,disk_probe,head_probe)),'fixture_override')
        observed=oc.release_identity(metadata);hashes={}
        # Actual bytes replace every expected file/hash; HEAD never substitutes.
        def refresh(value):
            if isinstance(value,dict):
                if 'path' in value and 'sha256' in value:
                    p=relative_file(root,value['path'])
                    value['sha256']=trace_io.digest(p) if p.is_file() else None
                    if value['sha256'] is not None:hashes[value['path']]=value['sha256']
                elif value and all(type(k) is str and '/' in k and type(v) is str for k,v in value.items()):
                    for rel in value:
                        p=relative_file(root,rel);value[rel]=trace_io.digest(p) if p.is_file() else None
                else:
                    for child in value.values():refresh(child)
        refresh(observed)
        for event in m['events']:
            ref=event['reference'];p=relative_file(root,ref['path'])
            if p.is_file():hashes[ref['path']]=trace_io.digest(p)
        for ref in [m['timing'],m['archive']['manifest'],
                    None if m['allocation'] is None else m['allocation']['reference'],
                    None if m['adoption'] is None else m['adoption']['reference']]:
            if ref is not None:
                p=relative_file(root,ref['path'])
                if p.is_file():hashes[ref['path']]=trace_io.digest(p)
        if m['implementation'] is not None:
            head=(head_probe() if fixture and head_probe else
                  subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip())
            observed['implementation']['commit']=head
            require(CLOSURE.issubset(m['implementation']['files']),'source_closure','I05 dependencies must be pinned')
        if m['runtime'] is not None:
            observed['runtime']=(runtime_probe(config) if fixture and runtime_probe else observe_runtime(config))
        used,ledger_hash=usage(root,ignore_run=ignore_run)
        evidence=dict(observed_release=observed,artifact_sha256=hashes,archive_available=False,
                      used_worker_seconds=used,ledger_sha256=ledger_hash)
        if request_path is None or request_sha256 is None:
            note('unavailable','trusted_request_missing','request')
        else:
            request=read_json(request_path,request_sha256)
            require(set(request)=={'request_version','evidence_kind','release_sha256','scientific_sha256',
                'adoption_sha256','allocation_sha256','task_decision','ledger_sha256','output_parent'},'request_fields')
            require(type(request['request_version']) is int and request['request_version']==1,'request_version')
            require(request['evidence_kind']==('synthetic_fixture' if fixture else 'scientific'),'evidence_kind')
            require(request['release_sha256']==oc.release_hash(metadata) and
                    request['scientific_sha256']==oc.scientific_hash(config),'reviewed_config_hash')
            require(request['ledger_sha256']==ledger_hash,'ledger_changed')
            require(request['output_parent']=='validation/oracle_runs','output_parent')
            evidence.update(trusted_adoption_sha256=request['adoption_sha256'],
                            trusted_allocation_sha256=request['allocation_sha256'])
            task=request['task_decision']
            require(type(task) is dict and set(task)=={'path','sha256'},'task_reference')
            decision=read_json(relative_file(root,task['path']),task['sha256'])
            require(set(decision)=={'decision_version','release_sha256','scientific_sha256','reference_role',
                'endpoint_id','approved_by','allowed_jobs','execution_authorized'},'task_fields')
            require(type(decision['decision_version']) is int and decision['decision_version']==1
                and decision['release_sha256']==request['release_sha256']
                and decision['scientific_sha256']==request['scientific_sha256']
                and type(decision['approved_by']) is str and bool(decision['approved_by'])
                and decision['allowed_jobs']==['oa1_gate_a'] and decision['execution_authorized'] is True,'task_decision')
            require(decision['reference_role']=='separate_strict_reference'
                and decision['endpoint_id']=='historical-strict-reference-v1','new_profile_required',
                'A balancing-aligned task requires a new scientific schema/proposal; never reinterpret OA-1')
            require(m['adoption'] is not None and m['allocation'] is not None,'not_adopted')
            adopt=read_json(relative_file(root,m['adoption']['reference']['path']),request['adoption_sha256'])
            tests=next((e['reference']['sha256'] for e in m['events'] if e['state']=='tested'),None)
            audit=next((e['reference']['sha256'] for e in m['events'] if e['state']=='independently_audited'),None)
            expected_adopt=dict(decision_version=1,release_sha256=request['release_sha256'],
                scientific_sha256=request['scientific_sha256'],allowed_jobs=['oa1_gate_a'],
                approved_by=m['adoption']['approved_by'],tests_sha256=tests,independent_audit_sha256=audit,
                allocation_sha256=request['allocation_sha256'],task_decision_sha256=task['sha256'],
                timing_sha256=None if m['timing'] is None else m['timing']['sha256'],
                archive_manifest_sha256=None if m['archive']['manifest'] is None else m['archive']['manifest']['sha256'],
                execution_authorized=True)
            require(oc._canonical(adopt)==oc._canonical(expected_adopt),'adoption_content')
            allocation=m['allocation'];alloc=read_json(relative_file(root,allocation['reference']['path']),request['allocation_sha256'])
            expected_alloc={k:allocation[k] for k in ['cap_seconds','support_seconds','queue_seconds',
                'fixed_overhead_seconds','archive_seconds','allowed_jobs']}
            expected_alloc.update(allocation_version=1,release_sha256=request['release_sha256'],adopted=True)
            require(oc._canonical(alloc)==oc._canonical(expected_alloc),'allocation_content')
            timing_ref=m['timing'];require(timing_ref is not None,'timing_missing')
            timing=read_json(relative_file(root,timing_ref['path']),timing_ref['sha256'])
            require(set(timing)=={'timing_version','release_sha256','measured','full_step_logging_included',
                'audit_scope','amended_tau_s','startup_seconds','audit_seconds','shutdown_seconds','projected_archive_bytes'},'timing_fields')
            require(type(timing['timing_version']) is int and timing['timing_version']==1 and timing['release_sha256']==request['release_sha256']
                and timing['measured'] is True and timing['full_step_logging_included'] is True
                and timing['audit_scope']=='full_candidate_replay','timing_scope')
            require(finite_number(timing['amended_tau_s'],positive=True)
                and all(finite_number(timing[k]) for k in ['startup_seconds','audit_seconds','shutdown_seconds']), 'timing_values')
            fixed=sum(timing[k] for k in ['startup_seconds','audit_seconds','shutdown_seconds'])
            require(fixed<=allocation['fixed_overhead_seconds'],'audit_overhead_does_not_fit')
            evidence['amended_tau_s']=timing['amended_tau_s']
            archive=m['archive'];ref=archive['manifest'];require(ref is not None,'archive_unbound')
            arch=read_json(relative_file(root,ref['path']),ref['sha256'])
            require(set(arch)=={'archive_version','release_sha256','locator','reserved_bytes','format','chunk_actions','status'},'archive_fields')
            require(type(arch['archive_version']) is int and arch['archive_version']==1 and arch['release_sha256']==request['release_sha256']
                and arch['locator']==archive['locator'] and arch['format']=='npz-tree-chunks-v1'
                and type(arch['chunk_actions']) is int and arch['chunk_actions']==32 and arch['status']=='available','archive_binding')
            require(type(timing['projected_archive_bytes']) is int and timing['projected_archive_bytes']>=11571200000
                and type(arch['reserved_bytes']) is int and arch['reserved_bytes']>=math.ceil(1.2*timing['projected_archive_bytes']), 'archive_reservation')
            parent=relative_file(root,request['output_parent'])
            require(Path(archive['locator']).resolve()==parent.resolve(),'archive_locator')
            ancestor=parent
            while not ancestor.exists():ancestor=ancestor.parent
            free=(disk_probe(ancestor).free if fixture and disk_probe else shutil.disk_usage(ancestor).free)
            require(free>=arch['reserved_bytes'],'archive_capacity')
            evidence['archive_available']=True
        checks.extend(oc.execution_eligibility(config,metadata,evidence).checks)
    except (LaunchError,oc.ConfigError,OSError,subprocess.SubprocessError,StopIteration,TypeError,KeyError,ValueError,OverflowError) as error:
        note(getattr(error,'status','fail'),getattr(error,'code',type(error).__name__),'preflight',str(error))
    return Preflight(oc.Report(tuple(checks)),config,metadata,root,request_path,request_sha256,used,fixture)


class QueueLock:
    def __init__(self,path):self.path=Path(path);self.file=None
    def __enter__(self):
        self.path.parent.mkdir(parents=True,exist_ok=True);self.file=self.path.open('a+b')
        try:fcntl.flock(self.file,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as error:
            self.file.close();raise LaunchError('queue_occupied',str(self.path)) from error
        return self.file.fileno()
    def __exit__(self,*exc):
        fcntl.flock(self.file,fcntl.LOCK_UN);self.file.close()


class Supervisor:
    """Serial gate/audit only; fake process/clock injection requires fixture mode."""
    def __init__(self,*,fixture=False,spawn=None,clock=None,sleep=None,terminate=None,kill=None):
        require(fixture or all(p is None for p in (spawn,clock,sleep,terminate,kill)),'fixture_override')
        self.fixture=fixture;self.spawn=spawn or subprocess.Popen;self.clock=clock or time.monotonic
        self.sleep=sleep or time.sleep;self.terminate=terminate or (lambda p:os.killpg(p.pid,signal.SIGTERM))
        self.kill=kill or (lambda p:os.killpg(p.pid,signal.SIGKILL));self.cancelled=False
    def cancel(self,*args):self.cancelled=True

    def job(self,command,log_path,deadline,lease_fd):
        started=self.clock();returncode=None;process=None;reason=None
        with Path(log_path).open('xb') as log:
            try:
                require(not self.cancelled and started<deadline-2,'job_budget')
                process=self.spawn(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,pass_fds=(lease_fd,))
                while process.poll() is None:
                    if self.cancelled or self.clock()>=deadline-2:
                        reason='interrupted' if self.cancelled else 'worker_cap'
                        self.terminate(process)
                        until=min(deadline,self.clock()+2.)
                        while process.poll() is None and self.clock()<until:self.sleep(min(.05,until-self.clock()))
                        if process.poll() is None:self.kill(process)
                        process.wait();break
                    self.sleep(min(.1,max(0.,deadline-2-self.clock())))
                returncode=process.poll()
            except BaseException:
                if process is not None and process.poll() is None:
                    self.kill(process);process.wait()
                raise
        return dict(returncode=returncode,reason=reason,worker_seconds=self.clock()-started,
                    command=command)

    def run(self,check_fresh,output):
        initial=check_fresh();require(initial.passed,'preflight_rejected')
        require(initial.fixture==self.fixture,'fixture_mismatch')
        root=initial.root;output=Path(output).absolute()
        require(output.parent.resolve()==(root/'validation/oracle_runs').resolve()
            and not output.exists() and not output.is_symlink(),'output_not_fresh')
        with QueueLock(root/'validation/overnight_runs/queue.lock') as lease:
            start=self.clock()
            try:
                fresh=check_fresh();require(fresh.passed,'fresh_preflight_rejected')
                require(oc.release_hash(fresh.metadata)==oc.release_hash(initial.metadata)
                    and fresh.request_sha256==initial.request_sha256,'preflight_changed')
            except (Exception,KeyboardInterrupt) as error:
                output.parent.mkdir(parents=True,exist_ok=True);output.mkdir(exist_ok=False)
                trace_io.atomic_json(output/'run.json',dict(status='stopped_awaiting_review',
                    worker_seconds=self.clock()-start,prior_stage0_seconds=initial.used_seconds,
                    evidence_kind='synthetic_fixture' if self.fixture else 'scientific',
                    reason=str(error),jobs=[],Gate_A_passed=False,downstream_jobs=[]))
                raise
            output.parent.mkdir(parents=True,exist_ok=True);output.mkdir(exist_ok=False)
            allocation=fresh.metadata.to_dict()['allocation'];cap=allocation['queue_seconds']
            state=dict(status='starting',worker_seconds=0.,prior_stage0_seconds=fresh.used_seconds,
                evidence_kind='synthetic_fixture' if self.fixture else 'scientific',supervisor_pid=os.getpid(),
                request_sha256=fresh.request_sha256,release_sha256=oc.release_hash(fresh.metadata),
                scientific_sha256=oc.scientific_hash(fresh.config),jobs=[],Gate_A_passed=False,downstream_jobs=[])
            state.update(queue_deadline_monotonic=start+cap,
                         gate_deadline_monotonic=start+cap-allocation['fixed_overhead_seconds'])
            trace_io.atomic_json(output/'run.json',state)
            try:
                # Snapshot executed Python and its full reviewed closure, not mutable working-tree code.
                m=fresh.metadata.to_dict();paths=set(m['implementation']['files'])|set(m['submitted']['files'])
                paths|={m['specification']['path'],m['derived_gain']['path'],m['trace_contract']['path']}
                for rel in sorted(paths):
                    src=relative_file(root,rel);dest=output/'source_snapshots'/rel
                    expected=(m['implementation']['files'].get(rel) or m['submitted']['files'].get(rel)
                              or next(ref['sha256'] for ref in (m['specification'],m['derived_gain'],m['trace_contract']) if ref['path']==rel))
                    require(trace_io.digest(src)==expected,'source_changed_after_preflight',rel)
                    dest.parent.mkdir(parents=True,exist_ok=True)
                    with dest.open('xb') as f:f.write(src.read_bytes())
                    require(trace_io.digest(dest)==expected,'snapshot_hash',rel)
                config_path=output/'scientific.json';metadata_path=output/'release_metadata.json'
                config_path.write_bytes(oc.canonical_bytes(fresh.config));metadata_path.write_bytes(fresh.metadata._canonical)
                request_path=output/'request.json';request_path.write_bytes(Path(fresh.request_path).read_bytes())
                entry=output/'source_snapshots/validation/gate_a_v1.py'
                base=[sys.executable,str(entry),'--repository-root',str(root),'--config',str(config_path),
                      '--metadata',str(metadata_path),'--request',str(request_path),'--request-sha256',fresh.request_sha256,
                      '--run-dir',str(output),'--lease-fd',str(lease)]
                # Audit gets the measured overhead slot even if gate used its full model allowance.
                gate_deadline=start+cap-allocation['fixed_overhead_seconds']
                state['status']='running';trace_io.atomic_json(output/'run.json',state)
                gate=self.job(base+['--worker','--execute'],output/'gate.log',gate_deadline,lease)
                state['jobs'].append(gate)
                if gate['reason'] or gate['returncode'] not in (0,3):raise LaunchError('gate_incomplete',str(gate))
                state['status']='auditing';trace_io.atomic_json(output/'run.json',state)
                manifest_hash=trace_io.digest(output/'run_manifest.json')
                audit=self.job(base+['--audit','--run-manifest-sha256',manifest_hash],output/'audit.log',start+cap,lease)
                state['jobs'].append(audit)
                require(audit['reason'] is None and audit['returncode']==0,'audit_failed')
                decision=read_json(output/'audit_result.json')
                report=decision['audit'];summary=decision['summary']
                require(set(report)=={'passed','checks'} and report['passed'] is True
                    and type(report['checks']) is list and bool(report['checks'])
                    and all(c['status']=='pass' for c in report['checks'])
                    and decision['run_manifest_sha256']==manifest_hash,'audit_result_invalid')
                require(type(summary['passed']) is bool and summary['passed']==(gate['returncode']==0)
                    and type(summary['cases']) is int and summary['cases']==200
                    and summary['evidence_kind']==state['evidence_kind'],'audit_result_mismatch')
                state['audit_result_sha256']=trace_io.digest(output/'audit_result.json')
                state['Gate_A_passed']=gate['returncode']==0
            except (Exception,KeyboardInterrupt) as error:
                state['reason']=str(error);state['Gate_A_passed']=False
            finally:
                state['worker_seconds']=self.clock()-start
                state['status']='passed_awaiting_review' if state['Gate_A_passed'] else 'stopped_awaiting_review'
                if state['worker_seconds']>cap:
                    state['Gate_A_passed']=False;state['status']='stopped_awaiting_review';state['cap_overrun_seconds']=state['worker_seconds']-cap
                trace_io.atomic_json(output/'run.json',state)
            return state


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repository-root',type=Path,default=ROOT);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--metadata',type=Path,required=True);p.add_argument('--request',type=Path)
    p.add_argument('--request-sha256');p.add_argument('--output-dir',type=Path);p.add_argument('--execute',action='store_true')
    args=p.parse_args(argv)
    def inspect():return preflight(args.repository_root,args.config,args.metadata,
        request_path=args.request,request_sha256=args.request_sha256)
    try:
        if args.execute:
            meta=oc.load_release_metadata(args.metadata)
            require(meta.to_dict()['adoption'] is not None,'not_adopted')
            require(args.request is not None and args.request_sha256 is not None,'trusted_request_missing')
            read_json(args.request,args.request_sha256)
            configure_settings(oc.load_scientific(args.config))
        report=inspect();print(json.dumps(report.report.to_dict(),indent=2,allow_nan=False))
        if not args.execute:return 0 if report.passed else 2
        require(report.passed,'preflight_rejected');require(args.output_dir is not None,'output_required')
        supervisor=Supervisor();signal.signal(signal.SIGTERM,supervisor.cancel);signal.signal(signal.SIGINT,supervisor.cancel)
        state=supervisor.run(inspect,args.output_dir)
        return 0 if state['Gate_A_passed'] else 3
    except (LaunchError,oc.ConfigError,OSError) as error:
        print(str(error),file=sys.stderr);return 2


if __name__=='__main__':raise SystemExit(main())
