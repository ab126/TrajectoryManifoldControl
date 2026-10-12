"""I05 config-selecting reference runner. Default is read-only preflight.

The current OA-1 proposal is unadopted and cannot execute. A revised balancing
endpoint requires a new profile; this entry point never silently selects one.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import signal
import sys
import time

import oracle_config as oc
import oracle_queue_v1 as queue
import oracle_trace_io_v1 as archive


def run_case(config,policy,step,initial,case,writer,*,deadline,stop,clock=time.monotonic,authorization=None):
    """Fixture-testable integration loop; production caller must pass preflight."""
    import numpy as np
    queue.require(policy.planner.fixture or (isinstance(authorization,queue.Preflight)
        and authorization.passed and not authorization.fixture
        and oc.scientific_hash(authorization.config)==oc.scientific_hash(config)),'case_not_authorized')
    x=initial.copy();nodes=[x.copy()];inputs=[];actions=[];status='interrupted'
    try:
        for k in range(config.to_dict()['endpoint']['transitions']):
            if stop() or clock()>=deadline:break
            action=policy.act(k,x)
            if action.value is None:
                writer.append(action.record,None);actions.append(action.record);status='nonfinite_stop';break
            u=float(action.value)
            if not math.isfinite(u) or abs(u)>10:raise queue.LaunchError('invalid_applied_input','no clipping of malformed policy output')
            nxt=step(x.copy(),u)
            if not isinstance(nxt,np.ndarray) or nxt.dtype!=np.float64 or nxt.shape!=(4,):
                raise queue.LaunchError('plant_output','CPU float64 shape-(4,) required')
            writer.append(action.record,nxt.copy());actions.append(action.record)
            inputs.append([u]);nodes.append(nxt.copy());x=nxt.copy()
        else:status='completed'
        # A final nonfinite state fails even if the last applied sample completed.
        if not np.isfinite(x).all():status='nonfinite_stop'
    finally:
        if status!='completed':policy.interrupt()
        checksum=writer.finish(status)
    X=np.array(nodes,dtype=np.float64);U=np.array(inputs,dtype=np.float64).reshape(-1,1)
    tail=X[850:1001]*np.array([1.,math.sqrt(9.81),1.,math.sqrt(9.81)])
    complete=status=='completed' and X.shape==(1001,4) and U.shape==(1000,1)
    finite=bool(np.isfinite(X).all() and np.isfinite(U).all())
    strict=bool(complete and finite and np.all(abs(X[:,0])<=2.4) and np.all(abs(X[:,2])<=math.pi/2)
        and np.all(abs(tail)<=np.array([.1,.05,.02,.05])) and not any(r['exception_latched'] for r in actions))
    return dict(case=case,status=status,strict_reference_success=strict,angle=float(initial[2]),
                index={'path':f'case_{case:03d}/index.json','sha256':checksum},
                evidence_kind='synthetic_fixture' if policy.planner.fixture else 'scientific')


def production_adapter(config):
    """Executed only by the leased, adopted worker; never by preflight/tests."""
    import numpy as np
    import torch
    # Preserve the original uniform-rod evaluation plant and documented workaround.
    source_root=Path(__file__).resolve().parents[1]
    sys.path.insert(0,str(source_root))
    import compat
    workaround=compat.install()
    from src.inverted_pendulum import rk4_step
    h=config.to_dict()['plant']['dt_normalized']
    def cpu(x,u):return rk4_step(x,u,dt=h,M=2.)
    def prediction(x,u):
        def flow(z):
            p,v,theta,w=z.unbind(-1)
            b=.5*theta.cos();r=u[...,0]+.5*theta.sin()*w*w;s=.5*theta.sin();det=1.-b*b
            return torch.stack((v,(r/3-b*s)/det,w,(3*s-b*r)/det),dim=-1)
        a=flow(x);b=flow(x+h*a/2);c=flow(x+h*b/2);d=flow(x+h*c)
        return x+h*(a+2*b+2*c+d)/6
    return cpu,prediction,workaround


def worker(ready,run_dir,stop):
    """One strict-reference development run; called only after leased preflight."""
    queue.require(isinstance(ready,queue.Preflight) and ready.passed and not ready.fixture,'worker_not_authorized')
    import numpy as np
    import oracle_math_v1 as math_impl
    import oracle_policy_v1 as policy_impl
    config=ready.config;m=ready.metadata.to_dict();c=config.to_dict()
    cpu,prediction,workaround=production_adapter(config)
    gain=math_impl.load_gain(config,ready.root/m['derived_gain']['path'],expected_sha256=m['derived_gain']['sha256'])
    planner=policy_impl.CEMPlanner(config,gain,cpu,prediction)
    initial=np.random.default_rng(c['cases']['case_seed']).uniform(-1,1,(200,4))*np.array(c['cases']['half_widths'])
    index=dict(run_version=1,release_sha256=oc.release_hash(ready.metadata),scientific_sha256=oc.scientific_hash(config),
               evidence_kind='scientific',status='running',CVXPY_IMPORT_WORKAROUND=workaround,cases=[])
    archive.atomic_json(run_dir/'run_manifest.json',index)
    deadline=queue.read_json(run_dir/'run.json')['gate_deadline_monotonic']-2.
    for case in range(200):
        if stop() or time.monotonic()>=deadline:break
        policy=policy_impl.OraclePolicy(config,planner,case=case)
        writer=archive.CaseArchive(run_dir/f'case_{case:03d}',config,case,initial[case],evidence_kind='scientific')
        row=run_case(config,policy,cpu,initial[case],case,writer,deadline=deadline,stop=stop,authorization=ready)
        index['cases'].append(row);archive.atomic_json(run_dir/'run_manifest.json',index)
        if row['status']=='interrupted':break
    index['status']='completed' if len(index['cases'])==200 and all(r['status']!='interrupted' for r in index['cases']) else 'interrupted'
    rows=index['cases'];successes=sum(r['strict_reference_success'] for r in rows)
    halves={name:[r for r in rows if (r['angle']<0 if name=='negative' else r['angle']>=0)] for name in ('negative','positive')}
    passed=index['status']=='completed' and successes>=196 and all(
        len(rs)>0 and sum(r['strict_reference_success'] for r in rs)*100>=95*len(rs) for rs in halves.values())
    index['summary']=dict(cases=len(rows),successes=successes,passed=bool(passed),
        halves={k:dict(n=len(rs),successes=sum(r['strict_reference_success'] for r in rs)) for k,rs in halves.items()},
        endpoint='historical-strict-reference-v1',revised_balancing_endpoint='pending')
    archive.atomic_json(run_dir/'run_manifest.json',index)
    return 0 if passed else (3 if index['status']=='completed' else 2)


def audit(ready,run_dir,manifest_sha256):
    import oracle_audit_v1 as audit_impl
    index=queue.read_json(run_dir/'run_manifest.json',manifest_sha256)
    m=ready.metadata.to_dict();config=ready.config
    queue.require(set(index)=={'run_version','release_sha256','scientific_sha256','evidence_kind','status',
        'CVXPY_IMPORT_WORKAROUND','cases','summary'},'run_manifest_fields')
    queue.require(type(index['run_version']) is int and index['run_version']==1
        and index['release_sha256']==oc.release_hash(ready.metadata)
        and index['scientific_sha256']==oc.scientific_hash(config)
        and index['evidence_kind']=='scientific' and index['status']=='completed','run_binding')
    auditor=audit_impl.Auditor(config,ready.root/m['derived_gain']['path'],gain_sha256=m['derived_gain']['sha256'])
    def cases():
        for row in index['cases']:
            queue.require(type(row['case']) is int and 0<=row['case']<200,'case_id')
            expected=f'case_{row["case"]:03d}/index.json'
            queue.require(row['index']['path']==expected,'case_path')
            data=archive.read_case(run_dir/Path(expected).parent,row['index']['sha256'])
            queue.require(data['case']==row['case'] and data['status']==row['status'],'case_identity')
            yield data
    result={}
    def row_check(verdict):
        rows=[row for row in index['cases'] if row['case']==verdict.case]
        queue.require(len(rows)==1,'case_duplicate')
        row=rows[0]
        queue.require(set(row)=={'case','status','strict_reference_success','angle','index','evidence_kind'},'case_row_fields')
        queue.require(type(row['strict_reference_success']) is bool and row['strict_reference_success']==verdict.success
            and type(row['angle']) is float and row['angle']==verdict.angle and row['evidence_kind']==verdict.evidence_kind,
            'case_row_mismatch')
    def replay():result.update(audit_impl.audit_run(auditor,cases(),case_callback=row_check))
    report=audit_impl.check(replay)
    if report.passed:
        claimed=index['summary'];observed={k:result[k] for k in claimed}
        if oc._canonical(claimed)!=oc._canonical(observed):
            report=oc.Report((oc.Check('fail','summary_mismatch','run.summary'),))
    output=dict(audit=report.to_dict(),summary=result if report.passed else None,run_manifest_sha256=manifest_sha256,
        author_independence='same-author independent-code replay; independent-author review pending')
    archive.atomic_json(run_dir/'audit_result.json',output)
    return 0 if report.passed else 2


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repository-root',type=Path,default=queue.ROOT);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--metadata',type=Path,required=True);p.add_argument('--request',type=Path);p.add_argument('--request-sha256')
    p.add_argument('--run-dir',type=Path);p.add_argument('--lease-fd',type=int);p.add_argument('--run-manifest-sha256')
    mode=p.add_mutually_exclusive_group();mode.add_argument('--worker',action='store_true');mode.add_argument('--audit',action='store_true')
    p.add_argument('--execute',action='store_true');args=p.parse_args(argv)
    try:
        if not args.worker and not args.audit:
            queue.require(not args.execute,'leased_worker_required')
            ready=queue.preflight(args.repository_root,args.config,args.metadata,
                request_path=args.request,request_sha256=args.request_sha256)
            print(json.dumps(ready.report.to_dict(),indent=2,allow_nan=False));return 0 if ready.passed else 2
        queue.require(args.run_dir is not None and args.lease_fd is not None,'lease_required')
        run_dir=args.run_dir.resolve();root=args.repository_root.resolve()
        queue.require(run_dir.parent==(root/'validation/oracle_runs').resolve(),'run_path')
        lease=root/'validation/overnight_runs/queue.lock'
        queue.require(os.fstat(args.lease_fd).st_ino==lease.stat().st_ino
            and os.fstat(args.lease_fd).st_dev==lease.stat().st_dev,'lease_identity')
        fcntl=__import__('fcntl');fcntl.flock(args.lease_fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state=queue.read_json(run_dir/'run.json')
        queue.require(state['supervisor_pid']==os.getppid() and state['request_sha256']==args.request_sha256,'supervisor_identity')
        queue.require(args.audit or args.execute,'execute_required')
        queue.configure_settings(oc.load_scientific(args.config))
        ready=queue.preflight(root,args.config,args.metadata,request_path=args.request,
            request_sha256=args.request_sha256,ignore_run=run_dir)
        queue.require(ready.passed,'worker_preflight_rejected')
        if args.audit:
            queue.require(args.run_manifest_sha256 is not None,'manifest_pin_required')
            return audit(ready,run_dir,args.run_manifest_sha256)
        stop_flag=[False]
        def stop(*unused):stop_flag[0]=True
        signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
        return worker(ready,run_dir,lambda:stop_flag[0])
    except (queue.LaunchError,oc.ConfigError,archive.TraceIOError,OSError) as error:
        print(str(error),file=sys.stderr);return 2


if __name__=='__main__':raise SystemExit(main())
