"""I05 fake-process/synthetic archive fixtures only. No scientific subprocess."""
import ast
import contextlib
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import torch

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
import oracle_config as oc
import oracle_queue_v1 as oq
import oracle_trace_io_v1 as tio
import gate_a_v1 as gate
import oracle_math_v1 as om
import oracle_policy_v1 as op
import oracle_audit_v1 as oa

GAIN='a0b7e1c67bc6746e8936cdf67959f76a04a1c2403418492e7c5600cc87f81c58'


class FutureFixture:
    """Imaginary adopted release in /tmp; never writes real approval records."""
    def __init__(self,root):
        self.root=root;self.config=oc.load_scientific(HERE/'reviews/i01_v1/scientific.json')
        self.m=oc.load_release_metadata(HERE/'reviews/i01_v1/release_metadata.json').to_dict()
        files=oq.CLOSURE|set(oc.SOURCE_HASHES)|set(oc.REQUIRED_IMPLEMENTATION_FILES)
        for rel in files:
            p=root/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((HERE.parent/rel).read_bytes())
        for rel,raw in [('validation/artifacts/usage.jsonl',b'{"worker_seconds":4000.0}\n'),
                        ('validation/diagnostics/usage.jsonl',b'{"worker_seconds":987.356492}\n'),
                        ('validation/fixtures/environment.txt',b'synthetic environment'),
                        ('validation/fixtures/native.so',b'synthetic native bytes')]:
            p=root/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
        def copy_ref(rel):
            p=root/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((HERE.parent/rel).read_bytes())
            return {'path':rel,'sha256':tio.digest(p)}
        self.m['implementation']={'commit':'a'*40,'files':{rel:tio.digest(root/rel) for rel in files}}
        self.m['environment']={'manifests':{'validation/fixtures/environment.txt':tio.digest(root/'validation/fixtures/environment.txt')},
            'native_files':{'validation/fixtures/native.so':tio.digest(root/'validation/fixtures/native.so')}}
        self.runtime=dict(prediction_device='cuda',gpu_uuid=self.config.to_dict()['execution']['gpu_uuid'],hardware='NVIDIA L4',
            python_version='3.11.9',driver='synthetic-driver',torch_version='synthetic-torch',cuda_runtime='synthetic-runtime',
            oracle_dtype='float64',plant_dtype='float64',local_dtype='float64',cpu_threads=1,workers=1,deterministic_algorithms=True,TF32=False)
        self.m['runtime']=self.runtime.copy();self.m['specification']=copy_ref('validation/oracle_amendment_proposal_v1.md')
        self.m['derived_gain']={**copy_ref('validation/reviews/i02_v1/gain.json'),'scientific_sha256':oc.scientific_hash(self.config)}
        self.m['trace_contract']=copy_ref('validation/oracle_launch_contract_v1.md')
        release=oc.release_hash(oc.metadata_from_dict(self.m));self.release=release
        self.m['events']=[dict(state=s,release_sha256=release,reference=self.artifact(s+'.json',{'synthetic':s}),
            reviewer='synthetic-independent-reviewer' if s=='independently_audited' else None,
            independent=s=='independently_audited',scope=list(oc.REQUIRED_SCOPES)) for s in oc.STATES[:5]]
        self.timing=dict(timing_version=1,release_sha256=release,measured=True,full_step_logging_included=True,
            audit_scope='full_candidate_replay',amended_tau_s=.2,startup_seconds=50.,audit_seconds=500.,
            shutdown_seconds=50.,projected_archive_bytes=12000000000)
        self.m['timing']={**self.artifact('timing.json',self.timing),'release_sha256':release}
        self.arch=dict(archive_version=1,release_sha256=release,locator=str(root/'validation/oracle_runs'),
            reserved_bytes=14400000000,format='npz-tree-chunks-v1',chunk_actions=32,status='available')
        self.m['archive']=dict(status='available',locator=self.arch['locator'],manifest=self.artifact('archive.json',self.arch))
        self.m['allocation']['status']='adopted'
        alloc={k:self.m['allocation'][k] for k in ['cap_seconds','support_seconds','queue_seconds','fixed_overhead_seconds','archive_seconds','allowed_jobs']}
        alloc.update(allocation_version=1,release_sha256=release,adopted=True)
        self.m['allocation']['reference']=self.artifact('allocation.json',alloc)
        self.task=dict(decision_version=1,release_sha256=release,scientific_sha256=oc.scientific_hash(self.config),
            reference_role='separate_strict_reference',endpoint_id='historical-strict-reference-v1',
            approved_by='synthetic-authority',allowed_jobs=['oa1_gate_a'],execution_authorized=True)
        task=self.artifact('task.json',self.task)
        adoption=dict(decision_version=1,release_sha256=release,scientific_sha256=oc.scientific_hash(self.config),
            allowed_jobs=['oa1_gate_a'],approved_by='synthetic-authority',tests_sha256=self.m['events'][2]['reference']['sha256'],
            independent_audit_sha256=self.m['events'][3]['reference']['sha256'],allocation_sha256=self.m['allocation']['reference']['sha256'],
            task_decision_sha256=task['sha256'],timing_sha256=self.m['timing']['sha256'],
            archive_manifest_sha256=self.m['archive']['manifest']['sha256'],execution_authorized=True)
        self.m['adoption']=dict(reference=self.artifact('adoption.json',adoption),release_sha256=release,
            approved_by='synthetic-authority',allowed_jobs=['oa1_gate_a'])
        self.request=dict(request_version=1,evidence_kind='synthetic_fixture',release_sha256=release,
            scientific_sha256=oc.scientific_hash(self.config),adoption_sha256=self.m['adoption']['reference']['sha256'],
            allocation_sha256=self.m['allocation']['reference']['sha256'],task_decision=task,
            ledger_sha256=oq.usage(root)[1],output_parent='validation/oracle_runs')
        self.config_path=root/'validation/fixtures/scientific.json';self.config_path.write_bytes(oc.canonical_bytes(self.config))
        self.metadata_path=root/'validation/fixtures/metadata.json';self.request_path=root/'validation/fixtures/request.json'
        self.persist()

    def artifact(self,name,value):
        rel='validation/fixtures/'+name;p=self.root/rel;p.parent.mkdir(parents=True,exist_ok=True)
        p.write_bytes(oc._canonical(value));return dict(path=rel,sha256=tio.digest(p))

    def persist(self):
        self.metadata_path.write_bytes(oc.metadata_from_dict(self.m)._canonical)
        self.request_path.write_bytes(oc._canonical(self.request));self.request_sha=tio.digest(self.request_path)

    def rebind_artifact(self,key,value):
        if key=='timing':
            self.m['timing']={**self.artifact('timing.json',value),'release_sha256':self.release}
        elif key=='archive':self.m['archive']['manifest']=self.artifact('archive.json',value)
        elif key=='task':self.request['task_decision']=self.artifact('task.json',value)
        ref=self.m['adoption']['reference'];adopt=json.loads((self.root/ref['path']).read_text())
        adopt.update(timing_sha256=self.m['timing']['sha256'],archive_manifest_sha256=self.m['archive']['manifest']['sha256'],
                     task_decision_sha256=self.request['task_decision']['sha256'])
        self.m['adoption']['reference']=self.artifact('adoption.json',adopt)
        self.request['adoption_sha256']=self.m['adoption']['reference']['sha256'];self.persist()

    def inspect(self,**changes):
        args=dict(request_path=self.request_path,request_sha256=self.request_sha,fixture=True,
                  runtime_probe=lambda c:self.runtime.copy(),head_probe=lambda:'a'*40,
                  disk_probe=lambda p:SimpleNamespace(free=10**12))
        args.update(changes)
        return oq.preflight(self.root,self.config_path,self.metadata_path,**args)


class FakeClock:
    def __init__(self):self.now=0.
    def __call__(self):return self.now
    def sleep(self,s):self.now+=s


class FakeProcess:
    def __init__(self,clock,*,duration=.2,code=0,ignore_term=False):
        self.clock,self.end,self.code=clock,clock()+duration,code;self.stopped=None;self.ignore_term=ignore_term
    def poll(self):return self.stopped if self.stopped is not None else (self.code if self.clock()>=self.end else None)
    def wait(self):return self.poll()
    def terminate(self):
        if not self.ignore_term:self.stopped=-15
    def kill(self):self.stopped=-9


class QueueTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='tmc-queue-fixture-');self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.f=FutureFixture(self.root)

    def test_preflight_is_read_only_and_future_fixture_is_explicit(self):
        before={str(p.relative_to(self.root)):tio.digest(p) for p in self.root.rglob('*') if p.is_file()}
        ready=self.f.inspect()
        self.assertTrue(ready.passed,ready.report.to_dict());self.assertTrue(ready.fixture)
        self.assertEqual(before,{str(p.relative_to(self.root)):tio.digest(p) for p in self.root.rglob('*') if p.is_file()})

    def test_inert_imports_and_production_injection_refusal(self):
        for name in ['gate_a_v1.py','oracle_queue_v1.py','oracle_trace_io_v1.py']:
            tree=ast.parse((HERE/name).read_text())
            eager={a.name for n in tree.body if isinstance(n,ast.Import) for a in n.names}
            self.assertFalse(eager & {'numpy','torch','scipy','src','common','models'})
        self.assertFalse(self.f.inspect(fixture=False).passed)
        with self.assertRaises(oq.LaunchError):oq.Supervisor(spawn=lambda *a,**k:None)

    def test_proposed_release_refuses_before_runtime_and_writes_nothing(self):
        for main in [gate.main,oq.main]:
            with mock.patch.object(oq,'observe_runtime',side_effect=AssertionError('runtime must not be reached')):
                with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
                    args=['--config',str(HERE/'reviews/i01_v1/scientific.json'),'--metadata',str(HERE/'reviews/i01_v1/release_metadata.json')]
                    self.assertEqual(main(args),2);self.assertEqual(main(args+['--execute']),2)

    def test_trusted_request_pin_required_and_wrong_hash_rejected(self):
        self.assertFalse(self.f.inspect(request_path=None,request_sha256=None).passed)
        self.assertFalse(self.f.inspect(request_sha256='0'*64).passed)
        self.f.request['extra']=True;self.f.persist();self.assertFalse(self.f.inspect().passed)

    def test_source_runtime_head_config_and_native_mismatch(self):
        p=self.root/'validation/oracle_policy_v1.py';p.write_text(p.read_text()+'\n# changed\n')
        self.assertFalse(self.f.inspect().passed)
        p.write_bytes((HERE/'oracle_policy_v1.py').read_bytes())
        self.assertFalse(self.f.inspect(head_probe=lambda:'b'*40).passed)
        self.f.runtime['gpu_uuid']='GPU-other';self.assertFalse(self.f.inspect().passed)
        self.f.runtime['gpu_uuid']=self.f.config.to_dict()['execution']['gpu_uuid']
        (self.root/'validation/fixtures/native.so').write_bytes(b'wrong native bytes')
        self.assertFalse(self.f.inspect().passed)

    def test_changed_ledger_missing_ledger_and_stale_active_job_block(self):
        p=self.root/'validation/artifacts/usage.jsonl';p.write_text('{"worker_seconds":4001.0}\n')
        self.assertFalse(self.f.inspect().passed);p.unlink();self.assertFalse(self.f.inspect().passed)
        p.write_text('{"worker_seconds":4000.0}\n')
        q=self.root/'validation/oracle_runs/stale/run.json';q.parent.mkdir(parents=True);q.write_text('{"status":"running","worker_seconds":0.0}')
        self.assertFalse(self.f.inspect().passed)

    def test_current_occupancy_and_missing_artifact_semantics(self):
        self.assertAlmostEqual(self.f.inspect().used_seconds,4987.356492)
        p=self.root/self.f.m['timing']['path'];p.unlink();result=self.f.inspect()
        self.assertFalse(result.passed);self.assertIn('unavailable',[c.status for c in result.report.checks])

    def test_unknown_nonterminal_run_is_stale_and_blocks(self):
        p=self.root/'validation/oracle_runs/stale/run.json';p.parent.mkdir(parents=True)
        p.write_text('{"status":"calibration_finished","worker_seconds":1.0}')
        result=self.f.inspect();self.assertFalse(result.passed)
        self.assertIn('active_or_stale_job',[c.code for c in result.report.checks])

    def test_missing_imported_helper_in_reviewed_closure_is_rejected(self):
        del self.f.m['implementation']['files']['validation/oracle_trace_io_v1.py'];self.f.persist()
        result=self.f.inspect();self.assertFalse(result.passed)
        self.assertIn('source_closure',[c.code for c in result.report.checks])

    def test_unreviewed_balancing_reinterpretation_is_rejected(self):
        task=self.f.task.copy();task['reference_role']='balancing_aligned_reference';task['endpoint_id']='balancing-only-v2'
        self.f.rebind_artifact('task',task);report=self.f.inspect()
        self.assertFalse(report.passed);self.assertIn('new_profile_required',[c.code for c in report.report.checks])

    def test_missing_independent_scope_and_adoption_binding_rejected(self):
        self.f.m['events'][3]['scope']=['configuration'];self.f.persist()
        self.assertFalse(self.f.inspect().passed)
        self.f.m['events'][3]['scope']=list(oc.REQUIRED_SCOPES)
        self.f.m['adoption']['approved_by']='someone-else';self.f.persist();self.assertFalse(self.f.inspect().passed)

    def test_tau_and_unmeasured_or_oversized_audit_reservation_rejected(self):
        for changes in [dict(amended_tau_s=.25001),dict(audit_seconds=1201.),dict(measured=False),dict(audit_scope='selected_only')]:
            timing={**self.f.timing,**changes};self.f.rebind_artifact('timing',timing)
            self.assertFalse(self.f.inspect().passed,changes)

    def test_storage_capacity_and_lower_bound_rejected(self):
        self.assertFalse(self.f.inspect(disk_probe=lambda p:SimpleNamespace(free=100)).passed)
        self.f.rebind_artifact('timing',{**self.f.timing,'projected_archive_bytes':100})
        self.assertFalse(self.f.inspect().passed)

    def supervisor(self,*,gate_code=0,audit_code=0,duration=.2,ignore_term=False,write_audit=True):
        clock=FakeClock();calls=[]
        def spawn(command,**kwargs):
            calls.append((command,kwargs))
            if '--worker' in command:
                run=Path(command[command.index('--run-dir')+1]);(run/'run_manifest.json').write_text('{"synthetic_fixture":true}')
            elif write_audit:
                run=Path(command[command.index('--run-dir')+1])
                decision=dict(audit=dict(passed=audit_code==0,checks=[dict(status='pass' if audit_code==0 else 'fail')]),
                    summary=dict(passed=gate_code==0,cases=200,evidence_kind='synthetic_fixture'),
                    run_manifest_sha256=command[command.index('--run-manifest-sha256')+1])
                (run/'audit_result.json').write_bytes(oc._canonical(decision))
            return FakeProcess(clock,duration=duration,code=gate_code if '--worker' in command else audit_code,ignore_term=ignore_term)
        s=oq.Supervisor(fixture=True,spawn=spawn,clock=clock,sleep=clock.sleep,
            terminate=lambda p:p.terminate(),kill=lambda p:p.kill())
        return s,clock,calls

    def test_exact_two_serial_jobs_source_snapshots_and_no_downstream(self):
        s,clock,calls=self.supervisor();out=self.root/'validation/oracle_runs/new'
        state=s.run(self.f.inspect,out)
        self.assertTrue(state['Gate_A_passed']);self.assertEqual(state['evidence_kind'],'synthetic_fixture')
        self.assertEqual(state['downstream_jobs'],[]);self.assertEqual(len(calls),2)
        self.assertIn('--worker',calls[0][0]);self.assertIn('--audit',calls[1][0]);self.assertIn('--run-manifest-sha256',calls[1][0])
        for rel in oq.CLOSURE:self.assertEqual((out/'source_snapshots'/rel).read_bytes(),(self.root/rel).read_bytes())
        self.assertGreater(state['worker_seconds'],0.)

    def test_occupied_lock_prevents_any_spawn(self):
        s,clock,calls=self.supervisor()
        with oq.QueueLock(self.root/'validation/overnight_runs/queue.lock'):
            with self.assertRaises(oq.LaunchError):s.run(self.f.inspect,self.root/'validation/oracle_runs/new')
        self.assertEqual(calls,[])

    def test_existing_output_refused_without_overwrite(self):
        out=self.root/'validation/oracle_runs/existing';out.mkdir(parents=True);(out/'sentinel').write_text('keep')
        s,clock,calls=self.supervisor()
        with self.assertRaises(oq.LaunchError):s.run(self.f.inspect,out)
        self.assertEqual(calls,[]);self.assertEqual((out/'sentinel').read_text(),'keep')

    def test_fresh_under_lock_recheck_blocks_new_ledger_change(self):
        n=[0]
        def fresh():
            n[0]+=1
            if n[0]==2:(self.root/'validation/artifacts/usage.jsonl').write_text('{"worker_seconds":4001.0}\n')
            return self.f.inspect()
        s,clock,calls=self.supervisor()
        with self.assertRaises(oq.LaunchError):s.run(fresh,self.root/'validation/oracle_runs/new')
        self.assertEqual(calls,[])

    def test_audit_failure_stops_without_promoting_gate_pass(self):
        s,clock,calls=self.supervisor(audit_code=2);state=s.run(self.f.inspect,self.root/'validation/oracle_runs/new')
        self.assertFalse(state['Gate_A_passed']);self.assertEqual(state['status'],'stopped_awaiting_review');self.assertEqual(len(calls),2)

    def test_zero_audit_exit_without_bound_decision_cannot_pass(self):
        s,clock,calls=self.supervisor(write_audit=False)
        state=s.run(self.f.inspect,self.root/'validation/oracle_runs/new')
        self.assertFalse(state['Gate_A_passed']);self.assertEqual(len(calls),2)

    def test_final_preflight_recheck_and_failed_recheck_are_charged(self):
        s,clock,calls=self.supervisor();n=[0]
        def inspect():
            n[0]+=1
            if n[0]==2:clock.sleep(7.)
            return self.f.inspect()
        state=s.run(inspect,self.root/'validation/oracle_runs/new')
        self.assertTrue(state['Gate_A_passed']);self.assertGreaterEqual(state['worker_seconds'],7.)

    def test_failed_gate_is_audited_but_incomplete_gate_stops(self):
        s,clock,calls=self.supervisor(gate_code=3);state=s.run(self.f.inspect,self.root/'validation/oracle_runs/failed')
        self.assertFalse(state['Gate_A_passed']);self.assertEqual(len(calls),2)
        # Another invocation needs a freshly reviewed usage hash.
        self.f.request['ledger_sha256']=oq.usage(self.root)[1];self.f.persist()
        s,clock,calls=self.supervisor(gate_code=2);state=s.run(self.f.inspect,self.root/'validation/oracle_runs/incomplete')
        self.assertFalse(state['Gate_A_passed']);self.assertEqual(len(calls),1)

    def test_cap_terminates_then_kills_and_charges_actual_occupancy(self):
        clock=FakeClock();process=FakeProcess(clock,duration=100.,ignore_term=True)
        s=oq.Supervisor(fixture=True,spawn=lambda *a,**k:process,clock=clock,sleep=clock.sleep,
            terminate=lambda p:p.terminate(),kill=lambda p:p.kill())
        result=s.job(['synthetic-command'],self.root/'job.log',5.,1)
        self.assertEqual(result['reason'],'worker_cap');self.assertEqual(result['returncode'],-9)
        self.assertAlmostEqual(result['worker_seconds'],5.)

    def test_interruption_cancels_without_second_job(self):
        s,clock,calls=self.supervisor(duration=100.)
        def sleep(seconds):
            clock.sleep(seconds)
            if clock()>=.3:s.cancel()
        s.sleep=sleep
        state=s.run(self.f.inspect,self.root/'validation/oracle_runs/new')
        self.assertFalse(state['Gate_A_passed']);self.assertEqual(len(calls),1);self.assertGreater(state['worker_seconds'],0.)


class ArchiveRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1);cls.config=oc.load_scientific(HERE/'reviews/i01_v1/scientific.json')
        cls.gain=om.load_gain(cls.config,HERE/'reviews/i02_v1/gain.json',expected_sha256=GAIN)

    def test_lossless_nonfinite_arrays_tuple_types_and_no_pickle(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'tree.npz';value={'tuple':(False,None,1,1.),'f64':np.array([np.nan,np.inf,-0.]),
                'f32':np.ones((2,3),dtype=np.float32),'int':np.arange(4,dtype=np.int64),'bool':np.array([True,False])}
            checksum=tio.write_tree(p,value);decoded=tio.read_tree(p,checksum)
            self.assertEqual(decoded['tuple'],value['tuple']);self.assertIsInstance(decoded['tuple'],tuple)
            for key in ['f64','f32','int','bool']:
                np.testing.assert_array_equal(decoded[key],value[key]);self.assertEqual(decoded[key].dtype,value[key].dtype)
            with self.assertRaises(FileExistsError):tio.write_tree(p,value)
            with self.assertRaises(tio.TraceIOError):tio.read_tree(p,'0'*64)

    def test_missing_raw_unavailable_object_and_json_nonfinite_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'missing.npz'
            report=oa.check(lambda:tio.read_tree(p,'0'*64));self.assertEqual(report.checks[0].status,'unavailable')
            with self.assertRaises(tio.TraceIOError):tio.write_tree(p,np.array([object()],dtype=object))
            with self.assertRaises(ValueError):tio.write_tree(p,{'bad':float('nan')})

    def fixture_run(self,tmp,stop=lambda:False,step=lambda x,u:x.copy()):
        planner=op.CEMPlanner.for_fixture(self.config,self.gain,lambda x,u:x.copy(),lambda x,u:x.clone())
        policy=op.OraclePolicy(self.config,planner,case=0)
        writer=tio.CaseArchive(Path(tmp)/'case',self.config,0,np.zeros(4),evidence_kind='synthetic_fixture')
        row=gate.run_case(self.config,policy,step,np.zeros(4),0,writer,deadline=float('inf'),stop=stop)
        return row,tio.read_case(Path(tmp)/'case',row['index']['sha256'])

    def test_full_synthetic_runner_archive_and_independent_replay(self):
        with tempfile.TemporaryDirectory() as tmp:
            row,data=self.fixture_run(tmp)
            self.assertTrue(row['strict_reference_success']);self.assertEqual(data['X'].shape,(1001,4));self.assertEqual(data['U'].shape,(1000,1))
            audit=oa.Auditor(self.config,HERE/'reviews/i02_v1/gain.json',gain_sha256=GAIN,fixture=True,
                step_cpu=lambda x,u:x.copy(),step_prediction=lambda x,u:x.clone())
            self.assertTrue(oa.check(lambda:audit.case(data,np.zeros(4))).passed)

    def test_interruption_flushes_exact_prefix_without_padding(self):
        n=[0]
        def stop():n[0]+=1;return n[0]>5
        with tempfile.TemporaryDirectory() as tmp:
            row,data=self.fixture_run(tmp,stop=stop)
            self.assertEqual(data['status'],'interrupted');self.assertEqual(data['X'].shape,(6,4));self.assertEqual(data['U'].shape,(5,1))
            self.assertFalse(row['strict_reference_success'])

    def test_nonfinite_state_preserved_and_case_stops(self):
        with tempfile.TemporaryDirectory() as tmp:
            row,data=self.fixture_run(tmp,step=lambda x,u:np.full(4,np.nan))
            self.assertEqual(data['status'],'nonfinite_stop');self.assertEqual(len(data['U']),1);self.assertEqual(len(data['actions']),2)
            self.assertFalse(row['strict_reference_success']);self.assertTrue(np.isnan(data['X'][-1]).all())

    def test_archive_traversal_and_float32_state_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            n=[0]
            def stop():n[0]+=1;return n[0]>1
            row,data=self.fixture_run(tmp,stop=stop);p=Path(tmp)/'case/index.json';index=json.loads(p.read_text())
            index['chunks'][0]['path']='../outside.npz';p.write_bytes(oc._canonical(index))
            with self.assertRaises(tio.TraceIOError):tio.read_case(p.parent,tio.digest(p))
        with tempfile.TemporaryDirectory() as tmp:
            writer=tio.CaseArchive(Path(tmp)/'case',self.config,0,np.zeros(4,dtype=np.float32),evidence_kind='synthetic_fixture')
            checksum=writer.finish('interrupted')
            with self.assertRaises(tio.TraceIOError):tio.read_case(Path(tmp)/'case',checksum)

    def test_production_case_and_worker_require_authorization(self):
        policy=SimpleNamespace(planner=SimpleNamespace(fixture=False))
        with self.assertRaises(oq.LaunchError):
            gate.run_case(self.config,policy,lambda x,u:x,np.zeros(4),0,None,deadline=1,stop=lambda:False)
        with self.assertRaises(oq.LaunchError):gate.worker(None,None,None)


if __name__=='__main__':unittest.main()
