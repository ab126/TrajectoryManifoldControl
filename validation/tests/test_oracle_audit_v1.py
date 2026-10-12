"""I04 adversarial engineering fixtures. No original plant/CUDA campaign."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
import oracle_config as oc
import oracle_math_v1 as om
import oracle_policy_v1 as op
import oracle_audit_v1 as oa

GAIN='a0b7e1c67bc6746e8936cdf67959f76a04a1c2403418492e7c5600cc87f81c58'


def zero_noise(config,case,sample):
    noise=torch.zeros((3,256,60,1),dtype=torch.float32)
    return noise,dict(key=['oracle_v1',case,0],seed=op.noise_seed(config,case,sample),
                     draw_sha256=hashlib.sha256(noise.numpy().tobytes()).hexdigest())


class AuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.config=oc.load_scientific(HERE/'reviews/i01_v1/scientific.json')
        cls.gain=om.load_gain(cls.config,HERE/'reviews/i02_v1/gain.json',expected_sha256=GAIN)
        cls.x=np.array([1.,0.,0.,0.]);cls.warm=np.ones((60,1))
        cls.trace=cls.planner().plan(cls.x,case=7,sample=0,warm=cls.warm).trace
        policy=op.OraclePolicy(cls.config,cls.planner(),case=7)
        cls.records=[policy.act(k,cls.x).record for k in range(7)]
        policy=op.OraclePolicy(cls.config,cls.planner(),case=0)
        cls.zero_case=dict(case=0,scientific_sha256=oc.scientific_hash(cls.config),evidence_kind='synthetic_fixture',
            status='completed',X=np.zeros((1001,4)),U=np.zeros((1000,1)),
            actions=[policy.act(k,np.zeros(4)).record for k in range(1000)])

    @classmethod
    def planner(cls,step=lambda z,u:z.copy(),pred=lambda z,u:z.clone()):
        return op.CEMPlanner.for_fixture(cls.config,cls.gain,step,pred,noise_provider=zero_noise)

    def auditor(self,step=lambda z,u:z.copy(),pred=lambda z,u:z.clone()):
        return oa.Auditor(self.config,HERE/'reviews/i02_v1/gain.json',gain_sha256=GAIN,fixture=True,
            step_cpu=step,step_prediction=pred,noise_provider=lambda c,s:torch.zeros((3,256,60,1),dtype=torch.float32))

    def test_import_closure_excludes_scoring_policy_sut_and_launcher(self):
        tree=ast.parse((HERE/'oracle_audit_v1.py').read_text())
        imports={n.module.split('.')[0] for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)}
        imports|={a.name.split('.')[0] for n in ast.walk(tree) if isinstance(n,ast.Import) for a in n.names}
        self.assertFalse(imports & {'oracle_math_v1','oracle_policy_v1','src','models','common','subprocess'})
        eager={a.name for n in tree.body if isinstance(n,ast.Import) for a in n.names}
        self.assertFalse(eager & {'numpy','scipy','torch'})

    def test_gain_independent_augmented_polynomial_and_dare(self):
        auditor=self.auditor()
        np.testing.assert_allclose(auditor.K,np.array(self.gain.to_dict()['K']),atol=1e-10,rtol=1e-10)
        bad=self.gain.to_dict();bad['K'][0][1]+=.001
        self.assertFalse(oa.check(lambda:oa.verify_gain(self.config,bad)).passed)

    def test_missing_artifact_unavailable_corrupt_artifact_failed(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'missing'
            report=oa.check(lambda:oa.read_pinned(p,GAIN))
            self.assertEqual(report.checks[0].status,'unavailable')
            p.write_bytes(b'corrupt')
            self.assertEqual(oa.check(lambda:oa.read_pinned(p,GAIN)).checks[0].status,'fail')

    def test_production_refuses_synthetic_override_without_cuda(self):
        with self.assertRaises(oa.AuditIssue):
            oa.Auditor(self.config,HERE/'reviews/i02_v1/gain.json',gain_sha256=GAIN,step_cpu=lambda x,u:x)

    def test_hand_cost_safety_rank_and_invalid_exclusion(self):
        X=np.zeros((4,61,4));U=np.zeros((4,60,1));X[0,:,0]=1.;X[1,:,0]=3.;U[2]=11;X[3,4,2]=np.nan
        row=oa.scores(self.config,X,U)
        self.assertEqual(row['J'][0],144.)
        self.assertEqual(oa.rank(row),[0,1,2,3])
        self.assertEqual(row['eligible'].tolist(),[True,True,False,False])
        self.assertEqual(row['feasible'].tolist(),[True,False,False,False])

    def test_all_candidates_slots_moments_and_incumbent_replay(self):
        self.assertTrue(oa.check(lambda:self.auditor().plan(self.trace,self.x,self.warm,7,0)).passed)

    def test_actual_indexed_random_draw_replay(self):
        planner=op.CEMPlanner.for_fixture(self.config,self.gain,lambda z,u:z.copy(),lambda z,u:z.clone())
        trace=planner.plan(self.x,case=7,sample=0,warm=self.warm).trace
        audit=oa.Auditor(self.config,HERE/'reviews/i02_v1/gain.json',gain_sha256=GAIN,fixture=True,
                         step_cpu=lambda z,u:z.copy(),step_prediction=lambda z,u:z.clone())
        self.assertTrue(oa.check(lambda:audit.plan(trace,self.x,self.warm,7,0)).passed)

    def test_mutations_seed_hash_cost_rank_input_and_moments_rejected(self):
        mutations=[lambda d:d.update(seed=d['seed']+1),lambda d:d.update(draw_sha256='0'*64),
            lambda d:d['iterations'][0]['J'].__setitem__(0,d['iterations'][0]['J'][0]+1),
            lambda d:d['iterations'][1]['first_seen'].__setitem__(3,0),
            lambda d:d['selected']['controls'].__setitem__((0,0),11),
            lambda d:d['iterations'][2]['std_after'].__setitem__((0,0),4),
            lambda d:d['iterations'][0]['feasible'].__setitem__(0,False),
            lambda d:d.update(extra_unreviewed_rule=True)]
        audit=self.auditor()
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                d=copy.deepcopy(self.trace);mutate(d)
                self.assertFalse(oa.check(lambda:audit.plan(d,self.x,self.warm,7,0)).passed)

    def test_queue_history_actual_shift_and_actions_replay(self):
        self.assertTrue(oa.check(lambda:self.auditor().actions(self.records)).passed)
        for field,value,index in [('warm_shift',4,5),('previous_plan_sample',1,2),('planning_calls',2,0),
                                  ('queue_after',5,0),('branch','local_enter',2),('safety_latched',True,6)]:
            d=copy.deepcopy(self.records);d[index][field]=value
            self.assertFalse(oa.check(lambda:self.auditor().actions(d)).passed,field)

    def test_queue_guard_failure_discards_both_memories(self):
        p=op.OraclePolicy(self.config,self.planner(),case=7)
        records=[p.act(0,self.x).record,p.act(1,self.x+np.array([.1,0,0,0])).record]
        self.assertTrue(oa.check(lambda:self.auditor().actions(records)).passed)
        records[1]['planning']['warm_controls'][0,0]=1.
        self.assertFalse(oa.check(lambda:self.auditor().actions(records)).passed)

    def test_local_entry_stay_exit_and_saturation_predicates_replay(self):
        step=lambda z,u:.5*z;pred=lambda z,u:.5*z
        p=op.OraclePolicy(self.config,self.planner(step,pred),case=7)
        states=[self.x,np.array([.1,0,0,0]),np.array([.05,0,0,0]),self.x]
        records=[p.act(k,z).record for k,z in enumerate(states)]
        audit=self.auditor(step,pred)
        self.assertTrue(oa.check(lambda:audit.actions(records)).passed)
        records[1]['local_guard']['transitions'][0]['rhs']+=1
        self.assertFalse(oa.check(lambda:audit.actions(records)).passed)

    def test_insufficient_population_partial_trace_and_exception_latch(self):
        def bad(z,u):
            nxt=z.clone();nxt[31:]=float('nan');return nxt
        p=op.OraclePolicy(self.config,self.planner(pred=bad),case=7)
        records=[p.act(0,self.x).record,p.act(1,np.zeros(4)).record]
        audit=self.auditor(pred=bad)
        self.assertTrue(oa.check(lambda:audit.actions(records)).passed)
        records[1]['exception_latched']=False
        self.assertFalse(oa.check(lambda:audit.actions(records)).passed)

    def test_no_feasible_zero_is_distinct_from_exception(self):
        p=op.OraclePolicy(self.config,self.planner(),case=7)
        x=np.array([3.,0.,0.,0.]);records=[p.act(0,x).record]
        self.assertEqual(records[0]['branch'],'no_feasible_plan')
        self.assertTrue(oa.check(lambda:self.auditor().actions(records)).passed)
        records[0]['zero_prediction']['safe']=True
        self.assertFalse(oa.check(lambda:self.auditor().actions(records)).passed)

    def test_nonfinite_stop_and_no_continuation(self):
        p=op.OraclePolicy(self.config,self.planner(),case=7)
        records=[p.act(0,np.array([np.nan,0,0,0])).record]
        self.assertTrue(oa.check(lambda:self.auditor().actions(records)).passed)
        records.append(p.act(1,np.zeros(4)).record)
        self.assertFalse(oa.check(lambda:self.auditor().actions(records)).passed)

    def test_unreproducible_exception_is_unavailable_not_certified(self):
        d=copy.deepcopy(self.records[:1]);d[0]['planning']=None
        report=oa.check(lambda:self.auditor().actions(d))
        self.assertEqual(report.checks[0].status,'unavailable')

    def test_backend_runtime_failure_is_unavailable_not_a_method_verdict(self):
        def runtime_error():raise RuntimeError('synthetic backend failure')
        result=oa.check(runtime_error)
        self.assertEqual(result.checks[0].status,'unavailable');self.assertFalse(result.passed)

    def test_full_synthetic_case_and_mutated_applied_input(self):
        self.assertTrue(oa.check(lambda:self.auditor().case(self.zero_case,np.zeros(4))).passed)
        d=copy.deepcopy(self.zero_case);d['U'][10,0]=.01
        self.assertFalse(oa.check(lambda:self.auditor().case(d,np.zeros(4))).passed)

    def test_final_window_includes_850_and_1000_physical_velocity_units(self):
        X=np.zeros((1001,4));U=np.zeros((1000,1))
        for index in [850,1000]:
            X[:]=0;X[index,3]=.05001/math_sqrt_g()
            result=oa.endpoint_components(self.config,X,U,complete=True,exception=False)
            self.assertFalse(result['balanced'])
        X[:]=0;X[849,2]=1
        self.assertTrue(oa.endpoint_components(self.config,X,U,complete=True,exception=False)['balanced'])
        self.assertFalse(oa.endpoint_components(self.config,X,U,complete=True,exception=True)['strict_reference_success'])

    def test_incomplete_and_false_completed_cannot_pass(self):
        d=copy.deepcopy(self.zero_case);d['X']=d['X'][:11];d['U']=d['U'][:10];d['actions']=d['actions'][:10]
        self.assertFalse(oa.check(lambda:self.auditor().case(d,np.zeros(4))).passed)
        d['status']='interrupted';result=self.auditor().case(d,np.zeros(4))
        self.assertFalse(result.complete);self.assertFalse(result.success)

    def test_aggregate_exact_counts_halves_and_199_refusal(self):
        results=[oa.CaseResult(i,-.1 if i<100 else .1,True,i>=4,{},'synthetic_fixture') for i in range(200)]
        self.assertTrue(oa.summarize(self.config,results)['passed'])
        results[4]=oa.CaseResult(4,-.1,True,False,{},'synthetic_fixture')
        self.assertFalse(oa.summarize(self.config,results)['passed'])
        with self.assertRaises(oa.AuditIssue):oa.summarize(self.config,results[:-1])
        results=[oa.CaseResult(i,-.1 if i<100 else .1,True,i>=5,{},'synthetic_fixture') for i in range(200)]
        results[-1]=oa.CaseResult(199,.1,True,True,{},'synthetic_fixture')
        # 195 total remains below the 196 overall threshold despite passing halves.
        self.assertFalse(oa.summarize(self.config,results)['passed'])


def math_sqrt_g():return np.sqrt(9.81)


if __name__=='__main__':unittest.main()
