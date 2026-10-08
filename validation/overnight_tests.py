"""Lifecycle/audit regression tests, synthetic fixtures, no calibration campaign."""
from common import *
import argparse
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import gate_a
from overnight_audit import audit,score
from overnight_queue import atomic,run_job

class OvernightTests(unittest.TestCase):
    def test_independent_physical_velocity_units(self):
        x=np.zeros((1001,4));self.assertTrue(score(x)[0])
        x[-1,1]=.050001/np.sqrt(9.81);self.assertFalse(score(x)[0])
        x[-1,1]=0.;x[-1,3]=.050001/np.sqrt(9.81);self.assertFalse(score(x)[0])
        self.assertFalse(score(np.zeros((50,4)))[0])

    def test_saved_endpoint_success_cannot_hide_wrong_plant(self):
        with tempfile.TemporaryDirectory(dir=OUT) as directory:
            root=Path(directory);X=np.zeros((1001,4));U=np.zeros((1000,1))
            np.save(root/'initial_states.npy',np.zeros((1,4)))
            atomic(root/'cases.json',[{'case':0,'strict_physical_success':True,'planning_calls':0,'executed_samples':1000}])
            atomic(root/'summary.json',{'passed':False,'incomplete':True,'cases':1})
            np.savez(root/'case_000.npz',X=X,U=U,branches=np.array(['local']*1000))
            good=audit(root);self.assertTrue(good['audit_passed']);self.assertFalse(good['Gate_A_passed'])
            X[111,0]=.01 # Still satisfies the endpoint; violates original RK4.
            np.savez(root/'case_000.npz',X=X,U=U,branches=np.array(['local']*1000))
            bad=audit(root);self.assertFalse(bad['audit_passed'])
            self.assertTrue(any('Plant replay' in message for message in bad['mismatches']))

    def test_termination_checkpoints_one_step_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory(dir=OUT) as directory:
            destination=Path(directory)/'calibration'
            args=SimpleNamespace(cases=200,starts=1,box_scale=1.,device='cpu',execute=True,max_worker_hours=7.7,output_dir=str(destination))
            original_step=gate_a.rk4_step
            def one_step(*args,**kwargs):
                answer=original_step(*args,**kwargs);gate_a.STOP_REQUESTED=True;return answer
            gen=SimpleNamespace(uniform=lambda *args,**kwargs:np.zeros((200,4)))
            gate_a.STOP_REQUESTED=False
            try:
                with patch.object(gate_a.np.random,'default_rng',return_value=gen),patch.object(gate_a,'check_oracle',return_value=0.),patch.object(gate_a,'rk4_step',side_effect=one_step):
                    with self.assertRaisesRegex(RuntimeError,'checkpoint saved'):gate_a.main(args)
                partial=json.loads((destination/'summary.json').read_text())
                self.assertTrue(partial['incomplete']);self.assertFalse(partial['passed'])
                with np.load(destination/'incomplete_case_0.npz') as data:
                    self.assertEqual(data['X'].shape,(2,4));self.assertEqual(data['U'].shape,(1,1))
                    np.testing.assert_allclose(data['X'][1],original_step(data['X'][0],float(data['U'][0,0]),.02/np.sqrt(1/9.81),2.))
                with patch.object(gate_a,'check_oracle',return_value=0.):
                    with self.assertRaisesRegex(RuntimeError,'overwrite'):gate_a.main(args)
            finally:gate_a.STOP_REQUESTED=False

    def test_failed_child_is_reported(self):
        with tempfile.TemporaryDirectory(dir=OUT) as directory:
            row=run_job('fixture',[sys.executable,'-c','raise SystemExit(7)'],60,Path(directory),time.perf_counter()+120)
            self.assertEqual(row['returncode'],7);self.assertFalse(row['timeout'])

if __name__=='__main__':
    configure()
    with Meter('overnight lifecycle/audit synthetic-fixture tests',reservation=60):
        result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(OvernightTests))
        save('overnight_tests.json',{'passed':result.wasSuccessful(),'tests':result.testsRun,
             'scope':'synthetic saved traces and process/checkpoint failures; one zero-input original-plant fixture step, no oracle calibration campaign'})
    raise SystemExit(0 if result.wasSuccessful() else 1)
