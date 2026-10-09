"""Numerical QP integration checks on existing E0 data; no closed-loop baseline."""
from campaign import *
from deepc import windows,compress,DeePC
import unittest

class RecordedDataQPChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path=OUT/'models/episodes3500_seed0/data.npz'
        if not path.exists():raise unittest.SkipTest('Existing authored-size data required')
        saved=np.load(path);cls.train=runner.split_episodes(saved['X'],saved['U'],0)['train']
    def test_full_state_30_step_qp_uses_recorded_units_and_bounds(self):
        Z,layout=windows(*self.train,30);basis=compress(Z,layout,30,1)
        for penalty in (.001,.01,.1):
            ctl=DeePC(basis,Q,R,lambda_per_column=penalty)
            for state in ([0.,0.,np.pi/3,0.],[1.5,-2.5,1.3,-3.5]):
                plan=ctl.solve(state)
                self.assertTrue(plan['finite'],plan['status'])
                np.testing.assert_allclose(plan['prediction'][0],state,atol=1e-4)
                self.assertLessEqual(np.max(np.abs(plan['u'])),10.)
    def test_io_qp_uses_actual_past_and_current_outputs(self):
        # Existing episodes are 30 steps; N=28 here checks the two-sample history.
        Z,layout=windows(*self.train,28,tini=2,output_indices=[0,2]);basis=compress(Z,layout,28,1,tini=2)
        ctl=DeePC(basis,Q,R,lambda_per_column=.01)
        state=np.array([0.,0.,np.pi/3,0.]);dt=runner.make_config('dev').dt
        for k in range(3):
            output=state[[0,2]].copy();u=ctl(k,output)
            state=runner._plant_step(state,u,dt)
        self.assertTrue(ctl.last_plan['finite'],ctl.last_plan['status'])
        np.testing.assert_allclose(ctl.last_plan['prediction'][0],output,atol=1e-4)
        self.assertEqual(ctl.diagnostics['solve_steps'],[2])

if __name__=='__main__':torch.set_num_threads(1);unittest.main(verbosity=2)
