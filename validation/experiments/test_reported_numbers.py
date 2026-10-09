"""Checks for quantities and arm wiring used in the scientific report."""
import unittest
from campaign import *

class StableTransition(torch.nn.Module):
    x_dim=4; u_dim=1
    def __init__(self):
        super().__init__()
        self.anchor=torch.nn.Parameter(torch.zeros(()))
        self.register_buffer('xu_std',torch.ones(5))
    def forward(self,x,u):
        return .9*x + torch.cat((u,torch.zeros_like(u).expand(*u.shape[:-1],3)),dim=-1)*.01

class ReportedNumbers(unittest.TestCase):
    def trajectory(self):
        t=np.arange(701)*.02
        X=np.zeros((701,4)); U=np.zeros((700,1))
        return t,X,U
    def test_primary_uses_physical_velocity_and_no_fall(self):
        t,X,U=self.trajectory()
        X[-151:,3]=.051/np.sqrt(9.81)
        self.assertFalse(metrics(t,X,U,[False]*700)['primary_success'])
        X[:,3]=0.; X[10,2]=np.pi/2
        result=metrics(t,X,U,[False]*700)
        self.assertTrue(result['balanced']); self.assertFalse(result['primary_success'])
        self.assertTrue(result['fell'])
    def test_cart_excursion_is_descriptive(self):
        t,X,U=self.trajectory(); X[:,0]=100.
        result=metrics(t,X,U,[True]*700)
        self.assertTrue(result['primary_success']); self.assertEqual(result['max_abs_p'],100.)
        self.assertEqual(result['final_p'],100.)
    def test_settling_requires_three_seconds_and_last_violation(self):
        t,X,U=self.trajectory(); X[:101,2]=.1
        result=metrics(t,X,U,[False]*700)
        self.assertAlmostEqual(result['angle_settle_seconds'],2.02)
        X[600,2]=.1
        self.assertIsNone(metrics(t,X,U,[False]*700)['angle_settle_seconds'])
    def test_switch_and_saturation_counts(self):
        t,X,U=self.trajectory(); U[:25]=10.; U[25:50]=-10.
        mask=[False]*100+[True]*100+[False]*500
        result=metrics(t,X,U,mask)
        self.assertEqual(result['switch_toggles'],2)
        self.assertAlmostEqual(result['local_fraction'],1/7)
        self.assertAlmostEqual(result['input_limit_seconds'],1.)
    def test_wilson_known_endpoints(self):
        low,high=wilson(0,10)
        self.assertAlmostEqual(low,0.,places=14)
        self.assertAlmostEqual(high,.2775327998628892)
        low,high=wilson(10,10)
        self.assertAlmostEqual(low,.7224672001371108)
        self.assertAlmostEqual(high,1.)
    def test_ic_counts_and_bounds(self):
        sets=ic_sets()
        self.assertEqual([len(sets[k]) for k in ('A','B','C','D')],[6,171,200,1681])
        C=np.asarray(sets['C'])
        self.assertTrue(np.all(C>=[-1.5,-2.5,-1.3,-3.5]))
        self.assertTrue(np.all(C<=[1.5,2.5,1.3,3.5]))
        self.assertEqual(sets,ic_sets())
    def test_M1_replans_every_step_and_M_queues_five(self):
        config=runner.make_config('smoke',device='cpu')
        decoder=CanonicalBehaviorDecoder(StableTransition(),config.horizon)
        gain=np.zeros((1,4)); state=np.array([0.,0.,.2,0.])
        M=controller(decoder,config,gain,'M'); M1=controller(decoder,config,gain,'M1')
        for k in range(7):M(k,state); M1(k,state)
        self.assertEqual(len(M.diagnostics['loss']),2)
        self.assertEqual(len(M1.diagnostics['loss']),7)
    def test_H_matches_original_factory(self):
        config=runner.make_config('smoke',device='cpu')
        decoder=CanonicalBehaviorDecoder(StableTransition(),config.horizon)
        gain=np.array([[.1,.2,.3,.4]])
        H=controller(decoder,config,gain,'H')
        original=runner.make_controller(decoder,config,gain)
        for k,state in enumerate((np.array([0.,0.,.2,0.]),np.zeros(4),np.array([0.,0.,.2,0.]))):
            np.testing.assert_array_equal(H(k,state),original(k,state))
        self.assertEqual(H.diagnostics['terminal_policy_used'],[False,True,False])
    def test_local_arm_clips_without_CEM(self):
        config=runner.make_config('smoke',device='cpu')
        L=controller(None,config,np.ones((1,4))*100,'L')
        self.assertEqual(float(L(0,np.ones(4))[0]),-10.)
        self.assertEqual(len(L.diagnostics['loss']),0)

if __name__=='__main__':
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
