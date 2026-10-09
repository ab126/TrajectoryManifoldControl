from ensemble_baseline import *
import unittest

class Member(torch.nn.Module):
    x_dim=4;u_dim=1
    def __init__(self,offset):
        super().__init__();self.anchor=torch.nn.Parameter(torch.tensor(float(offset)));self.register_buffer('xu_std',torch.ones(5))
    def forward(self,x,u):
        value=torch.zeros_like(x);value[...,2]=self.anchor;return value

class EnsembleChecks(unittest.TestCase):
    def members(self,horizon):return [CanonicalBehaviorDecoder(Member(v),horizon) for v in (.1,.2,.3,.4,.5)]
    def test_mean_trajectory_cost_not_cost_of_mean_prediction(self):
        members=self.members(1);Q=torch.diag(torch.tensor([0.,0.,1.,0.]));R=torch.zeros(1,1)
        solver=EnsembleCosts(members[0],Q,R,members=members,population=2,elite=1)
        u=torch.zeros(1,1,1);x=members[0].rollout(torch.zeros(1,4),u)
        self.assertAlmostEqual(float(solver._cost(x,u)),.11,places=7)
        self.assertNotAlmostEqual(float(solver._cost(x,u)),.3**2,places=6)
    def test_imported_factory_replans_each_sample_without_switch(self):
        config=runner.make_config('smoke',device='cpu',horizon=2)
        ctl=make_ensemble_controller(self.members(2),config)
        for k in range(7):ctl(k,np.zeros(4))
        self.assertEqual(len(ctl.diagnostics['loss']),7)
        self.assertEqual(ctl.diagnostics['terminal_policy_used'],[False]*7)
    def test_bootstrap_stays_in_training_episodes_and_varies_by_member(self):
        a=bootstrap_indices(2,0,100);b=bootstrap_indices(2,1,100)
        np.testing.assert_array_equal(a,bootstrap_indices(2,0,100))
        self.assertTrue(np.all((a>=0)&(a<100)));self.assertFalse(np.array_equal(a,b))
        self.assertLess(len(np.unique(a)),100)

if __name__=='__main__':torch.set_num_threads(1);unittest.main(verbosity=2)
