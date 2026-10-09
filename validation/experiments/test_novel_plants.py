from novel_plants import *
import tempfile
import unittest

class NovelPlantChecks(unittest.TestCase):
    def test_cart_adapter_identity(self):
        config=runner.make_config('smoke',episodes=4,horizon=3,device='cpu')
        plant=Plant('cart',4,1,np.array([-1.5,-2.5,-1.3,-3.5]),np.array([1.5,2.5,1.3,3.5]),Q,R,runner._plant_step)
        original=runner.collect_trajectory_data(config);adapted=collect(plant,config)
        for a,b in zip(original,adapted):np.testing.assert_array_equal(a,b)
    def test_dimension_parameter_is_used_by_imported_training(self):
        config=runner.make_config('smoke',episodes=8,horizon=3,epochs=1,device='cpu')
        plant=Plant('two',2,1,-np.ones(2),np.ones(2),np.eye(2),R,lambda x,u,dt:.9*x+float(u[0]))
        X,U=collect(plant,config)
        self.assertEqual(X.shape,(8,4,2));self.assertEqual(U.shape,(8,3,1))
        data=runner.split_episodes(X,U,config.seed)
        with tempfile.TemporaryDirectory() as temp:
            model,_=train(plant,config,data,Path(temp)/'test.pt')
            self.assertEqual(model.x_dim,2);self.assertEqual(model.u_dim,1)
            errors=prediction_rmse(CanonicalBehaviorDecoder(model,3),data['test'])
            self.assertEqual(len(errors['one_step_rmse_by_state']),2)
            self.assertEqual(np.asarray(errors['rmse_by_step_and_state']).shape,(4,2))
    def test_cost_adapter_changes_the_switch_in_the_imported_controller(self):
        config=runner.make_config('smoke',device='cpu')
        from src.manifold_control import DataDrivenTransition
        decoder=CanonicalBehaviorDecoder(DataDrivenTransition(4,1,(8,)),config.horizon)
        gain=np.zeros((1,4));state=np.array([1.,0.,0.,0.])
        original=runner.make_controller(decoder,config,gain);adapted=make_control(make_plant('linear'),decoder,config,gain,'H')
        original(0,state);adapted(0,state)
        self.assertEqual(original.diagnostics['terminal_policy_used'],[True])
        self.assertEqual(adapted.diagnostics['terminal_policy_used'],[False])
    def test_two_state_simulator_matches_hand_computed_discrete_rollout(self):
        config=runner.make_config('smoke',device='cpu',simulation_seconds=.2)
        plant=Plant('two',2,1,-np.ones(2),np.ones(2),np.eye(2),R,lambda x,u,dt:.9*x)
        t,X,U=simulate(plant,lambda k,x:np.zeros(1),np.array([1.,2.]),config)
        np.testing.assert_allclose(X,.9**np.arange(11)[:,None]*[1.,2.])
        self.assertAlmostEqual(t[-1],.2)
    def test_acrobot_collection_is_finite_under_stated_box_and_torque(self):
        plant=make_plant('acrobot')
        config=runner.make_config('dev',episodes=100,umax=plant.umax,device='cpu')
        X,U=collect(plant,config)
        self.assertTrue(np.isfinite(X).all());self.assertTrue(np.all(np.abs(U)<=1.))
    def test_uniform_rod_acrobot_mass_matrix_at_upright(self):
        derivative=acrobot_rhs(np.zeros(4),np.array([1.]))
        np.testing.assert_allclose(derivative,[0,0,-30/7,96/7],rtol=1e-14)
        np.testing.assert_array_equal(acrobot_rhs(np.zeros(4),np.zeros(1)),np.zeros(4))
    def test_acrobot_energy_conserved_with_zero_input(self):
        x=np.array([.2,-.3,.1,-.2]);initial=acrobot_energy(x)
        plant=make_plant('acrobot')
        for k in range(100):x=plant.step(x,np.zeros(1),.001)
        self.assertLess(abs(acrobot_energy(x)-initial),1e-10)
    def test_linear_reference_is_stabilizing(self):
        plant=make_plant('linear');A=np.diag([1.06,1.02,.96,.90]);B=np.array([.2,.2,.1,.1])[:,None]
        from scipy.linalg import solve_discrete_are
        P=solve_discrete_are(A,B,plant.Q,plant.R)
        gain=np.linalg.solve(plant.R+B.T@P@B,B.T@P@A)
        self.assertLess(max(abs(np.linalg.eigvals(A-B@gain))),1.)
    def test_known_linear_tail_cost_equals_infinite_horizon_optimum(self):
        from e3_novel import exact_linear_reference
        plant=make_plant('linear');gain,P=exact_linear_reference(plant)
        x=np.array([.1,-.2,.1,-.1]);optimal=x@P@x;cost=0.
        for _ in range(20):
            u=-gain@x
            cost+=x@plant.Q@x+u@plant.R@u
            x=plant.step(x,u,.02)
        self.assertAlmostEqual(cost+x@P@x,optimal,places=11)
    def test_acrobot_endpoint_uses_absolute_second_link_and_physical_rates(self):
        t=np.arange(701)*.02;X=np.zeros((701,4));U=np.zeros((700,1))
        plant=make_plant('acrobot')
        self.assertTrue(endpoint(plant,t,X,U,[True]*700)['primary_success'])
        X[:,0]=.015;X[:,1]=.015
        self.assertFalse(endpoint(plant,t,X,U,[True]*700)['primary_success'])
        X[:]=0.;X[:,3]=.051/np.sqrt(9.81)
        self.assertFalse(endpoint(plant,t,X,U,[True]*700)['primary_success'])

if __name__=='__main__':torch.set_num_threads(1);unittest.main(verbosity=2)
