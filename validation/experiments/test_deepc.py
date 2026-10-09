"""Hand-computed trajectories, MPC solutions, regularizer and IO access checks."""
from deepc import *
import unittest


def scalar_data():
    rng=np.random.default_rng(7);U=rng.normal(size=(80,4,1));X=np.empty((80,5,1));X[:,0]=rng.normal(size=(80,1))
    for k in range(4):X[:,k+1]=1.2*X[:,k]+U[:,k]
    Z,layout=windows(X,U,2)
    return compress(Z,layout,2,1),Z


class DeePCChecks(unittest.TestCase):
    def test_time_channel_order_and_episode_boundaries(self):
        X=np.array([[[0,10],[1,11],[2,12],[3,13]],[[100,110],[101,111],[102,112],[103,113]]])
        U=np.array([[[20],[21],[22]],[[120],[121],[122]]])
        Z,layout=windows(X,U,2)
        np.testing.assert_array_equal(Z[:,0],[0,10,1,11,2,12,20,21])
        np.testing.assert_array_equal(Z[:,1],[100,110,101,111,102,112,120,121])
        np.testing.assert_array_equal(Z[:,2],[1,11,2,12,3,13,21,22])
        self.assertEqual(Z.shape,(8,4))
        Z,_=windows(X,U,2,tini=1,output_indices=[0,1])
        np.testing.assert_array_equal(Z[:,0],[20,0,10,1,11,2,12,3,13,21,22])
        self.assertEqual(Z.shape,(11,2))
    def test_unconstrained_scalar_mpc_equals_hand_riccati_solution(self):
        basis,_=scalar_data();self.assertEqual(basis.B.shape[1],3)
        control=DeePC(basis,np.eye(1),np.eye(1),terminal_factor=1,lambda_per_column=0,input_rate_weight=0)
        plan=control.solve([1.])
        self.assertTrue(plan['finite'])
        # P_2=1; P_1=1+1.2²/(1+1)=1.72; K_0=1.2*1.72/(1+1.72).
        first=-1.2*1.72/2.72;second=-.6*(1.2+first)
        np.testing.assert_allclose(plan['u'][:,0],[first,second],atol=2e-6)
        np.testing.assert_allclose(plan['prediction'][:,0],[1.,1.2+first,1.2*(1.2+first)+second],atol=2e-6)
    def test_active_input_bound_in_hand_scalar_case(self):
        basis,_=scalar_data()
        control=DeePC(basis,np.eye(1),np.eye(1),umax=.1,terminal_factor=1,lambda_per_column=0,input_rate_weight=0)
        plan=control.solve([1.]);self.assertTrue(plan['finite'])
        np.testing.assert_allclose(plan['u'][:,0],[-.1,-.1],atol=2e-6)
        np.testing.assert_allclose(plan['prediction'][:,0],[1.,1.1,1.22],atol=2e-6)
    def test_svd_preserves_minimum_g_regularization(self):
        basis,Z=scalar_data();alpha=np.array([.2,-.1,.3]);trajectory=basis.B@alpha
        g=np.linalg.lstsq(Z,trajectory,rcond=1e-8)[0]
        self.assertAlmostEqual(float(g@g),float(alpha@alpha/basis.columns),places=12)
        np.testing.assert_allclose(Z@g,trajectory,atol=1e-12)
    def test_output_only_history_startup_and_linear_trajectory(self):
        rng=np.random.default_rng(11);dt=.1;U=rng.normal(size=(120,6,1));X=np.empty((120,7,4));X[:,0]=rng.normal(size=(120,4))
        A=np.eye(4);A[0,1]=dt;A[2,3]=dt;B=np.array([0.,1.,0.,.7])
        for k in range(6):X[:,k+1]=X[:,k]@A.T+U[:,k]*B
        Z,layout=windows(X,U,3,tini=2,output_indices=[0,2]);basis=compress(Z,layout,3,1,tini=2)
        self.assertEqual(basis.B.shape[1],9) # four initial states plus five inputs
        ctl=DeePC(basis,np.eye(4),np.eye(1),lambda_per_column=.01,dt=dt)
        state=np.array([.2,.1,-.1,.2])
        for k in range(3):
            u=ctl(k,state[[0,2]])
            if k<2:np.testing.assert_array_equal(u,[0.])
            state=A@state+B*u[0]
        plan=ctl.last_plan;self.assertTrue(plan['finite'])
        # y_(j+2)-2y_(j+1)+y_j = dt*[1,.7]*u_j.
        Y=plan['prediction'];expected=dt*plan['u'][:-1]*[1.,.7]
        np.testing.assert_allclose(Y[2:]-2*Y[1:-1]+Y[:-2],expected,atol=1e-8)
        with self.assertRaises(ValueError):ctl(3,state)
    def test_nonfinite_observation_is_recorded_as_failed_solve(self):
        basis,_=scalar_data();ctl=DeePC(basis,np.eye(1),np.eye(1))
        plan=ctl.solve([np.nan]);self.assertFalse(plan['finite'])
        np.testing.assert_array_equal(plan['u'],np.zeros((2,1)))

if __name__=='__main__':unittest.main(verbosity=2)
