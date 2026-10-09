from e3_robustness import *
import unittest

class RobustnessWiring(unittest.TestCase):
    def test_nominal_transform_is_exact_original_step(self):
        x=np.array([.1,-.2,.3,-.4]);u=np.array([2.])
        np.testing.assert_array_equal(physical_mismatch_step(x,u,.0626418390534633),NOMINAL_STEP(x,u,.0626418390534633))
    def test_length_scales_initial_angular_acceleration(self):
        # At upright/rest with constant input, the uniform-rod mass equations
        # give pdd=u/(M+1/4), thetadd=-3*pdd/(2*l).
        dt=1e-5;u=np.array([1.]);x=np.zeros(4)
        for length in (.75,1.25):
            xn=physical_mismatch_step(x,u,dt,mass=2.,length=length)
            self.assertAlmostEqual(xn[1]/dt,1/2.25,places=8)
            self.assertAlmostEqual(xn[3]/dt,-1.5/(length*2.25),places=8)
    def test_noise_uses_nominal_units_and_is_reproducible(self):
        scale=np.array([1.,2.,3.,4.])
        a=normal_sequence(1,'measurement_0.03',3,20,scale)
        b=normal_sequence(1,'measurement_0.03',3,20,scale)
        np.testing.assert_array_equal(a,b)
        unit=normal_sequence(1,'measurement_0.03',3,20,np.ones(4))
        np.testing.assert_allclose(a,unit*scale)
        self.assertFalse(np.array_equal(a,normal_sequence(1,'measurement_0.03',4,20,scale)))

if __name__=='__main__':unittest.main(verbosity=2)
