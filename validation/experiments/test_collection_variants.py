"""Scientific checks on the changed collection distributions."""
from distribution_ablation import *
import unittest

class CollectionVariants(unittest.TestCase):
    def test_identity_parameterization_matches_submitted_data(self):
        config=runner.make_config('smoke',episodes=10,horizon=3,device='cpu')
        X,U=runner.collect_trajectory_data(config)
        Xa,Ua=collector()(config)
        np.testing.assert_array_equal(X,Xa);np.testing.assert_array_equal(U,Ua)
    def test_half_box_changes_initial_states_exactly(self):
        config=runner.make_config('smoke',episodes=10,horizon=3,device='cpu')
        X,U=runner.collect_trajectory_data(config);Xa,Ua=collector(.5)(config)
        np.testing.assert_array_equal(Xa[:,0],X[:,0]*.5)
        np.testing.assert_array_equal(Ua,U)
    def test_uniform_branch_ignores_normal_draws(self):
        # Run the adapted collector with an RNG that rejects normal() calls.
        fn=collector(uniform_only=True)
        original=np.random.default_rng
        class UniformRNG:
            def __init__(self,seed):self.rng=original(seed)
            def uniform(self,*a,**kw):return self.rng.uniform(*a,**kw)
            def integers(self,*a,**kw):return self.rng.integers(*a,**kw)
            def normal(self,*a,**kw):raise AssertionError('Correlated/noisy policy used')
        from unittest.mock import patch
        config=runner.make_config('smoke',episodes=10,horizon=3,device='cpu')
        with patch.object(np.random,'default_rng',UniformRNG):X,U=fn(config)
        self.assertTrue(np.isfinite(X).all());self.assertTrue(np.all(np.abs(U)<=10))

if __name__=='__main__':unittest.main(verbosity=2)
