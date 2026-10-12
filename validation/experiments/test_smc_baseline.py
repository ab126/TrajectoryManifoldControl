"""Scientific checks for the proposed sliding law, calibration and effort metrics."""
import unittest
import numpy as np
from scipy.linalg import solve_discrete_are
from smc_baseline import surface, calibrate, SlidingFeedback, effort


class SlidingNumbers(unittest.TestCase):
    def setUp(self):
        self.A=np.diag([1.04,1.01,.97,.93])
        self.B=np.array([[.1],[.2],[.1],[.15]])
        self.Q=np.diag([.02,30.,100.,10.]);self.R=np.array([[.03]])
        self.C,self.P,self.E=surface(self.A,self.B,self.Q,self.R)

    def test_full_state_surface_has_unit_input_gain(self):
        np.testing.assert_allclose(self.C@self.B,[[1.]],atol=1e-12)
        self.assertEqual(self.C.shape,(1,4))
        self.assertTrue(np.all(np.abs(self.C)>0))

    def test_equivalent_map_reaches_surface_and_is_stable(self):
        np.testing.assert_allclose(self.C@self.E,np.zeros((1,4)),atol=1e-12)
        self.assertLess(max(abs(np.linalg.eigvals(self.E))),1.)
        self.assertLess(max(np.linalg.eigvalsh(self.E.T@self.P@self.E-self.P)),0.)

    def test_equivalent_norm_is_no_worse_than_DARE_feedback(self):
        K=np.linalg.solve(self.R+self.B.T@self.P@self.B,self.B.T@self.P@self.A)
        F=self.A-self.B@K
        self.assertLessEqual(max(np.linalg.eigvalsh(self.E.T@self.P@self.E-F.T@self.P@F)),1e-10)

    def test_hand_computed_inside_and_outside_reaching(self):
        law=SlidingFeedback([[1.1]],[[1.]],phi=2.,bound=100.)
        for x,expected in ((1.,.45),(4.,2.7),(-1.,-.45),(-4.,-2.7)):
            u,_,_=law.command([x])
            self.assertAlmostEqual(1.1*x+u[0],expected)

    def test_fitted_multistate_reaching_matches_scalar_law(self):
        law=SlidingFeedback(self.A,self.C,phi=.3,bound=1e6)
        for x in (np.array([.2,-.1,.1,.3]),np.array([-2.,1.,-1.,-3.])):
            u,s,_=law.command(x)
            expected=.9*s-.135*np.clip(s/.3,-1,1)
            self.assertAlmostEqual(float((self.C@(self.A@x+self.B@u)).item()),expected)

    def test_zero_correction_is_same_rho_linear_control(self):
        law=SlidingFeedback(self.A,self.C,phi=.3,correction=False,bound=1e6)
        x=np.array([.2,-.1,.1,.3]);u,s,_=law.command(x)
        self.assertAlmostEqual(float((self.C@(self.A@x+self.B@u)).item()),.9*s)
        self.assertEqual(law.beta,0.)

    def test_clipping_and_nonfinite_input_fallback(self):
        law=SlidingFeedback([[1.1]],[[1.]],phi=.01,bound=10.)
        u,_,raw=law.command([1000.])
        self.assertLess(raw,-10.);self.assertEqual(u[0],-10.)
        self.assertEqual(law.command([float('nan')])[0][0],0.)

    def test_validation_calibration_excludes_distant_states(self):
        train_x=np.linspace(-1,1,21).reshape(21,1,1)
        train=(np.concatenate((train_x,train_x),axis=1),np.zeros((21,1,1)))
        val_x=np.r_[np.linspace(-.5,.5,20),100.].reshape(21,1,1)
        vn=val_x+.02;vn[-1]=10000.
        validation=(np.concatenate((val_x,vn),axis=1),np.zeros((21,1,1)))
        result=calibrate(np.eye(1),np.ones((1,1)),np.ones((1,1)),train,validation)
        self.assertEqual(result['validation_samples'],20)
        self.assertAlmostEqual(result['residual_abs_q95'],.02)
        self.assertAlmostEqual(result['phi'],.08)
        self.assertFalse(result['uncertainty_bound_certified'])

    def test_noiseless_calibration_has_positive_width_floor(self):
        x=np.linspace(-.5,.5,20).reshape(20,1,1)
        data=(np.concatenate((x,x),axis=1),np.zeros((20,1,1)))
        self.assertEqual(calibrate(np.eye(1),np.ones((1,1)),np.ones((1,1)),data,data)['phi'],.001)

    def test_input_variation_uses_applied_not_requested_input(self):
        result=effort([0.,1.,-1.,0.],[0.,1.,-1.,1.])
        self.assertEqual(result['input_total_variation'],4.)
        self.assertEqual(result['command_total_variation'],5.)
        self.assertAlmostEqual(result['input_rms'],np.sqrt(.5))

    def test_cpu_feedback_adapter_has_no_planner_or_LQR_branch(self):
        from e6_smc import make_control
        control=make_control(self.A,self.C,.3,'SMC')
        x=np.array([.2,-.1,.1,.3])
        expected=SlidingFeedback(self.A,self.C,.3).command(x)[0]
        np.testing.assert_allclose(control(0,x),expected)
        self.assertEqual(control.diagnostics['loss'],[])
        self.assertEqual(control.diagnostics['terminal_policy_used'],[False])
        self.assertEqual(len(control.diagnostics['evaluation_seconds']),1)

    def test_feedback_pairs_include_linear_correction_control(self):
        from e6_smc import paired_feedback
        from pathlib import Path
        import csv,tempfile
        rows=[]
        for arm,successes in [('L',[True,False]),('SMC_linear',[False,False]),('SMC',[True,True])]:
            for index,success in enumerate(successes):
                rows.append(dict(seed=0,ic_set='nominal_A',seconds=14.,arm=arm,ic_index=index,primary_success=success))
        with tempfile.TemporaryDirectory() as temp:
            paired_feedback(Path(temp),rows)
            with (Path(temp)/'feedback_paired_comparisons.csv').open() as handle:
                pairs=list(csv.DictReader(handle))
        pair=next(r for r in pairs if r['reference']=='SMC_linear' and r['arm']=='SMC')
        self.assertEqual(pair['n_paired'],'2')
        self.assertEqual(pair['reference_only'],'0');self.assertEqual(pair['arm_only'],'2')
        self.assertEqual(float(pair['success_rate_difference']),-1.)

    def test_scientific_sample_counts_and_pilot_membership(self):
        from e6_smc import tasks
        nominal,stress=tasks()
        self.assertEqual(len(nominal),354)
        self.assertEqual(len(stress),3696)
        self.assertEqual({t[2] for t in nominal[:4]},{'SMC','SMC_linear'})
        self.assertTrue(all(t[0]==0 and t[-1]==14. for t in nominal[:4]))
        self.assertEqual(len({(t[0],t[1],t[2],t[3],t[4],t[6]) for t in nominal+stress}),4050)

    def exercise_adapter(self,condition):
        import e6_smc as experiment
        from pathlib import Path
        from unittest.mock import patch
        import tempfile
        config=experiment.runner.make_config('dev',device='cpu',seed=0)
        state=np.array([0.,0.,.02,0.]);applied=[]
        def fake_plant(x,u,dt):
            applied.append(np.asarray(u).copy())
            return self.A@x+self.B@np.asarray(u)
        def fake_case(out,decoder,gain,config,arm,label,index,state,seconds,observe=None,control=None):
            X=[np.asarray(state)];U=[]
            for k in range(2):
                observed=X[-1] if observe is None else observe(k,X[-1])
                value=control(k,observed);U.append(value)
                X.append(experiment.runner._plant_step(X[-1],value,config.dt))
            key=f'seed0_{arm}_{label}_{index:04d}_{seconds:g}s'
            raw=out/'raw'/(key+'.npz');raw.parent.mkdir(parents=True)
            np.savez_compressed(raw,t=np.arange(3)*.02,X=X,U=U)
            return dict(case=key,wall_seconds=.01,raw_path=str(raw))
        with tempfile.TemporaryDirectory() as temp:
            with patch.object(experiment,'OUT',Path(temp)),patch.object(experiment,'ROOT',Path(temp)),\
                 patch.object(experiment.runner,'_plant_step',experiment.runner._plant_step),\
                 patch.object(experiment,'model',return_value=(self.A,self.C,.3,np.ones(4),config)),\
                 patch.object(experiment,'NOMINAL_STEP',fake_plant),patch.object(experiment,'run_case',fake_case):
                row=experiment.evaluate((0,condition,'SMC','A',0,state.tolist(),.04))
                with np.load(row['raw_path']) as data:
                    arrays={key:data[key] for key in data.files}
        return row,arrays,np.asarray(applied),state

    def test_delay_adapter_records_inputs_actually_applied_to_plant(self):
        row,arrays,applied,_=self.exercise_adapter('input_delay')
        np.testing.assert_allclose(arrays['U_applied'],applied)
        np.testing.assert_allclose(applied[0],[0.])
        np.testing.assert_allclose(applied[1],arrays['U'][0])
        self.assertEqual(row['max_abs_u'],float(np.max(np.abs(applied))))

    def test_measurement_adapter_uses_observed_state_and_matched_noise(self):
        from e6_smc import normal_sequence
        _,arrays,applied,state=self.exercise_adapter('measurement_0.03')
        noise=normal_sequence(0,'measurement_0.03',0,2,np.ones(4)*.03)
        np.testing.assert_allclose(arrays['measurement_noise'],noise)
        expected=SlidingFeedback(self.A,self.C,.3).command(state+noise[0])[0]
        np.testing.assert_allclose(arrays['U'][0],expected)
        np.testing.assert_allclose(arrays['U_applied'],applied)


if __name__=='__main__':unittest.main(verbosity=2)
