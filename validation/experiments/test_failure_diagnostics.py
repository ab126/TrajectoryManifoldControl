from failure_diagnostics import *
import unittest

class ObjectiveChecks(unittest.TestCase):
    def test_hand_cost_includes_terminal_and_only_within_plan_input_differences(self):
        X=np.eye(4);U=np.array([[1.],[2.],[3.]])
        # .02 + 30 + 100 + 12*10 + .03*(1+4+9) + .002*(1+1)
        self.assertAlmostEqual(objective(X,U),250.444,places=11)
        from test_reported_numbers import StableTransition
        decoder=CanonicalBehaviorDecoder(StableTransition(),3)
        solver=DataDrivenCEMSolver(decoder,torch.tensor(Q,dtype=torch.float32),torch.tensor(R,dtype=torch.float32),
            Q_terminal=torch.tensor(12*Q,dtype=torch.float32),population=2,elite=1,input_rate_weight=.002)
        value=float(solver._cost(torch.tensor(X[None],dtype=torch.float32),torch.tensor(U[None],dtype=torch.float32)))
        self.assertAlmostEqual(value,250.444,places=4)
    def test_cost_witnesses_do_not_invent_missing_true_plan_costs(self):
        costs=dict(chosen_model_cost=10.,chosen_true_cost=None,local_candidate_model_cost=7.,
                   local_candidate_true_cost=6.,local_candidate_inputs_bounded=True)
        result=compare_costs(costs)
        self.assertTrue(result['feasible_better_model_cost']);self.assertEqual(result['model_cost_gap_over_feasible_alternative'],3.)
        self.assertIsNone(result['model_preference_reversal'])
        costs.update(chosen_model_cost=2.,chosen_true_cost=9.,local_candidate_model_cost=3.)
        result=compare_costs(costs)
        self.assertFalse(result['feasible_better_model_cost']);self.assertTrue(result['model_preference_reversal'])
        costs['local_candidate_inputs_bounded']=False
        self.assertIsNone(compare_costs(costs)['feasible_better_model_cost'])
    def test_representative_contrasts_match_seed_and_IC(self):
        rows=[]
        for index,theta in enumerate((15.,60.)):
            for arm in ('H','L','M','M1','Hstar'):
                success=arm=='L' or (arm=='Hstar' and index==0)
                rows.append(dict(case=str(index)+arm,seed=0,ic_set='B',ic_index=index,arm=arm,seconds=14.,
                    initial_state=[0.,0.,float(np.deg2rad(theta)),0.],primary_success=success,no_fall=True,
                    fell_within_1s_of_switch_exit=False,switch_toggles=2 if arm=='H' else 0))
        chosen=select_representative_failures(rows)
        self.assertEqual(len(chosen),5);self.assertEqual(len({r['case'] for r in chosen}),5)
        self.assertEqual(chosen[0]['case'],'0H');self.assertEqual(chosen[-1]['case'],'1Hstar')
        self.assertTrue(all(not r['primary_success'] for r in chosen))
    def test_from_rest_spans_preserve_failed_and_missing_points(self):
        items=[dict(ic_index=i,primary_success=success,initial_state=[0.,0.,float(np.deg2rad(i)),0.])
               for i,success in [(0,True),(1,False),(2,True),(3,True),(5,True)]]
        self.assertEqual(from_rest_spans(items),[
            dict(start_degrees=0.,end_degrees=0.,tested_points=1),
            dict(start_degrees=2.,end_degrees=3.,tested_points=2),
            dict(start_degrees=5.,end_degrees=5.,tested_points=1)])
    def test_candidate_is_feasible_and_uses_same_opaque_plant(self):
        state=np.array([0.,0.,.3,0.]);gain=np.ones((1,4))*100
        X,U=local_candidate(state,gain,30)
        self.assertTrue(np.all(np.abs(U)<=10.));np.testing.assert_array_equal(X,replay(state,U))
        self.assertEqual(X.shape,(31,4));self.assertEqual(U.shape,(30,1))

if __name__=='__main__':torch.set_num_threads(1);unittest.main(verbosity=2)
