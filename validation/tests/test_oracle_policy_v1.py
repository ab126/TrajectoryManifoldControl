"""I03 synthetic engineering fixtures only; no plant/CUDA/scientific cases."""
import ast
import hashlib
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import oracle_config as oc
import oracle_math_v1 as om
import oracle_policy_v1 as op


def zero_noise(config, case, sample):
    noise = torch.zeros((3, 256, 60, 1), dtype=torch.float32)
    return noise, dict(key=['oracle_v1', case, 0], seed=op.noise_seed(config, case, sample),
                       draw_sha256=hashlib.sha256(noise.numpy().tobytes()).hexdigest())


class FakePlanner:
    fixture = True
    def __init__(self, local, step):
        self.local, self.step_cpu, self.calls = local, step, []
        self.error, self.feasible, self.bad_flag, self.bad_first = None, True, False, False

    def plan(self, x, *, case, sample, warm):
        self.calls.append(dict(sample=sample, warm=warm.copy()))
        if self.error:
            raise self.error
        u = np.linspace(0., 1., 60, dtype=np.float64)[:, None]
        states = [x.copy()]
        for control in u[:, 0]:
            states.append(self.step_cpu(states[-1], float(control)))
        z = np.array(states)
        if self.bad_flag:
            z[-1, 0] = 3.
        if self.bad_first:
            z[0, 0] += .1
        return op.PlanResult(u, z, self.feasible, (False, 0., 0., 1., 0), {'fixture_solver': True})


class PolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.config = oc.load_scientific(HERE/'reviews/i01_v1/scientific.json')
        cls.gain = om.load_gain(cls.config, HERE/'reviews/i02_v1/gain.json',
                               expected_sha256='a0b7e1c67bc6746e8936cdf67959f76a04a1c2403418492e7c5600cc87f81c58')
        cls.local = om.LocalArithmetic(cls.config, cls.gain)

    def policy(self, step=lambda x, u: x.copy()):
        planner = FakePlanner(self.local, step)
        return op.OraclePolicy(self.config, planner, case=7), planner

    def test_import_is_inert_and_has_no_sut_or_launcher(self):
        tree = ast.parse((HERE/'oracle_policy_v1.py').read_text())
        eager = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
        names = {n.module for n in eager if isinstance(n, ast.ImportFrom)}
        names |= {a.name for n in eager if isinstance(n, ast.Import) for a in n.names}
        self.assertFalse(names & {'torch', 'numpy', 'src', 'common', 'subprocess', 'models'})

    def test_shift_uses_actual_elapsed_and_repeats_final_input(self):
        controls = np.linspace(-10., 10., 60)[:, None]
        for d in (0, 1, 5, 60, 75):
            shifted = op.shift_warm(controls, d)
            expected = np.array([controls[min(j+d, 59), 0] for j in range(60)])[:, None]
            np.testing.assert_array_equal(shifted, expected)
        with self.assertRaises(op.PolicyError): op.shift_warm(controls, True)
        with self.assertRaises(op.PolicyError): op.shift_warm(controls.astype(np.float32), 1)

    def test_indexed_noise_is_float32_full_and_does_not_change_global_rng(self):
        before = torch.random.get_rng_state().clone()
        a, ra = op.indexed_noise(self.config, 7, 12)
        op.indexed_noise(self.config, 8, 1)  # branch/call order does not alter key
        b, rb = op.indexed_noise(self.config, 7, 12)
        self.assertTrue(torch.equal(a, b)); self.assertEqual(ra, rb)
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        self.assertEqual(a.dtype, torch.float32)
        self.assertEqual(tuple(a.shape), (3, 256, 60, 1))
        payload = b'[20261006,"cem",["oracle_v1",7,0],12]'
        expected = int.from_bytes(hashlib.sha256(payload).digest()[:8], 'little') % (2**63-1)
        self.assertEqual(ra['seed'], expected)

    def test_queue_five_inputs_and_shift_on_next_solve(self):
        policy, planner = self.policy(); x = np.array([1., 0., 0., 0.])
        actions = [policy.act(k, x) for k in range(6)]
        self.assertEqual([a.record['branch'] for a in actions], ['planned']+['queued']*4+['planned'])
        self.assertEqual([a.record['queue_after'] for a in actions[:5]], [4, 3, 2, 1, 0])
        self.assertEqual(len(planner.calls), 2)
        self.assertEqual(actions[5].record['warm_shift'], 5)
        np.testing.assert_array_equal(planner.calls[1]['warm'], op.shift_warm(policy.previous.controls, 5))

    def test_queue_guard_failure_clears_both_and_replans_once_zero_warm(self):
        policy, planner = self.policy(); x = np.array([1., 0., 0., 0.])
        policy.act(0, x)
        a = policy.act(1, x+np.array([.1, 0., 0., 0.]))
        self.assertIn('queue_guard_failed', a.record['invalidations'])
        self.assertEqual(a.record['planning_calls'], 1)
        np.testing.assert_array_equal(planner.calls[-1]['warm'], np.zeros((60, 1)))
        self.assertIsNone(a.record['warm_shift'])

    def test_unsafe_next_queue_guard_clears_memories(self):
        policy, planner = self.policy(); x = np.array([1., 0., 0., 0.])
        policy.act(0, x)
        def unsafe_once(z, u):
            if u > 0:
                return np.array([3., 0., 0., 0.])
            return z.copy()
        planner.step_cpu = policy.step_cpu = unsafe_once
        a = policy.act(1, x)
        self.assertIn('queue_guard_failed', a.record['invalidations'])
        np.testing.assert_array_equal(planner.calls[-1]['warm'], np.zeros((60, 1)))
        self.assertEqual(a.record['planning_calls'], 1)

    def test_local_entry_precedes_queue_and_local_exit_discards_old_plan(self):
        policy, planner = self.policy(step=lambda z, u: .5*z)
        policy.act(0, np.array([1., 0., 0., 0.]))
        local = policy.act(1, np.array([.1, 0., 0., 0.]))
        self.assertEqual(local.record['branch'], 'local_enter')
        self.assertIsNone(policy.previous); self.assertEqual(local.record['queue_after'], 0)
        stay = policy.act(2, np.array([.05, 0., 0., 0.]))
        self.assertEqual(stay.record['branch'], 'local_stay')
        exited = policy.act(3, np.array([1., 0., 0., 0.]))
        self.assertIn('local_exit', exited.record['invalidations'])
        self.assertEqual(exited.record['planning_calls'], 1)
        np.testing.assert_array_equal(planner.calls[-1]['warm'], np.zeros((60, 1)))

    def test_actual_gap_shift_beyond_horizon(self):
        policy, planner = self.policy(); x = np.array([1., 0., 0., 0.])
        policy.act(0, x)
        old = policy.previous.controls.copy()
        a = policy.act(75, x)  # fixture-only sampling gap
        self.assertEqual(a.record['warm_shift'], 75)
        np.testing.assert_array_equal(planner.calls[-1]['warm'], op.shift_warm(old, 75))

    def test_no_feasible_plan_zero_is_not_itself_endpoint_failure(self):
        policy, planner = self.policy(); planner.feasible = False
        a = policy.act(0, np.array([1., 0., 0., 0.]))
        self.assertEqual(a.value, 0.); self.assertEqual(a.record['branch'], 'no_feasible_plan')
        self.assertFalse(a.record['exception_latched']); self.assertFalse(a.record['safety_latched'])
        self.assertTrue(a.record['zero_prediction']['safe']); self.assertIsNone(policy.previous)

    def test_exception_zero_and_latched_failure_survive_later_settling(self):
        policy, planner = self.policy(step=lambda z, u: .5*z)
        planner.error = op.PolicyError('fixture_solver_error', 'synthetic')
        a = policy.act(0, np.array([1., 0., 0., 0.]))
        self.assertEqual(a.record['branch'], 'exception_zero'); self.assertTrue(a.record['exception_latched'])
        planner.error = None
        b = policy.act(1, np.zeros(4))
        self.assertEqual(b.record['branch'], 'local_enter'); self.assertTrue(b.record['exception_latched'])

    def test_finite_safety_violation_latches_but_nonfinite_stops(self):
        policy, planner = self.policy(); planner.feasible = False
        a = policy.act(0, np.array([3., 0., 0., 0.]))
        self.assertEqual(a.value, 0.); self.assertTrue(a.record['safety_latched'])
        b = policy.act(1, np.array([1., 0., 0., 0.]))
        self.assertTrue(b.record['safety_latched']); self.assertEqual(b.record['first_violation'], 0)
        c = policy.act(2, np.array([np.nan, 0., 0., 0.]))
        self.assertIsNone(c.value); self.assertEqual(c.record['branch'], 'nonfinite_stop')

    def test_new_plan_guard_failure_is_exception_without_second_solve(self):
        policy, planner = self.policy(); planner.bad_first = True
        a = policy.act(0, np.array([1., 0., 0., 0.]))
        self.assertEqual(a.record['error']['code'], 'new_plan_guard_failed')
        self.assertEqual(len(planner.calls), 1); self.assertEqual(a.record['planning_calls'], 1)

    def test_false_feasible_flag_is_rejected(self):
        policy, planner = self.policy(); planner.bad_flag = True
        a = policy.act(0, np.array([1., 0., 0., 0.]))
        self.assertEqual(a.record['error']['code'], 'plan_feasibility')

    def test_interrupt_and_duplicate_samples_are_rejected(self):
        policy, _ = self.policy(); x = np.array([1., 0., 0., 0.])
        policy.act(0, x)
        with self.assertRaises(op.PolicyError): policy.act(0, x)
        policy.interrupt()
        with self.assertRaises(op.PolicyError): policy.act(1, x)
        self.assertIsNone(policy.previous)

    def test_fixture_CEM_slots_warm_retention_and_full_three_iterations(self):
        planner = op.CEMPlanner.for_fixture(self.config, self.gain, lambda z, u: z.copy(),
                                          lambda z, u: z.clone(), noise_provider=zero_noise)
        result = planner.plan(np.array([1., 0., 0., 0.]), case=7, sample=4, warm=np.ones((60, 1)))
        self.assertTrue(result.feasible); self.assertEqual(result.rank[-1], 1)  # zero slot, earliest tie
        self.assertEqual(len(result.trace['iterations']), 3)
        self.assertEqual(result.trace['evidence_kind'], 'synthetic_fixture')
        for i, row in enumerate(result.trace['iterations']):
            np.testing.assert_array_equal(row['first_seen'], np.arange(256*i, 256*(i+1)))
            self.assertEqual(row['J'][1], 144.)
            self.assertAlmostEqual(row['J'][3], 145.8)  # original warm survives each iteration
            self.assertTrue(row['moments_updated'])
            self.assertEqual(len(row['elites']), 32)
        np.testing.assert_array_equal(result.controls, np.zeros((60, 1)))

    def test_insufficient_CEM_population_preserves_partial_trace(self):
        def bad_batch(z, u):
            nxt = z.clone(); nxt[31:] = float('nan'); return nxt
        planner = op.CEMPlanner.for_fixture(self.config, self.gain, lambda z, u: z.copy(),
                                          bad_batch, noise_provider=zero_noise)
        with self.assertRaises(op.PolicyError) as error:
            planner.plan(np.array([1., 0., 0., 0.]), case=7, sample=0, warm=np.zeros((60, 1)))
        self.assertEqual(error.exception.code, 'insufficient_finite_candidates')
        row = error.exception.evidence['iterations'][0]
        self.assertFalse(row['moments_updated']); self.assertEqual(row['eligible_count'], 31)

    def test_scientific_noise_override_is_refused_without_initializing_CUDA(self):
        with self.assertRaises(op.PolicyError):
            op.CEMPlanner(self.config, self.gain, lambda z, u: z, lambda z, u: z, noise_provider=zero_noise)


if __name__ == '__main__': unittest.main()
