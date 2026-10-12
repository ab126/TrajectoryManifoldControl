"""I02 engineering fixtures: no SUT import, plant rollout, CUDA or RNG draws."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import oracle_config as oc
import oracle_math_v1 as om


class OracleMathTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.config = oc.load_scientific(HERE/'reviews/i01_v1/scientific.json')
        cls.gain = om.construct_gain(cls.config)
        cls.local = om.LocalArithmetic(cls.config, cls.gain)

    def plans(self, m=1):
        return np.zeros((m, 61, 4), dtype=np.float64), np.zeros((m, 60, 1), dtype=np.float64)

    def score(self, states, controls, **kwargs):
        with np.errstate(over='ignore', invalid='ignore'):
            return om.score_candidates(self.config, states, controls, **kwargs)

    def test_import_has_no_eager_scientific_or_sut_dependencies(self):
        tree = ast.parse((HERE/'oracle_math_v1.py').read_text())
        eager = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
        names = {n.module.split('.')[0] for n in eager if isinstance(n, ast.ImportFrom)}
        names |= {a.name.split('.')[0] for n in eager if isinstance(n, ast.Import) for a in n.names}
        self.assertFalse(names & {'numpy', 'scipy', 'torch', 'common', 'src', 'models', 'subprocess'})
        all_names = {n.module.split('.')[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
        self.assertFalse(all_names & {'src', 'common', 'oracle_diagnostics', 'models', 'gate_a'})

    def test_hand_computable_two_step_objective(self):
        x = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=np.float64)
        u = np.array([[2], [-1]], dtype=np.float64)
        p = om.objective_components(self.config, x, u)
        self.assertEqual(p.stage_state, 32.)
        self.assertEqual(p.stage_input, .15)
        self.assertEqual(p.terminal, 1200.)
        self.assertAlmostEqual(p.slew, .018, places=15)
        self.assertAlmostEqual(p.total, 1232.168)

    def test_one_step_has_no_first_input_slew_or_dt_factor(self):
        x = np.array([[0, 0, 0, 1], [1, 0, 0, 0]], dtype=np.float64)
        u = np.array([[10]], dtype=np.float64)
        p = om.objective_components(self.config, x, u)
        self.assertEqual(p.total, 37.)  # 10 + .03*100 + 24
        self.assertEqual(p.slew, 0.)

    def test_full_horizon_static_position_cost_is_144(self):
        x, u = self.plans(2)
        x[0, :, 0], x[1, :, 0] = 1., -1.
        p = om.objective_components(self.config, x, u)
        np.testing.assert_array_equal(p.total, [144., 144.])
        np.testing.assert_array_equal(p.terminal, [24., 24.])

    def test_cost_uses_12Q_and_never_riccati_P(self):
        x, u = self.plans()
        x[0, -1, 0] = 1.
        p = om.objective_components(self.config, x, u)
        self.assertEqual(p.total[0], 24.)
        self.assertNotEqual(p.terminal[0], self.gain.to_dict()['P'][0][0])

    def test_batched_components_match_hand_sum_without_wrapping(self):
        x, u = self.plans(2)
        x[0, :, 2], x[1, :, 2] = 2*np.pi, 0.
        p = om.objective_components(self.config, x, u)
        self.assertAlmostEqual(p.total[0], 72*100*(2*np.pi)**2)
        self.assertEqual(p.total[1], 0.)

    def test_shapes_dtype_backend_and_config_are_strict(self):
        x, u = self.plans()
        for a, b in ((x.astype(np.float32), u), (x, u.astype(np.int64)), (x, u[..., 0]),
                     (x[:, :-1], u), (x[:, :, :3], u), (x, np.zeros((2, 60, 1))),
                     (x.tolist(), u), (x, torch.from_numpy(u))):
            with self.subTest(shape=type(a)), self.assertRaises(om.MathError):
                om.objective_components(self.config, a, b)
        with self.assertRaises(TypeError):
            om.objective_components(self.config.to_dict(), x, u)

    def test_production_scoring_locks_horizon_and_batch(self):
        x, u = self.plans()
        for a, b in ((x[:, :3], u[:, :2]), (x[0], u[0]), (x[:0], u[:0])):
            with self.assertRaises(om.MathError):
                self.score(a, b)
        for index in (True, 1., -1, 2**63-1):
            with self.assertRaises(om.MathError):
                self.score(x, u, first_seen_offset=index)

    def test_safety_and_input_bounds_are_inclusive_at_all_nodes(self):
        x, u = self.plans(4)
        x[0, 0, 0] = 2.4
        x[1, -1, 2] = -np.pi/2
        u[2, :, 0] = 10.
        u[3, :, 0] = -10.
        s = self.score(x, u)
        np.testing.assert_array_equal(s.feasible, [True]*4)
        np.testing.assert_array_equal(s.D, [0.]*4)
        np.testing.assert_array_equal(s.W, [0.]*4)
        x[0, 0, 0] = np.nextafter(2.4, np.inf)
        x[1, -1, 2] = np.nextafter(-np.pi/2, -np.inf)
        u[2, 25, 0] = np.nextafter(10., np.inf)
        self.assertFalse(self.score(x, u).feasible[:3].any())

    def test_excess_D_W_include_initial_and_terminal_nodes(self):
        x, u = self.plans()
        x[0, 0, 0], x[0, -1, 2] = 3.6, np.pi
        s = self.score(x, u)
        self.assertAlmostEqual(s.D[0], 1.)
        self.assertAlmostEqual(s.W[0], 1.25)

    def test_cheaper_unsafe_cannot_displace_feasible(self):
        x, u = self.plans(2)
        x[0, 0, 0] = 2.5
        x[1, :, 1] = 2.
        s = self.score(x, u)
        self.assertLess(s.components.total[0], s.components.total[1])
        np.testing.assert_array_equal(om.rank_candidates(s), [1, 0])

    def test_ranking_order_D_then_W_then_cost_then_first_seen(self):
        x, u = self.plans(5)
        x[:, 0, 0] = 4.8  # D=1, W=1 for all
        x[0, 1, 0] = 4.8  # W=2, loses even with lower cost
        x[1, :, 1] = 10.  # high J but W=1
        x[2, 0, 0] = 7.2  # D=2, always last
        s = self.score(x, u, first_seen_offset=256)
        np.testing.assert_array_equal(s.first_seen, np.arange(256, 261))
        np.testing.assert_array_equal(om.rank_candidates(s), [3, 4, 1, 0, 2])

    def test_nonfinite_unbounded_and_overflow_are_excluded(self):
        x, u = self.plans(6)
        x[0, 1, 1] = np.nan
        u[1, 2, 0] = np.inf
        u[2, 4, 0] = 10.0001
        x[3, :, 1] = 1e308  # finite arrays; cost arithmetic overflows
        x[4, 0, 0] = 1000.  # valid but unsafe, ahead of all invalid
        s = self.score(x, u)
        np.testing.assert_array_equal(s.eligible, [False]*4+[True, True])
        self.assertTrue(s.arrays_finite[3])
        self.assertFalse(s.arithmetic_finite[3])
        np.testing.assert_array_equal(om.rank_candidates(s), [5, 4, 0, 1, 2, 3])

    def test_elites_follow_same_safety_rank_and_exclude_nan(self):
        x, u = self.plans(34)
        x[0, :, 0] = 2.5  # cheaper unsafe candidate must be excluded from 32 elites
        x[1:33, :, 1] = 2.
        u[33, 0, 0] = np.nan
        s = self.score(x, u)
        moments = om.elite_moments(self.config, s, u)
        np.testing.assert_array_equal(moments.indices, np.arange(1, 33))
        self.assertTrue(np.isfinite(moments.mean).all())
        np.testing.assert_array_equal(moments.std, np.full((60, 1), .02))

    def test_unbiased_elite_std_and_mean_are_unsmoothed(self):
        x, u = self.plans(32)
        u[:, :, 0] = np.linspace(-1., 1., 32)[:, None]
        moments = om.elite_moments(self.config, self.score(x, u), u)
        np.testing.assert_allclose(moments.mean, 0., atol=1e-16)
        np.testing.assert_allclose(moments.std, np.linspace(-1., 1., 32).std(ddof=1))

    def test_insufficient_finite_candidates_raises_before_any_moment_update(self):
        x, u = self.plans(32)
        u[0, 0, 0] = np.nan
        snapshot = u.copy()
        with self.assertRaises(om.SearchError) as raised:
            om.elite_moments(self.config, self.score(x, u), u)
        self.assertEqual(raised.exception.finite_count, 31)
        self.assertEqual(raised.exception.code, 'insufficient_finite_candidates')
        np.testing.assert_array_equal(u, snapshot)

    def test_numpy_and_cpu_torch_components_rank_and_moments_agree(self):
        x, u = self.plans(34)
        x[:, :, 0] = np.linspace(0., 3., 34)[:, None]
        u[:, :, 0] = np.linspace(-2., 2., 34)[:, None]
        x[2, 3, 2], u[5, 2, 0] = np.nan, 11.
        sn = self.score(x, u)
        tx, tu = torch.from_numpy(x), torch.from_numpy(u)
        st = om.score_candidates(self.config, tx, tu)
        for name in ('arrays_finite', 'input_bounded', 'arithmetic_finite', 'eligible', 'feasible', 'D', 'W', 'first_seen'):
            np.testing.assert_allclose(getattr(sn, name), getattr(st, name).numpy(), equal_nan=True)
        for name in ('stage_state', 'stage_input', 'terminal', 'slew', 'total'):
            np.testing.assert_allclose(getattr(sn.components, name), getattr(st.components, name).numpy(), equal_nan=True)
        np.testing.assert_array_equal(om.rank_candidates(sn), om.rank_candidates(st).numpy())
        mn, mt = om.elite_moments(self.config, sn, u), om.elite_moments(self.config, st, tu)
        np.testing.assert_allclose(mn.mean, mt.mean.numpy(), atol=1e-15)
        np.testing.assert_allclose(mn.std, mt.std.numpy(), atol=1e-15)
        self.assertEqual(st.D.dtype, torch.float64)
        self.assertEqual(st.D.device, tx.device)
        self.assertEqual(mt.mean.device, tu.device)

    def test_scoring_and_ranking_have_no_explicit_cpu_transfers(self):
        source = (HERE/'oracle_math_v1.py').read_text()
        tree = ast.parse(source)
        nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                 and n.name in ('objective_components', 'score_candidates', 'rank_candidates')]
        calls = {n.func.attr for node in nodes for n in ast.walk(node)
                 if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
        self.assertFalse(calls & {'cpu', 'numpy', 'tolist', 'item'})

    def test_exact_RK4_derivative_matches_independent_linear_RK4_application(self):
        A, B = om.upright_rk4_matrices(self.config)
        h = self.config.to_dict()['plant']['dt_normalized']
        # Direct slope application, independently of matrix-power formula.
        def f(z, u):
            return np.array([z[1], -z[2]/3+4*u/9, z[3], 2*z[2]-2*u/3])
        def step(z, u):
            k1 = f(z, u); k2 = f(z+h*k1/2, u)
            k3 = f(z+h*k2/2, u); k4 = f(z+h*k3, u)
            return z+h*(k1+2*k2+2*k3+k4)/6
        independent_A = np.column_stack([step(z, 0.) for z in np.eye(4)])
        independent_B = step(np.zeros(4), 1.)[:, None]
        np.testing.assert_allclose(A, independent_A, atol=2e-16, rtol=0)
        np.testing.assert_allclose(B, independent_B, atol=2e-16, rtol=0)

    def test_gain_matches_fixed_reference_residual_and_linear_stability(self):
        record = self.gain.to_dict()
        ref = json.loads((HERE/'oa1_config_proposed.json').read_text())['gain']
        np.testing.assert_allclose(record['K'], ref['K_reference'], atol=1e-10, rtol=1e-10)
        np.testing.assert_allclose(record['P'], ref['P_reference'], atol=1e-10, rtol=1e-10)
        self.assertLess(record['dare_relative_residual'], 2e-10)
        self.assertAlmostEqual(record['spectral_radius'], .9785039640, places=9)
        self.assertEqual(record['scientific_sha256'], oc.scientific_hash(self.config))
        self.assertEqual(self.gain.sha256, hashlib.sha256(self.gain.canonical_bytes()).hexdigest())
        record['K'][0][0] = 0.
        self.assertNotEqual(record['K'], self.gain.to_dict()['K'])

    def test_dare_failure_propagates_without_ridge_or_retry(self):
        with patch('scipy.linalg.solve_discrete_are', side_effect=np.linalg.LinAlgError('fixture')) as solve:
            with self.assertRaises(np.linalg.LinAlgError):
                om.construct_gain(self.config)
            self.assertEqual(solve.call_count, 1)

    def load_record(self, record, *, raw=None, expected_hash=None):
        content = raw if raw is not None else json.dumps(record, sort_keys=True, ensure_ascii=True,
                                                       separators=(',', ':'), allow_nan=False).encode()
        with tempfile.TemporaryDirectory(dir=HERE/'tests') as temp:
            path = Path(temp)/'gain.json'
            path.write_bytes(content)
            return om.load_gain(self.config, path, expected_sha256=expected_hash or hashlib.sha256(content).hexdigest())

    def test_gain_round_trip_requires_external_hash_and_canonical_bytes(self):
        self.assertEqual(self.load_record(self.gain.to_dict()).sha256, self.gain.sha256)
        with self.assertRaises(om.MathError) as raised:
            self.load_record(self.gain.to_dict(), expected_hash='0'*64)
        self.assertEqual(raised.exception.code, 'gain_hash')
        with self.assertRaises(om.MathError):
            self.load_record(self.gain.to_dict(), raw=self.gain.canonical_bytes()+b'\n')

    def test_gain_rejects_wrong_config_reference_shape_numbers_and_diagnostics(self):
        original = self.gain.to_dict()
        mutations = [lambda r: r.update(extra=True), lambda r: r.update(scientific_sha256='0'*64),
                     lambda r: r.update(reference_verified=False), lambda r: r.update(artifact_version=True),
                     lambda r: r['K'][0].__setitem__(0, 0.), lambda r: r['P'][0].__setitem__(0, -1.),
                     lambda r: r['A'][0].__setitem__(0, 2.), lambda r: r.update(B=[[0.]]),
                     lambda r: r['K'][0].__setitem__(0, True), lambda r: r.update(spectral_radius=1.01),
                     lambda r: r.update(dare_relative_residual=.1)]
        for mutate in mutations:
            record = copy.deepcopy(original); mutate(record)
            with self.subTest(record=record), self.assertRaises(om.MathError):
                self.load_record(record)

    def test_gain_json_missing_duplicate_and_nonfinite_are_rejected(self):
        with self.assertRaises(om.MathError) as raised:
            om.load_gain(self.config, HERE/'tests/no-such-gain.json', expected_sha256='0'*64)
        self.assertEqual(raised.exception.code, 'unavailable')
        for raw in (b'{"A":1,"A":2}', b'{"P":NaN}', b'\xff', b'['):
            with self.assertRaises(om.MathError):
                self.load_record(None, raw=raw)

    def test_local_origin_and_clipping_sign(self):
        z = np.zeros(4)
        self.assertEqual(self.local.input(z), (0., 0.))
        x = np.array([0., 0., 1., 0.])
        applied, raw = self.local.input(x)
        self.assertEqual(applied, 10.)
        self.assertGreater(raw, 10.)
        self.assertEqual(self.local.input(-x)[0], -10.)

    def test_local_safe_boundaries_and_dtype(self):
        x = np.array([2.4, 0., -np.pi/2, 0.])
        self.assertTrue(self.local.safe(x))
        x[0] = np.nextafter(2.4, np.inf)
        self.assertFalse(self.local.safe(x))
        x[0] = np.nan
        self.assertFalse(self.local.safe(x))
        with self.assertRaises(om.MathError):
            self.local.input(x)
        with self.assertRaises(om.MathError):
            self.local.safe(np.zeros(4, dtype=np.float32))
        with self.assertRaises(om.MathError) as raised:
            self.local.transition(np.zeros(4), 0., x)
        self.assertEqual(raised.exception.code, 'local_nonfinite')

    def test_local_decrease_records_exact_clipped_input_rule(self):
        x = np.array([.2, .03, .01, -.02])
        u, raw = self.local.input(x)
        nxt = .5*x  # SYNTHETIC prediction; no original-plant viability claim
        record = self.local.transition(x, u, nxt)
        P = np.array(self.gain.to_dict()['P'])
        v, vn = float(x@P@x), float(nxt@P@nxt)
        expected = -.5*(2*x[0]**2+30*x[1]**2+100*x[2]**2+10*x[3]**2+.03*u*u)+1e-10*max(1., v)
        self.assertEqual(record['lhs'], vn-v)
        self.assertEqual(record['rhs'], expected)
        self.assertEqual(record['raw_input'], raw)
        self.assertTrue(record['passed'])
        with self.assertRaises(om.MathError):
            self.local.transition(x, u+.01, nxt)

    def test_decrease_is_inclusive_and_near_zero_tolerance_is_preserved(self):
        x = np.zeros(4)
        # P00 * next_p^2 == 1e-10 to roundoff; test both sides explicitly.
        P00 = self.gain.to_dict()['P'][0][0]
        near = np.array([math_sqrt(1e-10/P00)*.999999, 0., 0., 0.])
        far = near.copy(); far[0] *= 1.00001
        self.assertTrue(self.local.transition(x, 0., near)['passed'])
        self.assertFalse(self.local.transition(x, 0., far)['passed'])
        self.assertTrue(self.local.transition(x, 0., x)['passed'])
        # Isolate exact lhs==rhs with synthetic V/c values; '<' must fail this.
        with patch.object(self.local, 'V', side_effect=[1e10, 1e10]), patch.object(self.local, 'state_cost', return_value=2.):
            equality = self.local.transition(x, 0., x)
        self.assertEqual(equality['lhs'], equality['rhs'])
        self.assertTrue(equality['passed'])

    def prediction(self, state, mode):
        steps = 60 if mode == 'entry' else 1
        return np.array([state*(.5**i) for i in range(steps+1)], dtype=np.float64)

    def test_entry_and_stay_thresholds_are_inclusive(self):
        # Q_angular_speed=10; sqrt thresholds bracketed by adjacent floats.
        for mode, threshold in (('entry', .75), ('stay', 1.)):
            x = np.zeros(4); x[3] = np.sqrt(threshold/10)
            while self.local.state_cost(x) > threshold:
                x[3] = np.nextafter(x[3], 0.)
            self.assertTrue(self.local.predicate(x, self.prediction(x, mode), mode=mode)['passed'])
            while self.local.state_cost(x) <= threshold:
                x[3] = np.nextafter(x[3], np.inf)
            self.assertFalse(self.local.predicate(x, self.prediction(x, mode), mode=mode)['passed'])
        # Exactly representable equality: cost = 2*.5^2 + 100*.05^2 = .75.
        x = np.array([.5, 0., .05, 0.])
        self.assertEqual(self.local.state_cost(x), .75)
        self.assertTrue(self.local.predicate(x, self.prediction(x, 'entry'), mode='entry')['passed'])

    def test_full_entry_and_one_step_stay_contract(self):
        x = np.array([.1, 0., 0., 0.])
        entry = self.local.predicate(x, self.prediction(x, 'entry'), mode='entry')
        stay = self.local.predicate(x, self.prediction(x, 'stay'), mode='stay')
        self.assertEqual(len(entry['transitions']), 60)
        self.assertEqual(len(stay['transitions']), 1)
        self.assertTrue(entry['passed'] and stay['passed'])
        self.assertAlmostEqual(entry['minimum_rail_margin_m'], 2.3)
        self.assertEqual(entry['minimum_angle_margin_rad'], np.pi/2)
        self.assertAlmostEqual(entry['minimum_safety_margin'], 2.3/2.4)
        with self.assertRaises(om.MathError):
            self.local.predicate(x, self.prediction(x, 'stay'), mode='entry')
        wrong = self.prediction(x, 'entry'); wrong[0, 0] += 1e-5
        with self.assertRaises(om.MathError):
            self.local.predicate(x, wrong, mode='entry')

    def test_predicted_safety_decrease_and_nonfinite_override_threshold(self):
        x = np.array([.1, 0., 0., 0.])
        for sample in (1, 30, 60):
            pred = self.prediction(x, 'entry'); pred[sample, 0] = 2.5
            self.assertFalse(self.local.predicate(x, pred, mode='entry')['passed'])
        pred = np.repeat(x[None, :], 61, axis=0)
        self.assertFalse(self.local.predicate(x, pred, mode='entry')['passed'])
        pred[10, 2] = np.nan
        with self.assertRaises(om.MathError) as error:
            self.local.predicate(x, pred, mode='entry')
        self.assertEqual(error.exception.code, 'local_nonfinite')


def math_sqrt(value):
    return value**.5


if __name__ == '__main__':
    unittest.main()
