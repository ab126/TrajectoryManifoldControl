"""Bounded development diagnosis, not calibration or a recipe search.

Fixed position-cost candidate Qp=2 versus submitted Qp=.02 for local tests.
At 20 fixed states compare common-prefix one/five CEM starts and two warm starts.
Submitted source and original plant remain unchanged. Supervised by run_diagnostics.
"""
from common import *
import argparse
from collections import Counter
import signal
from scipy.linalg import solve_discrete_are
from unittest.mock import patch
from src.inverted_pendulum import rk4_step, linearize_upright_dynamics
from src.manifold_control import CanonicalBehaviorDecoder, DataDrivenCEMSolver
from models import OracleTransition, check_oracle
from gate_a import strict_physical

STOP = False


def save_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def local_gain(Q, h):
    Ac, Bc = linearize_upright_dynamics(2.)
    A = sum(h**j/fact*np.linalg.matrix_power(Ac, j) for j, fact in enumerate((1, 1, 2, 6, 24)))
    B = sum(h**(j+1)/fact*np.linalg.matrix_power(Ac, j) for j, fact in enumerate((1, 2, 6, 24))) @ Bc
    R = np.array([[.03]])
    P = solve_discrete_are(A, B, Q, R)
    K = np.linalg.solve(R+B.T@P@B, B.T@P@A)
    return K, np.linalg.eigvals(A-B@K)


def fixed_states(run_dir):
    audit = json.loads((run_dir/'calibration/independent_audit.json').read_text())
    rows = audit['case_diagnostics']
    selected = [1, 2, 9, 0, 22, 72, 89]
    initial_ids = selected + [row['case'] for row in rows if row['case'] not in selected]
    states = []
    for case in initial_ids[:12]:
        with np.load(run_dir/'calibration'/f'case_{case:03d}.npz') as trace:
            states.append({'label': f'case_{case:03d}_initial', 'case': case, 'sample': 0, 'x': trace['X'][0].tolist()})
    unsafe = sorted((row for row in rows if row['first_angle_sample'] is not None or row['first_rail_sample'] is not None),
                    key=lambda row: (min(v for v in [row['first_angle_sample'], row['first_rail_sample']] if v is not None), row['case']))
    for row in unsafe[:6]:
        first = min(v for v in [row['first_angle_sample'], row['first_rail_sample']] if v is not None)
        k = max(0, first-1)
        with np.load(run_dir/'calibration'/f'case_{row["case"]:03d}.npz') as trace:
            states.append({'label': f'case_{row["case"]:03d}_pre_violation_{k}', 'case': row['case'], 'sample': k, 'x': trace['X'][k].tolist()})
    states.extend({'label': f'pure_position_{sign:+d}', 'case': f'pure_position_{sign:+d}', 'sample': 0,
                   'x': [float(sign), 0., 0., 0.]} for sign in (-1, 1))
    assert len(states) == 20
    return states


def objective_parts(X, U):
    Qdiag = np.array([.02, 30., 100., 10.])
    per_state = np.sum(X[:-1]**2 * Qdiag, axis=0)
    terminal = 12*float(np.sum(X[-1]**2*Qdiag))
    control = .03*float(np.sum(U**2))
    rate = .002*float(np.sum(np.diff(U, axis=0)**2))
    return {'position': float(per_state[0]), 'speed': float(per_state[1]), 'angle': float(per_state[2]),
            'angular_speed': float(per_state[3]), 'terminal': terminal, 'control': control, 'rate': rate,
            'total': float(np.sum(per_state))+terminal+control+rate}


def main(args):
    global STOP
    configure()
    begin = time.perf_counter()
    output = Path(args.output_dir).resolve(); output.mkdir(parents=True, exist_ok=True)
    if not output.is_relative_to(HERE if 'HERE' in globals() else ROOT/'validation'):
        raise ValueError('Diagnostic outputs must remain under validation/')
    run_dir = Path(args.run_dir).resolve()
    config = json.loads((run_dir/'calibration/config.json').read_text())
    h = config['dt_normalized']
    Q = np.diag([.02, 30., 100., 10.]); K, poles = local_gain(Q, h)
    np.testing.assert_allclose(K, config['LQR_gain'], rtol=1e-12, atol=1e-12)
    states = fixed_states(run_dir)
    save_json(output/'config.json', {'mode': args.mode, 'device': args.device, 'states': states,
              'local_position_candidate_Qp': 2., 'candidate_fixed_before_execution': True,
              'CEM': {'population': 256, 'elites': 32, 'iterations': 3, 'horizon': 60, 'starts': [1, 5]},
              'CEM_warm_starts': ['zeros', 'oracle_local_rollout'], 'root_seed': SEED,
              'scope': 'diagnosis only; no calibration, tuning search, box shrink, H changes or dependent stages',
              'historical_warm_start_limitation': 'full historical plans were not stored; tests start from explicit zero/local-rollout warm starts, not reconstructed historical plans',
              'dtype': 'float64 oracle/plant; original float32 RNG tensors cast to float64 for replay',
              'determinism': determinism_manifest()})
    records = {'local': [], 'cem': [], 'incomplete': False, 'new_closed_loop_diagnostic_plant_samples': 0}
    def checkpoint():
        records['analysis_wall_seconds'] = time.perf_counter()-begin
        save_json(output/'results.json', records)
    def guard():
        if STOP:
            records['incomplete'] = True; checkpoint(); raise RuntimeError('Stopped; partial diagnostic checkpoint saved')
    if args.mode in ('local', 'all'):
        local_states = [{'label': f'pure_p_{p:+g}', 'x': [p, 0., 0., 0.]} for p in [-1., -.5, .5, 1.]]
        local_states += [state for state in states[:3]]
        for recipe, matrix in [('submitted_weights', Q), ('fixed_position_candidate', np.diag([2., 30., 100., 10.]))]:
            gain, eigenvalues = local_gain(matrix, h)
            for state in local_states:
                guard(); X = np.empty((1001, 4)); U = np.empty((1000, 1)); X[0] = state['x']
                for k in range(1000):
                    guard(); U[k] = np.clip(-gain@X[k], -10, 10)
                    X[k+1] = rk4_step(X[k], float(U[k, 0]), h, 2.)
                    records['new_closed_loop_diagnostic_plant_samples'] += 1
                np.savez_compressed(output/f'local_{recipe}_{state["label"]}.npz', X=X, U=U)
                physical = X*np.array([1., np.sqrt(9.81), 1., np.sqrt(9.81)])
                records['local'].append({'label': state['label'], 'recipe': recipe, 'strict_physical_success': strict_physical(X),
                                        'safety_passed': bool(np.all(abs(X[:, 0])<=2.4) and np.all(abs(X[:, 2])<=np.pi/2)),
                                        'terminal_max_abs_physical': np.max(abs(physical[-151:]), axis=0).tolist(),
                                        'gain': gain.tolist(), 'poles': [[float(z.real), float(z.imag)] for z in eigenvalues],
                                        'slow_mode_time_constant_s': float(-.02/np.log(max(abs(eigenvalues)))),
                                        'note': 'local-only diagnostic; local gain applied even outside the original switch region; not a hybrid success estimate'})
                checkpoint()
                print(f'Local {recipe}: {state["label"]}, strict={strict_physical(X)}', flush=True)
    if args.mode in ('cem', 'all'):
        if args.device != 'cuda' or not torch.cuda.is_available():
            raise RuntimeError('CUDA required for CEM diagnosis; no CPU fallback')
        check_oracle('cuda')
        device = 'cuda'; transition = OracleTransition(device)
        Qt = torch.tensor(Q, dtype=torch.float64, device=device)
        solver = DataDrivenCEMSolver(CanonicalBehaviorDecoder(transition, 60), Qt,
                                    torch.tensor([[.03]], dtype=torch.float64, device=device),
                                    Q_terminal=12*Qt, u_bounds=(-10, 10), population=256, elite=32,
                                    iterations=3, input_rate_weight=.002, seed=SEED)
        gain = torch.tensor(K, dtype=torch.float64, device=device)
        for state_spec in states:
            for warm_kind in ('zeros', 'oracle_local_rollout'):
                guard(); x = torch.tensor(state_spec['x'], dtype=torch.float64, device=device)
                warm = torch.zeros(60, 1, dtype=torch.float64, device=device)
                if warm_kind == 'oracle_local_rollout':
                    predicted = x.clone()
                    for k in range(60):
                        warm[k] = (-gain@predicted).clamp(-10, 10)
                        predicted = transition(predicted, warm[k])
                warm_X = solver.decoder.rollout(x, warm).cpu().numpy()
                warm_cost = objective_parts(warm_X, warm.cpu().numpy())['total']
                attempts = []
                for attempt in range(5):
                    guard(); attempt_warm = warm.clone()
                    if attempt:
                        generator = torch.Generator(device='cpu').manual_seed(indexed_seed('oracle_start', state_spec['case'], state_spec['sample'], attempt))
                        perturbation = 5*torch.randn(warm.shape, generator=generator, dtype=torch.float64).to(device)
                        attempt_warm = (warm+perturbation).clamp(-10, 10)
                    candidate_warm_cost = objective_parts(solver.decoder.rollout(x, attempt_warm).cpu().numpy(), attempt_warm.cpu().numpy())['total']
                    draws = normals(['oracle', state_spec['case'], attempt], state_spec['sample'], horizon=60, device=device).double()
                    iterator = iter(draws); sync(device); solve_start = time.perf_counter()
                    with patch('torch.randn', side_effect=lambda shape, **kw: next(iterator)):
                        solution = solver.solve(x, attempt_warm)
                    sync(device); duration = time.perf_counter()-solve_start
                    prediction, controls = solution.x.cpu().numpy(), solution.u.cpu().numpy()
                    parts = objective_parts(prediction, controls)
                    if solution.finite:
                        np.testing.assert_allclose(parts['total'], solution.loss, rtol=1e-9, atol=1e-8)
                        assert solution.loss <= candidate_warm_cost + 1e-7*max(1., abs(candidate_warm_cost))
                    attempts.append({'attempt': attempt, 'finite': solution.finite, 'loss': solution.loss,
                                     'objective_parts': parts, 'solve_seconds': duration,
                                     'warm_candidate_cost': candidate_warm_cost,
                                     'predicted_rail_safe': bool(np.all(abs(prediction[:, 0])<=2.4)),
                                     'predicted_angle_safe': bool(np.all(abs(prediction[:, 2])<=np.pi/2)),
                                     'first_force_N': float(controls[0, 0]*9.81)})
                    np.savez_compressed(output/f'cem_{state_spec["label"]}_{warm_kind}_{attempt}.npz', X=prediction, U=controls)
                best_index = min(range(5), key=lambda i: attempts[i]['loss'])
                records['cem'].append({'state': state_spec, 'warm_kind': warm_kind, 'initial_warm_cost': warm_cost,
                                       'attempts': attempts, 'one_start_loss': attempts[0]['loss'],
                                       'five_start_loss': attempts[best_index]['loss'], 'best_attempt': best_index})
                checkpoint()
                print(f'CEM {state_spec["label"]} {warm_kind}: one={attempts[0]["loss"]:.3g}, five={attempts[best_index]["loss"]:.3g}', flush=True)
    checkpoint()
    print(json.dumps({'local_tests': len(records['local']), 'cem_state_warm_pairs': len(records['cem']),
                      'new_closed_loop_diagnostic_plant_samples': records['new_closed_loop_diagnostic_plant_samples']}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['local', 'cem', 'all'], required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    def stop(*unused):
        global STOP
        STOP = True
    signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
    main(args)
