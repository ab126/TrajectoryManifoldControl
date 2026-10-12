"""Read-only evidence review: no SUT imports, solves, training or trajectories.

NumPy rescoring and pointwise algebra on saved states only. Missing archived
files are reported as unavailable; existing files with wrong hashes fail.
Writes a new review manifest, never the historical runs or their reports.
"""
import hashlib
import argparse
import json
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.linalg import solve_discrete_are

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEST = HERE / "reviews/oracle_amendment_v1/evidence.json"
inputs = {}
unavailable = []


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def remember(path):
    inputs[str(path.relative_to(ROOT))] = digest(path)


def read(path):
    remember(path)
    return json.loads(path.read_text())


def verify_manifest(base, manifest):
    counts = Counter()
    for name, expected in manifest.items():
        path = base / name
        if not path.exists():
            unavailable.append(str(path.relative_to(ROOT)))
            counts['unavailable'] += 1
            continue
        remember(path)
        assert digest(path) == expected, path
        counts['matched'] += 1
    return dict(counts)


def arrays(path):
    if not path.exists():
        unavailable.append(str(path.relative_to(ROOT)))
        return None
    remember(path)
    with np.load(path, allow_pickle=False) as saved:
        return {name: saved[name] for name in saved.files}


def flags(X):
    assert X.shape == (1001, 4)
    Y = X * [1, np.sqrt(9.81), 1, np.sqrt(9.81)]
    finite = np.isfinite(Y).all()
    end = np.max(np.abs(Y[850:1001]), axis=0)
    angular = finite and end[2] <= .02 and end[3] <= .05
    speed = angular and end[1] <= .05
    settled = speed and end[0] <= .1
    rail = finite and np.max(np.abs(Y[:, 0])) <= 2.4
    angle = finite and np.max(np.abs(Y[:, 2])) <= np.pi / 2
    return dict(finite=bool(finite), angular=bool(angular), speed=bool(speed),
                settled=bool(settled), rail=bool(rail), angle=bool(angle),
                strict=bool(settled and rail and angle))


def saved_step_residual(X, U):
    # Independently solve the normalized uniform-rod mass matrix in closed form.
    # Evaluate each SAVED state separately; no generated state feeds the next.
    def rhs(z):
        c = .5 * np.cos(z[:, 2])
        a = U[:, 0] + .5 * np.sin(z[:, 2]) * z[:, 3]**2
        b = .5 * np.sin(z[:, 2])
        d = 1 - c*c
        return np.column_stack((z[:, 1], (a/3-c*b)/d, z[:, 3], (3*b-c*a)/d))
    h = .02 * np.sqrt(9.81)
    z = X[:-1]
    k1 = rhs(z)
    k2 = rhs(z + h*k1/2)
    k3 = rhs(z + h*k2/2)
    k4 = rhs(z + h*k3)
    expected = z + h*(k1+2*k2+2*k3+k4)/6
    np.testing.assert_allclose(expected, X[1:], atol=1e-10, rtol=1e-10)
    return float(np.max(np.abs(expected-X[1:])))


def main(destination=DEST):
    assert destination.is_relative_to(HERE), 'Review output must stay under validation/'
    assert not destination.exists(), 'Choose a new filename; preserve completed reviews'
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    status = subprocess.check_output(['git', 'status', '--porcelain=v1'], cwd=ROOT, text=True)
    changes = subprocess.check_output(['git', 'diff', '--name-only', '17d42cd', 'HEAD'], cwd=ROOT, text=True).splitlines()
    assert all(p.startswith('validation/') for p in changes)
    source = {}
    original = read(HERE/'artifacts/source_hashes_initial.json')
    for name in ('scripts/run_inverted_pendulum_mc.py', 'src/manifold_control.py', 'src/inverted_pendulum.py'):
        path = ROOT/name
        remember(path)
        value = digest(path)
        assert value == original['TrajectoryManifoldControl/'+name]
        submitted = subprocess.check_output(['git', 'show', '325beae:'+name], cwd=ROOT)
        assert hashlib.sha256(submitted).hexdigest() == value
        source[name] = value
    recorded = {name: read(HERE/'artifacts'/name) for name in
                ('plant_checks.json', 'equivalence.json', 'equivalence_cuda.json', 'existing_tests.json')}
    for name in ('plant_checks.json', 'equivalence.json', 'equivalence_cuda.json'):
        assert recorded[name]['passed']
    assert recorded['existing_tests.json']['exit_code'] == 0
    log = HERE/'artifacts/existing_tests.log'
    if log.exists():
        remember(log)
        assert '29 passed, 1 warning' in log.read_text()
    else:
        unavailable.append(str(log.relative_to(ROOT)))

    run = HERE/'overnight_runs/20261007T054549.695689Z'
    checksums = verify_manifest(run, read(run/'checksums.json'))
    calibration = run/'calibration'
    config = read(calibration/'config.json')
    initial_path = calibration/'initial_states.npy'
    seed = int.from_bytes(hashlib.sha256(json.dumps([20261006, 'gate_a_cases', 1.0], separators=(',', ':')).encode()).digest()[:8], 'little') % (2**63-1)
    assert seed == config['case_seed']
    expected_initial = np.random.default_rng(seed).uniform(-1, 1, (200, 4))*[1.5, 2.5, 1.3, 3.5]
    if initial_path.exists():
        remember(initial_path)
        initial = np.load(initial_path, allow_pickle=False)
        np.testing.assert_array_equal(initial, expected_initial)
    else:
        unavailable.append(str(initial_path.relative_to(ROOT)))
        initial = None
    rows = read(calibration/'cases.json')
    totals, categories, branches = Counter(), Counter(), Counter()
    ids, signs = [], {False: [0, 0], True: [0, 0]}
    residual = 0.
    for row in rows:
        saved = arrays(calibration/f'case_{row["case"]:03d}.npz')
        if saved is None:
            continue
        X, U = saved['X'], saved['U']
        np.testing.assert_array_equal(X[0], expected_initial[row['case']])
        assert U.shape == (1000, 1) and np.isfinite(U).all() and np.max(np.abs(U)) <= 10
        f = flags(X)
        assert f['strict'] == row['strict_physical_success']
        totals.update({k: int(v) for k, v in f.items()})
        totals['reviewed_cases'] += 1
        categories[(f['rail'], f['angle'])] += 1
        branches.update(saved['branches'].tolist())
        sign = bool(X[0, 2] >= 0)
        signs[sign][0] += int(f['strict'])
        signs[sign][1] += 1
        if f['strict']:
            ids.append(row['case'])
        residual = max(residual, saved_step_residual(X, U))
    if totals['reviewed_cases'] == 200:
        assert [totals[k] for k in ('strict', 'angular', 'speed', 'settled')] == [3, 147, 67, 12]
        assert ids == [22, 72, 89]

    local = HERE/'diagnostics/20261007T175948.296063Z'
    local_rows = read(local/'results.json')['local']
    local_counts = Counter()
    gains = {}
    for row in local_rows:
        gains[row['recipe']] = row['gain']
        saved = arrays(local/f'local_{row["recipe"]}_{row["label"]}.npz')
        if saved is None:
            continue
        f = flags(saved['X'])
        assert f['strict'] == row['strict_physical_success'] and f['rail'] and f['angle']
        np.testing.assert_allclose(saved['U'][:, 0], np.clip(-saved['X'][:-1] @ np.asarray(row['gain'])[0], -10, 10), atol=1e-12)
        local_counts[row['recipe']+'_reviewed'] += 1
        local_counts[row['recipe']+'_successes'] += int(f['strict'])

    cem = HERE/'diagnostics/20261008T190703.853434Z'
    old_review = read(HERE/'diagnostics/cem_review.json')
    diagnostic_config = read(cem/'config.json')
    runtime = read(cem/'cuda_runtime_check.json')
    assert runtime['passed'] and runtime['device'] == 'NVIDIA L4'
    for state in diagnostic_config['states']:
        if isinstance(state['case'], int):
            original_saved = arrays(calibration/f'case_{state["case"]:03d}.npz')
            if original_saved is not None:
                np.testing.assert_array_equal(state['x'], original_saved['X'][state['sample']])
        else:
            np.testing.assert_array_equal(state['x'], [int(state['label'].split('_')[-1]), 0, 0, 0])
    cem_hashes = verify_manifest(HERE, old_review['input_sha256'])
    snapshot_checks = {}
    for directory in (local, cem):
        job = read(directory/'job.json')
        assert job['returncode'] == 0 and job['status'] == 'finished'
        snapshot_checks[directory.name] = verify_manifest(directory/'source_snapshots', job['source_SHA256'])
    results = read(cem/'results.json')
    assert not results['incomplete'] and results['new_closed_loop_diagnostic_plant_samples'] == 0
    groups = {w: Counter() for w in ('zeros', 'oracle_local_rollout')}
    max_error, max_residual, durations = 0., 0., []
    for pair in results['cem']:
        group = groups[pair['warm_kind']]
        assert pair['state'] in diagnostic_config['states']
        safe = []
        for attempt in pair['attempts']:
            saved = arrays(cem/f'cem_{pair["state"]["label"]}_{pair["warm_kind"]}_{attempt["attempt"]}.npz')
            if saved is None:
                safe.append(None)
                continue
            X, U = saved['X'], saved['U']
            assert X.shape == (61, 4) and U.shape == (60, 1)
            assert np.isfinite(X).all() and np.isfinite(U).all() and attempt['finite']
            np.testing.assert_array_equal(X[0], pair['state']['x'])
            assert np.max(np.abs(U)) <= 10
            parts = np.sum(X[:-1]**2 * [.02, 30, 100, 10], axis=0)
            end = 12*np.sum(X[-1]**2 * [.02, 30, 100, 10])
            control, rate = .03*np.sum(U**2), .002*np.sum(np.diff(U, axis=0)**2)
            objective = float(parts.sum()+end+control+rate)
            expected = attempt['objective_parts']
            np.testing.assert_allclose(list(parts)+[end, control, rate], [expected[k] for k in
                ('position', 'speed', 'angle', 'angular_speed', 'terminal', 'control', 'rate')], atol=1e-8, rtol=1e-9)
            np.testing.assert_allclose(objective, attempt['loss'], atol=1e-8, rtol=1e-9)
            max_error = max(max_error, abs(objective-attempt['loss']))
            max_residual = max(max_residual, saved_step_residual(X, U))
            rail, angle = np.max(np.abs(X[:, 0])) <= 2.4, np.max(np.abs(X[:, 2])) <= np.pi/2
            assert bool(rail) == attempt['predicted_rail_safe'] and bool(angle) == attempt['predicted_angle_safe']
            assert attempt['loss'] <= attempt['warm_candidate_cost'] + 1e-7*max(1, abs(attempt['warm_candidate_cost']))
            np.testing.assert_allclose(U[0, 0]*9.81, attempt['first_force_N'], atol=1e-10)
            if (pair['state']['label'].startswith('pure_position') and pair['warm_kind'] == 'zeros'
                    and attempt['attempt'] in (0, pair['best_attempt'])):
                np.testing.assert_array_equal(U, np.zeros((60, 1)))
                np.testing.assert_array_equal(X, np.tile(pair['state']['x'], (61, 1)))
                np.testing.assert_allclose(objective, 60*.02+12*.02, atol=1e-14)
            safe.append(bool(rail and angle))
            durations.append(attempt['solve_seconds'])
            group['reviewed_solves'] += 1
        best = min(range(5), key=lambda i: pair['attempts'][i]['loss'])
        assert best == pair['best_attempt']
        group['pairs'] += 1
        group['cost_improved'] += int(pair['five_start_loss'] < pair['one_start_loss']*(1-1e-12))
        if all(s is not None for s in safe):
            group['one_safe'] += int(safe[0])
            group['five_safe'] += int(safe[best])
            group['safety_changed'] += int(safe[0] != safe[best])

    # Fixed-matrix algebra only: independently verify the already-declared gain.
    Ac = np.array([[0,1,0,0],[0,0,-1/3,0],[0,0,0,1],[0,0,2,0]], dtype=float)
    Bc = np.array([[0],[4/9],[0],[-2/3]])
    h = .02*np.sqrt(9.81)
    A = sum(h**j/den*np.linalg.matrix_power(Ac,j) for j,den in enumerate((1,1,2,6,24)))
    B = sum(h**(j+1)/den*np.linalg.matrix_power(Ac,j) for j,den in enumerate((1,2,6,24))) @ Bc
    algebra = {}
    for recipe, q in [('submitted_weights', .02), ('fixed_position_candidate', 2.)]:
        Q, R = np.diag([q,30,100,10]), np.array([[.03]])
        P = solve_discrete_are(A,B,Q,R)
        K = np.linalg.solve(R+B.T@P@B, B.T@P@A)
        np.testing.assert_allclose(K, gains[recipe], atol=1e-12, rtol=1e-12)
        poles = np.linalg.eigvals(A-B@K)
        algebra[recipe] = {'K': K.tolist(), 'P': P.tolist(), 'spectral_radius': float(np.max(np.abs(poles))),
            'slow_time_constant_physical_s': float(-.02/np.log(np.max(np.abs(poles))))}
    readiness = sum(r['worker_seconds'] for r in map(json.loads, (HERE/'artifacts/usage.jsonl').read_text().splitlines()))
    overnight = read(run/'run.json')['worker_seconds']
    diagnosis = sum(r['worker_seconds'] for r in map(json.loads, (HERE/'diagnostics/usage.jsonl').read_text().splitlines()))
    notebook = read(HERE/'notebooks/gate_a_examples_source.ipynb')
    assert all(not cell.get('outputs') and cell.get('execution_count') is None for cell in notebook['cells'] if cell['cell_type']=='code')
    output = {'review_version': 1, 'reviewed_HEAD': head, 'status_at_review': status,
        'scope': 'saved-array arithmetic, hashes and fixed Riccati algebra only; zero new experiments; same-author review',
        'changes_since_17d42cd': changes, 'submitted_source_SHA256': source,
        'recorded_correctness': recorded, 'original_run_checksums': checksums,
        'gate_a': {'counts': dict(totals), 'success_ids': ids, 'angle_halves_success_total': {str(k):v for k,v in signs.items()},
                   'branches': dict(branches), 'pointwise_RK4_max_abs_residual': residual},
        'local': dict(local_counts), 'cem': {'groups': {k:dict(v) for k,v in groups.items()},
            'hashes': cem_hashes, 'source_snapshots': snapshot_checks, 'max_cost_discrepancy': max_error,
            'pointwise_RK4_max_abs_residual': max_residual, 'reviewed_solves': len(durations)},
        'gain_algebra': algebra, 'usage': {'readiness_s':readiness,'overnight_s':overnight,'diagnosis_s':diagnosis,
            'stage0_hours': (readiness+overnight+diagnosis)/3600, 'remaining_15h':15-(readiness+overnight+diagnosis)/3600},
        'output_free_notebook_verified': True,
        'original_seed_and_available_case_provenance_verified': True,
        'diagnostic_available_state_provenance_verified': True,
        'all_raw_review_inputs_available': not unavailable,
        'unavailable': unavailable, 'input_SHA256': inputs}
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k:output[k] for k in ('reviewed_HEAD','gate_a','local','cem','usage','unavailable')}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEST)
    main(parser.parse_args().output.resolve())
