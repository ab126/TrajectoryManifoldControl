"""Review existing Gate A traces; no controller solves or plant acquisition.

Use the private validation Python with its native-library launcher settings.
Writes new review artifacts outside the immutable overnight run directory.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import numpy as np

HERE = Path(__file__).resolve().parent


def sha256(path):
    value = hashlib.sha256()
    with path.open('rb') as file:
        for block in iter(lambda: file.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def quantiles(values):
    array = np.asarray(values, dtype=float)
    if not len(array):
        return None
    return dict(zip(('minimum', 'median', 'p95', 'p99', 'maximum'),
                    map(float, np.quantile(array, [0, .5, .95, .99, 1]))))


def review(run_dir):
    started = time.perf_counter()
    run_dir = Path(run_dir).resolve()
    calibration = run_dir / 'calibration'
    run = json.loads((run_dir / 'run.json').read_text())
    config = json.loads((calibration / 'config.json').read_text())
    summary = json.loads((calibration / 'summary.json').read_text())
    audit = json.loads((calibration / 'independent_audit.json').read_text())
    checksum_manifest = json.loads((run_dir / 'checksums.json').read_text())
    checksum_mismatches = [name for name, expected in checksum_manifest.items()
                           if not (run_dir / name).is_file() or sha256(run_dir / name) != expected]
    source_mismatches = [name for name, expected in run['submitted_source_hashes'].items()
                        if not (HERE.parent.parent / name).is_file()
                        or sha256(HERE.parent.parent / name) != expected]
    validation_mismatches = [name for name, expected in run['validation_source_hashes'].items()
                            if not (HERE / name).is_file() or sha256(HERE / name) != expected]
    branches = Counter()
    durations, cases = [], []
    for row in audit['case_diagnostics']:
        case = row['case']
        with np.load(calibration / f'case_{case:03d}.npz', allow_pickle=False) as trace:
            X, U, flags = trace['X'], trace['U'], trace['branches']
            durations.extend(trace['solve_seconds'].tolist())
            branches.update(flags.tolist())
            physical = X * [1., np.sqrt(9.81), 1., np.sqrt(9.81)]
            tail_max = np.max(np.abs(physical[-151:]), axis=0)
            local = np.flatnonzero(flags == 'local')
            cases.append({
                **row, 'initial_normalized': X[0].tolist(),
                'final_physical': physical[-1].tolist(),
                'tail_max_abs_physical': tail_max.tolist(),
                'first_local_s': float(local[0] * .02) if len(local) else None,
                'saturated_action_fraction': float(np.mean(np.abs(U[:, 0]) >= 10 - 1e-12)),
            })
    complete = audit['calibration_complete'] and not summary['incomplete']
    safe = [row for row in cases if not any(name in row['failure_flags']
                                          for name in ('finite', 'rail', 'angle'))]
    position_only_safe = [row['case'] for row in safe
                          if row['failure_flags'] == ['terminal_position']]
    safe_by_failures = dict(Counter(tuple(row['failure_flags']) for row in safe))
    h = config['dt_normalized']
    # Analytic upright Jacobians for the ORIGINAL uniform-rod plant, M/m=2.
    Ac = np.array([[0., 1., 0., 0.], [0., 0., -1/3, 0.],
                   [0., 0., 0., 1.], [0., 0., 2., 0.]])
    Bc = np.array([[0.], [4/9], [0.], [-2/3]])
    A = sum(h**j / fact * np.linalg.matrix_power(Ac, j)
            for j, fact in enumerate((1, 1, 2, 6, 24)))
    B = sum(h**(j + 1) / fact * np.linalg.matrix_power(Ac, j)
            for j, fact in enumerate((1, 2, 6, 24))) @ Bc
    F = A - B @ np.asarray(config['LQR_gain'])
    eigenvalues = np.linalg.eigvals(F)
    linear_samples = np.array([np.linalg.matrix_power(F, k) @ [1., 0., 0., 0.]
                               for k in range(1001)])
    result = {
        'run_directory': str(run_dir), 'review_scope': 'existing traces and linear algebra only; no new experimental run',
        'integrity': {'checksums_checked': len(checksum_manifest), 'checksum_mismatches': checksum_mismatches,
                      'submitted_source_mismatches': source_mismatches,
                      'validation_source_snapshot_mismatches': validation_mismatches,
                      'note': 'validation documentation can legitimately change after this review; run snapshot remains authoritative'},
        'complete': complete, 'Gate_A_passed': audit['Gate_A_passed'],
        'audit_passed': audit['audit_passed'], 'plant_replay_max_abs_error': audit['plant_replay_max_abs_error'],
        'successes': [row['case'] for row in cases if row['success']],
        'success_count': sum(row['success'] for row in cases), 'case_count': len(cases),
        'angle_halves': [{'positive_angle': sign,
                         'cases': sum(row['positive_angle'] == sign for row in cases),
                         'successes': sum(row['success'] and row['positive_angle'] == sign for row in cases)}
                        for sign in (False, True)],
        'failure_counts_overlapping': audit['failure_counts'],
        'throughout_safe_cases': len(safe), 'safe_position_only_failure_cases': position_only_safe,
        'safe_failure_combinations': [{'failures': list(key), 'cases': value} for key, value in safe_by_failures.items()],
        'branches': dict(branches), 'total_executed_samples': sum(branches.values()),
        'replan_fraction': branches['plan'] / sum(branches.values()),
        'local_fraction': branches['local'] / sum(branches.values()),
        'solve_latency_seconds': {'n': len(durations), 'mean': float(np.mean(durations)), **quantiles(durations)},
        'first_violation_seconds': {name: quantiles([row[name] * .02 for row in cases if row[name] is not None])
                                   for name in ('first_rail_sample', 'first_angle_sample')},
        'local_linearization': {
            'closed_loop_eigenvalues': [[float(z.real), float(z.imag)] for z in eigenvalues],
            'slow_mode_time_constant_physical_s': float(-.02 / np.log(max(abs(eigenvalues)))),
            'initial_normalized': [1., 0., 0., 0.],
            'initial_tracking_cost': .02,
            'max_linear_tracking_cost_20s': float(np.max(np.einsum('bi,ij,bj->b', linear_samples, config['cost_Q'], linear_samples))),
            'p_at_17s_m': float(linear_samples[850, 0]), 'p_at_20s_m': float(linear_samples[1000, 0]),
            'limitation': 'linearized diagnostic, not a new nonlinear closed-loop test or attainability proof'},
        'jobs': [{k: row[k] for k in ('job', 'returncode', 'wall_s', 'timeout', 'cancelled')} for row in run['jobs']],
        'usage': {'overnight_worker_hours': run['worker_seconds'] / 3600,
                  'readiness_worker_hours': run['prior_stage0_seconds'] / 3600,
                  'total_stage0_worker_hours': (run['worker_seconds'] + run['prior_stage0_seconds']) / 3600,
                  'allocated_hours': run['stage0_cap_hours'],
                  'calibration_worker_hours': summary['worker_seconds'] / 3600},
        'cases': cases,
        'input_manifests_SHA256': {name: sha256(run_dir / name) for name in
                                  ('run.json', 'checksums.json', 'calibration/config.json',
                                   'calibration/summary.json', 'calibration/independent_audit.json')},
    }
    result['review_analysis_wall_seconds'] = time.perf_counter() - started
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_directory')
    args = parser.parse_args()
    result = review(args.run_directory)
    destination = HERE / 'reviews' / Path(args.run_directory).name
    destination.mkdir(parents=True, exist_ok=True)
    (destination / 'trace_review.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'cases'}, indent=2))
    if result['integrity']['checksum_mismatches'] or result['integrity']['submitted_source_mismatches']:
        raise SystemExit('Saved evidence integrity failed; do not rely on its conclusions')
