"""Compare saved terminal balance with balance requiring no horizontal crossing.

Reads completed original pendulum CSVs only. No training, simulation, or changes
to primary metrics. Uses the unchanged supervisor's usage/credit telemetry.
"""
import csv
import hashlib
import importlib.util
import json
import os
from collections import defaultdict
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[3]
VALIDATION = ROOT / 'validation'
OUT = VALIDATION / 'results_literature/horizontal_sensitivity'
SOURCES = ['E0/summary.csv', 'E0/authored_cases.csv', 'E1/summary.csv',
           'E3/summary.csv', 'E4/summary.csv', 'E6_SMC/summary.csv']


def boolean(value):
    if str(value).lower() not in ('true', 'false'):
        raise ValueError(f'Missing or invalid recorded boolean: {value!r}')
    return str(value).lower() == 'true'


def usage_gate():
    guard_path = VALIDATION / 'codex_supervisor.py'
    manifest = json.loads((VALIDATION / 'results_literature/manifest.json').read_text())
    assert hashlib.sha256(guard_path.read_bytes()).hexdigest() == manifest['protected_source_hashes']['validation/codex_supervisor.py']
    spec = importlib.util.spec_from_file_location('original_guard', guard_path)
    guard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guard)
    limits = guard.latest_limits()
    if limits is None or guard.balance(limits) is None:
        raise RuntimeError('Missing usage telemetry; preserve work and stop.')
    if guard.balance(limits) < manifest['credit_balance_baseline'] - 1e-6:
        raise RuntimeError('Credit guard: stop.')
    used = [guard.window_state(limits.get(k))[0] for k in ('primary', 'secondary')]
    if used[0] >= 85 or used[1] >= 90:
        raise RuntimeError(f'Usage pause: {used}')
    return dict(five_hour_percent=used[0], weekly_percent=used[1], credits=guard.balance(limits))


def main():
    assert set(os.sched_getaffinity(0)) <= set(range(80, 128))
    started = time.process_time()
    telemetry = usage_gate()
    groups = defaultdict(list)
    sources = []
    for source in SOURCES:
        path = VALIDATION / 'results_v1' / source
        content = path.read_bytes()
        sources.append(dict(path=str(path.relative_to(ROOT)), sha256=hashlib.sha256(content).hexdigest()))
        rows = list(csv.DictReader(content.decode().splitlines()))
        for row in rows:
            finite = not boolean(row['nonfinite_state'])
            balanced = boolean(row['balanced'])
            no_fall = boolean(row['no_fall'])
            primary = boolean(row['primary_success'])
            assert primary == (balanced and no_fall), row['case']
            relaxed = balanced and finite
            assert not primary or relaxed, row['case']
            key = (source, int(row['seed']), row.get('episodes', ''), row['arm'],
                   row.get('ic_set', 'authored'), float(row['seconds']))
            groups[key].append((row['case'], primary, relaxed, no_fall, finite))
        usage_gate()
    result = []
    for key, rows in sorted(groups.items()):
        assert len({r[0] for r in rows}) == len(rows), key
        source, seed, episodes, arm, cohort, seconds = key
        strict = sum(r[1] for r in rows)
        relaxed = sum(r[2] for r in rows)
        result.append(dict(source=source, seed=seed, episodes=episodes, arm=arm,
                           ic_set=cohort, seconds=seconds, n=len(rows),
                           primary_no_fall_successes=strict,
                           terminal_balance_successes=relaxed,
                           additional_recoveries=relaxed-strict,
                           horizontal_crossings=sum(r[4] and not r[3] for r in rows),
                           nonfinite_cases=sum(not r[4] for r in rows)))
    OUT.mkdir(parents=True, exist_ok=True)
    csv_path = OUT / 'saved_metric_comparison.csv'
    with csv_path.with_suffix('.csv.tmp').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result[0]))
        writer.writeheader()
        writer.writerows(result)
    csv_path.with_suffix('.csv.tmp').replace(csv_path)
    report = dict(analysis='Saved-metric comparison; not independent raw-trajectory rescoring.',
                  primary_metric='Recorded terminal angular balance AND no horizontal crossing.',
                  alternative_metric='Recorded terminal angular balance AND finite trajectory, allowing earlier horizontal crossings.',
                  sources=sources, script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  usage=telemetry, cpu_seconds=time.process_time()-started, groups=result,
                  primary_files_modified=False, controllers_executed=0)
    path = OUT / 'saved_metric_comparison.json'
    path.with_suffix('.json.tmp').write_text(json.dumps(report, indent=2) + '\n')
    path.with_suffix('.json.tmp').replace(path)
    for r in result:
        if r['source'] == 'E1/summary.csv' and r['ic_set'] == 'C' and r['seconds'] == 14 and r['arm'] in ('H', 'L'):
            print(json.dumps(r))
    print('Saved analysis:', OUT)


if __name__ == '__main__':
    main()
