#!/usr/bin/env python3
"""Apply the explicitly authorized restart after preserving the prior credit stop.

No model or numerical job is launched here. The historical guard and thresholds
are unchanged. A fixed approved credit balance prevents automatic rebaselining
after another deduction.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path

VALIDATION = Path(__file__).resolve().parent
OUT = VALIDATION / 'results_literature'


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true', help='Validate only; no state changes.')
    args = parser.parse_args()
    if not set(os.sched_getaffinity(0)) <= set(range(80, 128)):
        raise SystemExit('Run with taskset -c 80-127.')
    approval = json.loads((OUT / 'RESUME_AUTHORIZATION_20261014.json').read_text())
    if approval.get('authorization') != 'explicit_user_request' or approval.get('paid_credits_authorized') is not False:
        raise SystemExit('Expected explicit included-usage-only resumption authorization.')
    manifest = json.loads((OUT / 'manifest.json').read_text())
    guard_path = VALIDATION / 'codex_supervisor.py'
    if hashlib.sha256(guard_path.read_bytes()).hexdigest() != manifest['protected_source_hashes']['validation/codex_supervisor.py']:
        raise SystemExit('Original guard changed; refusing restart.')
    spec = importlib.util.spec_from_file_location('original_guard', guard_path)
    guard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guard)
    limits = guard.latest_limits()
    current = guard.balance(limits)
    approved = approval['approved_credit_balance']
    if limits is None or current is None or abs(current - approved) > 1e-6:
        raise SystemExit(f'Credit telemetry differs from approved restart balance {approved}; refusing to rebaseline.')
    if guard.window_state(limits.get('primary'))[0] >= 85 or guard.window_state(limits.get('secondary'))[0] >= 90:
        raise SystemExit('Usage threshold reached; wait for included usage to reset.')
    status = (OUT / 'STATUS').read_text().splitlines()[0]
    allowed = ('STOPPED_CREDIT_GUARD', 'READY_TO_RESUME', 'READY')
    if status not in allowed:
        raise SystemExit(f'Task state {status!r} does not permit credit-stop preparation.')
    incident = OUT / 'reviews/credit_stop_documentation_checkpoint.json'
    if not incident.is_file():
        raise SystemExit('Historical credit-stop evidence missing.')
    record_path = OUT / 'reviews/resumption_20261014.json'
    archive = OUT / 'reviews/credit_alert_before_resume_20261014.txt'
    if record_path.exists():
        previous = json.loads(record_path.read_text())
        if previous['new_credit_baseline'] != approved or manifest['credit_balance_baseline'] != approved:
            raise SystemExit('Existing resumption record conflicts with task baseline.')
        if (OUT / 'CREDIT_ALERT').exists():
            raise SystemExit('A new credit alert exists after the recorded restart; no automatic second reset.')
        print('This authorized resumption was already prepared; existing baseline retained.')
        return 0
    old = manifest['credit_balance_baseline']
    alert = OUT / 'CREDIT_ALERT'
    if not alert.is_file():
        raise SystemExit('Expected original credit alert is missing; review state before restart.')
    if alert.read_text().strip() != approval['historical_credit_alert']:
        raise SystemExit('Credit alert differs from the approved historical incident.')
    if args.check:
        print(f'Resumption preflight passed: credit baseline {old} -> {approved}; guard/thresholds unchanged; no state changed.')
        return 0
    # The launcher holds .launch.lock; this also excludes a running supervisor.
    with (OUT / '.supervisor.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('A supervisor is already active; refusing restart preparation.')
        if archive.exists():
            raise SystemExit('Historical archive already exists without resumption record; review interrupted preparation.')
        record = dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),
                      authorization_file='RESUME_AUTHORIZATION_20261014.json',
                      previous_credit_baseline=old, new_credit_baseline=approved,
                      historical_credit_alert=alert.read_text().strip(),
                      guard_sha256=hashlib.sha256(guard_path.read_bytes()).hexdigest(),
                      thresholds_unchanged=True, paid_credits_authorized=False)
        # Copy the incident first. Retain the active alert until the manifest and
        # restart record are durable, so an interruption cannot silently clear it.
        archive.write_text(alert.read_text())
        manifest['credit_balance_baseline'] = approved
        manifest['resumption_authorization'] = 'validation/results_literature/RESUME_AUTHORIZATION_20261014.json'
        manifest['resumption_record'] = 'validation/results_literature/reviews/resumption_20261014.json'
        manifest['current_checkpoint'] = 'Authorized resumption prepared; awaiting supervised launch.'
        atomic_json(OUT / 'manifest.json', manifest)
        atomic_json(record_path, record)
        alert.unlink()
        temporary = OUT / 'STATUS.tmp'
        temporary.write_text('READY\nExplicit user-authorized resumption prepared; prior credit alert archived; awaiting supervised launch.\n')
        temporary.replace(OUT / 'STATUS')
    print('Authorized resumption prepared; prior incident retained; further credit decreases still stop work.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
