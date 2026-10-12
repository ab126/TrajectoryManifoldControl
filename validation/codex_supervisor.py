#!/usr/bin/env -S python3.11
"""Run Codex unattended on the validation handoff without touching paid credits.

Codex records account usage in its session logs (~/.codex/sessions/**/rollout-*.jsonl)
as ``rate_limits``: ``primary`` (5-hour window) and ``secondary`` (weekly window), each
with ``used_percent`` and ``resets_at``, plus ``credits.balance``. This supervisor:

* starts (or resumes) a ``codex exec`` session on the handoff;
* polls the newest usage record every ``--poll`` seconds (no model tokens used);
* interrupts Codex when either window reaches its threshold and sleeps until that
  window resets, then resumes the same session with "continue";
* kills Codex and exits immediately if the credit balance drops below its starting value;
* stops when ``results_v1/STATUS`` no longer starts with RUNNING, or Codex stops
  making progress.

Usage:
    .venv/bin/python codex_supervisor.py --check           # print current usage and decision
    nohup .venv/bin/python codex_supervisor.py > /dev/null 2>&1 &
Extra Codex flags go after ``--``, e.g. ``-- -s workspace-write``.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

VALIDATION = Path(__file__).resolve().parent
REPO = VALIDATION.parent
RESULTS = VALIDATION / "results_v1"
SESSIONS = Path.home() / ".codex" / "sessions"
UUID = re.compile(r"([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})\.jsonl$")

INITIAL_PROMPT = (
    "Read validation/codex_handoff_from_claude_v1.md and execute it; it supersedes the "
    "earlier implementation plans and handoffs. Follow section 6: create "
    "validation/results_v1/STATUS (first line RUNNING), detach long jobs, and keep work "
    "resumable. You are supervised for usage limits, so do not track usage yourself."
)
RESUME_PROMPT = (
    "Continue executing validation/codex_handoff_from_claude_v1.md. You may have been "
    "paused mid-turn for usage limits. Re-read validation/results_v1/STATUS, the JOB files "
    "and the end of RESULTS.md, check detached jobs, then continue from where you left off."
)


def log(message: str) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    line = f"{datetime.now().isoformat(timespec='seconds')} {message}"
    print(line, flush=True)
    with open(RESULTS / "supervisor.log", "a") as handle:
        handle.write(line + "\n")


def _find_key(obj, key):
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for value in obj.values():
            found = _find_key(value, key)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for value in obj:
            found = _find_key(value, key)
            if found is not None:
                return found
    return None


def rollout_files() -> list[Path]:
    return sorted(SESSIONS.rglob("rollout-*.jsonl"), key=lambda p: p.stat().st_mtime)


def latest_limits() -> dict | None:
    """Return the most recent rate_limits record across all Codex sessions."""
    for path in reversed(rollout_files()[-5:]):
        with open(path, "rb") as handle:
            handle.seek(max(0, path.stat().st_size - 4_000_000))
            lines = handle.read().decode("utf-8", "replace").splitlines()
        for line in reversed(lines):
            if '"rate_limits"' not in line:
                continue
            try:
                limits = _find_key(json.loads(line), "rate_limits")
            except json.JSONDecodeError:
                continue
            if isinstance(limits, dict) and limits.get("primary"):
                return limits
    return None


def window_state(window: dict | None) -> tuple[float, float]:
    """(used_percent, resets_at); a window whose reset time has passed counts as empty."""
    if not window:
        return 0.0, 0.0
    used, resets = float(window.get("used_percent") or 0), float(window.get("resets_at") or 0)
    return (0.0, resets) if resets and time.time() > resets else (used, resets)


def balance(limits: dict | None) -> float | None:
    credits = (limits or {}).get("credits") or {}
    try:
        return float(credits["balance"])
    except (KeyError, TypeError, ValueError):
        return None


def pause_until(limits: dict | None, args) -> float:
    """Unix time to sleep until, or 0 if Codex may run now."""
    if limits is None:
        return 0.0
    wait = 0.0
    for name, cap in (("primary", args.max_5h), ("secondary", args.max_week)):
        used, resets = window_state(limits.get(name))
        if used >= cap:
            wait = max(wait, resets + args.reset_slack)
    return wait


def status() -> str:
    try:
        return (RESULTS / "STATUS").read_text().split("\n", 1)[0].strip()
    except FileNotFoundError:
        return "MISSING"


def progress_marker() -> float:
    paths = [RESULTS / "STATUS", RESULTS / "RESULTS.md"]
    return max((p.stat().st_mtime for p in paths if p.exists()), default=0.0)


def stop_codex(process: subprocess.Popen) -> None:
    for sig, grace in ((signal.SIGINT, 30), (signal.SIGTERM, 15), (signal.SIGKILL, 5)):
        if process.poll() is not None:
            return
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            continue


def launch(session: str | None, extra: list[str]) -> subprocess.Popen:
    if session:
        command = ["codex", "exec", *extra, "resume", session, RESUME_PROMPT]
    else:
        command = ["codex", "exec", "-C", str(REPO), *extra, INITIAL_PROMPT]
    log("launch: " + " ".join(command[:-1]) + " <prompt>")
    output = open(RESULTS / "codex_exec.log", "a")
    return subprocess.Popen(command, cwd=REPO, stdout=output, stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL, start_new_session=True)


def find_new_session(since: float) -> str | None:
    for path in reversed(rollout_files()):
        if path.stat().st_mtime < since:
            break
        match = UUID.search(path.name)
        if match:
            with open(path) as handle:
                head = handle.readline()
            if str(REPO) in head:
                return match.group(1)
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--max-5h", type=float, default=85.0, help="pause at this 5-hour used %%")
    parser.add_argument("--max-week", type=float, default=90.0, help="pause at this weekly used %%")
    parser.add_argument("--poll", type=float, default=20.0, help="seconds between usage checks")
    parser.add_argument("--reset-slack", type=float, default=120.0,
                        help="extra seconds to wait after a window's reset time")
    parser.add_argument("--session", help="Codex session id to resume (default: saved state)")
    parser.add_argument("--max-idle-runs", type=int, default=3,
                        help="stop after this many consecutive Codex runs without progress")
    parser.add_argument("--check", action="store_true", help="print usage and decision, then exit")
    args, extra = parser.parse_known_args()
    extra = [arg for arg in extra if arg != "--"]

    RESULTS.mkdir(parents=True, exist_ok=True)
    state_path = RESULTS / ".supervisor_state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    session = args.session or state.get("session")

    limits = latest_limits()
    start_balance = balance(limits)
    if args.check:
        for name in ("primary", "secondary"):
            used, resets = window_state((limits or {}).get(name))
            when = datetime.fromtimestamp(resets).isoformat(timespec="minutes") if resets else "-"
            print(f"{name:9s} used={used:5.1f}%  resets={when}")
        print(f"credits balance={start_balance}  session={session}  STATUS={status()}")
        until = pause_until(limits, args)
        print("decision:", "run now" if not until else
              f"wait until {datetime.fromtimestamp(until).isoformat(timespec='minutes')}")
        return 0
    if limits is None:
        log("no rate_limits record found in Codex session logs; refusing to run unguarded")
        return 2
    log(f"start: credit balance baseline={start_balance}, thresholds 5h={args.max_5h}% "
        f"week={args.max_week}%")

    idle_runs = 0
    while True:
        if session and status() not in ("RUNNING", "MISSING"):
            log(f"STATUS is {status()}; stopping")
            return 0
        until = pause_until(latest_limits(), args)
        if until:
            log(f"usage near limit; sleeping until {datetime.fromtimestamp(until).isoformat()}")
            time.sleep(max(0.0, until - time.time()))
            continue

        marker, started = progress_marker(), time.time()
        process = launch(session, extra)
        interrupted = False
        while process.poll() is None:
            time.sleep(args.poll)
            limits = latest_limits()
            now_balance = balance(limits)
            if start_balance is not None and now_balance is not None \
                    and now_balance < start_balance - 1e-6:
                stop_codex(process)
                log(f"CREDITS CONSUMED ({start_balance} -> {now_balance}); Codex killed, exiting")
                (RESULTS / "CREDIT_ALERT").write_text(f"{start_balance} -> {now_balance}\n")
                return 3
            if pause_until(limits, args):
                log("threshold reached during run; interrupting Codex")
                stop_codex(process)
                interrupted = True
            if not session:
                session = find_new_session(started)
                if session:
                    state_path.write_text(json.dumps({"session": session}))
                    log(f"session id {session}")
        log(f"codex exited with code {process.returncode}")
        if not session:
            log("could not identify the Codex session id; pass --session and rerun")
            return 2
        if interrupted:
            continue
        idle_runs = 0 if progress_marker() > marker else idle_runs + 1
        if idle_runs >= args.max_idle_runs:
            log(f"{idle_runs} consecutive runs without progress; stopping for review")
            return 4


if __name__ == "__main__":
    sys.exit(main())
