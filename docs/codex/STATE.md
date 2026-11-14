# Current checkpoint — 2026-11-14

## Current objective
Completed saved TMC/L results report with historical map and convergence figures.

## Completed work
- src/validation_report.py and scripts/report_tmc.py render saved cases only:
  recovery scatter projection, all shared starts in paginated convergence plots,
  report and manifest/case/raw-hash evidence. No validation source hashes changed.
- docs/validation/tmc-seed0-results/REPORT.md and PNG/evidence artifacts generated
  from saves/validation/tmc-c200-seed0-v1; historical user screenshot included.
- docs/validation/TMC.md and README link report/figure commands for smoke,
  tuning and C200. No simulation rerun needed to produce current plots.

## Verified results and tests
- Actual saved run is partial:7/400 cases; TMC3/4 recoveries,L2/3; shared3
  starts yield1 TMC-only win,1 L-only win,1 joint recovery. No superiority claim.
- Median of case median controller latency:TMC232.680ms,L0.014ms; ideal feedback.
- Legacy phase-map plotting remains in validation/experiments/figures.py. User
  screenshot exists under saves/validation/temp and labels hybrid, three seeds;
  exact provenance unverified, never presented as new standalone TMC evidence.
- Rendered real report with control311 Python; inspected convergence PNG.
- Focused report test passed:1 test covering partial/missing, failed empty traces,
  no shared starts and paired rendering. git diff --check passed.

## Decisions, assumptions, blockers
- C starts vary four coordinates: map is a projection, no basin interpolation.
- .venv lacks numpy; use activated control311 or its explicit interpreter.
- No new training or simulations launched. Existing validation can resume exact
  original command; report regeneration does not affect resumability.

## Uncommitted changes and next actions
- Task source, tests, docs and report committed as cf4c80d3 and pushed to origin/main.
- Resume C200 run if desired, then regenerate report; lock tuning protocol before
  final evaluation and repeat across seeds. Preserve historical snapshot context.
