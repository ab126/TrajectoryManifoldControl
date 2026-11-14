# Current checkpoint — 2026-11-14

## Current objective
Complete original screenshot experiment reproduction and launchable IC sweep.

## Completed work
- Inspected requested upstream repo at053b14890a107063db9b58859884f655173528b8
  in /tmp/tmc-upstream-figure. Campaign bytes match pinned migrated copy.
- Exact screenshot pattern is TERMINAL RECOVERY (earlier crossings allowed),
  not stricter primary_success. Both endpoints implemented and plotted.
- src/validation_sweep.py + scripts/sweep_tmc.py: saved D41x41 fixed-cart grid,
  seeds0/1/2, matched per-seed dev checkpoint/data, original CEM hybrid/L default;
  optional explicitly labeled standalone Adam20/.15/3start/replan1. Raw traces,
  diagnostics, hashes, source/artifact protocol guards, budget/resume and plots.
- scripts/plot_legacy_grid.py imports complete archived E1 CSV to a separate
  output, validates1681states x3seeds x2arms and reproduces screenshot format.
- docs/validation/grid-reproduction/README.md includes exact definitions,
  full training/sweep/resume/plot/archive commands and verified PNG/count data.
  TMC.md and prior report link corrected provenance and new sweep workflow.

## Verified results and tests
- Rebuilt archived all10086cases: terminal H2864/5043,L3247/5043; strict primary
  H1485/5043,L3247/5043. Inspected both maps, terminal matches supplied pattern.
- Nine focused sweep/report/controller tests pass in16.21s. Grid orientation,
  incomplete-seed masking, finite/short traces, crossing/recovery distinctions.
- Real seed0 matched artifacts:14s original H/L first D pair verified both
  endpoints against legacy campaign; H~15.64s wall,140CEM solves. Exact resume
  skipped cases and verified raw hashes; plot-only regenerated both maps.
- Real original hybrid and standalone .04s smoke paths passed. Latest concise
  console-output version checked with separate grid-launch-check-v1 output.
- git diff --check passed; no frozen/old launcher changes.

## Important decisions and assumptions
- Screenshot default is legacy HYBRID; standalone is optional different experiment.
- No new full10086-case sweep launched. Full command provided to user; training
  seeds1/2 and long simulation remain. Original archived maps are not new runs.
- Reconstructed weights can differ from archived weights; exact archived counts
  are reproducible via imported CSV, not guaranteed by retraining.
- 48h per-invocation rollout budget suggested; training excluded. Exact rerun
  resumes. Use one writer/output and new output for changed scientific settings.

## Uncommitted changes and next actions
- Task source/tests/docs/maps committed as24fef25c and pushed to origin/main.
- User can launch documented --train full sweep, or separate training and sweep.
  Check summary.complete and saved10086; repeat exact command if budget expires.
