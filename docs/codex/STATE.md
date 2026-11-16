# Current checkpoint — 2026-11-16

## Objective
Use core TMC versus matched data-driven LQR for new launches and tuning.
Active grid must finish before starting the proposed tuning/training plan.

## Completed work and decisions
- New entry point docs/launch.md links concise tuning/launch plan at
  docs/validation/investigation-2026-11-16/LAUNCH.md. README, TMC guide and grid
  guide explicitly select core TMC; historical hybrid evidence retains labels.
- Candidate launcher now defaults to --arms TMC L; hybrid requires explicit opt-in.
  Core TMC receives no linear callable; invalid plans record failed cases.
- Sweep source/default unchanged to preserve running manifest/resumability;
  always pass --controller standalone. Candidate source changed: old hybrid
  candidate screens require old source for resume; new protocols need new output.
- Historical completed hybrid grid:2880/5043 versus LQR3247/5043; seed2 hybrid
  1225 versus LQR1085. These are not core-TMC outcomes.
- Larger candidate remains unverified:14000episodes,256x3,200epochs plus20
  multistep/window10,85deg domain,25%near-upright data. No guaranteed improvement.

## Active simulation
- PID1499380 verified live during docs update; first TMC/L pair saved.
- scripts/sweep_tmc.py --seeds 0 1 2 --controller standalone
  --models-root saves/validation/models --seconds 14
  --output saves/validation/grid-standalone-3seed-v1 --max-wall-seconds 172800
- Manifest verified:TMC/L,seeds0/1/2,1681starts/seed,10086total cases.
- Adam20,lr.15,3starts,replan1; previous hybrid used CEM. Not a switch-only ablation.
- Log: saves/validation/grid-standalone-3seed-v1/launch.log. Budget checked between
  cases; exact command resumes. Never run two writers on this output.

## Verification / next actions
- Focused candidate/controller tests pending; record outcome before commit.
- No final recovery outcome yet. Monitor process/log; check completeness before
  interpreting maps. After completion review docs/launch.md tuning plan.
- Current changes: concise docs, candidate core-default support, workflow test.
  No blockers. Commit/push after checks; leave running grid unchanged.

## Verification and next actions
- control311 pytest -q tests/test_candidate_workflow.py
  tests/test_validation_controllers.py:7passed in29.34s. Covers default core arms,
  short-trace failure, exact resume, protocol guard and standalone failure behavior.
- git diff --check passed. No simulation restart or training launched.
- Changes ready to commit/push: launch docs, candidate CLI, workflow test, handoff.
- Next inspect active grid log/cases and completion; only then review tuning plan.
  No blockers; no final core-TMC recovery conclusion yet.
