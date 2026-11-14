# Current checkpoint — 2026-11-14

## Current objective
Document clean-checkout TMC validation after missing-model error.

## Completed work and decisions
- Updated docs/validation/TMC.md: train matching seed-0 checkpoint/data first,
  artifact check, separate smoke/tuning outputs, existing dev artifacts,
  unmatched archived gain limits, fresh evaluation versus resumability.
- Verified saves/saved_models is empty; saves/ is Git-ignored. Prior session's
  local seed234 checkpoint is unavailable in this checkout.
- Controller code and frozen validation sources unchanged.

## Verified results and test status
- Both launchers' --help pass with .venv/bin/python; documented options checked
  against implementations. No expensive training or validation launched.
- Documentation-only change; git diff --check checked before commit.

## Important assumptions and blockers
- User must generate or supply trained artifacts before TMC validation.
- Training reconstruction uses 3500 episodes/100 epochs; no training time limit.
- Sandbox execution fails at bwrap network setup; escalated commands work.

## Uncommitted changes and next actions
- Task documentation and checkpoints to be committed/pushed.
- Follow guide to train, smoke-check, then screen. Lock tuning settings before
  C/fresh evaluation. No new recovery/performance result claimed.
