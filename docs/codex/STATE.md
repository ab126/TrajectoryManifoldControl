# Current checkpoint — 2026-11-13

## Current objective
Completed renaming standalone controller/validation terminology to TMC
(trajectory manifold control), verified renamed CLI/tests and pushed to origin/main.
Shared reference-aware LQR/hybrid implementation remains unchanged.

## Completed work and decisions
- src/validation_controllers.py: validation L law generalized to x_ref/u_ref;
  hybrid uses the same callable for local switch and failed manifold solves.
  Switch region, tracking objective and initial input plan use references.
- TMC uses learned NativeSolver, no LQR or failed-plan substitution. Invalid
  plans terminate the case; launcher records failure and continues other arms.
- scripts/validate_tmc.py: TMC/L default, optional hybrid, tuning/A/C/fresh starts,
  reference flags, solver knobs, paired summary, raw traces, latency, explicit
  failures, source/artifact/config manifest and guarded resumability.
- Frozen validation sources and old native/frozen launchers unchanged. New
  experiment uses archived validation gain or fits L from supplied training data.
- Local checkpoint saves/saved_models/canonical_decoder_dev.pt has seed234;
  its training data absent. --archived-gain-seed explicitly labels unmatched
  comparison; --data is required for a matched refit.
- README links docs/validation/TMC.md with runnable local screen, matched
  reconstruction/evaluation commands, reference/hybrid flags and tuning advice.
- Advice from REPORT.md: Adam20/.15/3starts, replan1, plain cost/window1 as
  starting protocol; screen10/20/40 iterations and .05/.10/.15 rates on tuning
  only. Earlier two-start ties do not establish superiority over L.

## Verified results and test status
- Six new tests pass: exact hybrid/L actions including nonzero references and
  clipping, reference-shifted objective, strict invalid solve after valid plan,
  nonfinite-plan rejection, real CLI failure/resume/config-change guard.
- Focused old/native plus initial new tests:11 passed. Full suite:68 passed,
  2 failures in unchanged tests: primary notebook compare configuration conflicts
  with test expecting empty comparisons; known upstream exact-array mismatch.
  Notebook source matches HEAD and was not edited in this task.
- Actual seed234 checkpoint: zero-reference .04s smoke and nonzero x_ref/u_ref
  .04s smoke for TMC/L/hybrid passed; exact L action replay and resume verified.
  Historical outputs saves/validation/tmx-smoke and tmx-reference-smoke remain
  unchanged as prior-run evidence; new runs use tmc names.
- Rename verification:12 focused controller/CLI/native tests passed in9.08s;
  renamed CLI --help exposes TMC/L/hybrid; git diff --check passes.
- Earlier verify_snapshot and py_compile pass; frozen sources unchanged.
- No 14s recovery benchmark or matched training reconstruction executed.

## Uncommitted changes and blockers
- Preserve user's pre-existing manifold_control.ipynb outputs/metadata.
- Renamed CLI to scripts/validate_tmc.py and guide to docs/validation/TMC.md;
  arm identifiers, tests, README and current checkpoint use TMC. Historical LOG
  entries retain original names as a record of actual earlier runs.
- User explicitly authorized publication. Rename commit65c82e4f pushed to
  origin/main successfully; audit5ab64b0b and implementation62e9eb84 are included
  in published ancestry. Only user notebook edit remains unrelated to task.
- Missing local matched training dataset is not a code blocker; explicit archived
  gain smoke path is supported and scientific reconstruction commands documented.

## Specific next actions
1. Run new CLI tuning screen using docs/validation/TMC.md; exact command resumes.
2. For matched comparison reconstruct/provide transition.pt + data.npz, tune
   only on tuning starts, then lock protocol for C/fresh and repeat across seeds.
3. Assess recovery AND controller latency; ideal simulation does not inject
   computational delay. Arbitrary references may not be equilibria; angle-only
   recovery does not certify cart regulation.
