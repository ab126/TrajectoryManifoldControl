# Current checkpoint — 2026-11-16

## Current objective
Run pure standalone TMC versus matched data-driven LQR on the previous three-seed
D grid. Active launch details below supersede the earlier completed investigation.

## Previous objective
Investigate completed TMC hybrid versus linear grid and publish concise visual
report, measured tuning outcomes and executable candidate-training/evaluation.

## Completed work
- Completed grid verified10086cases: hybrid2880/5043(57.1%),L3247(64.4%).
  Seed2 hybrid1225/1681(72.9%) beats matchedL1085(64.5%);seed0/1 lose.
  H-only1275,L-only1642;no controller failures/nonfinite CEM recorded.
- scripts/investigate_tmc.py audits models/data and raw witness/replay hashes,
  measures forecast errors, local diagnostic spectral radii, and24 independent
  tuning seed/start pairs per optimizer. Four reproducible visual figures.
- docs/validation/investigation-2026-11-16/REPORT.md concise visuals and seed/
  endpoint explanation; LAUNCH.md investigation, tuning, larger model and grid codes.
- scripts/train_tmc_candidate.py: configurable measured-data collection, broader
  angle domain and near-upright mixture, width/epochs and existing paired
  one-step/multistep fine-tuning. No original/frozen source changes.
- scripts/screen_tmc_candidate.py: flexible candidates, original/safe CEM/Adam,
  current hybrid + same-data fitted L, independent tuning orDgrid, resumability.

## Verified findings and tests
- Forecast endpoint angle RMSE .054-.069rad vs one-step .0012-.0015rad.
  L-only-case median .6s replay angle RMSE .029initial,.156at8s; selected cases,
  association not isolated causality. Equilibrium biases and near-critical gains.
- 24 tuning probes: CEM10 median true replay cost ratio .869 vs CEM3;
  Adam40 ratio1.233. Cold-start .6s plans, not coverage evidence.
- Launched and completed2-start14s seed0 screens: CEM3queue5,CEM10queue5,
  Adam20replan1 all hybrid1/2 vs L2/2. No improvement established; only2starts.
- Candidate collection/train40episodes/1epoch/16x16 plus fine-tune smoke passed;
  candidate fine-tuned checkpoint evaluated; short traces correctly fail recovery.
- Nine focused candidate workflow/sweep/controller tests pass in30.73s:
  CLI train/evaluate/resume/changed-protocol guard plus controller/metric checks.
- Final investigation rerun checks all model/data identities, raw replay hashes,
  preserves script hash and generates all four plots. No full candidate training
  or new dense grid launched. git diff --check before publication.

## Decisions and assumptions
- No credible almost-sure hyperparameter guarantee. Seed2 already wins existing
  grid; selecting it post hoc is not independent evidence or general guarantee.
- Candidate hypothesis:14000episodes,256x3,200epochs+20multistep/window10,
  initialtheta85deg,25%upright/small-input collection;CEM10,replan1vs5screen.
  Test each factor separately and refit matchedL on the same candidate data.
- Three plotted runs are seeds0/1/2 (whole fit/planner), not3convergence rules;
  terminal recovery finite +final3s angle/rate tolerances, cart not constrained.
- L/H are simulated with ideal feedback; measured compute delay not injected.

## Uncommitted changes and next actions
- Final audit succeeded; scripts/tests/docs/evidence committed as c75c9f39
  and pushed to origin/main. All tuning jobs finished; no long job remains active.
- Screen all8tuning starts before locking parameters; run expensive proposed
  training only using provided commands. Final dense candidate confirmation can
  take days. Continue to keep scientific hypotheses distinct from verified wins.

## Active simulation — 2026-11-16
- User requested pure TMC (no hybrid) versus LQR, as before.
- Verified scripts/sweep_tmc.py standalone passes no linear callable to
  trajectory_controller: no LQR switch/fallback; invalid plans fail the case.
- Launched PID1499380; verified live process and manifest: standalone, arms TMC/L,
  seeds0/1/2,14s,1681starts/seed,10086total cases. Existing matched models reused.
- Standalone uses Adam20,lr.15,3starts,replan1; prior hybrid used CEM.
  This is a controller comparison, not an isolated switching ablation.
- Output/log: saves/validation/grid-standalone-3seed-v1/{manifest.json,launch.log}.
- Budget172800s checked between cases; resume exact launch command if incomplete.
- No outcome or completion verified yet; next inspect log/cases/process and
  generate final maps only after checking completion. No source changes.
