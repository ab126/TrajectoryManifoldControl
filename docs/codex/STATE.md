# Codex Project State

## Current objective
Integrate the independent validation benchmark, establish baseline parity, then
diagnose and improve standalone nonlinear learned trajectory-manifold control.

## Working state
Phases1–3 and bounded Phase4/5 screens complete; Phase6 partial. Submodule at external/TrajectoryManifoldControl,
pin e66a2b79f27f347286576f678f424625f60787b4. Original local controller/decoder
are byte-identical to upstream. Added isolated adapter and explicit A/B/C metrics;
47 tests pass. See docs/validation/REPORT.md and PLAN.md. Reconstructed original seed0 H gives
4/6 authored successes, matched L 6/6; all authored binary outcomes agree with
archive. Exact H trajectories differ (retrained CPU versus archived CUDA).

## Decisions and assumptions
Preserve originals and all historical data. Do not redesign before baseline and
diagnostic gates. No project-level upstream license found; call functions locally,
do not copy/adapt upstream source. Canonical decoder has 34 coordinates (30 free
inputs for fixed x0), no active learned encoder. Force bound is normalized ±10.

## Verified results
Git branch main, starting commit 325beae. Only pre-existing untracked AGENTS.md.
Upstream manuscript/source inspected. Missing upstream raw arrays/checkpoints
confirmed with git ls-files. Adapter contracts pass, true CVXPY installed.
Nine archived-gain L replays agree exactly on compared numerical values.
Seed0 decoder test NRMSE [0.01018,0.01695,0.00910,0.01839]; rank34, exact initial/
input/predicted-prefix consistency. Four validation candidate replays show angle
terminal error up to0.124 rad; one CEM cost exceeds diagnostic L candidate cost.
All four measured CEM solves ~83ms, above20ms. Artifacts saves/validation/.
Full archived-gain L seed0/1/2:122/130/128 of200, exact600-case compared scalar
agreement and A/B agreement. H9-case binary agreement. Opt-in native projected
Adam/safe CEM and periodic/window objectives implemented and tested, originals
unchanged. Completed16-case tuning screen: Adam2/2, periodic1/2, window2/2;
M0/2,H1/2,L2/2,CEM5/CEM1 both0/2. Adam~0.2s/solve, not real-time.
Multistep5 fine-tuning did not improve common validation criterion or M0/2
successes. Static20iter Adam objective beats CEM1/4, worse3/4; no blanket claim.
Frozen nominal C0/1: native1/2,H1/2,L2/2 under all endpoints; native0pairedwins,
1loss vsL. Case0 angle meets tolerance but speed0.06899rad/s exceeds0.05.
Delay1 sample, authored−15deg: native1/1,H0/1,L0/1. Cartmass+25% same start:
all1/1. Baseline60s±15deg H/L both2/2; native60s not run. No target achievement
claimed. Figures/raw data/manifests under saves/validation; compact EVIDENCE.json
and full report/reproduction commands under docs/validation.

## Blockers
Sandbox bwrap fails before commands; approved escalated execution works.
System Python lacks scientific dependencies and ensurepip. Temporary environment
/tmp/tmc-validation-env created via micromamba; dependencies installed.
Original upstream CUDA checkpoints unavailable in Git; CPU reconstruction is
distinct evidence and cannot be called exact checkpoint reproduction.

## Next steps
1. Tuning16/16 complete: plain Adam2/2, periodic1/2, terminalwindow2/2,
   original M0/2,H1/2,L2/2, safe CEM5/1 both0/2. Plain Adam selected using
   predeclared original-objective tie rule, frozen-native.json SHA256
   97a3d77c12d69a30a8ef799f8703c77f425f0dc94f762c88a0403c7a01ad9f41.
2. Frozen verification passed; bounded C200 stopped after2paired starts in306.2s
   (180s between-case budget permits current case to finish). Resume unchanged:
   /tmp/tmc-validation-env/bin/python -u scripts/frozen_manifold_validation.py evaluate --bundle saves/validation/frozen-native.json --arms L H native --set C --output saves/validation/final-seed0-C --max-wall-seconds 180
   Completed delay/mass/60s-baseline commands and outputs are in REPORT.md.
3. Frozen evaluator supports C/fresh/A60 and H/native/L; full native C200 estimated
   ~7.8hours per fitted seed on this CPU. Do not claim it ran. Seed1/2 model
   reconstruction and full H/native validation remain incomplete. Never retune
   this frozen configuration using the arriving final outcomes.
4. Original seed1/2 models still need independent reconstruction; use seed0
   training command with explicit --seed and distinct paths. Frozen evaluator
   can pin all3models in a new bundle with unchanged selected settings.
5. Full200/fresh200/native60s remain unrun. Further method development may use
   only training/validation and independent tuning data. Existing broad-validation
   cold-start costs, prediction precision and CPU latency suggest next studies;
   final C outcomes must not be used to select a passing revision.

## Last checkpoint
2026-10-18 (work started October17). Uncommitted .gitmodules/gitlink staged by git submodule add;
local metrics, adapter, native solver/controller, training/ablation/diagnostic/
freeze scripts, tests, pytest config, .gitignore, README, plan/report/checkpoints.
All experimental processes finished or stopped at their own saved budget boundary;
no background controller/training run is left active.47tests pass; upstream clean.
No commit or push made. Two directories L-seed1-C200 and L-seed2-C200 were
mistakenly launched without --seed (both are seed0 duplicates); exclude them.
Corrected explicit seed runs use *-corrected paths. No data deleted.
