# Codex Progress Log

Append dated entries for meaningful project milestones.
Include tested results, important decisions, and commit IDs when available.

## 2026-10-17 — Integration protocol and source audit

- Started main at 325beae; untracked AGENTS.md belongs to user. Read STATE/LOG.
- `git submodule add https://github.com/aroyphillips/TrajectoryManifoldControl.git external/TrajectoryManifoldControl`
  pins e66a2b79f27f347286576f678f424625f60787b4. No commit/push.
- `git diff --no-index` confirmed identical original manifold_control.py and
  run_inverted_pendulum_mc.py. Read full implementation TeX in upstream handoff.
- No project-level LICENSE/COPYING, checkpoint .pt, or dataset/raw .npz tracked.
  Do not confuse third-party licenses or archived result tables with these.
- Added PLAN.md, isolated subprocess entry point, A/B/C scoring and metric tests.
  Test discovery explicitly excludes external to avoid upstream import collisions.
- System python3=3.12 lacks torch and ensurepip; existing fmri311 also lacks torch
  and could not bootstrap venv. Created /tmp/tmc-validation-env via micromamba;
  dependency installation underway. Sandbox bwrap network setup fails; commands
  run through approved escalation. No performance/success claim yet.

## 2026-10-17 — Baseline/metric verification and decoder diagnostics

- Isolated scientific Python at /tmp/tmc-validation-env (Python3.11.17,
  torch2.14.1+cpu, numpy2.4.6, scipy1.17.1, real cvxpy1.9.3).
- `python -m pytest -q`:40 passed in6.22s after correcting one malformed test
  helper. Discovery targets tests/, excludes external and root test_lqr_dd.py
  (an executable simulation/animation script, not a unit-test module).
- `scripts/validation_benchmark.py --verify`: saved C identities exactly match
  IC_SEED20261009; direct upstream imports, held-input RK4 timing, normalized dt
  and episode splits pass. No CVXPY workaround.
- Archived-gain L, six A + first3 C: binary and compared scalar trajectory values
  exactly agree with archive. Figures saved under saves/validation/initial-replay.
- Reconstructed seed0 original dev:3500episodes,100epochs,76036parameters;
  train2450episodes/73500transitions; validation525; test525. H4/6,L6/6 on A;
  archived binary agreement12/12. CPU reconstruction is not original CUDA replay.
- `scripts/diagnose_validation_decoder.py`: test NRMSE0.01018/0.01695/0.00910/
  0.01839; initial,input,predicted-prefix residual0; numerical rank34. Candidate
  input replay on4 validation states: terminal angle errors0.029/0.071/0.124/0.022rad.
  One CEM cost762.67 exceeds L diagnostic candidate702.23 under same learned cost.
  No L candidate used as manifold action. Four CEM solves~83ms, all miss20ms.
- Full L launches named seed1/2 initially omitted --seed: summary duplicate guard
  caught this. Preserve and exclude these seed0 duplicates. Relaunched explicit
  seed1/2 into *-corrected paths; final summary pending.

## 2026-10-17 — Matched baselines and native implementation milestone

- Corrected full L reproduction:122/200,130/200,128/200, all three endpoints;
  all600 A/B records match upstream, compared final_theta/final_p/max_abs_p
  differences0. Reconstructed H also agrees on C3 (H1/3,L2/3).
- Original checkpoint SHA256 c3f6032f5e35b6bb36395a4eec23813ac41e44a4c72a04037b03629f5cbf4b4f;
  measured data SHA2560e825cda7821de4afbe839f5da022b5942c6f8b14dd5208a040f05278cbde119.
- Added opt-in native safe CEM/projected Adam, actual-age warm starts, one-action
  feedback, optional periodic and terminal-window objectives. Original sources
  untouched.46tests passed. Native controller never invokes L.
- Static matched validation-state solver screen: safe/original CEM same costs;
  Adam20iter beats CEM on1/4, worse3/4;~0.21s after startup versus~0.08sCEM.
- Matched10epoch fine-tuning5step loss: common validation criterion0.00398455
  versus0.00397709 one-step control. Original M on tuning starts0/2 for both;
  L2/2. No multistep improvement claimed; no expensive direct decoder/geometric
  penalty introduced without evidence.
- Fixed-model tuning screen in progress, independent seed20261017. First state:
  M,H,cem5,cem1 fail; L and adam1 recover. Adam terminal angle max0.00321rad,
  rate max0.00983rad/s, cart final4.304m, no fallback,~0.200s median solve.
  This is tuning evidence only, not final benchmark superiority.
- Added Markdown report/reproduction commands and a frozen-protocol evaluator
  (not yet executed). Full improved C200/fresh200 cost remains high on CPU.

## 2026-10-17 — Resumability and preliminary objective ablation

-47tests pass (6.17s), including freeze rejection for incomplete evidence and
  changed source. Syntax/whitespace checks pass; upstream submodule stays clean.
- Tuning invocation1 stopped at a between-case budget boundary after379.96s
  (a complete periodic case ran across the300s boundary). Seven of16cases saved.
  Both quadratic Adam and periodic Adam recover the first start, no advantage
  of periodic cost demonstrated. Same command resumed; prior cases skipped by
  verified unchanged protocol/model/source hashes.
- Full native C200 estimated7.8CPU hours per fitted seed at0.2s×700×200;
  final three-seed + fresh-holdout evaluation is beyond this bounded screen.
  `scripts/frozen_manifold_validation.py` provides hash-checked resumable final
  evaluation and within-seed paired endpoints; not a claim it has run.

## 2026-10-18 — Second tuning start and final-screen boundary

- Invocation2 finished through adam1 on second tuning state;14/16cases complete.
  Native quadratic Adam2/2 versus original M0/2,H1/2,L2/2, safe CEM5/1 both0/2.
  Second Adam terminal-angle maximum0.00622rad. This is tuning evidence only.
- Resumed invocation3 for the remaining periodic/window second-start cases.
  Both model-training ablations are completed negative results (M0/2,L2/2).
- Verified eight tuning states distinct from pinned fresh200, in addition toC200.
  Success/failure five-panel figures generated and Adam success figure inspected.
- Frozen evaluator now pins original H/L artifacts separately from candidate
  artifacts and refuses a fine-tuned checkpoint as the original H baseline.

### Selection rule recorded before remaining objective results

Maximize terminal recovery on the declared independent tuning screen; on tied
binary outcomes retain the original quadratic objective/terminal weight instead
of adding an unproven objective change. Plain Adam already2/2; periodic/window
second-start outcomes not yet available. No final-score selection is allowed.
Upstream robustness helper nominal-mismatch identity and delayed-input timing
checked directly; frozen evaluator supports those interventions without passing
parameters to controllers.

### Completed screen and frozen release

- All16 tuning cases complete. M0/2,H1/2,L2/2,CEM5/CEM1 both0/2, Adam2/2,
  periodic1/2, periodic+terminalwindow2/2 for each A/B/C endpoint.
- Selected plain Adam by the recorded tie rule; original model/objective/terminal
  weight, one-action feedback,20iterations,3starts,lr0.15. Freeze SHA256
  97a3d77c12d69a30a8ef799f8703c77f425f0dc94f762c88a0403c7a01ad9f41.
- `frozen_manifold_validation.py evaluate --verify-only`: all hashes/config pass.
  Began bounded C200 evaluation, arms L H native,180s budget checked between
  cases, preserving partial results. Also evaluating predeclared authored index2
  under input_delay and mass_plus25. No retuning allowed from these outcomes.
- Completed inexpensive baseline60s Aindices2,3: originalH/L both2/2 under all
  endpoints. Native60s remains pending; do not imply these were native tests.
- Compact EVIDENCE.json exported; original-arm forward counts derived from
  raw CEM records (ablation's original-arm zero counter means uninstrumented).

### Frozen robustness and first nominal result (no retuning)

- Delay1 sample, fixed authored−15deg: native1/1,H0/1,L0/1 under all endpoints.
  One native-only paired win; exact paired p=1 with n=1, no broad inference.
- Cartmass+25%, same start: native/H/L all1/1. Upstream interventions reused.
- First nominal C state: native fails all endpoints (angle max0.01271rad passes,
  physical speed max0.06899rad/s exceeds0.05), L succeeds,H fails. No numerical
  failure/fallback; native median controller call0.2115s,100%20ms misses.
  Do not change tolerances or tune from this result. Second nominal native case
  still running; C200 partial and resumable.

### Final bounded comparison and session handoff

- Bounded frozen C200 invocation completed2paired starts in306.20s, then stopped
  at its own between-case boundary. A/B/C: native1/2,H1/2,L2/2. Native versusL
  0wins/1loss, exact paired p=1.0 (n=2); target not achieved. No retuning.
- Native case1 settles angularly at2.40s (L0.98,H1.26); terminal angle/speed maxima
  0.00715rad/0.02587rad/s. Native C0/1 finalcart−6.213/4.259m, energy22.817/1.794,
  median call0.2115/0.2035s,100%20ms deadline misses, no fallback/numerical failure.
-47tests pass; syntax/whitespace checks pass, submodule clean. Figures generated
  for successful and failed tuning/frozen/robust/long-baseline trajectories.
  `EVIDENCE.json` preserves compact machine-readable data and hashes; raw/model
  artifacts remain ignored under saves/validation. No commit/push or deletion.
- Full native/H C200, remaining fitted seeds1/2, independent fresh200 and native
  60s not completed. Frozen protocol and exact resume commands in REPORT/STATE.
  Estimated native C200~7.8CPU hours/seed before other arms/holdouts. No active
  controller experiment remains after the bounded run. Future configuration
  selection must not use the observed final C scores.
