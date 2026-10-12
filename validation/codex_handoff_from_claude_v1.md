# Codex handoff from Claude review — redirect to the user's questions

October 9, 2026. Written by Claude after reviewing `claude_review_handoff_v3.md`,
`implementation_plan_v3.md`, `collaborator_memo_v3.md`, the critique/stress-test plans,
the evidence under `validation/`, the submitted source, the author notebooks, the
arXiv paper and the three PDFs in `control_literature/`. Claude verified the
load-bearing facts below against primary files; file paths are given for each.

**This document supersedes implementation_plan_v3 and the claude_review_handoff
series as the operating plan.** Do not create v4 of those documents. Keep them
for history and do not delete or rewrite them.

## 0. The user's questions (the only acceptance criteria)

The user was asked by their lab to *validate* a labmate's method. Every work item
below must answer one of these:

- **Q1** Does the model work as claimed?
  **Q1.1** When does it fail? How much data does it need? Does it work on novel datasets/systems?
- **Q2** Does `trajectory_manifold_implementation.tex` contain errors (unexplained
  variables, incorrect equations or claims, unsupported novelty)?
- **Q3** How does the method differ from DeePC, Deep Koopman (Lusch 2018), DFINE
  (Abbaspourazad 2024) and deep RL (Baek 2024)? What are its benefits and drawbacks,
  and what gap does it fill?

Authorization: the user asked for this plan so that Codex does the bulk of the
compute and writing. The experiments in §3 are in scope once the user hands
this document over; you do not need further per-stage authorization. Stay
inside the caps in §5.

## 1. What to stop, and why

The current campaign has made a true-model "Gate A" (≥196/200 on a strict task the
reviewer defined) a prerequisite for ever training the submitted controller H.
Gate A scored 3/200. Since then, most of the effort has gone into repairing an
*oracle* controller (OA-1, Qp=2, I01–I05, a leased queue, archive chunk formats,
budget arithmetic). By line count, about 70–80% of `validation/` is oracle,
ops or handoff machinery, and about 10–15% addresses Q1–Q3. Changing Qp from
0.02 to 2 modifies the authors' method; it does not validate it. Specifically:

- **Stop** OA-1, I01–I05 development, Gate A/amended Gate A, `npz-tree-chunks`,
  queue leases, worker-hour ledgers to 15 decimals, IUT/Holm/power-simulation
  machinery, and multi-version handoff/memo/plan documents. Leave the code in
  place, frozen and unused. Add a one-line note to `README.md` saying it is
  parked.
- **Drop the gating.** H is measured directly. The true-model oracle remains
  useful only as a *diagnostic arm* (§3, E1), never as a gate.
- **Replace the endpoints (user decision, October 9).** The real goal is to
  **drive the pendulum angle to 0 and eventually stabilize the system**. The
  collaborator never guaranteed anything else. The user confirms that the rail
  ±2.4 m, the terminal |p|≤0.1 m, the recentering requirement and the ≥196/200
  gate were Codex's own inventions; remove them as criteria everywhere.
  - **Primary: angle stabilization.** Use the authors' own
    `success_metrics(...)["balanced"]` (`scripts/run_inverted_pendulum_mc.py:276`):
    |θ|≤0.02 and |θ̇|≤0.05 over the last 3 s. Add a *no-fall* condition
    (|θ|<π/2 throughout).
  - **Secondary: eventual full stabilization.** This is ṗ→0 (the authors'
    `cart_rest`) and p settling. Q_p=0.02 makes recentering slow by design (local
    time constant about 12 s), so judge it on a **long-horizon run** (60 s) for a subset of
    each IC set, in addition to the authored 14 s.
  - **The user explicitly wants railing and slow stabilization documented.**
    For every run, report:
    - max |p| (cart excursion);
    - time to angle settle (first time after which the primary criterion holds);
    - time to full stabilization (also |ṗ|≤0.05, and |p| change below a stated
      small rate);
    - final p;
    - max |u| and time at the ±10 input limit;
    - the fraction of steps on local feedback.

    Present these as distributions and maps, not pass/fail thresholds.
- **Replace the statistics.** Report successes/n with Wilson 95% CIs per
  condition. Compare arms on the *same* initial conditions with paired tests
  (McNemar, or a paired bootstrap on the difference). That is enough for this
  question.
- **Provenance:** one `results/<exp>/manifest.json` per experiment, holding the
  git commit, the config dict, the seeds, the device and the wall time. No hash
  pin chains or source snapshots beyond that.

## 2. Errors in the existing validation documents (fix these first)

| # | Error | Evidence | Fix |
|---|---|---|---|
| V1 | The handoff and memo say no development reproduction exists, but **the authors' own dev-profile closed loop is saved in the notebook**. All 6 initial angles (±15/30/60° from rest, 14 s) have balanced=True. cart_rest=True holds in 3/6 cases. The local-feedback fraction is 0.50–0.97 (0.97 at −15°, 0.94 at +15°, 0.50 at ±60°), and there are 6–70 CEM solves per case. | `manifold_control.ipynb` cells 10 and 12 | State this as the authored claim/evidence baseline in the results doc. Reproduce it (E0). |
| V2 | `run_inverted_pendulum_mc.log` is from an earlier autoencoder/latent-MPC version. Its output reads "one-step latent MPC", 10000 trajectories, manifold error 175 vs LQR 0.0024. Treat it as historical and do not cite it as evidence about the active CEM path. The same applies to `inverted_pendulum_mc_debug.ipynb` and `linear_system_mc.ipynb` outputs. | The log text vs the current runner | Note it once in the results doc. |
| V3 | The Gate A "3/200 / failed" framing rests on Codex-invented criteria (rail, recentering, ≥196/200), which the user has now withdrawn. Its count under the real goal is **147/200 angle-stabilized** for the *true model* from the full collection box. That box includes cart speeds up to 7.8 m/s and \|θ\|≤74°, far beyond the authored ±60°-from-rest demonstration. Separately, 67/200 also reach cart rest. | `overnight_runs/20261007T054549.695689Z/calibration/independent_audit.json` | Restate it in RESULTS.md as "oracle, wide box: 147/200 angle-stabilized, 67/200 also cart at rest; 103/200 exceed π/2 at some point", plus the excursion statistics. Remove "Gate A failed" wording from any new document. This is a diagnostic only. |
| V4 | The handoff told Claude that the paper was unreviewed and left R01–R04 to Claude. Claude has now retrieved and pinned it (see §4). | `reviews/paper_review_v1/source_manifest.json` | Finish R01–R04 using §4, with the reduced scope given there. |
| V5 | The existing TeX annotations (`reviews/implementation_tex_v2/annotations.json`) are accurate but incomplete. R16's "no confirmed false equation or prose/code mismatch" is true for the equations, but it misses the items in §4.2. | — | Produce annotations v3 (Q2 deliverable). |

## 3. Experiments (Q1, Q1.1, Q3)

Hardware: NVIDIA L4, 128 CPUs, and the existing `.venv-gpu`.

Measured costs:
- Dev training is about 14 s on CUDA, extrapolated from `artifacts/benchmark_training_cuda_1.json`.
- An H planning solve is about 0.07 s (notebook cell 12, `benchmark_control_*`).
- An oracle N=60 solve is about 0.21 s.

Everything below is cheap. Parallelize across CPU workers or GPU streams freely. Use the
**unchanged** submitted runner and classes for H (`scripts/run_inverted_pendulum_mc.py`,
`src/manifold_control.py`). New code goes in `validation/experiments/` and
imports the runner's functions rather than copying them. The only thing you
may parameterize is something the runner hard-codes, such as the IC list, the
episode count, the seed, or the plant parameters for mismatch tests.

Initial-condition sets (fixed, saved to disk, shared by all arms):
- **IC-A** (authored): θ0 ∈ {±15, ±30, ±60}°, all else 0.
- **IC-B** (from rest): θ0 on a 1° grid over [−85°, 85°], p0=ṗ0=θ̇0=0.
- **IC-C** (in-box random): 200 draws uniform from the collection box (`run_inverted_pendulum_mc.py:74-75`), seed fixed.
- **IC-D** (2-D maps): a (θ0, θ̇0) grid, 41×41, with p0=ṗ0=0. Add a (θ0, ṗ0) grid if budget allows.

Simulate for 14 s (authored). Use 20 s only if a result looks horizon-limited, and say so.

Arms. Each arm uses the same trained model and data unless the arm itself changes them.

- **H**: submitted hybrid, unchanged.
- **L**: the ridge/DARE local law alone, clipped (`fit_local_feedback`). It is the key ablation, because L runs 50–97% of H's steps on IC-A.
- **M**: the CEM-MPC alone, with no local switch.
- **M1**: the "vanilla NN-MPC" arm. M, but replanning every step (s=1, no queue). This is the Nagabandi-2018 / single-model-PETS equivalent.
- **H\***: H with the learned transition replaced by the true RK4 plant (the existing `OracleTransition` in `validation/models.py`), N=30, the same cost and switch. This separates model error from controller design. It is a diagnostic only.
- **E5** (Q3, optional after E2): a 5-member bootstrap ensemble with mean cost and no switch, as already specified. Treat it as PETS-lite.

### E0 — Reproduce the authored claim (do first)
- Run the runner as authored (`--profile dev`), then run training seeds 0–9.
- Report:
  - one-step and N-step test RMSE per state;
  - L-fit RMSE and the radius of the 6000-sample neighbourhood (max |θ| and max ‖·‖ inside it). Claude estimates it reaches |θ| of about 0.9 rad; measure it;
  - IC-A results for all arms.
- Answers: does the notebook result reproduce, and across how many seeds? How much of the success is L?

### E1 — Operating region and failure modes (Q1.1 "when does it fail")
- Run H, L, M, M1 and H\* on IC-B, IC-C and IC-D, using 3 training seeds.
- Produce success maps, plus failure categories:
  - fell (|θ|≥π/2);
  - never entered the switch region;
  - chattered at the switch boundary (count switch toggles);
  - balanced with cart drift;
  - nonfinite CEM.
- Diagnose 5 representative failures with prediction-vs-actual rollout plots. These separate model error from optimizer error from switch error.
- The existing CEM/local diagnostics may be cited as context.

### E2 — Data efficiency (Q1.1 "how much data")
- Episodes ∈ {35, 100, 350, 1000, 3500, 10000} × 5 training seeds.
- For each model, report:
  - test N-step rollout RMSE;
  - L-fit quality (the 6000-nearest rule becomes nonlocal at small data, and silently uses all data when L<6000; record this);
  - success on IC-A plus a fixed 50-IC subset of IC-C, for H, L and M1.
- Plot the success rate against training transitions.
- Also run a collection-distribution ablation: a box half as wide, and inputs only from the odd/uniform policy. This tests sensitivity to the data distribution.

### E3 — Robustness and novel data/systems (Q1.1 "novel")
Run at the dev data size on IC-A plus 50 IC-C, with 3 seeds:
- (a) measurement noise of 0.01/0.03/0.1 × state std on the controller input;
- (b) process noise of the same levels, applied as additive state noise per step in training data *and* in the plant;
- (c) plant mismatch: mass ratio and length ±25%, with the model trained on nominal data;
- (d) a one-sample input delay.

Novel systems: refactor the harness minimally so that n, m, box, costs and the plant are inputs, and run on:
- (i) a linear system: the existing `linear_system_mc.ipynb` plant, or a 4-state LTI. The optimal LQR is known here, so it gives a clean reference;
- (ii) one harder nonlinear plant (a double inverted pendulum on a cart, or an acrobot), with cost weights chosen by a stated rule rather than by tuning per arm.

Keep the method otherwise unchanged and record every per-plant choice.

### E4 — Baselines for Q3 (after E0–E2)
Run on matched data, the same IC sets and the same endpoint:
- (a) **M1**, the vanilla NN-MPC, already listed above;
- (b) **E5**, PETS-lite;
- (c) **DeePC**, regularized, on the cart-pole using input/state data (and an output-only variant measuring (p, θ)). `/home/ap60/github/data-driven-control/inverted_pendulum/src/pe.py` and `dataset.py` may be reused after a quick audit. External code in `external_repos/` may be used if it is checked against a hand-computed small case;
- (d) optionally, Koopman/EDMD-MPC (Korda–Mezić style). Lusch 2018 has no controller, so treat Deep Koopman as a *model* comparison: N-step prediction error at equal data, not closed loop;
- (e) optionally and last, SAC sample-efficiency on the same task, for a rough comparison of data budget against Baek 2024.

DFINE is an unforced, noisy, partial-observation latent model. Compare it in prose and, at most, on N-step prediction under measurement noise. Do not build a DFINE controller.

Cut order if time runs short: E4(e) → E4(d) → E3(ii) → E5 → IC-D second grid.

## 4. Verified paper and TeX findings (Q2, Q3) — build on these, do not redo them

### 4.1 Paper (pinned: `reviews/paper_review_v1/`, sha256 `87179e0c…08fa`, 13 pp, 8 refs)

- **Thm 3.3 is correct but elementary.** B_N^+ is the graph of the rollout
  map over the open domain D_N (Lemma 3.2), so the embedded C^r manifold of
  dimension n+Nm follows directly from f ∈ C^r. The paper does carefully handle
  openness of the domain. Cor 4.1, the fixed-x0 slice of dimension Nm, follows
  the same way. Cor 4.3 (E\*=π, D\*=Φ_N) is a restatement. Cor 4.4 (exact C¹
  autoencoder latent ≥ n+Nm) is a correct rank argument but weak; Remark 4.5
  concedes that approximate low-dimensional codes are not excluded.
- **The paper's own §9 reduces the program to "approximation of the ordinary
  finite-dimensional rollout map"** and defers learning, predictive control and
  stability to Part II (lines 60–63, 605–625 of `bayer_v1.txt`). In the
  full-state setting, that is standard learned-dynamics shooting MPC. The novelty
  must therefore come from somewhere other than the full-state geometry. The
  candidates are the output-only/IO case (Remark 5.1 states the injectivity
  issue but gives no result) and noise (the §8 noise discussion is informal).
- **The citations are thin** (8 references: Willems ×3, Markovsky, Willems et al.
  2005, Coulson 2019, Lee, Hartman). There are no nonlinear DeePC references
  (Berberich & Allgöwer; kernel/Hankel lifting; Lazar; Alsalti), no Koopman MPC
  (Korda & Mezić 2018), no learned-model MPC (Nagabandi 2018, PETS 2018,
  MPPI/PlaNet), and no delay embedding (Takens). Verify each omission from the
  primary sources (arXiv PDFs) before asserting it in R03.
- **Counterexamples to record for the output-only extension**, which the paper does
  not claim but which a reader may infer:
  - x⁺ = x, y = 0 gives an output behavior of dimension Nm, not n+Nm;
  - non-injective y = h(x) needs lag/observability conditions, as in Takens' theorem.
- **Bridge to the implementation:** the code uses Part I only as coordinates and
  dimension bookkeeping (34/154, the fixed-x0 slice searched by CEM).
  π∘Φ̂ = id holds by construction even for a useless model (critique-plan C04).
  No Part I result constrains or predicts closed-loop performance.
- Record the paper's AI-assistance disclosure only as metadata. It is not a
  ground for critique.
- **Reduced R01–R04 scope:** a claim ledger with about 15 rows (above), a
  proof audit, the literature matrix (§4.3), and an assessment. Reuse
  `handoff/plans/critique_plan.md` C01–C12, which Claude checked and found
  accurate.

### 4.2 Implementation TeX (Q2) — add to annotations v3
No equation contradicts the code: every equation (1)–(23) and every table value
matches `run_inverted_pendulum_mc.py` and `manifold_control.py`. Errors and
omissions:

1. **Symbol overloads:**
   - D is the episode count (5) and also the decoder D\*;
   - 𝒟 is the dataset, while 𝒟_N is an undefined domain;
   - j indexes training triples and also CEM iterations;
   - μ_z/μ_d vs the CEM mean μ^{(j)};
   - h is the network and also the hold integer;
   - p is the cart position and also the candidate index;
   - a is the SiLU argument and also the input bound;
   - l (length) vs ℓ_QR.
2. **Undefined or implicit items:**
   - 𝒟_N comes from Part I only;
   - x_ref = u_ref = 0 is stated in prose only, and the switch test x^TQx silently assumes it;
   - Q, R and Q_f act on the **normalized** coordinates of eq. (23), which the note never says. Q is also defined twice in code (`:201`, `:338`).
3. **Hard-coded equation numbers** ("(6)", "(13)"): use `\eqref`.
4. **Cosine schedule:** it is stepped per epoch (`T_max=epochs`), so the last epoch trains at a small nonzero lr. "Ending at zero" is approximate.
5. **Dead parameters in the collection rule:** k=0 always satisfies `k % h == 0`, so the initial h is never used. The initial u matters only for even episodes (`:79-87`). The note presents both as meaningful.
6. **"Near the origin" overstates the local fit.** The 6000 nearest of about 73.5k training transitions plausibly span |θ| up to about 0.9 rad; measure this in E0. The inputs are not localized. When L < 6000 (small-data runs) the fit uses all data.
7. **Switch-region geometry.** x^TQx<0.75 corresponds to |θ|<0.087 rad (5°), |ṗ|<0.50 m/s, |θ̇|<0.86 rad/s, but |p| up to 6.1 m, which is outside the 1.5 collection box. The test runs in float32 while the DARE runs in float64.
8. **CEM NaN edge:** `argmin` over costs containing NaN can hide the best finite candidate. If every cost is NaN, `best_u` is the warm-start mean and `finite=False` (`manifold_control.py:304-306`). The note's "nonfinite → zero" is correct but incomplete.
9. **Omissions:**
   - the evaluation protocol (six angles from rest, 14 s, no angle wrapping);
   - the CEM RNG is reseeded to 234 for every case;
   - CUDA nondeterminism;
   - `--reuse-data` does not check the stored config;
   - the plant is a *uniform rod* (COM at l/2, I = ml²/3), stated only in code;
   - the normalized step uses 1/√9.81, valid only for l = 1.
10. **Terminology:** "trajectory manifold" for im(Φ̂) is valid only on an open
    coordinate domain (or as a manifold with corners on the input box). Its
    dimension of 34 is true by construction, not an empirical intrinsic-dimension
    finding. It concerns the learned model, not the true B_N^+.
11. **Stale warm start after local feedback** (eq. 22): already annotated as R12.
    E1 should measure whether it matters, by counting failures that occur right
    after leaving the switch region.

Keep annotations to errors, qualifications and disclosed limitations. Do not
edit the labmate's text. Write a separate annotated copy, as v2 did.

### 4.3 Literature (Q3) — starting matrix, verify from primary sources
- **Lusch 2018:**
  - autonomous, full-state, noise-free Koopman autoencoder;
  - multi-step and linearity losses;
  - **no control**.
- **DFINE 2024:**
  - unforced, noisy, *partially observed*, high-dimensional data;
  - a manifold-plus-linear-latent model with a Kalman filter;
  - control is motivation only.
  - Its "manifold" is a state embedding, not a trajectory manifold.
- **Baek 2024:**
  - model-free SAC on a *triple* pendulum, in simulation and on hardware;
  - 0.8–2 M steps.
  - The tasks differ, so compare data budgets only qualitatively.
- **DeePC:**
  - IO data, nonparametric Hankel, a convex QP;
  - exact for noise-free LTI under persistent excitation;
  - regularized for noise.
- **The implementation:**
  - a full-state, one-step-trained NN, plus deterministic CEM shooting MPC with 5-step open-loop queues, plus a ridge-LQR switch;
  - algorithmically, that is Nagabandi 2018 / single-model PETS with an LQR catch.
- **Claimed benefits worth testing:**
  - sample efficiency compared with deep RL;
  - nonlinear reach compared with linear DeePC/LQR;
  - no physics in the model.
- **Drawbacks:**
  - needs the full state;
  - no uncertainty handling;
  - no guarantees;
  - one-step training only;
  - the queue reduces feedback;
  - a hard switch with no hysteresis.
- **The gap the framing could fill:** a nonlinear analogue of DeePC for IO data
  and noise. The implementation does not exercise it.

The final verdict must rest on E0–E4, not on this prior.

## 5. Caps, cadence and deliverables

- **Compute cap:** 40 GPU-hours plus the matching CPU hours, total. Log wall time
  per experiment in its manifest. If an experiment's projected time exceeds
  3× its estimate, stop it and report. Do not build an allocation framework.
- **Order:** E0 → E1 → E2 → paper/TeX deliverables (in parallel, no compute)
  → E3 → E4.
- **Deliverables**, all under `validation/results_v1/`:
  - `RESULTS.md`, a single living document organized by Q1, Q1.1, Q2 and Q3. Each claim links to a figure, table or file. Lead with the answers. Mark items `unresolved` where evidence is insufficient;
  - one folder per experiment, holding the manifest, the summary CSV and the figures. Raw arrays may stay local; record their paths;
  - `reviews/paper_review_v1/`: `claim_ledger.csv`, `proof_audit.md`, `literature_matrix.csv`;
  - `reviews/implementation_tex_v3/`: the annotated copy plus `annotations.json`.
- **Checkpoint with the user after E0+E1.** Give a short summary of whether the
  authored claim reproduces, how much L accounts for, and the operating region.
  Then continue.
- **Tests:** only for new experiment code that computes a reported number
  (success metric, IC generation, arm wiring). Do not test infrastructure.
- **Style:** short plain-language summaries. Do not write per-stage contracts,
  decision records or versioned plans. If you must deviate from this plan,
  add a dated line to `RESULTS.md` under "Deviations".

## 6. Running unattended under the usage supervisor

**Hard rule from the user: use only the 5-hour and weekly usage windows. Never
consume paid credits.**

You may be run in either of two ways:

- **Under the supervisor** (`validation/codex_supervisor.py`). It pauses you near
  the limits, resumes you after a reset, and kills you if credits drop. In this
  case, do not track usage yourself.
- **Self-managed.** Before each work chunk, and at least every ~15 minutes, read the
  newest `rate_limits` record in `~/.codex/sessions/**/rollout-*.jsonl`. It holds
  `primary` (the 5-hour window), `secondary` (the weekly window), each with a
  `used_percent` and a `resets_at` unix time, plus `credits.balance`.
  - Record the starting balance.
  - At ≥85% of the 5-hour window or ≥90% of the weekly window, finish the current
    step, leave detached jobs running, and wait until that window's `resets_at`.
    Use a single shell `sleep` until then, not repeated polling turns.
  - If the balance ever falls below its starting value, stop all model work,
    write `BLOCKED: credits consumed` to STATUS, and end.
  - Keep this as a short check. Do not build accounting around it.

Either way, your session may be interrupted mid-turn and resumed later with
"continue". Work so that this is harmless:

- **Status file.** Keep `validation/results_v1/STATUS`. Its first line is exactly
  one of the following; the rest of the file is a two-line note on what you are
  doing and what comes next:
  - `RUNNING`;
  - `CHECKPOINT_USER` (written after E0+E1, so the user can review);
  - `BLOCKED` (you need a user decision; say what);
  - `DONE`.

  The supervisor stops on anything except `RUNNING`.
- **Long jobs.** Launch any job longer than about 2 minutes **detached**
  (`setsid nohup … > results_v1/<exp>/run.log 2>&1 &`), and record its PID and
  command in `results_v1/<exp>/JOB`. Jobs then keep running while you are paused.
  Make jobs resumable, with per-case output files, and skip cases that are
  already done.
- **On every resume:** read `STATUS`, the `JOB` files and the tail of
  `RESULTS.md`. Check whether jobs finished or died, then continue. Don't redo
  finished work.
- **Commits.** Commit to git after each experiment completes. Code, summaries
  and figures are committed; raw arrays are not.

## 7. User addition — SMC baseline, October 11, 2026

After E2–E4 finish, automatically run the approved same-data discrete
boundary-layer sliding-mode feedback comparison and its zero-correction control.
The request is `results_v1/E6_SMC/REQUEST.json`, the research specification is
`results_v1/smc_research.md`, and the implementation is `experiments/e6_smc.py`.
The future E4 entry point and resumed continuation both call the idempotent
`experiments/queued_smc.py` launcher; do not restart active E2–E4 jobs to adopt it.
Reuse saved nominal/process-trained linear fits, IC-A and the fixed 50 IC-C,
three seeds, the existing long-horizon subset and affordable E3 stress conditions.
Use CPU feedback only, with at most one additional CPU-hour/wall-hour inside
the original overall cap. Preserve all usage-supervisor thresholds and its
credit guard. A recorded cap deferral or partial result is acceptable; do not
silently enlarge the cap or cut required E2–E4 to make room.
On resume, also inspect `results_v1/E6_SMC/JOB` and its manifest. Keep STATUS
RUNNING while that job is pending or active; finalize interpretation and commit
the SMC code, request, summaries and figures after it reaches DONE, PARTIAL or
SKIPPED. An ERROR requires diagnosis. No SMC experiment is authorized to start
in this interactive setup session.
