# Executable stress-test protocol: TrajectoryManifoldControl

Revised October 6, 2026. CPU/CUDA readiness measured. No scientific validation campaign run. Lock amendments before final evaluation; experiments cannot prove guarantees.

## Repository, scope, and claims

**System under test: `TrajectoryManifoldControl/`, active runner `scripts/run_inverted_pendulum_mc.py` and `src/manifold_control.py`.** Pin `325beae718ab093d3ca48b18a59ae61fd93798a3` (record any replacement). Preserve submitted controller; new harness/artifacts belong in `TrajectoryManifoldControl/validation/`. `inverted_pendulum/` supplies candidate baselines only: audit/port algorithms, units, plant/data access before reuse; never substitute its plant or archived scores. `external_repos/` is reference code, not automatically a controlled baseline.

Part I assumes deterministic full-state Cr dynamics (r≥1) on an open domain. Claims: T1, nonempty N-step behavior is an embedded manifold of dimension n+Nm; T2, (x0,u) is a global Cr chart; T3, nonempty fixed-state slice dimension Nm; T4, exact C1 reconstruction requires latent p≥n+Nm; T5, smooth held-input flows/admissible RK4 inherit this. Learning/convergence/stability are deferred. Boundaries/noise/output-only need qualifications. Audit proofs separately; experiments cannot prove T1–T5.

Part I addresses behavior geometry; the implementation specifies computations, not comparative performance. Assess gap closure/novelty in `review/critique_plan.md`. The labmate's verbal efficiency/interpretability claims are not PDF claims.

Reviewer-defined empirical questions: Q1 superiority; Q2 effects of switch/local feedback, queue, predictor, planning, collection; Q3 robustness/failures; Q4 acquisition cost. Targets/margins are ours, not document promises. Jacobian/counterfactual errors are predictor diagnostics, not interpretability tests. Map runs to unresolved practical questions; keep correctness, novelty, clarification, and performance verdicts separate.

| Stage | Tests | Permitted conclusion |
|---|---|---|
| 0 | Implemented T2/T3 coordinates, T5 plant scope, endpoint | Numerical correctness/equivalence and attainability; no geometric proof |
| 1 | Reviewer-defined Q1/Q2 plausibility | Development compute triage; no document/novelty verdict |
| 2 | Reviewer-defined Q1/Q4 | Distribution-specific superiority/efficiency; not a theorem or necessary condition for novelty |
| 3 | Q2/Q3; full-state/deterministic assumption boundaries | Component effects/practical limits; out-of-scope failures do not refute theory |
| 4 | Q3 transfer/theorem applicability | New-task evidence under a separate protocol |

## Primary hypothesis and multiplicity

At **105000 newly acquired transitions per seed bundle**, under the common split and locked recipes below, define pH, pL, pE as marginal nominal strict-success probabilities for original hybrid H, identified LQR L, and ensemble NN-MPC E. The **one primary contrast** is Δ=pH−max(pL,pE)=min(pH−pL,pH−pE). Test **superiority**, H 0: Δ≤0.05 versus H 1: Δ>0.05. The 5-point margin screens trivial gains; nonrejection is not equivalence. Our target requires pH estimate≥0.95/one-sided 95% LCB≥0.90, not a 95%-success guarantee. Full-budget superiority does not decide scientific contribution.

Use paired outcomes/crossed bootstrap: both one-sided **95% component bounds** for pH−pL and pH−pE must exceed 0.05 (intersection-union test, alpha 0.05). Under H0 at least one component null is true; rejecting both requires rejecting it. These are not simultaneous 95% bounds. Simulate boundary and mixed-null type-I calibration. Report Δ/both bounds/seed results once; no final-panel peeking.

The acquisition ratio is one key-secondary test at one-sided 0.05, gated on primary rejection (fixed sequence). Other full-budget tests—conditional common cost, rail/fall violation probability, deadline-miss probability versus each baseline—form six tests, Holm-adjusted at 0.05 and gated on both primary and key-secondary rejection (sequential family control). Report failures separately from cost among successes. Ablations, stresses, other families, fidelity, and curve sensitivity are exploratory with descriptive intervals, not uncorrected significance claims.

## Data access, splits, and common randomness

Keep collection episodes at 30 transitions, independent of MPC horizon. Coarse budgets are E={10,35,100,350,1000,3500} episodes, T=30 E. **For fair validation, remove the in-budget test split and use 80% training/20% validation; final testing uses a separate frozen set.** Thus full-budget training uses 84000 transitions, unlike the original reproduction's 73500. Label this shared-split evaluation of the original algorithm; it is not the untouched dev experiment.

Assign episode IDs 5 j+4 to validation and other IDs to training before window extraction, with randomized episode order fixed per bundle. At E=10 use 8 train/2 validation; E=35 use 28/7. Finer E uses floor(E/5) validation (E≥10); record counts. Fit scaling/local ID on training only. No cross-episode windows; failed DARE remains failure, without changing H’s nearest-6000 rule.

Seed bundles independently randomize acquisition/training; methods share episodes/split IDs. Freeze IID nominal case states shared across bundles/methods/budgets. Save dataset_seed[b], training_seed[b,m], case_seed[c]. **Share indexed per-case/per-solve CEM standard-normal tensors across CEM methods**; index by case/time so branching cannot cause RNG drift. Share per-case/time noise/disturbance arrays; ensemble bootstrap/member randomness has separate indices. Share observations/actions/resets; no true-state feedback under noise. Label oracle/units priors. Match costs/horizon/planning unless explicitly adapted.

## Stage 0: correctness, attainable scoring, and equivalence

Pin sources/configs/dependencies/device/precision and reuse hashes; run relevant existing tests. Check rod inertia/mass matrix, signs, equilibria, units, zero-input energy, and RK4 convergence against a high-accuracy solver. Active dt=0.02 s,N=30,s=5,|u|≤10 (98.1 N). Reproduce original modulo collection/dev split and six angles once; not final evidence.

**Equivalence test only here:** implement an independent ordinary-shooting wrapper with the exact same transition weights, cost, CEM draws, clipping, switch, queue, and warm-start rule. Test 50 scripted cases ×100 samples, including bounds, switch crossings, local exits, queued execution, and nonfinite-cost fallback. Require identical branch flags/queue lengths and allclose on actions, plans/queue values, and costs: atol=1e−5, rtol=1e−5; use the same device/dtype and replayed RNG tensors. Record max errors. Tolerance allows float32 ordering; diagnose failures. On passing, **remove from sweeps**. Passing establishes tested numerical agreement, not all-input equivalence or causal attribution. The ordinary-shooting hybrid shares measured benefits; the test isolates no distinct geometric algorithm. Geometry may still clarify/motivate the construction; assess that separately.

**Endpoint:** 20-second balance-recovery episodes, 1000 plant samples. Score on the **true plant state**, including under observation noise. Strict success requires, throughout, finite states, |p|≤2.4 m and |theta|≤pi/2, and throughout the final 3 seconds |theta|≤0.02 rad, |theta_dot|≤0.05 rad/s, |p_dot|≤0.05 m/s, |p|≤0.1 m. These defaults add reviewer-defined rail/position safety; also report original metrics. Exceptions/deadline misses fail; report failures and success-conditional cost separately.

**Attainability calibration:** on 200 development draws from the normalized box with half-widths (1.5,2.5,1.30,3.5) in (p/l,p_dot/sqrt(lg),theta,theta_dot/sqrt(g/l)), use true-model MPC plus local oracle feedback,N=60,max 5 planning starts/state. Require≥98% physical strict success and≥95% in each angle-sign half; oracle timing is exempt here. If not, stop Gate A, diagnose optimizer defects, then amend nominal task before final data. Default fallback: multiply all original box half-widths by 0.5, then 0.25 if needed; retain thresholds and broad-box stress reporting. Freeze the box before pilot. This is an attainable reference task, not a controllability certificate or permission to tailor to H.

**Simulated deadline rule:** predictive deadline is 0.1 s. Measure synchronized end-to-end control/planning latency on fixed hardware. If late, discard the late plan, hold the last applied input for the next five plant samples (zero if none), clear pending/previous plans, then replan from the new observation; do not jump into a privileged rescue controller. Continue the trajectory but strict success is false. Local feedback has a 0.02-s deadline; on a miss hold the prior input for one sample and mark failure. Log all fallback steps; simulation still advances at dt=0.02 s. The deadline verdict is hardware-specific.

**Pending amendment:** server readiness should consider timing-agnostic primary success with separate deadline-enforced operational scoring. Keep the current endpoint until accepted before lock; hardware failures do not refute geometry.

Gate A: save evidence; stop dependent work on failure.

## Stage 1: full-budget kill check and pilot

Freeze E’s recipe first. Use 3 development bundles/50 shared cases at T=105000 for **H/L/E**. E is five episode-bootstrap NN members, mean member trajectory cost, CEM (256 candidates,32 elites,3 iterations,N=30,s=5), no switch; **not full PETS**. Single-NN-MPC without switch: optional full-budget diagnostic.

**Gate B kill defaults:** let d=pH−max(pL,pE) on development data. Stop broad confirmation if (i) d≤0.03 and the one-sided 90% crossed-bootstrap upper bound for d is ≤0.05; or (ii) H has ≥10 percentage points more all-time rail/fall violations than either baseline. If d≤0.03 but uncertainty is wider, add exactly 2 bundles and 50 new common cases (5×100 total), once; unresolved evidence means pause/inconclusive, not “matched.” Go if d>0.03, no safety kill, and cap feasible; else apply the one expansion/pause. Triage is not inference.

On kill, cut full confirmation, optional families, and transfer; continue the separate claim/proof/gap critique. Retain H/L/E at 1050/10500/105000 on existing bundles/cases within 4 additional worker-hours; report and stop. This checks a masked low-data benefit without claiming equivalence. Full-budget matching does not settle contribution; exploratory low-data gains can motivate a separate preregistered study, not a revised primary hypothesis.

If Gate B passes, pilot H/L/E at 1050/10500 using the same 3 bundles/50 cases. Record failures, paired variance, timings, local-fit radius, coverage, equilibrium residual, and rollout errors at 1/5/30/60 steps. Development evidence never enters final testing.

## Concrete tuning protocol

H keeps submitted architecture/training/cost/switch/queue/CEM; L uses its local identification. Audit adapters; keep H fixed and disclose unknown historical tuning costs.

Tune each eligible family **once, exactly 12 configs** (bounded equal search), not per budget, on one shared independent pool of 3×350 episodes=31500 transitions (80/20). Select mean standardized 30-step validation RMSE across 3 bundles; ties: smaller then faster. Reuse data/freeze recipes; per-budget checkpointing allowed. No extra plant queries.

| Family | Exactly 12 configurations (Cartesian product) |
|---|---|
| Single NN (exploratory) | hidden widths {64,192,256}, three layers × weight decay {1e−6,1e−4} × LR {3e−4,1e−3}; SiLU residual, batch=min(2048,T_train), max 100 epochs, best validation checkpoint |
| Ensemble E (required) | same 12 NN recipes; five independently initialized episode-bootstrap members; mean-cost MPC, no uncertainty penalty; identical epoch cap/CEM |
| GP (optional) | ARD RBF/Matérn-3/2 kernels × max training points {256,1024,2048} × increment/output-noise floors {1e−4,1e−2} in standardized units; deterministic max 200 marginal-likelihood iterations; inducing/subset recipe fixed and labeled |
| SINDy (optional) | standardized discrete increment libraries {degree 2, degree 3, degree 2+sin/cos(theta)} × sparsity thresholds {0.001,0.03} × ridge {1e−6,1e−3}; max 20 threshold/refit iterations; no true derivatives |
| Controlled Koopman (optional) | state lifts {degree 2, degree 3, degree 2+sin/cos(theta)} × ridge {1e−6,1e−3} × control structure {additive,bilinear}; retain state coordinates, discrete least squares, MPC adapter |

Lock subset/features/scales/decoding; physics features are priors. Adapters are not paper reproductions. DFINE/SAC require separate protocols.

Charge D=31500 tuning transitions/method to A=D+T, including H/L (label unused data); report actual-overhead sensitivity and fit budget T. Count development/oracle/final evaluations separately in total project interactions. If development outcomes change recipes, add their acquired transitions to D and recompute. No evaluation data returns to training. Count resets/failed collection; model rollouts are synthetic.

## Stage 2: resolution, power, and locked confirmation

**Success curves:** fit marginal success versus log T by method with a monotone piecewise-linear logit spline, knots at six coarse budgets, nonnegative slopes, ridge 1e−4 on slopes, logits bounded±12, binomial composite likelihood over seed×case outcomes. This is a working interpolation, not a learning theorem. Compute intervals from **5000 crossed bootstraps**: independently resample seed bundles and case IDs, preserve pairing across methods/budgets, refit each time. Uncertainty covers **both new training bundles and new nominal cases**; repeated cases are shared factors.

T95,m is the fitted p=0.95 crossing within observed budgets. Qualify reliability only if fitted p≥0.95 and one-sided 95% LCB≥0.90 there. Missing crossings/failures are censored, never extrapolated or removed from bootstrap draws. Unbounded ratio bounds mean efficiency unestablished.

After pilot add≤4 common finer budgets: rounded whole-episode crossings×{0.9,1.1} for H and predicted best baseline, clipped to 10…3500 episodes and deduplicated. Unknown crossings use two log-uniform points in the nearest target bracket, or near the upper boundary if unbracketed. Freeze all≤10 budgets before final testing. Show points/fits; sensitivity: unconstrained spline and logit-linear fits. Material nonmonotonicity or decision-changing fit sensitivity makes efficiency unresolved.

**20% key-secondary:** R=(D+T95,H)/min(D+T95,L,D+T95,E). Require one-sided 95% upper bootstrap bound≤0.80, reliability-qualified H/denominator, agreement under both sensitivity fits, and primary rejection. This is our 20% criterion, not a document guarantee. Unresolved crossings are inconclusive; if primary fails, report exploratory curves/ratio.

**Power sizing:** fit a multivariate crossed-effects Bernoulli model to pilot data: correlated Gaussian method-specific seed effects, shared Gaussian case effects, Gaussian seed×case residuals; choose intercepts by integration to match marginal success. Simulate 2000 experiments/design for S∈{8,12,16,24},C∈{100,200,400}, applying the 95%-component IUT and absolute-target rules (1000 bootstraps for sizing,5000 final), including boundary-null type-I checks. Stress pilot uncertainty: logit seed SD{0.25,0.5,1.0}, method correlation{0.3,0.7}; three seeds are imprecise.

Select least measured worker-time design achieving≥80% power at Δ=0.10 and≥80% probability of estimate≥0.95/LCB≥0.90 at true pH=0.97, with baseline probabilities 0.87 and pilot heterogeneity. Publish power at 5/7.5/10-point gains and pH=0.95 (boundary target attainment lacks high assurance). Require conservative scenarios to pass. If no capped design qualifies, pause/insufficient power; do not lower margins or replace seeds with cases.

Forecast S=12,C=200,B=10 awaits sizing. H/L/E mandatory; final IID cases independent/hidden until lock. Save source/data/split hashes, configs/checkpoints, true/observed states, actions/branches/queues/plans, predictions, failures, timings, seeds, and interaction counts.

## Compute contract and cut order

### GPU execution amendment — October 6, 2026

The intended device for H/E training and CEM prediction, and Gate A oracle
prediction, is **CUDA on the host NVIDIA L4 (24 GiB, approximately 3 GiB used
according to the user's host nvidia-smi reading)**. This is a host observation,
not a completed GPU benchmark. The submitted NumPy plant and local LQR remain
on CPU; never replace the plant or change oracle float64 to fit a device.
Use one CPU thread per worker, float32 learned models, deterministic PyTorch
algorithms, and TF32 off. Freeze GPU UUID, driver, CUDA wheel/runtime, software
hashes and concurrency before development comparisons and confirmation.

The current managed execution session has a private /dev mount without NVIDIA
device nodes; its nvidia-smi/CUDA failure does not contradict host availability.
Use the host terminal in which nvidia-smi succeeds. A private offline CUDA venv
is prepared under validation/.venv-gpu (PyTorch 2.5.1+cu121, bundled CUDA 12.1);
no driver/toolkit reinstall is justified by the session-visibility failure.
The launcher must successfully allocate/synchronize CUDA tensors and validate
oracle RK4 numerics; no silent CPU fallback. In a managed execution context,
GPU device exposure must be supplied by that environment's administrator.

Run from the workspace root:

```bash
bash TrajectoryManifoldControl/validation/run_gpu.sh probe
bash TrajectoryManifoldControl/validation/run_gpu.sh readiness
```

Readiness stops on failed GPU equivalence (50 cases, same 1e-5 tolerances and
branch/queue requirements) before smoke or downstream benchmarks. It shares
the existing **one-worker-hour readiness ledger**, including the completed CPU
work; if the remaining reservation is insufficient, report skipped jobs rather
than reset usage. It performs only tiny smoke and performance measurements,
not oracle calibration, tuning, confirmation, or transfer. Save synchronized
H/E latency quantiles including switch/LQR, queue and transfers; N=60 float64
oracle cost; full-size 84000-transition training throughput; RAM/VRAM; and
1/2/4 workers on the same GPU with CPU threads=1. Start with **one GPU worker**;
choose concurrency from measured aggregate throughput and memory headroom,
then verify end-to-end deadline behavior under that concurrency before lock.
Do not infer free VRAM from the earlier approximately 3 GiB observation.

Replace CPU forecasts with device-specific measured tau_H/tau_E/tau_oracle
and GPU-smoke replan rate. Report BOTH the s=1 reservation and projected
runtime versus unchanged stage caps, plus wall-time from aggregate solves/s;
concurrency does not reduce worker-hours. The CPU readiness estimates are
reference only (Stage 2: approximately 2601/526 worker-hours, cap 48).
Host GPU readiness has now completed: mean tau_H/tau_E/tau_oracle are
**0.074332/0.399123/0.214193 s**; H/E planning p99 **0.10133/0.43733 s** misses
the 0.1 s deadline. Training is faster, planning is slower than CPU in these
performance workloads. Four GPU workers achieve **56.89/14.81 H/E solves/s**;
end-to-end deadlines under contention remain unmeasured. Readiness used
**0.42929 worker-hours** total. CUDA Stage 2 is approximately **3158/633
worker-hours** worst/projected versus cap 48. No cap feasibility is inferred.
Detailed measurements and both-device stage tables are in
[server_readiness.md](../TrajectoryManifoldControl/validation/server_readiness.md).
Performance-only results are not scientific validation. Deadline misses still
count as strict failures under the current endpoint; the proposed timing-agnostic
primary amendment remains pending, so device choice can change primary outcomes.

An unattended Gate A queue is prepared in
[validation/overnight.md](../TrajectoryManifoldControl/validation/overnight.md).
It checks runtime/source provenance, calibrates 200 cases with one start, audits
saved true-plant traces and writes a morning report. No Stage 1, tuning,
confirmation, extra starts, box shrink or transfer is queued. The default Stage-0
cap remains 8 h. A launch explicitly requesting **--stage0-cap-hours 15** is a
reviewable allocation override for this Gate-A-only queue, effective only when
the user executes it. Prior readiness/queue occupancy is included; calibration
is capped internally at 14.4 h (process slot 14.42 h). A failed/incomplete gate
stops experiments; a passed gate also waits for morning review before Stage 1.

After GPU readiness and allocation review, preflight Gate A on CUDA:

```bash
bash TrajectoryManifoldControl/validation/run_gpu.sh gate_a \
  --cases 200 --starts 1 --box-scale 1 --max-worker-hours 7
```

This command does not execute calibration. Its cap is a conservative initial
proposal; compute the available Stage-0 allocation as 8 hours minus actual
readiness usage and any other Stage-0 work. The CUDA timing file is required;
never use CPU timings to reserve GPU work. **One-start CUDA Gate A needs 11.90
planning hours, or 14.28 hours with 20% overhead**, exceeding this proposal and
the entire 8-hour Stage-0 cap. The preflight was run without --execute; do not
launch until an explicit allocation/device amendment resolves this. A CPU
oracle alternative would require an explicit device-rule amendment and a
review of its 7.65-hour reservation against the 7.57-hour Stage-0 remainder. Keep one start initially (within
up-to-five allowance); allow extra starts only after a new measured reservation
fits. Add --execute only for the separately authorized calibration allocation.


For stage j, approximate CEM solves by

    solves_j = sum over (method,budget,bundle,case) ceil(steps_j / s_method)
    hours_j = [sum(solves_j,m * tau_j,m) + training_j + collection_j
               + scoring/logging_j + statistical_analysis_j] / 3600

Times are worker-seconds; tau_m measures synchronized end-to-end solves. Switching increases replans: **reserve steps_j solves (s=1) as the hard upper bound**, not steps/s. Meter solves/worker-time; reject blocks exceeding remaining worst-case reservation and stop incomplete at cap. Caps sum worker-time across concurrency. Include tuning/power/bootstrap compute.

| Stage | Planning solve estimate | Hard cap (worker-hours) |
|---|---|---|
| 0 | 2×50×20 equivalence solves +200×200 oracle replans×up to 5 starts; reproduction extra logged | 8 h |
| 1 | 3 bundles×50 cases×3 budgets×2 CEM arms×200 replans =180000; single-NN diagnostic adds 30000; extension separately projected | 12 h, including required ensemble tuning; kill follow-up≤4 h within this cap |
| 2 | S×C×B×2 CEM arms×200; default 12×200×10×2×200=9.6 million | 48 h |
| 3 | maximum 3 stresses×50 cases×3 bundles×2 CEM arms×200=180000, plus optional diagnostics | 8 h |
| 4 | no committed solves | 0 h until Gate C approval |

With `tau_H = [measured CUDA pilot seconds]`, `tau_E = [measured CUDA pilot seconds]`, default Stage 2 planning takes 4.8 million×(tau_H+tau_E)/3600 hours plus training/collection/statistics. Example only:0.005/0.015 s yields 26.7 h. Worst-case48 million solves. Caps bound feasibility work; check before launch. CPU performance readiness completed; CUDA campaign execution remains pending.

If capped: cut optional GP/SINDy/Koopman training and sweeps first, then single-NN diagnostics, then Stage 3/transfer, then curve refinement/coarse budgets far from crossings (minimum four common budgets bracketing targets). Never cut L/E, replace E with a weaker baseline, reduce the power-selected full-budget primary design, or discard failures to fit. If mandatory tuning or the primary design alone exceeds cap, stop and request an explicit revised compute allocation. Statistical analysis retains its own reservation before simulation starts.

## Stage 3 and Gate C

Exploratory diagnostics: full and one target-adjacent budget,50 cases/3 bundles; switch removal first, then s=1 and fresh warm-start. Label extensions; predictor fidelity uses 100 held-out states, not an interpretability verdict.

Prioritize 3 stresses within 8 h: noise SD 0.05×fixed nominal physical state scales, one-sample action delay, mass/inertia+25%; shared seeds/true-state scoring/deadline rules, no retraining. Label shifts and empirical failure maps, not attraction certificates.

Gate C requires resolved correctness and a remaining transfer question. Stage4 placeholder: Pendulum-v1 or MountainCarContinuous-v0, with pinned Gymnasium version, shared objective, topology/clipping/smoothness audit, and a separate capped protocol. No transfer runs committed; discrete actions require declared planner changes.

---

## One-page decision summary

**Scope:** pinned active `TrajectoryManifoldControl/`; other repos are audited baseline sources. Empirical tests assess reviewer-defined practical improvement, not promises attributed to the PDFs. Gap/novelty critique remains independent.

**Primary:** H versus better of LQR/ensemble at 105000 additional transitions, superiority margin 5 points. Both one-sided 95% component bounds exceed 0.05 (IUT, not simultaneous coverage). Target: estimate≥95%/LCB≥90%. Failure is not equivalence or no contribution.

**Endpoint:** true-state 20 s recovery, |p|≤2.4 m/|theta|≤pi/2 throughout; final 3 s tolerances0.02 rad/0.05 rad/s/0.05 m/s/0.1 m. Exceptions/deadline misses fail. Oracle≥98% on 200 development cases locks nominal box. Timing-agnostic scoring is a pending amendment.

**Kill:** Gate A stops on correctness/calibration; B cuts broad confirmation at gain≤3 points/90% upper≤5 points or≥10 points extra violations. One expansion 3×50→5×100; then pause. Retain≤4 h three-budget diagnostics and document critique, no transfer.

**Efficiency/inference:** monotone logit spline,5000 crossed seed/case bootstraps,≤4 finer budgets; tuning overhead 31500/method. Gated secondary ratio upper≤0.80 plus reliability/sensitivity meets our 20% criterion; otherwise exploratory/inconclusive. Shared episodes/cases/indexed CEM-noise streams;10/35splits8/2,28/7. No extrapolation/dropped censoring.

**Device:** CUDA on the host L4 for learned/oracle prediction and training; original NumPy plant/local LQR on CPU. Complete GPU readiness and freeze measured concurrency/software before launch; no CPU fallback or assumed GPU speedup.

**Power/compute:** simulate 8–24 bundles×100–400 cases: 80% power at 10-point gain,80% target attainment at true 97%, conservative variance/type-I checks. Default 12×200×10:9.6 million solves;4.8 million×(tau_H+tau_E)/3600h plus overhead,48 million worst-case. Caps0/1/2/3=8/12/48/8 worker-hours; transfer 0. Cut optional arms/stresses/nonessential budgets first; preserve powered H/L/E or stop insufficient-power.
