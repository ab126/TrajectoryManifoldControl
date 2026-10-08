# TrajectoryManifoldControl validation handoff

Prepared October 8, 2026; updated after the completed host CUDA diagnosis. This is a handoff of completed work and proposed remaining work, not authorization to launch every experiment below. **Gate A failed; Stage 1, tuning, confirmation, robustness sweeps, and transfer remain stopped.** Both bounded diagnostic phases are complete. Specify a versioned oracle amendment and prepare its configuration/audit support before another calibration. The submitted controller and original plant remain unchanged.

## 1. Scope, provenance, and plans to carry forward

System under test: `scripts/run_inverted_pendulum_mc.py` and its active path through `src/manifold_control.py`, originally evaluated at **`325beae718ab093d3ca48b18a59ae61fd93798a3`**. Current HEAD is **`17d42cd83dea810af5b2979e657f0074244f2cc4`** (`Validation work`): the user committed the validation package; the submitted source is unchanged relative to the evaluated commit. At review start, only `validation/diagnostics/usage.jsonl` was modified and the new CUDA diagnostic directory was untracked. This follow-up additionally changes validation reports/review code. The handoff manifest records current status and checks submitted source hashes against readiness. Raw artifacts are ignored by Git and require a separate archive. No applicable repository instructions were found during readiness. The parent directory is not a Git repository.

Full plans, rather than only this summary, are included as byte-for-byte snapshots:

| Document | Snapshot included with this handoff | Role |
|---|---|---|
| Initial stress-test plan, before GPU amendment | [stress_test_plan_original.md](handoff/plans/stress_test_plan_original.md) | Original campaign, gates, endpoint, statistical and compute contracts |
| Current stress-test plan | [stress_test_plan_current.md](handoff/plans/stress_test_plan_current.md) | Authorized GPU execution amendment and allocation history |
| Original critique plan | [critique_plan.md](handoff/plans/critique_plan.md) | Claim ledger, proof audit, literature comparison, contribution assessment |
| Editable implementation-note transcription | [trajectory_manifold_implementation.tex](handoff/sources/trajectory_manifold_implementation.tex) | Equations (1)–(23), page markers, annotation macro; reconstructed typography |

Live plans are [stress_test_plan.md](../../review/stress_test_plan.md) and [critique_plan.md](../../review/critique_plan.md). Snapshot/source hashes and current evidence inventory are in [manifest.json](handoff/manifest.json). Neither plan nor the original transcription was edited for this handoff. The original implementation and Part I PDFs were not found in the present workspace; obtain and hash the exact supplied PDFs before a page-by-page proof/transcription audit. Earlier review statements describe their inspection at that time, not current PDF availability.

The request initially mentioned a 48-character invalid commit and 97.5% Bonferroni component bounds. The plan actually available on disk already used the valid 40-character original evaluated commit above and the correct **95% intersection-union component bounds**. Preserve this version distinction. Historical readiness documents contain statements such as “calibration unexecuted”; their later follow-up sections and the completed run's `run.json` supersede those statements.

`inverted_pendulum/` contains candidate baseline implementations only. None has been audited or reused. Never substitute its plant or archived results. `external_repos/` is reference material. New adapters, model variants, and artifacts belong under `validation/` and need explicit labels distinguishing original H, oracle reference, and experimental alternatives.

## 2. Original campaign and current completion status

The stress campaign tests reviewer-defined practical performance; it does not convert geometric statements into learning or stability guarantees. The independent document critique can proceed while experimental gates are stopped.

| Original work package | Planned deliverable / decision | Current result or gap |
|---|---|---|
| Readiness and cheap Stage 0 | Plant/units checks, ordinary-shooting equivalence, indexed RNG, existing tests, smoke, resource/timing estimates | Implemented and executed on CPU and host CUDA where applicable; limitations below |
| Original dev reproduction | Original collection/split/training/evaluation at submitted dev budget | Tiny smoke only; complete 3500-episode/100-epoch reproduction not performed |
| Gate A | 200 development cases, original plant, true-model MPC N=60, up to five starts; at least 98% overall and 95% in each angle-sign half | One-start, full-box CUDA run completed: **3/200**; audit passed; gate failed |
| Failure diagnosis | Separate local recentering, objective/search, safety and recoverability explanations | Local and fixed-state GPU CEM phases completed; lower search cost did not improve combined predicted safety; broad-box viability remains unresolved |
| Amended Gate A | Specify reference/task changes, audit them, recalibrate, freeze a passing nominal box | Proposal only; no amendment adopted, no box shrink or rerun |
| Stage 1 / Gate B | Freeze E; H/L/E full-budget kill check on 3 bundles × 50 common cases; then lower-budget pilot if informative | Not implemented as a complete campaign; no tuning or H/L/E comparison executed |
| Stage 2 | Power-selected, locked H/L/E confirmation and qualified data-efficiency curves | Inference rule scaffold only; simulation, curve fitting, final-case lock, and campaign runner absent |
| Stage 3 | Switch/queue ablations, model fidelity, at most three prioritized robustness stresses | Not executed; some model metric routines exist, but no study protocol/adapter/results |
| Stage 4 / Gate C | Separately justified Gymnasium transfer with pinned version/topology audit | Not implemented or authorized; committed cap is zero |
| Critique A | Claim-to-proof-to-code ledger and gap-to-contribution matrix | C01–C12 preliminary assessments exist in critique plan; final CSV/matrix absent |
| Critique B | Independent proof audit and annotated implementation note | Preliminary local reasoning and plant derivation only; formal independent audit/annotations absent |
| Critique C | Primary-source, assumption-matched literature/novelty review | Leads and comparison categories exist; verified full review absent |
| Critique D | Separate correctness, novelty, faithful specification, and practical added-value verdicts | No final verdict; avoid deriving a novelty verdict from failed control experiments |

Stage 1's mandatory methods are **H** (submitted neural MPC + identified local LQR), **L** (the same data-identified LQR), and **E** (five independently initialized, episode-bootstrap neural models, mean trajectory cost, CEM, **no local switch**). E is not full PETS. Optional single-network MPC, GP, SINDy, Koopman, DFINE adaptations, and SAC cannot replace E or L.

Coarse acquisition budgets are 10/35/100/350/1000/3500 episodes of **30 transitions**, i.e. 300/1050/3000/10500/30000/105000 transitions. Campaign splitting is 80/20 by episode; the original runner uses 70/15/15 and has 73500 training transitions at its dev budget. The campaign's full training count is 84000. Both protocols must be recorded without conflation. Fixed tuning overhead is D=31500 transitions per method, including unused access charged to H/L; acquisition comparisons use A=D+T. Recipe-changing development interactions require an additional accounting review.

The locked endpoint is 20 physical seconds / 1000 samples: all states finite, all-time |p|≤2.4 m and |θ|≤π/2; throughout the final three seconds |p|≤0.1 m, |ṗ|≤0.05 m/s, |θ|≤0.02 rad, |θ̇|≤0.05 rad/s. Exceptions and deadline misses fail H/L/E under the current rule. The original runner's narrower metrics must also be reported, but do not substitute for this endpoint. Oracle calibration alone is timing-exempt.

**Endpoint provenance:** the rail/position safety requirements were added by the validation plan, not derived from the rod equations or a modeled physical rail. The active CEM solver and local LQR enforce only input clipping, |u|≤10 (98.1 N); neither enforces hard state bounds. Quadratic position/angle costs are soft penalties. The simulator continued after rail/angle crossings. Keep this added strict safe-recovery claim separate from the original balancing metric; any endpoint change must be explicitly adopted before lock.

## 3. Completed evidence and what it establishes

### Plant, equivalence, tests, and smoke

See [plant_review.md](plant_review.md), [server_readiness.md](server_readiness.md), and the raw JSON/logs under `artifacts/`.

The original uniform rod has cart mass 2 kg, rod mass 1 kg, full length 1 m, COM distance l/2, I_COM=ml²/12, and I_pivot=ml²/3. With θ positive toward positive cart displacement from upright:

\[
\begin{bmatrix}M_c+m&ml\cos\theta/2\\ml\cos\theta/2&ml^2/3\end{bmatrix}
\begin{bmatrix}\ddot p\\\ddot\theta\end{bmatrix}
=\begin{bmatrix}F+ml\sin\theta\,\dot\theta^2/2\\mg l\sin\theta/2\end{bmatrix}.
\]

Normalized x=(p/l, ṗ/√(lg), θ, θ̇/√(g/l)), u=F/(mg), and τ=t/√(l/g). The normalized mass matrix is `[[3, cos(θ)/2], [cos(θ)/2, 1/3]]`, determinant at least 0.75. **u=10 means 98.1 N**, and normalized RK4 dt≈0.0626418391 means physical dt=0.02 s. Signs and zero-force upright/downward equilibrium families agree with the source.

| Check | Recorded result | Interpretation |
|---|---|---|
| Independent physical RHS | Maximum discrepancy 1.42e−14 | Governing-equation agreement on checked cases |
| Zero-input energy, 20 s | Normalized drift 1.675e−5, below 1e−4 threshold | Numerical energy diagnostic on tested trajectory |
| RK4 vs DOP853, 1 s | Errors 4.402e−6 / 1.972e−7 / 9.800e−9 at 50/100/200 substeps | Convergence evidence, not a uniform integration bound |
| Float64 vectorized oracle vs original RK4 | Maximum discrepancy 4.44e−16 over 100 draws | Oracle prediction implementation check; evaluation still uses original NumPy plant |
| Ordinary-shooting equivalence, CPU and CUDA | 50×100 samples each, atol=rtol=1e−5; maximum recorded numeric errors zero; identical branch flags/queue lengths | Computational correspondence for the replayed inputs/weights |
| Relevant existing tests | 29 passed once; one torch.load warning | Existing suite passed; plotting experiment `test_lqr_dd.py` excluded |
| Tiny original-pipeline smoke | 120×30 collection, three epochs, six angle cases × two physical seconds; 120/600 replans, no local steps; all six original balance flags false | Pipeline smoke only, not trained-method performance |

Equivalence exercised 1020 solves per implementation, local/queued/planning branches, clipping, switch interruptions, stale warm starts, and nonfinite fallback. `independent_equivalence.py` separately implements rollout/objective/CEM/controller logic, sharing transition weights, numerical specification, PyTorch/NumPy and specified RNG tensors. **It was authored by the same orchestrator, not a fresh-context independent reviewer.** Requested Terra selection was unavailable; configured inherited model/effort and LLM usage were not exposed. Do not call this an independent-author certification.

Indexed randomness is implemented: SHA256-derived case/physical-sample streams, shared float32 CPU CEM tensors replayed across devices/methods, cast to float64 for oracle. Dataset, training, member, and observation-noise streams are separate. Deterministic Torch algorithms are enabled; TF32 and cuDNN benchmark are off; scientific CPU/CUDA environments are locked; learned models are float32 and plant/oracle float64; OMP/MKL/OpenBLAS/Torch threads are one. This does not promise bitwise identity across future library/hardware changes.

### Why the calibration passed only 3/200

Original evidence: [morning_review.md](overnight_runs/20261007T054549.695689Z/morning_review.md), final [run.json](overnight_runs/20261007T054549.695689Z/run.json), and [gate_a_review.md](gate_a_review.md). All 200 cases completed, totaling 200000 original-plant transitions. Success IDs: **22, 72, 89**. Angle halves: **0/94 negative, 3/106 positive**. Calibration exit 3 is the intentional failed-gate code; the audit/runtime checks exited zero. No timeout or nonfinite-state explanation was found. Independent saved-state scoring and original-plant replay had zero error; 212 checksums passed at review time.

| Unique outcome | Cases |
|---|---:|
| Strict success | 3 |
| Safe throughout, position-only settling failure | 27 |
| Safe throughout, position and speed settling failure | 5 |
| Rail-only safety violation | 62 |
| Angle-only safety violation | 20 |
| Both rail and angle violations | 83 |

Overlapping failed-condition counts: rail 145, angle 103, terminal position 188, speed 133, angle settling 52, angular-speed settling 53. Among violating cases, median first rail/angle crossings were 0.52/0.22 s. **The 3/200 count is a failed true-model reference, not a test of a fully trained H or E.** It does not establish physical impossibility, optimizer optimality, or absence of a scientific contribution.

Independent rescoring of the saved physical trajectories makes the distinction explicit; see [endpoint_breakdown.json](reviews/20261007T054549.695689Z/endpoint_breakdown.json). The original run's 212 checksums were reverified during this follow-up:

| Nested endpoint requirement | Passed / failed |
|---|---:|
| Upright angle and angular velocity settled, ignoring earlier safety | **147 / 53** |
| Above, plus cart velocity settled | **67 / 133** |
| Above, plus cart centered within ±0.1 m | **12 / 188** |
| All four settling conditions plus both all-time safety limits | **3 / 197** |

Nine cases met all final settling conditions but violated safety earlier. In total 165 crossed at least one safety bound. The 53 angular-settling failures describe this 20-second window; they do not establish inability ever to stabilize.

Two distinct mechanisms need attention:

1. **Slow position recovery is supported by direct evidence.** The reference uses Q=diag(0.02,30,100,10), R=0.03, and local entry xᵀQx<0.75. Pure cart displacement at the rail costs only 0.1152, so local entry is not a near-target position condition. The original known-model discrete LQR is linearly stable but has a slow pole 0.998382/sample (12.35 s time constant). A pure p=1 m start predicts p≈0.270 m at 17 s and 0.212 m at 20 s. Saved nonlinear case 1 is safe but has terminal-window maximum |p|≈0.505 m.
2. **Early safety failures are unresolved.** The reference has no explicit rail/angle constraints, a 1.2 s N=60 horizon and 256/32/3 CEM settings; the full box permits initial |ṗ| up to 7.83 m/s and |θ̇| up to 10.96 rad/s. Cost mismatch, horizon length, search quality, and recoverability are competing explanations. More CEM starts cannot directly repair time spent exclusively in the slow local law.

The bounded local-only diagnostic applied each of two fixed known-model gains throughout 20 s on seven selected states: p=±1,±0.5 and saved initial cases 1,2,9. Original Qp=0.02 passed **0/7**; predeclared oracle-only Qp=2 passed **7/7**; all fourteen traces remained safe. For p=±1, terminal-window |p| maxima were 0.270384 m versus about 1.03e−7 m. See [local_findings.md](diagnostics/local_findings.md) and paired diagnostic figures. These are selected probes, applied outside the original switch when needed; **neither a new Gate A pass nor an adopted H/reference change**.

The GPU phase completed at `diagnostics/20261008T190703.853434Z`: **200/200 finite solves**, job/runtime checks passed, 20 selected states × two explicit warm starts × five starts. See [cem_findings.md](diagnostics/cem_findings.md) and [cem_review.json](diagnostics/cem_review.json). The saved-data review independently recomputed objective components, input bounds, force conversion and safety for all 200 predictions; maximum objective discrepancy was 1.019e−10, and all reported warm-candidate loss bounds passed.

| Warm start | Best-five improves cost | Median / maximum reduction | Both predicted safety limits: one / best-five |
|---|---:|---:|---:|
| Zero controls | 11/20 | 4.06% / 71.69% | 9/20 / 9/20 |
| Original oracle local-feedback rollout | 11/20 | 8.87% / 30.20% | 9/20 / 9/20 |

Local warm had lower best-five cost on 8/20 states; zero warm on 12/20. All six pre-violation probes remained predicted unsafe for both warm starts. These are **1.2-second open-loop prediction limits**, not closed-loop strict successes or a viability certificate. Combined safety classification did not change in any pair; individual rail counts changed (zero 9→11 failures, local 11→10), while angle failures remained 10 per group.

For pure p=±1 m, the zero-warm solve retained zero force, cost **1.44**, leaving p unchanged. The original local-feedback plan cost **2.105506** and ended at |p|=0.960875 m. Neither best-five improved its first cost. Thus the short-horizon cost ranks the tested no-motion plan below the tested recentering plan; zero control is not established as a global optimum. This supports objective/endpoint misalignment alongside the independent local-gain diagnosis. More starts alone did not repair predicted safety on this selected cohort. Historical full warm plans were not saved, so these are controlled synthetic warm-start comparisons, not exact historical solve replay. No new closed-loop transitions or reference/H changes occurred.

## 4. What already exists for visualization

[gate_a_examples.ipynb](notebooks/gate_a_examples.ipynb) has prepopulated offline animations of all three successes and five original failure types; a case selector covers any of the 200 cases. A new **angle-nonsettling** section adds three examples (063/139/028), a cohort plot of all 52 angle failures, 17–20 s angle/angular-speed zooms, and animations opening at 17 s. Case 139 ends near zero but fails to stay in tolerance; case 051 is separately identified as speed-only. The [gallery](figures/20261007T054549.695689Z/index.html) includes **200 PNG + 200 PDF dashboards**, complete-cohort views, **eleven standalone HTML animations** (eight original, three additional), raw/derived NPZ exports, and seven paired local-feedback comparisons. Subfolders distinguish successes, rail/angle/both violations, and settling failures. The new section is available directly at [angle_nonsettling/index.html](figures/20261007T054549.695689Z/angle_nonsettling/index.html); its generator is `angle_failure_media.py`.

Each case displays physical states, normalized/physical input, force slew, energy/work/residual, tracking/switch cost, terminal tolerances, safety margins, branch and solve time. Rod geometry uses the original full-length rod and COM, and the complete 20 seconds remain visible. Energy, power, work, acceleration, and geometry are **derived measurables, not simulated noisy sensor observations**. There are no learned-model uncertainty bands in these original oracle figures.

Recorded original export checks verified all 200 traces/force/energy conversions, original-source integrity, and 629 generated-artifact checksums. After adding the angle section, all **27 notebook Python cells** ran offline and both notebook schemas validated; source outputs are cleared. JavaScript playback/scrub/reset passed a mocked DOM/canvas check. **Actual browser rendering remains untested.** The display kernel is a separately frozen existing Python 3.12 environment; it reads saved data and does not define the scientific numerical environment. Selected animations illustrate mechanisms, not representative success probabilities. No new controller/plant experiments were used to add this section.

## 5. Prioritized remaining control and numerical tests

P0 means next before recalibration; P1 means needed before pilot/lock; P2 means exploratory after the campaign remains informative. CT01 is complete; remaining experimental designs need a measured allocation. Use the original plant for evaluation, common cases/RNG, declared variants, and all failures in denominators.

| ID / priority | Question and minimal design | Evidence / decision to save |
|---|---|---|
| CT01 / completed | Fixed CUDA diagnosis, all 200 solves; do not repeat | Cost improved in 22/40 pairs, combined predicted safety stayed 18/40; saved-data arithmetic/source review passed; findings above |
| CT02 / P0 | After CT01, specify the oracle local gain, position-aware switch, planning cost and any safety handling. Check proposed entry states with exact nonlinear dynamics, clipping and original strict thresholds | Versioned proposal; entry/saturation/failure maps. Qp=2 probes motivate a candidate but do not validate a whole switch region |
| CT03 / P0 | Separate search deficiency from objective deficiency using fixed-state comparisons before any horizon/population/start sweep | If warm candidates already outperform returned plans, diagnose implementation; if lower-cost plans remain unsafe, diagnose objective. A feasible open-loop plan is not closed-loop attainability |
| CT04 / P0 | Investigate broad-box viability only where needed: saved early failures, input saturation, short-horizon reachable-state envelopes; any longer-horizon/high-accuracy reference is separately allocated | Report unresolved cases. Failure of sampled search does not certify nonviability; simple cart braking estimates ignore rod coupling and cannot certify safety |
| CT05 / P0 | Generalize audit/launcher assumptions, freeze the amended reference, then a fresh 200-case Gate A. If it fails after diagnosis, consider sequential half/quarter boxes as written in the plan | At least 196/200 and 95% per angle half, complete traces/replay/hash audit, independent physical scoring; freeze passing box before pilot; retain broad-box stress evidence |
| CT06 / P1 | Inspect data-identified local models by acquisition budget: selected state/input radius, singular values/conditioning of Z, ridge sensitivity, residuals, fitted equilibrium, A/B and closed-loop eigenvalues | Diagnose weak excitation and nonlocal nearest-6000 selection; count DARE failures. Compare to exact plant linearization diagnostically, without supplying truth to H/L training |
| CT07 / P1 | Check learned equilibrium and local derivatives at/near origin; use autograd and finite-difference agreement with the plant and each other | f̂(0,0), ∂f̂/∂x, ∂f̂/∂u, physical units and step size, perturbation-size sensitivity. Stable fitted A−BK does not certify the nonlinear hybrid system |
| CT08 / P1–P2 | Map local return/decrease with saturation and sampled dynamics; separately test both sides of the switch surface, chatter, disturbance-triggered entry/exit and stale-plan age | Empirical V(x⁺)−V(x), saturation, dwell times and rail/angle margins; label empirical maps, not certified region of attraction. Any formal certificate needs explicit domain/robustness bounds |
| CT09 / P2 | Queue ablations: submitted s=5 versus s=1, fresh plan after local feedback, actual-elapsed-step shift, and optionally hysteresis. Freeze each variant before paired execution | Full hybrid traces, solve counts, strict success, cost and latency. These are developmental controller variants, not silent corrections to submitted H |
| CT10 / P1 | Implement deadline fallback exactly and fault-inject late planning/local calls, exceptions, nonfinite predictions and interrupted runs | Late plan discarded; five-sample hold and queue/previous reset for planning; one-sample hold for local; failure recorded. No privileged rescue; boundary-time and partial-run tests |
| CT11 / P1 | Expand integration checks near saturation/high angular speed and safety boundaries, using held-input DOP853 and dt refinement | Numerical error versus state/margin; distinguish interpolation/sample scoring from any proposed continuous-time safety claim. Current sampled endpoint must not change silently |
| CT12 / P2 | Prioritized robustness: fixed-scale observation noise, one-sample action delay, and +25% mass/inertia, no retraining | True and observed states, delayed/applied controls, indexed common noise, empirical failure map. Specify physically consistent parameter changes in the original plant; never borrow another repository's plant |

Do not presume standard MPC stability conditions hold: terminal weight is **12Q, not the DARE P**, no invariant terminal set is enforced, the CEM search is approximate, input limits can saturate, and local switching/queueing changes the policy. This is a limitation already disclosed in the implementation note, not an unclaimed theorem to refute. A certified controller could be a separate research extension after the practical comparison.

## 6. Mathematical and implementation-note audit

Active source anchors: [transition](../src/manifold_control.py#L29), [canonical decoder](../src/manifold_control.py#L78), [CEM](../src/manifold_control.py#L231), [local identification](../scripts/run_inverted_pendulum_mc.py#L180), [hybrid controller](../scripts/run_inverted_pendulum_mc.py#L199). Git/blob hashes, rather than line numbers alone, define the snapshot. The module's historical joint-optimization docstring and autoencoder/curvature/latent-solver classes are **inactive** in this runner.

For n=4, m=1, N=30, q=(x₀,u₀,…,u₂₉) has dimension **34**, the fixed-state online decision has dimension **30**, and terminal-augmented w has dimension **154**. This is explicit rollout parameterization; it is not an active learned bottleneck or a lower-dimensional decision space than ordinary shooting. Experiments can test implementation correspondence and plant fidelity, but do not prove general theorems or historical novelty.

| ID / priority | Remaining audit / test | Required distinction or artifact |
|---|---|---|
| MA01 / P1, read-only | Build the critique's claim ledger C01–C12 with exact page/equation, assumptions, code mapping and evidence | `claim_ledger.csv`, annotated TeX copy, gap/contribution matrix; separate document claims from verbal data-efficiency/interpretability claims |
| MA02 / P1, read-only | Independently derive graph and regular-level-set arguments: nonempty open recursive domain, Cʳ dynamics, inverse projection, full rank, fixed-state slice | Proof memo. Dynamic constraints have a block-triangular next-state derivative with identity diagonal; verify ordering/domain rather than relying on numerical rank alone |
| MA03 / P1, read-only | Audit exact differentiable latent lower bound p≥n+Nm, including decoder differentiability on its latent domain and hypotheses used in the rank argument | Do not extend the exact bound to approximate encodings, restricted policies, finite datasets or state-only Koopman embeddings |
| MA04 / P1, read-only | Verify sampled-flow existence/smoothness and RK4 admissibility; distinguish open smooth domains from input-box boundaries/corners, safety-restricted sets, output-only and stochastic observations | Mark needed qualifications, not a blanket rejection of the deterministic full-state theorem |
| MA05 / P1, read-only | Resolve process-noise dimension wording | If noise is retained as an ambient coordinate, graph construction applies. If only states/inputs are observed, noise may be noninjective; dynamics independent of noise is a counterexample to automatic observed-dimension increase |
| MA06 / P1, cheap harness | Causality and packing property checks: perturb future inputs, batch/nonbatch shape, horizon boundaries, coordinate extraction and ordering | Exact/appropriate float tolerance tests; identity πΦ̂=q and its identity Jacobian block are architectural invariants, not learned fidelity evidence |
| MA07 / P1, cheap harness | Independent objective/gradient checks on tiny horizons: state/input/terminal/rate indexing, clipping and finite-difference gradients of differentiable interiors | Cost decomposition already has equivalence coverage; add meaningful corner cases. No slew penalty couples first planned input to last applied input in submitted cost |
| MA08 / P1, read-only + held-out diagnostics | Explain conditional rollout-error recurrence eₖ₊₁≤L eₖ+ε under stated bounds, then measure local sensitivities/error growth | Empirical sensitivities do not establish a uniform L or ε, closed-loop robustness, learnability or data-efficiency theorem |
| MA09 / P1, read-only | Closest primary-source literature by matched assumptions, before September 2026; verify theorem/algorithm overlap and source history | Source-backed novelty matrix. Ordinary-shooting equivalence establishes computation, not historical non-novelty; the unverified OpenReview lead remains unresolved |

The note explicitly disclaims consistency, optimizer convergence and closed-loop stability. Keep absent guarantees separate from incorrect statements. A graph identity remains true even with a wrong transition model. Use a deliberately inaccurate smooth predictor as a demonstration of this distinction if useful; do not treat that demonstration as a model-performance experiment. No independent proof reviewer was used in the completed work.

## 7. Machine-learning tests that the current evidence lacks

No full-budget trained H/E nominal success estimate exists. Timing checkpoints trained on repeated smoke data and the largely untrained ensemble are not substitutes. The following tests must use actual acquired episode data and independent held-out evaluation; ground-truth plant derivatives are diagnostic references only.

| ID / priority | Design | Output / interpretation |
|---|---|---|
| ML01 / P1 | Implement fixed 30-transition collection independent of planning N; nested episode budgets, deterministic 80/20 splits, train-only scaling/ID, episode bootstrap members, separate final cases | Data/split/bootstrap manifests, no crossing-episode windows, tests for leakage/accidental seed reuse; record attempted/failed acquisition and resets |
| ML02 / P1 | Complete original dev reproduction separately, then campaign H/L training. Freeze E with the plan's exactly 12 recipes on independent tuning data | Checkpoints, histories, training/member seeds and selection evidence. Original reproduction and evaluation-protocol adaptation are distinct |
| ML03 / P1 | Held-out one-step and free-run errors at 1/5/30/60 steps, with episode-boundary-aware windows and declared longer probe sequences where 30-step episodes cannot supply a 60-step target | Per-coordinate physical/standardized RMSE, quantiles/maxima, error-vs-horizon and region; compare zero-increment and local-linear diagnostic references |
| ML04 / P1 | Compare random-collection validation with closed-loop visited state/actions and near-boundary/saturated inputs, across independent training seeds | Coverage/distance summaries and stratified rollout errors; good IID validation need not imply safe controller-induced extrapolation |
| ML05 / P1 | Equilibrium/kinematic/physical-symmetry diagnostics: cart translation, simultaneous state/input sign reflection, and periodic-angle tests where the declared domain supports them | True-reference vs learned residuals. Periodicity outside training/nominal domains is an extrapolation diagnostic, not an automatic in-domain specification error |
| ML06 / P1–P2 | Check physical energy/work consistency of predicted rollouts, separating learned-map defect from original RK4 integration defect | Work-energy residual distribution against oracle reference, units and signed conventions; no claim that architecture enforces energy conservation |
| ML07 / P1 | For trained E, evaluate member disagreement, bias, held-out error/disagreement correlation and interval coverage by horizon/region | Coverage/reliability plots; ensemble spread is not calibrated uncertainty. The mandatory E policy is mean-cost, without risk penalty or trajectory sampling |
| ML08 / P2 | Same-state/input counterfactual replay under learned model and original plant, including optimizer-selected controls; compare predicted ranking with realized cost/safety | Distinguish approximation error from objective mismatch/search. Historical plans are missing, so new explicit plans and predictions must be logged |
| ML09 / P1 | Across bundles and budgets, report training convergence, checkpoint sensitivity, local-fit quality, failure attribution and common-case successes | Seed variation, not just extra cases at one seed; archive nonfinite/DARE/collection failures and never silently retry until success |
| ML10 / P2 | Separate declared variants: multistep loss, equilibrium/structure-aware features, or genuinely learned compressed representations | Matched data/tuning/cost accounting and held-out evaluation. These change the method and cannot retrospectively improve the submitted-H result |

If practical interpretability is assessed, separately define physical coordinate semantics, dynamical explanation and action explanation. Held-out derivative signs/magnitudes, unit-rescaling stability and counterfactual accuracy are useful; a two-dimensional picture, identity extraction or full Jacobian rank alone is insufficient. This is an optional contextual audit unless explicitly attributed to a document claim.

## 8. Inference and power implementation still needed

The primary is Δ=min(pH−pL,pH−pE), testing Δ>0.05 at T=105000. **Both one-sided 95% component lower bounds must exceed 0.05.** Under the union null at least one component null is true, and rejecting both entails rejecting that true null; valid component tests at alpha=0.05 control the intersection-union test at 0.05. A Bonferroni split to 97.5% is unnecessary and costs power. These are not simultaneous-coverage 95% bounds. `power_rules.py` implements this decision rule and Holm, but passing rule tests do not establish bootstrap validity or power.

Remaining statistical work:

1. Fit/simulate the plan's multivariate crossed-effects Bernoulli model, preserving correlated method-specific seed effects, shared case effects and seed×case variation. Match marginal probabilities by integration. Check boundary and mixed-null type-I behavior, degeneracy/all-fail/all-pass samples, and the dependence on small seed counts.
2. Size S∈{8,12,16,24}, C∈{100,200,400} using 2000 simulated experiments/design and 1000 bootstrap draws for sizing; use 5000 in final analysis. Stress seed SD 0.25/0.5/1.0 and method correlation 0.3/0.7. Require conservative scenarios to meet 80% power at Δ=0.10 and target attainment at pH=0.97 versus baselines 0.87. Publish sensitivity to 5/7.5/10-point gains and pH=0.95. More cases cannot replace independent training bundles.
3. Implement Gate B's development kill/one-expansion rules, keeping development data out of final inference. Full-budget gain≤0.03 with one-sided 90% upper≤0.05, or ≥10 points extra H safety violations, stops broad confirmation; uncertain evidence allows only the specified 3×50→5×100 expansion before pause.
4. Implement monotone piecewise-linear logit success curves, six coarse knots, nonnegative slopes/ridge/logit bounds; 5000 crossed seed/case bootstraps preserve pairing across methods and budgets. Freeze at most four finer budgets before final cases. Keep censored/missing crossings, avoid extrapolation, and show unconstrained/logit-linear sensitivity.
5. Preserve the absolute H target (estimate≥0.95 and one-sided 95% LCB≥0.90), gated acquisition-ratio criterion (95% upper≤0.80), and then **Holm for the six secondary tests**. Conditional cost among successes must be accompanied by all failures; nonrejection is not equivalence. Stress/ablation results are exploratory.
6. Lock recipes, box, endpoint/device, budgets, inference scripts and independently hidden final cases before confirmation. Reserve statistical CPU work in the allocation. If mandatory powered work cannot fit, report insufficient power and revise allocation explicitly; do not weaken E/L or margins to fit.

## 9. Harness pieces to complete or reimplement

**Preserve and reuse:** locked environments/manifests; original-plant evaluation and validated oracle vectorization; indexed RNG; physical scoring; separate equivalence wrapper; supervised lifecycle/occupancy guards; original trace review and media exporters. Review every new diff and source hash before reliance.

**Complete:** H/L/E campaign adapters, true trained E recipe and mean-cost/no-switch policy, plan-compliant collection/splits, full deadline fallback logging, robust interaction accounting, crossed-effects sizing/curve inference, hidden final-case workflow, and claim/proof/literature deliverables. Existing `PROFILES['full']={}` falls back to base defaults; it is **not** the planned confirmatory campaign.

**Reimplement carefully where current assumptions would be wrong:** `overnight_audit.py` currently assumes original full box, seed/config, 200 cases and one start; `overnight_queue.py` launches the original reference. Before an amended calibration, make the independent audit derive expectations from a frozen reviewed config, while keeping an independent units/RHS/scoring path. Add incorrect-seed/scale/reference rejection cases and partial-run tests. Preserve the original run directory and source snapshots. A configurable audit must not merely trust a run's own success labels.

**Strengthen reproducibility:** save every plan, warm plan/age/shift, queue length, candidate/objective component, observed/true state, applied input, runtime/fallback and RNG key needed to reproduce later failure decisions. Store execution source snapshots separately from subsequently edited working files. Verify common RNG prefixes under different branch histories. Avoid full tensor dumps for every solve if storage is excessive; store indexed keys and selected forensic tensors with a frozen replay contract.

**Environment gap:** genuine CVXPY is absent and network DNS blocked. `compat.py` omits two unused plant imports in memory with fail-closed checks; this supports the active path, not an unqualified full environment or legacy CVXPY methods. Install the real dependency when available and validate the normal-import active path. No plant/controller files were modified. Model/effort/LLM telemetry was unavailable; do not invent usage counts.

**Independent review gap:** the current equivalence implementation is separate code from the same author. An independently authored replay or concise adversarial code review would strengthen confidence. Preserve the user's bounded delegation rules: no nested delegation, no server calls by other agents during timing, shared compute cap, and no silent model substitution.

**Optional performance refactor:** after correctness gates, profile actual CPU/CUDA work, scalar synchronizations and small sequential kernels. Batched inference/vectorization or fewer host-device transfers may help, but that is a hypothesis. Put optimized controllers in validation variants; rerun relevant equivalence before comparing them, remeasure synchronized end-to-end latency and precision, and label scientific outcomes by implementation. Do not retroactively replace submitted H.

**Historical code:** autoencoder/latent optimization/curvature routines exist but are not called by the active runner. Testing or reviving them is a new method study with an explicit decoder, identification objective, causal state/input interface, dimension object and comparison protocol. Their mere presence does not mean the active controller already uses learned manifold compression.

## 10. Additional visualizations, in order of usefulness

| ID / priority | Visualization to add | Data prerequisite / interpretation |
|---|---|---|
| V01 / P0 | Inspect existing success/failure animations in a real browser; verify play/pause/scrub/reset, physical time, angle sign and rail/COM geometry | Existing artifacts only; record browser/version and screenshots; this closes a display check, not a scientific gate |
| V02 / P0 | GPU diagnostic small multiples: one/best-five loss, state/input/terminal/rate decomposition, warm candidate vs returned plan, predicted rail/angle margins | CT01 data now available; visualization still unimplemented. Show lower cost and safety separately, and mark each warm-start recipe |
| V03 / P0–P1 | Position-feedback comparison with switch ellipsoid, terminal tolerances, saturation and p–ṗ / θ–θ̇ phase portraits | Existing local probes plus any allocated entry tests; phase projections do not certify a full 4-D safe region |
| V04 / P1 | Predicted-vs-true 1/5/30/60-step trajectories, per-coordinate error fans, and optimizer-selected counterfactual replay | Actual trained checkpoints and independently held-out inputs; show model/oracle errors separately from control outcomes |
| V05 / P1 | State-action coverage and error maps: training vs closed-loop, near boundaries, nearest-6000 local subset/radius, design singular values | Actual datasets/visited states; mark projections and density, avoid presenting 2-D coverage as full-state coverage |
| V06 / P1 | Hybrid timeline: switch cost, branch, queue length, warm age, plan shift, saturation, deadlines and first safety/settling violation | Expanded forensic logging; original traces support only a subset. Show physical deadline failures separately from offline optimizer times |
| V07 / P1 | Trained-E uncertainty calibration: disagreement vs error, empirical interval coverage by horizon/region, reliability diagrams | Trained members and true outcomes; spread is not a probability without calibration |
| V08 / P1 | Learned/true local Jacobian and equilibrium/energy-work defects with physical-unit annotations | CT07/ML05/ML06; identity coordinate-block rank panels should be labeled structural, not fidelity |
| V09 / P1–P2 | Learning curves with all seed points, paired contrasts, crossed-bootstrap bands, censored T95 crossings and fit sensitivity; power/type-I heatmaps | Implemented inference and real pilot/final data; no invented bands from the 200 oracle cases |
| V10 / P2 | Paired controller-variant animations and noise/delay/parameter-shift failure maps; observed state overlaid with true state | Declared new study and allocated runs; retain nominal original trajectories and show every tested outcome |

Use physical seconds/units consistently, label raw versus wrapped angle, display both normalized input and newtons, retain safety and final-window bands, and use labels/line styles as well as colors. Exemplars should be selected by a recorded rule and accompanied by cohort summaries. Store numerical data and PNG/PDF exports alongside interactive displays; derived quantities must be named as such.

## 11. Hardware, measured cost, caps, and endpoint amendment

Readiness recorded two EPYC 7543 sockets, **64 physical/128 logical cores**, approximately 1 TiB RAM. Host CUDA verified an **NVIDIA L4**, 23034 MiB VRAM at that snapshot, driver 610.43.02, driver/toolkit CUDA 13.3, with private Torch 2.5.1+cu121/runtime CUDA 12.1. Memory availability/shared load is time-specific. This managed session lacks NVIDIA device nodes; use the GPU-enabled host terminal, not a silent CPU fallback. No scheduler policy was discoverable; establish lab allocation requirements before larger jobs.

N=30, CEM population/elites/iterations=256/32/3, dev 192/192/192 SiLU network; oracle N=60. Readiness includes warm-up, synchronization and end-to-end controller branches. Quantiles are empirical, not hard deadline guarantees.

| Device | Mean τ_H / τ_E / τ_oracle per start (s) | H / E planning p99 (ms) | H / E local p99 (ms) | H epochs/s / five-member ensemble epochs/s |
|---|---|---|---|---|
| CPU, one thread | 0.061848 / 0.327245 / 0.114787 | 69.78 / 370.00 | 0.072 / 0.083 | 1.396 / 0.297 |
| L4 | 0.074332 / 0.399123 / 0.214193 | 101.33 / 437.33 | 0.166 / 0.169 | 7.025 / 1.420 |

E's local column is a performance-only hybrid branch measurement; scientific E has no local switch. CPU planning misses H 0/120, E 120/120; L4 H 2/120, E 120/120. Thus CPU H's measured p99 meets 0.1 s, GPU H and both E workloads do not; the tested local branches meet 0.02 s. Contended end-to-end deadlines remain unmeasured. Training used 84000/21000 repeated smoke transitions, and four ensemble control members were untrained: these measurements cannot establish scientific performance.

CPU peak RSS was about 287–292 MiB for control and 399–407 MiB for training; GPU RSS about 876–888 MiB control and 949–953 MiB training, with allocator peaks about 35 MiB control and 104 MiB training. CUDA context/other users were excluded; **continuous whole-device peak VRAM was not measured**.

Measured aggregate H/E solves/s: CPU workers 1/2/4/8 =16.45/32.23/65.18/128.53 and 3.42/6.74/13.32/26.44; L4 workers 1/2/4 =14.89/29.36/56.89 and 3.86/7.62/14.81. Eight one-thread CPU workers or four persistent GPU workers are measured offline-throughput choices; use one GPU worker until deadline behavior under contention is verified. Fixed-state solver throughput is not a replacement for randomized end-to-end latency. GPU training is about five times faster, but submitted end-to-end planning was slower.

Historical forecasts use K=1000 samples/episode, τ measured above, H smoke replan fraction r_H=0.20 and E fixed r_E=0.20. For M=bundles×cases×budgets, H/E planning projection is `M*K*(r_H*τ_H+r_E*τ_E)/3600`; worst-case solve-count reservation sets both rates to one. Oracle uses `200*K*starts*r_oracle*τ_oracle/3600`. Add modeled acquisition/training and all presently unmeasured overhead. The mean-latency formula is a reservation model, not a hard runtime bound.

| Stage | CPU worst / smoke projection (worker-h) | L4 worst / smoke projection (worker-h) | Original cap (worker-h) |
|---|---:|---:|---:|
| 0, readiness forecast + up to five-start oracle | 32.31 / 6.81 | 59.93 / 12.33 | 8; user-launched Gate-A-only queue explicitly used 15 |
| 1, required tuning and pilot model | 49.35 / 10.44 | 59.34 / 11.99 | 12 |
| 2, provisional 12 bundles × 200 cases × 10 budgets | **2601.46 / 526.30** | **3158.06 / 632.97** | **48** |
| 3, three prioritized stresses | 48.64 / 9.73 | 59.18 / 11.84 | 8 |
| 4 | No committed workload | No committed workload | 0 |

Source: [stage_estimates.json](artifacts/stage_estimates.json) and [stage_estimates_cuda.json](artifacts/stage_estimates_cuda.json). These are historical model outputs, not fresh remaining-time reservations. Stage-0 forecast JSON used the readiness usage available then. Every worst-case stage exceeds its original cap; Stage 2 projections exceed cap on both devices. Stage 1 GPU projection leaves effectively no overhead allowance. Measured concurrency gives roughly 60.8 CPU-eight-worker or 113.4 GPU-four-worker wall-hours for projected Stage-2 arm solves alone; **parallelism does not reduce worker-hours**.

The failed Gate A observed r_oracle=0.08064 (16128 replans), 60.294% local and 31.642% queued. Its solve mean/p99 were 193.47/231.42 ms, with scope limited to CEM solves. Do not apply this rate to H/E or an amended oracle. One-start CUDA worst planning plus 20% overhead was about 14.28 h; five starts about 71.40 h, exceeding current Stage-0 remainder. Any amended reference needs a new feasible reservation, including startup/failures/audit/statistics.

After completed diagnosis, the 15-hour Stage-0 allocation has **13.614623 h remaining**. Even the unchanged one-start CUDA planning-plus-20%-overhead model (~14.28 h) exceeds that remainder. A fresh calibration therefore needs a reviewed allocation/device/protocol decision; neither short actual prior runs nor the unused diagnostic balance justify bypassing preflight.

**Recommended endpoint amendment, still pending:** use timing-agnostic physical success as primary, with latency and deadline-enforced operational success as secondary/descriptive endpoints. Under the current primary, misses are strict failures and device choice can change primary outcomes; this may mainly compare implementation/hardware rather than control/data efficiency. Accept and freeze any change before pilot/final lock, retaining the original operational rule in separate reporting. Changing device, precision, objective, safety constraints, nominal box or primary endpoint must be explicit.

Recorded experimental occupancy: readiness **1715.784 s (0.476607 h)** within its one-worker-hour cap; user-launched overnight queue **3192.586 s (0.886830 h)**; local diagnostic **7.076882 s**; completed CUDA diagnosis **71.909122 s**. Total Stage-0 occupancy **4987.356492 s (1.385377 h)** under the user-run 15 h allocation. Shared diagnostic usage is **78.986003/900 s**, leaving **821.013997 s** numerically; both phases are complete and unused time does not authorize additional experiments. Includes startup/failures as logged. CEM solve-only mean/median/p95/p99 were 214.27/213.26/217.72/220.28 ms; this is not end-to-end latency. Read-only review/export/documentation is separate; figure generation recorded 520.576 s. Readiness interaction manifest is phase-local: add the later 200000 calibration and 14000 diagnostic plant transitions rather than interpreting its old zero-calibration field as current. CUDA diagnosis added model-prediction solves but no closed-loop acquisition. No experimental jobs were launched for this follow-up; LLM usage is unexposed.

## 12. Concrete next actions and runnable commands

Both bounded diagnostic phases are complete; **do not rerun `run_diagnostics.sh cem --execute`**. The next work is a reviewed oracle-amendment specification and configuration/audit support, with no automatic recalibration. From `/home/ap60/github/data-driven-control`, the saved-data CEM review can be reproduced without GPU/controller execution:

```bash
validation_dir="$PWD/TrajectoryManifoldControl/validation"
LD_LIBRARY_PATH="$validation_dir/.venv/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
LD_PRELOAD="$validation_dir/.venv/lib/libstdc++.so.6" \
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
"$validation_dir/.venv/bin/python" "$validation_dir/review_cem_diagnostic.py"
```

This reads original diagnostic arrays and writes the aggregate `diagnostics/cem_review.json` outside the run directory. It requires the separately archived NPZ files, which are intentionally ignored by Git. Completed local/CUDA directories are `diagnostics/20261007T175948.296063Z` and `diagnostics/20261008T190703.853434Z`; preserve both. The experimental supervisor still enforces the shared cap and rejects duplicate completed phases.

View saved visualizations without any new control experiments:

```bash
bash TrajectoryManifoldControl/validation/notebooks/open_notebook.sh --no-browser
```

Or open `validation/figures/20261007T054549.695689Z/index.html` directly in a browser. Do not regenerate the full figure archive merely to read the handoff.

After this completed review: (1) finalize the claim ledger/proof-to-code checks alongside read-only work; (2) specify the reference gain/switch/cost/safety amendment, keeping balancing and added safe-recentering requirements distinct; (3) implement reviewed configurable audit/queue and meaningful regressions; (4) preflight a fresh 200-case Gate A with explicit feasible allocation; (5) execute only the separately specified gate and review; (6) freeze a passing nominal task, resolve E recipe/endpoint/power/allocation, then consider Stage 1. **There is no ready-to-copy amended-calibration command yet**, because that reference and audit have not been finalized. Do not reuse the old overnight launcher to silently run Qp=2 or a smaller box.

The next deliverable is a concrete, versioned oracle-amendment proposal with reviewed configuration/audit requirements and a feasible allocation. The completed diagnostic does not authorize a new calibration or confirmatory sweep.

A copyable continuation prompt is saved in [next_session_prompt.md](next_session_prompt.md).
