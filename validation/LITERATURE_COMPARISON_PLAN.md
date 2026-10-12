# Plan: implement the missing literature comparisons

October 12, 2026 · **For user review; no experiments or automatic launches authorized by this document.**

**Recommendation: start with controlled Koopman/EDMD and SINDy-MPC, then implement full PETS and one nonlinear DeePC method. Follow with separate Deep Koopman and DFINE prediction benchmarks, then controlled adaptations of both as stages 7–8.** This order prioritizes applicability to our measured-state control task, information gained beyond existing results, and implementation cost. It is a research judgment, not a prediction of which method will win.

The completed [campaign](results_v1/RESULTS.md) remains intact. Its [operating handoff](codex_handoff_from_claude_v1.md) continues to define the original validation; this is a proposed extension, not a replacement or a v4 plan. Stability certification remains a [separate task](results_v1/STABILITY_GUARANTEE_HANDOFF.md).

## Questions and boundaries

We want to establish whether a nonlinear model with more suitable structure or uncertainty handling improves on **H**, the submitted neural-planning/linear-feedback hybrid, and **L**, its fitted linear feedback used alone. We also want to distinguish prediction quality from control quality, quantify data and computation requirements, and compare noisy/partial-observation modeling. Stages 5–6 reproduce prediction/inference scope; stages 7–8 add our own input-aware control adaptations, clearly distinguished from the original papers.

The old baseline results do not settle these questions. E4 tested a simplified deterministic ensemble and linear DeePC on nonlinear data; it did not implement Koopman, SINDy or DFINE. Three seeds were used on 200 starting states—not 200 seeds. A correct but unsuccessful implementation remains a reported result; success is never a prerequisite for including it.

## Priority and proposed implementations

| Order | Comparison | What we will implement | Reason for this priority |
|---|---|---|---|
| 1 | Controlled Koopman/EDMD-MPC | Nonlinear state features, a fitted linear predictor in feature space, and quadratic-program control | Direct full-state/input comparator; separates richer prediction from expensive sampled planning |
| 2 | SINDy with control / SINDy-MPC | Sparse nonlinear state/input regression and predictive control | Small, interpretable models may suit this four-state system and expose data-efficiency differences |
| 3 | Full PETS | Probabilistic model ensemble, uncertainty propagation and sampled predictive control | Closest uncertainty-aware neural comparator; corrects what the old ensemble omitted |
| 4 | Nonlinear DeePC | One source-faithful kernelized formulation | Tests nonlinear trajectory prediction rather than repeating the inappropriate linear-span assumption |
| 5 | Lusch Deep Koopman | Autonomous multi-step prediction | Directly implements the cited representation method, while respecting its original scope |
| 6 | DFINE | Causal inference and prediction with noisy/partial observations | Relevant to the observation/noise gap, but not a ready-made forced controller |
| 7 | Controlled Deep Koopman | Input-aware learned latent predictor plus sampled MPC, with/without common L | Turns the learned representation into a direct forced-control comparator |
| 8 | Controlled DFINE | Input-aware latent dynamics and causal filtering plus sampled MPC, with/without common L | Tests whether learned inference improves noisy and partially observed control |
| Later | SAC / Baek-related comparison | Same-plant online reinforcement learning under a separate allocation | Needed for a matched RL sample-efficiency claim; different data collection and larger scope |

### 1. Controlled Koopman/EDMD-MPC

Extended Dynamic Mode Decomposition (**EDMD**) fits approximately linear dynamics after mapping measured states into nonlinear features. Follow [Korda–Mezić](https://arxiv.org/abs/1611.03537): fit `z_next = A z + B u`, reconstruct state from `z`, and solve a bounded-input quadratic program at each sample. Reinitialize `z` from the current observation each time; do not assume the fitted lifted space is invariant.

Start with identity state features plus degree-two polynomial features. Test a second dictionary adding sine/cosine of the measured angle; this uses the known periodic meaning of angle, not analytic plant dynamics. Disclose that inductive bias. Prespecify three ridge strengths, giving at most six configurations per seed. Standardize features on training episodes only; retain state coordinates explicitly so state costs can be represented directly.

Use the same horizon, physical sampling interval, input bounds and state/input/terminal costs as H. Include the adjacent-input rate cost with its existing indexing. Verify a small linear example against an independently computed solution, then check saved-input predictions against plant replay. Report lifted dimension, conditioning, rollout error, optimization residuals and solve time. Do not call a finite feature approximation the exact Koopman operator.

### 2. SINDy-MPC

Sparse Identification of Nonlinear Dynamics (**SINDy**) selects a small number of terms from a candidate function library. [SINDYc](https://arxiv.org/abs/1605.06682) includes actuation; [Kaiser, Kutz and Brunton](https://arxiv.org/abs/1711.05501) develop the MPC comparison.

First implement a **discrete-time SINDYc adaptation** predicting next-state increments from measured state/input samples. This avoids giving it true derivatives unavailable to the neural model. Use degree-two/three polynomials, with and without measured-angle sine/cosine features, and bounded sparse-regression settings: no more than twelve configurations per seed. Publish the selected terms and coefficients. No true mass, length, Jacobian, kinematic equation or analytic derivative enters the learned model.

For model attribution, first substitute this predictor into the existing sampled planner with the same search budget. For the practical SINDy-MPC arm, use a differentiable finite-horizon nonlinear optimizer, with fixed iteration/time limits and recorded constraint residuals. Keep these arms separate: one tests the predictor; the other tests a different predictor/optimizer combination. A derivative-based continuous-time reproduction is a later labeled variant, with numerical differentiation and its noise sensitivity explicitly accounted for.

Check exact recovery of a small sparse discrete system, input dependence, scaling and independent horizon-cost evaluation. Coefficient sparsity is not evidence of physical correctness or stability.

### 3. Full PETS

[Probabilistic Ensembles with Trajectory Sampling (PETS)](https://arxiv.org/abs/1805.12114) combines probabilistic neural dynamics with ensemble/particle uncertainty propagation. Implement five bootstrap models predicting state-increment means and variances, with the paper's variance handling, and trajectory sampling that preserves the selected member along each particle rollout. Start at twenty particles; pilot timing may reduce this explicitly to five. Use cross-entropy input search and first-action feedback. Limit timing-based population/particle choices to three declared configurations.

Audit the author's implementation before porting. Fix the ensemble/particle semantics and publish differences from the paper. Compare against the old deterministic ensemble and a deterministic first-action neural planner on the same data and search budget. Report both a matched-search configuration and the best configuration within the shared compute allowance; extra particles or candidates are not free. Record predictive calibration as well as error.

The primary experiment is **offline-trained PETS-style control on our fixed collection**. It does not reproduce PETS's complete iterative data-collection procedure or settle online RL sample efficiency. No catch controller is silently included.

### 4. Nonlinear DeePC

Data-enabled Predictive Control (**DeePC**) predicts future trajectories directly from recorded data. Implement one method from [Robust and Kernelized DeePC](https://arxiv.org/abs/2206.01866), auditing the full formulation and author code first. Do not describe an arbitrary kernel-regression predictor plus MPC as a reproduction of its robust optimization problem.

Pilot full-state input/output windows first. Select a bounded training-window subset by a fixed training-only rule; cap it initially at 512 windows. Standardize before kernel construction, and limit bandwidth/regularization search to six configurations. Report the entire retained-data rule. Use the same subset for a linear-kernel/linear-DeePC control so that any improvement is attributable to nonlinear structure rather than window selection alone.

Check the linear-kernel limiting case, objective/residual calculations and matched-input replay. If the source-faithful solver is unavailable or cannot fit the allowance, report the method as blocked or partial, and specify any simpler adaptation separately. Do not convert an implementation failure into a scientific zero-success result. Output-only control follows only after the full-state implementation is verified and its extra startup/history requirements are documented.

### 5–6. Deep Koopman and DFINE: prediction first

The cited [Lusch Deep Koopman model](https://doi.org/10.1038/s41467-018-07210-0) learns autonomous embeddings; [DFINE](https://www.nature.com/articles/s41551-023-01106-1) combines nonlinear observation embeddings with linear latent dynamics and Kalman inference. Their original models do not accept the planned force sequence needed by our controller. Feeding them the forced data while withholding inputs would be an unfair control comparison.

Create one **separate passive, zero-input prediction dataset**, using the same pendulum simulator, whole-episode splits and physical units. Initially collect 1,000 episodes of 100 samples, with a fixed training/validation/test split and three model seeds. Use two predeclared initial-state ranges—one near upright and one broader—without selecting cases based on later prediction success. Passive trajectories may fall; that is not a data exclusion rule. The new dataset must be charged to this extension and cannot be compared directly to forced-control success counts.

For Lusch, preserve reconstruction, multi-step prediction and latent-linearity losses, and audit whether the original pendulum example needs its frequency-dependent latent evolution rather than a constant linear matrix. Compare state reconstruction and 1/10/30-step open-loop forecasts against the submitted transition architecture retrained on those same passive data, EDMD and linear regression. Use up to six latent-size/loss configurations. The input-aware adaptation is planned explicitly in stage 7, regardless of whether passive prediction wins; stage 5 remains separately reported.

For DFINE, implement the cited nonlinear inference method, up to six latent/noise settings, and causal prediction at the same horizons. Add fixed observation-noise levels and missing-observation masks shared by all methods. Include raw-observation and linear Kalman baselines; provide the neural predictor the same available history through a declared causal adapter. Start with full-state observations, then `(position, angle)` history; score observed channels unless hidden-state readouts receive the same supervised training labels across methods. No future observations may enter the causal result. Noncausal smoothing, if reported, gets its own label and table.

The local source check found TensorFlow-era DeepKoopman scripts with large random-search loops and torchDFINE expecting older Python versions. Use isolated environments or a tested port; never launch those original search loops. Preserve licenses and record the external revision and compatibility changes. A small source-reproduction check precedes task adaptation.

### 7. Controlled Deep Koopman predictor plus MPC

Use the public Lusch encoder, decoder and latent evolution as the starting point. Add measured control inputs to the latent update, initially `z_next = g(z) + B u`, where `g` is the audited autonomous latent map (including frequency dependence when applicable), and learn `B` from forced trajectories. First check the fixed-encoder/input-fit adapter; then jointly train reconstruction, latent consistency and multi-step input-conditioned prediction, with at most six configurations. A state-dependent input map is a separately labeled later variant, not an unreported change after a poor result.

The primary matched experiment trains on the same forced training episodes as H; stage-5 passive pretraining is excluded. Any pretrained variant must account for those extra data and be reported separately. Encode the current measured state, roll each candidate force sequence through the controlled latent dynamics, decode future states, and use the same physical-state cost, bounds and horizon. Use first-action sampled MPC initially, avoiding a new optimization assumption: a nonlinear decoder or frequency-dependent latent map does not generally give a convex quadratic program.

Report **Deep Koopman + MPC** and **Deep Koopman + MPC + common L**, using H's same data-fitted gain and switch. Include the matching neural-predictor/MPC arms so an improved result is not automatically credited to the representation. Audit force dependence, zero-input reduction to the underlying predictor, state reconstruction at initialization, and multi-step predictions on identical saved inputs. These are our controlled adaptations of a published predictor, not a reproduction of a controller in Lusch 2018.

### 8. Controlled DFINE predictor plus MPC

Reuse torchDFINE's nonlinear observation mappings and causal latent filtering. Add an explicit learned control channel to its linear dynamic state, `z_next = A z + B u + w`, where `w` is the modeled process noise. Training, filtering and multi-step forecast code must all consume the correctly time-aligned applied inputs. Refit input-aware dynamics and covariance parameters on the same forced episodes as H; simply wrapping an unforced predictor in an optimizer would give it no identified response to candidate forces.

Start with measured full state and the same data/splits/three seeds. At each sample, update the latent estimate using only current/past observations and applied inputs, then forecast candidate input sequences through the controlled latent model and nonlinear observation decoder. Use first-action sampled MPC with the shared horizon and costs. The initial planner scores decoded conditional-mean forecasts; label this approximation explicitly. A nonlinear decoder means this is not the exact expected state cost, and retained covariance estimates alone do not make it robust or chance-constrained MPC. No future-data smoothing is allowed in control.

Report **DFINE + MPC** and **DFINE + MPC + common L**, plus **DFINE-filtered L alone** to separate inference benefit from planning benefit. The catch uses the same gain/threshold on the declared reconstructed state; record reconstruction bias and switching frequency. For the output-only `(position, angle)` track, use a causal history/state-readout trained on permitted labels, and give competing observation adapters the same history/labels. Compare this track separately from full-state H/L; never give DFINE hidden simulated states at evaluation. Bound architecture/control-channel choices to six configurations before final tests. Reuse the stage-6 noise/missing-data masks where the protocols match.

On an independently constructed linear controlled state-space example, check the control-aware filter against standard Kalman predictions, including missing observations. Also verify force indexing, no future leakage, covariance validity, zero-input behavior and matched-input replay. Both stages receive the same nominal, delay, mismatch and initial-state reporting as other controllers, within their allocation. If an input model cannot be fitted or its solver fails, report that status rather than calling an unimplemented controller unsuccessful.

## Public code and isolation

Prefer author or maintained public implementations over reimplementing algorithms from descriptions:

| Method | Public starting point | Reuse boundary |
|---|---|---|
| EDMD-MPC | [MilanKorda/KoopmanMPC](https://github.com/MilanKorda/KoopmanMPC) | Audit the reference; port only the needed regression/control adapter if its environment differs |
| SINDy | [dynamicslab/pysindy](https://github.com/dynamicslab/pysindy) | Reuse sparse identification and feature libraries; keep our data/control adapter separate |
| PETS | [kchua/handful-of-trials](https://github.com/kchua/handful-of-trials) | Reuse/audit probabilistic ensemble and trajectory sampling; isolate legacy dependencies |
| Deep Koopman | [BethanyL/DeepKoopman](https://github.com/BethanyL/DeepKoopman) | Reuse representation/loss code; identify our input channel and control additions explicitly |
| DFINE | [ShanechiLab/torchDFINE](https://github.com/ShanechiLab/torchDFINE) | Reuse embeddings/filtering/training; identify the added control-aware updates explicitly |
| Nonlinear DeePC | Author implementation linked from the chosen paper, if available and verified | Record any unavailable code, solver limitation or necessary independent implementation |

Existing `external_repos/DeepKoopman` and `external_repos/torchDFINE` are read-only reference checkouts. Their inspected revisions are `20a9f2a1e55ebc53ef1b5706c7a3ff8cc91fd879` and `37d25e332753eaba49f42e91766d0db8e186aefe`, respectively; these are local audit revisions, not claims about latest releases.

Use separate source checkouts under `/tmp/trajectory-literature-vendor/`, environments under `/tmp/trajectory-literature-envs/`, and scratch/dry-run outputs under `/tmp/trajectory-literature-work/`. If a separate project checkout is needed, use an isolated worktree/checkout there. Do not install into or upgrade the project's existing `.venv`/`.venv-gpu`, change root dependency files, alter H/L source, modify reference checkouts, or mix new outputs into `results_v1`. Do not run upstream batch/search entry points automatically. Invoke external code by explicit paths or subprocess adapters, avoiding global path modifications and editable installs into shared environments.

Durable additions are confined to the proposed validation implementation/results directories. Record source URL, commit, license, environment lock and a small patch/port record for each reuse; include a mathematical or saved-example parity check before adapting it. Cache only what can be reconstructed from that record, because `/tmp` can disappear. If compatibility requires a port, preserve the original semantics and record the parity evidence; do not silently substitute a different model. Before accepting changes, check the submitted source/environment files against the pre-work state and inspect the diff for unrelated changes.

## Common comparison protocol

**Data and references.** Reuse the saved original `3500 × 30` collections for seeds 0–2 and their whole-episode splits under [models](results_v1/models/episodes3500_seed0/diagnostics.json). All forced full-state methods receive identical training data and measured inputs. H, L, the first-action neural planner, SMC and its zero-correction control are the references. E4's `3281 × 32` collection is a separate cohort; reuse its scores only when that exact collection is shared. Any window-based method unable to use a 30-step episode gets an explicitly matched new cohort, not stitched episodes.

**Selection and fitting.** Use training/validation data only for configuration selection; leave the final initial-state outcomes out of tuning. Report each prespecified candidate's validation score and the selection rule. Begin with normalized 30-step validation error; probabilistic models also report calibration. Keep a zero-input/equilibrium check and short control pilot as diagnostics, not opportunities for unreported tuning. Different optimization methods may need different solvers; match physical cost/constraints and record solver-specific allowances.

**Feedback attribution.** Report native controllers without L as the primary literature baselines. For EDMD, SINDy and the stage-7/8 controlled predictors, also test the same fixed L gain and switch used by H, labeled `baseline + common L`. Include the every-step neural planner with that same catch as a reference; otherwise changing H's five-step queue would confound the comparison. Any model-specific terminal controller is a separately labeled variant. Do not repair H during this comparison; keep possible implementation fixes as another experiment.

**Starting states and endpoints.** Start at six authored angles plus the existing fixed 50 random states, 14 s, all three seeds. Keep the current angular-tail/no-fall endpoint. Separately report falls, tail angle/angular-speed fluctuations, cart travel, full settling, input saturation and effort. Run the prespecified three-angle subset for 60 s. No recentering or rail acceptance gate is reintroduced.

For detailed operating regions, evaluate computationally affordable methods on the existing 200-state set and 171-angle grid, then on every second row/column of the saved 41×41 angle/rate grid (441 points). Add a fresh fixed 200-state holdout, seed `20261012`, drawn from the same declared box: the old operating-region results are already public to this project. No test-state selection based on observed outcomes. Expensive methods can receive fewer completed sets under the cap, with untested cells visibly missing. Preserve a per-set comparison rather than pooling unlike starting-state distributions.

**Stress and data efficiency.** After nominal implementation/evaluation, test the shared one-sample input delay, ±25% parameter mismatches and the smallest existing measurement/process-noise levels on six authored plus fifty random states. Match disturbance arrays and the original training-noise protocol. Report physical noise scales and fluctuation sizes beside strict endpoint counts. Next, within remaining allowance, use nested whole-episode training subsets at three prespecified budgets with a fixed held-out set; report training, validation and total acquired transitions separately. Retrain H/L on those same subsets before making matched data-efficiency claims. Transfer to the existing linear/acrobot plants is an optional expansion, not a prerequisite for measuring cart-pole baselines.

**Statistics and figures.** Give successes/denominator for each seed, paired discordant counts against H and L, and descriptive intervals. Treat grids as fixed tests and three-seed frequencies as descriptive, not calibrated probabilities. Primary comparisons are each new method against L on the fresh 200-state holdout; adjust that family of tests together if claiming statistical significance. Mark smaller diagnostic comparisons exploratory. Produce one initial-angle/angular-rate panel per controller with common axes/colors, per-state comparisons for random cases, and prediction/data/latency plots. Do not interpolate untested outcomes.

## Staging, costs and supervision

These are **proposed additional compute limits**, separate from the completed campaign's accounting. They require user review; they are not measured runtime estimates. Adding stages 7–8 increases the earlier proposal by 10 GPU-hours and 20 CPU core-hours, from 20/64 to 30/84. This is a proposed allocation change, not permission to spend it; if the earlier ceiling is retained, later work must be deferred or the scope reduced explicitly.

| Stage | GPU-hours ceiling | Aggregate CPU core-hours ceiling |
|---|---:|---:|
| Preparation and bounded source/solver checks | 0 | 2 |
| EDMD | 0 | 8 |
| SINDy | 0 | 12 |
| PETS | 8 | 8 |
| Nonlinear DeePC | 2 | 12 |
| Core reserve | 2 | 6 |
| Deep Koopman passive prediction | 3 | 8 |
| DFINE inference/prediction | 5 | 8 |
| Controlled Deep Koopman + MPC | 4 | 8 |
| Controlled DFINE + MPC | 6 | 12 |
| **Total proposed ceiling** | **30** | **84** |

CPU accounting sums CPU time across worker processes/threads; GPU accounting conservatively reserves elapsed device-occupied job time, without counting simultaneous jobs as free. Record elapsed time separately. Start with one GPU job and at most four single-thread CPU workers. Lower-priority work is deferred when the allocation is exhausted; do not silently borrow more. SAC is outside these ceilings and needs a separate bounded proposal, including online interaction budgets, reward and validation protocol.

At each stage: verify the implementation, time a bounded seed-0 pilot on six authored plus ten predeclared random states (prediction-only stages use a bounded train/validation batch instead), project the requested matrix, then run only what fits the remaining allocation. These pilots are part of the compute budget. Stop a job at its explicit ceiling; if measured cost exceeds three times its pilot projection, retain partial outputs and report the discrepancy. A checkpoint after EDMD/SINDy reports findings and costs; the reviewed execution scope determines whether subsequent stages continue automatically. Do not substitute endless optimization for an unsuccessful result.

Execution must use the usage supervisor's existing **85% five-hour / 90% weekly thresholds and unchanged credit guard**. The completed supervisor hard-codes its old prompt and results path: simply restarting it would be wrong. Prepare a task-specific entry point reusing its guard/defaults, with this plan's prompt and an isolated state/log directory; verify that it starts a new task rather than resuming the completed campaign. Preserve the original supervisor and `results_v1/STATUS=DONE`. Refuse unguarded launches when usage telemetry is unavailable; stop on a recorded credit decrease. No automatic launch is part of this planning task.

## Execution checklist and fixed defaults

The following defaults make this executable after the user returns their review. Incorporate those updates first. Approval of the resulting scope authorizes routine setup, adapter implementation, bounded correctness checks, pilots and evaluations within the accepted caps; it does not require per-method permission. Announce progress at the EDMD/SINDy checkpoint and continue in order unless the review explicitly makes that a stopping point. Do not start the work merely because this planning document exists.

1. **Record the reviewed scope.** Put the reviewed plan revision, selected stages and approved resource ceilings in `results_literature/manifest.json`; create its own status and supervised task prompt. Snapshot the existing tracked-file diff and checksums of submitted source and shared dependency files so pre-existing edits can be preserved. Verify the task supervisor targets this extension, not the old session or results. No change to the original supervisor's thresholds/guard is permitted.
2. **Inventory before training.** Verify the three saved collections, split identities, physical units and reusable reference cases. Record missing raw files or environments explicitly. Reuse artifacts only on an identity match. If a matched artifact is absent, regenerate only the necessary data/reference under the current allocation and record the cost; do not silently import scores from another collection. Pin external sources and create isolated environments before importing legacy training scripts.
3. **Implement one method at a time.** Use a small `fit / predict_inputs / act` adapter boundary, with stateful `reset` for filtering. `predict_inputs` must consume an explicit candidate force sequence. Keep source-faithful modules separate from our task adapters and input-aware extensions. Validate units, input influence, causal indexing, reset behavior, independently calculated objective values and failure reporting on small examples before scientific cases.
4. **Freeze selection.** Save the candidate grid, validation scores, selected configuration and selection rule before final evaluation. The initial selection score is mean per-state standardized 30-step rollout squared error; normalization uses training statistics. Report probabilistic calibration separately. Numerical nonfiniteness is a diagnostic/implementation issue, not a reason to invent a different metric after inspecting the test outcomes.
5. **Run the bounded pilot, then the minimum matrix.** Use seeds 0–2; six authored initial states and the fixed fifty random states; 14 s, plus the existing three-angle subset at 60 s. This is **177 runs per control arm**: `3 × (6 + 50 + 3)`, with horizons kept separate. Reuse only matching reference cases. For stages 1, 2, 7 and 8, native versus common-L variants are distinct arms; stage 2's shared sampled-planner arm is also distinct. Stage 8's filtered-L control is a separate inference ablation. Forecast all required arms, training candidates and postprocessing—not just one successful case—against the stage cap.
6. **Expand only within the stage allocation.** Priority after the minimum matrix is fresh 200-state holdout, then delay/mismatch/noise tests, then old random/from-rest sets, then the 441-point map, then data-efficiency/transfer. For a method too costly to expand, retain its complete minimum matrix and show sparse tested cases; do not make a dense success map. If even the minimum cannot fit, preserve partial counts with exact denominators and a cap explanation. No automatic allocation increase or borrowing from another stage; the explicit reserve is available to complete core work before optional expansions.
7. **Close each stage.** Require a source/adapter change record, passing relevant correctness checks, model/config/data identities, completed or explicitly partial case inventory, per-seed summary, comparisons to H/L, computation/effort diagnostics, and a short interpretation with failures retained. Prediction-only stages instead require reconstruction/forecast tables and causal-inference checks, not invented control success counts. Update the single report/status and save a reviewable code diff before moving on.
8. **Finish the extension.** Consolidate controller maps, baseline case matrices and prediction/inference comparisons; update the collaborator briefing and replay inventory. Verify local links and independently recompute a stratified sample of saved endpoints. Record skipped/blocked work separately from scientific failures. Stop the supervised task on its own DONE status while preserving the original completed campaign.

### Prespecified small candidate grids

These are defaults to freeze before fitting, not claims of globally optimal tuning. Public-code conventions and exact scaling must be recorded; a compatibility-driven change gets a dated amendment before test evaluation.

- **EDMD:** the two feature dictionaries above × ridge strengths `1e−6, 1e−4, 1e−2` after feature standardization; six candidates.
- **Discrete SINDYc:** polynomial degree `2 or 3` × with/without angle sine/cosine × sparse coefficient thresholds `1e−3, 1e−2, 1e−1` in standardized regression; twelve candidates. Retain a small fixed ridge for the thresholded refits and record the resulting raw-coordinate coefficients.
- **PETS:** five probabilistic networks; start with twenty trajectory particles and the existing 256-candidate/three-iteration search. The three timing configurations are `(particles, candidates) = (20,256), (5,256), (5,128)`, with three iterations and first-action feedback. Choose the largest that fits the measured allowance; disclose the reduction. Fit models with the audited bounded-variance objective and validation early stopping, capped at 100 epochs initially.
- **Kernelized DeePC:** three bandwidths around the training-window median-distance scale (`0.5, 1, 2` times that scale) × two regularization values fixed in the source's scaled formulation before fitting. The source audit must resolve that formulation; if unavailable, this stage is blocked rather than substituted silently.
- **Deep Koopman/DFINE:** latent dynamic dimensions `4, 8, 16` × two declared loss/regularization settings; at most six configurations per stage. Use the source's reconstruction/prediction terms and record their exact coefficients in the stage manifest before fitting. Start at 100 training epochs with validation early stopping. Stage 7/8 full-state adapters train on forced data without free passive pretraining. Keep noise covariance fitting and latent/control dimensions visible in the record.

The initial nonlinear-optimizer SINDy arm uses a maximum of fifty iterations per solve and reports termination/residuals; it is not assumed globally optimal. A finite bounded control proposal can be applied only when it meets declared input/solver acceptance checks. Otherwise record the solver failure and the fixed zero-input fallback. Apply analogous explicit fallback reporting to every baseline; no undisclosed L fallback is allowed in a native arm. Wall-time checks cannot manufacture real-time feasibility: report latency distributions and separate the existing 20 ms input-delay stress from nominal fixed-step simulation.

### What needs renewed input, and what does not

Routine dependency isolation, parity-preserving compatibility fixes, bounded implementation debugging and pilot-driven use of the declared particle configurations are included in the reviewed scope. A failed controller, disappointing score or a small certified operating region is a result to document, not a reason to stop for approval. Pause for a genuine scope decision only if work requires more than the accepted total budget, a different physical task/data-privilege assumption, substantial unplanned source modification, or a replacement method where the specified one cannot be implemented. Report the concrete obstacle and completed evidence. Missing optional software should first be handled with the declared isolated/public-code route.

## Deliverables and next-session handoff

After review, put new implementation under `validation/experiments/literature/` and results under `validation/results_literature/`; scratch stays under `/tmp`. Keep one experiment manifest containing source/configuration, external revisions, data/split/model identity, seeds, device, elapsed/CPU/GPU accounting and completion state. Resume only when configuration and data/model identity match, not merely because a filename exists. Save per-case outputs and raw-artifact locations; distinguish scientifically failed runs from solver errors and untested cases. Preserve baseline code and old results.

Deliver a single results report, tested adapters, prediction tables, per-seed/paired control counts, regime figures, source-faithfulness notes and an artifact inventory sufficient to transfer the replay bundle. Link the report from the collaborator briefing, explicitly marking which methods ran, which were adapted, and which were deferred.

**Recommended review decision:** approve the ranked protocol and a first checkpoint after EDMD/SINDy; confirm or revise the proposed 30 GPU-hour / 84 CPU core-hour allocation (including the added controller stages). The default is to continue through the reviewed stages automatically after reporting the first checkpoint, with optional expansions constrained by remaining budget. The user can annotate this document or provide updates before implementation. Nothing is queued yet.
