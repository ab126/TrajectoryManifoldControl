# Validation of the submitted trajectory-manifold controller

## Q1 — Does it work as claimed?

**The authored seed-234 run reproduces all six angle/no-fall successes, but the hybrid is sensitive to seed and its local-feedback ablation is stronger on the authored cases.** The fresh unchanged `--profile dev` run is saved in [E0/authored_summary.json](E0/authored_summary.json); expanded endpoints and ten-seed ablations are complete. Nine focused [metric and arm tests](E0/metric_arm_tests.txt) passed. The authored notebook already contains a development-profile demonstration: all six initial angles (±15°, ±30°, ±60° from rest, 14 s) have `balanced=True`; three also have `cart_rest=True`. Local feedback supplies 50–97% of steps, with 6–70 CEM solves per case. The notebook is the authored baseline; the fresh independent reproduction is reported below. Source: [notebook](../../manifold_control.ipynb), cells 10 and 12.

The primary endpoint is the runner's `balanced` metric (|θ|≤0.02 rad and physical |θ̇|≤0.05 rad/s in the last 3 s) **and no fall** (|θ|<π/2 throughout). Cart excursion, drift, settling, final position, input saturation and local feedback use are reported separately. There is no rail limit, recentering deadline or acceptance gate.

The fresh authored reproduction gives **6/6 primary successes** (Wilson 95% interval 0.610–1.000), with cart rest in 4/6 rather than the notebook's 3/6. Local feedback supplies 43.1–96.7% of steps and CEM solves range from 6 to 81. Maximum cart excursions span 0.413–7.858 m. These differences from the saved notebook should be retained as reproduction variability, not overwritten. The authored local-fit selection reaches |θ|=1.775 rad (101.7°), with standardized radius 0.809 and inputs spanning the full ±10 limit. Sources: [authored endpoints](E0/authored_cases.csv), [angle/cart/input figure](E0/authored_excursions.png), [model diagnostics](E0/authored_model_diagnostics.json).

On IC-A at 14 s, H succeeds in 4–5/6 cases per seed, while L succeeds in 6/6 for every seed. M and M1 each have 0/6 for every seed; H* has 4/6 for every seed. At 60 s, H improves to 4–6/6, L remains 6/6, and H* gives 4–5/6. This supports the local catch's importance; it does not yet determine whether H adds reach on harder ICs. The true-model failures show that learned-model error is not the only limitation. Sources: [per-seed rates](E0/rates.csv), [paired comparisons](E0/paired_comparisons.csv), [figures](E0/success_rates.png).

M and M1 can approach upright while missing the authored settling limits. In the illustrative seed-0 −15° case, M1's final-three-second maximum angle is 0.0388 rad and maximum physical angular speed is 1.047 rad/s; L gives 0.000275 rad and 0.000102 rad/s. The [terminal-window figure](E0/terminal_behavior.png) and [all 300 seeded 14 s terminal summaries](E0/terminal_behavior.csv) show the distinction between approaching upright and meeting the authored settling limits.

A [feasible-alternative diagnostic](E0/feasible_alternative_diagnostics.json) examines the last saved plans of three illustrative seed-0 misses. It found no lower internal-model cost among the local-law input sequences tried; that does not establish global optimality. Some older E0 arrays lack full planned inputs, so their chosen true-plan costs are left unavailable. E1 saves full plans for matched-input replays. Five [objective, missing-data, interval and matched-selection checks](failure_diagnostic_tests.txt) pass.

## Q1.1 — Failure region, data needs, novel systems

**E1 is complete. Local feedback alone (L) succeeds more often than H in every tested IC set and seed. Data efficiency and robustness remain unresolved pending E2–E3.**

The table gives the range of primary success counts across seeds 0–2, with each seed kept separate in the [rates and Wilson intervals](E1/rates.csv). These are tested ICs, not continuous-domain guarantees.

| IC set | H | L | M | M1 | H* |
|---|---|---|---|---|---|
| B: from rest | 72–107/171 | 137–141/171 | 0/171 | 0/171 | 85–97/171 |
| C: random box | 31–78/200 | 122–130/200 | 0/200 | 0/200 | 42–53/200 |
| D: angle/rate grid | 356–717/1681 | 1073–1089/1681 | 0/1681 | 0/1681 | 454–548/1681 |

On IC-B and IC-D, every H success also succeeds under L in all three seeds. L adds 54, 69 and 34 matched from-rest successes, and 661, 733 and 368 grid successes. In the random box, H adds 3, 0 and 1 catches beyond L, while L adds 77, 99 and 51 beyond H. Thus H occasionally catches an IC that L misses, but it does not expand the tested from-rest or angle/rate-grid reach over L. Together with the zero primary successes for M and M1, this supports the identified feedback law as the useful control component in this configuration. Sources: [matched comparisons](E1/paired_comparisons.csv), [rates and per-seed Wilson intervals](E1/rates.csv).

On the tested 1° from-rest grid, L succeeds from −68° to +68° in seed 0 and −70° to +70° in seeds 1–2. H’s largest uninterrupted successful spans are respectively [−36°, +34°], [−39°, +30°] and [−38°, +49°]. Additional isolated successful spans remain in the [exact interval table](E1/from_rest_intervals.csv); failed points are not bridged, and the grid establishes no guarantee between tested angles. H* is not a consistent upper bound: it improves seed 1, while seed 2 gives H 16 additional from-rest and 31 additional random-box successes over H*, against 6 and 4 in the opposite direction. These are matched per-seed comparisons on fixed ICs, not a pooled population estimate.

The [five selected failure diagnoses](E1/representative_failures.json) show both optimizer and model limitations. Every selected initial CEM plan admits a bounded local-law input sequence with lower internal-model cost. For the true-model −60° case, those costs are 7671.793 versus 6681.425, witnessing optimizer suboptimality despite an accurate transition. All four learned-model examples also have later model/true-cost preference reversals. For H at −43°, step 698 prefers its chosen inputs internally (95.216 versus 330.518), while true-plant replay favors the alternative (286.494 versus 122.846). The [−60° H trace](E1/failure_2.png) also shows repeated local-feedback exits where matched L succeeds. These are illustrative optimizer/model witnesses and switch associations, not proof of a sole closed-loop failure cause or global optimality. Sources: [matched replay for H](E1/failure_1_prediction_errors.json), [matched replay for H*](E1/failure_5_prediction_errors.json), [overlapping failure categories and toggle counts](E1/failure_categories.csv).

The 60 s subsets do not uniformly resolve misses. On the fixed −60°, 0°, +60° from-rest subset, L gives 3/3 primary successes in every seed, H gives 1–3/3 and H* gives 1/3. Angle success and full cart settling remain distinct: four of the six nonzero-angle L runs have no full-settle time by 60 s under the stated position-change criterion. Conversely, seed-2 H from −60° records a full-settle time of 29.1 s with final cart position −5.603 m; settling does not require recentering. Earlier falls remain disqualifying even when a run later balances. These small, prespecified subsets are descriptive; the [60 s per-seed counts](E1/rates.csv), [settling/final-position case metrics](E1/summary.csv) and [excursion, saturation and local-use distributions](E1/distributions.csv) retain all runs, including failed simulations with large excursions.

[Exact tested success intervals from rest](E1/from_rest_intervals.csv) preserve holes in the 1° grid. The [D map for seed 0](E1/map_D_seed0.png) and [cart-excursion map](E1/cart_excursion_D_seed0.png) have counterparts for seeds 1–2. [Paired comparisons](E1/paired_comparisons.csv), [overlapping failure categories](E1/failure_categories.csv) and [five selected failure diagnoses and evidence paths](E1/representative_failures.json) ([example plot](E1/failure_1.png)) separate the endpoints, switch associations and model/optimizer witnesses. Cart excursions remain descriptive; no rail criterion is used.

The previous true-model wide-box diagnostic has 147/200 angle-balanced runs, 67/200 also reaching cart rest, and 103/200 crossing π/2. The first two counts omit the newly required no-fall condition; they must not be presented as primary successes. Initial speeds reach 7.8 m/s and angles reach 74°, unlike the authored demonstration. Source: [independent audit](../overnight_runs/20261007T054549.695689Z/calibration/independent_audit.json); recomputation from saved trajectories gives **65/200** angle-balanced with no fall (Wilson 95% interval 0.264–0.393). Cart-excursion median is 3.829 m, 95th percentile 14.196 m and maximum 20.009 m. This historical run used N=60 and 20 s, so it is not a matched comparison with the new N=30, 14 s arms. Sources: [recomputed summary](historical_oracle/endpoint_summary.json), [all cases](historical_oracle/summary.csv), [manifest](historical_oracle/manifest.json).

`run_inverted_pendulum_mc.log`, `inverted_pendulum_mc_debug.ipynb`, and `linear_system_mc.ipynb` contain historical latent-MPC outputs. They are not evidence about the active CEM controller.

### E2 model diagnostics (all 30 models complete)

The held-out 30-step prediction errors decrease substantially as the data budget grows. The nearest-6000 local fit uses every training transition at 35 and 100 episodes. Those unwrapped training trajectories reach well beyond a single pendulum revolution; the small-data fit is not local to upright. Even at 3,500 episodes the selected angles reach 98–104°. At 10,000 episodes they reach 78–84°. These are collection-neighborhood diagnostics, not successful closed-loop angles. Sources: [per-model prediction and local-fit diagnostics](E2/model_diagnostics.csv), [saved models' diagnostics](models/episodes35_seed0/diagnostics.json).

The table gives min–max across the five whole-run seeds. RMSE columns use stored normalized state coordinates (physical velocities multiply by √9.81); no aggregate across unlike state units is used. Collected transitions are 30 × episodes; the fitted counts below follow the actual whole-episode split.

| Episodes | Fitted transitions | Local-fit samples | Selected max abs θ [degrees] | N-step p RMSE | N-step v RMSE | N-step θ RMSE | N-step ω RMSE |
|---|---|---|---|---|---|---|---|
| 35 | 750 | 750 | 420.6–506.0 | 0.5581–0.7051 | 0.3314–0.8815 | 1.48–2.858 | 1.671–3.086 |
| 100 | 2100 | 2100 | 445.0–517.9 | 0.4035–0.6097 | 0.4916–0.6615 | 1.561–3.422 | 1.396–3.26 |
| 350 | 7380 | 6000 | 343.6–372.9 | 0.4318–0.608 | 0.446–0.6766 | 0.5935–1.995 | 0.8578–2.311 |
| 1000 | 21000 | 6000 | 154.8–196.0 | 0.1224–0.134 | 0.1761–0.2471 | 0.1895–0.2952 | 0.2947–0.4151 |
| 3500 | 73500 | 6000 | 98.3–103.7 | 0.03114–0.04086 | 0.04457–0.07288 | 0.04959–0.06936 | 0.08783–0.1553 |
| 10000 | 210000 | 6000 | 78.5–83.5 | 0.01044–0.01186 | 0.01396–0.02541 | 0.0185–0.02189 | 0.02975–0.08188 |

### Completed E2 budgets through 3,500 episodes

At 35, 100 and 350 episodes, H, L and M1 each score **0/6 on IC-A and 0/50 on IC-C50 in every seed**. All 840 runs fall at each of the 35- and 100-episode budgets. At 350 episodes, 736/840 fall; the 104 runs without a fall still miss the angular settling limits at 14 seconds. These totals describe matched cases across five seeds and three arms; they are not pooled population estimates.

At 1,000 episodes (30,000 collected and 21,000 fitted transitions), performance is highly seed-dependent. H scores 0–4/6 on IC-A and 0–16/50 on IC-C50; L scores 0–6/6 and 0–37/50. M1 has no primary successes. In seed 2, H adds 11 random-subset catches beyond L, while L adds none (exact matched McNemar p=0.000977). Thus component attribution depends on the data budget and fitted gain: the broader E1 conclusion concerns its 3,500-episode configuration, and does not establish that H can never help when L misses.

| Seed at 1,000 episodes | IC-A H | IC-A L | IC-A M1 | IC-C50 H | IC-C50 L | IC-C50 M1 |
|---|---|---|---|---|---|---|
| 0 | 3/6 | 4/6 | 0/6 | 15/50 | 25/50 | 0/50 |
| 1 | 4/6 | 6/6 | 0/6 | 10/50 | 37/50 | 0/50 |
| 2 | 0/6 | 0/6 | 0/6 | 11/50 | 0/50 | 0/50 |
| 3 | 2/6 | 6/6 | 0/6 | 16/50 | 37/50 | 0/50 |
| 4 | 0/6 | 2/6 | 0/6 | 0/50 | 0/50 | 0/50 |

At 3,500 episodes (105,000 collected and 73,500 fitted transitions), H scores 4–5/6 on IC-A and 6–19/50 on IC-C50; L scores 6/6 and 30–36/50. M1 again has no primary successes. The table preserves each whole-run seed rather than pooling the repeated ICs.

| Seed at 3,500 episodes | IC-A H | IC-A L | IC-A M1 | IC-C50 H | IC-C50 L | IC-C50 M1 |
|---|---|---|---|---|---|---|
| 0 | 4/6 | 6/6 | 0/6 | 11/50 | 33/50 | 0/50 |
| 1 | 4/6 | 6/6 | 0/6 | 6/50 | 36/50 | 0/50 |
| 2 | 4/6 | 6/6 | 0/6 | 16/50 | 36/50 | 0/50 |
| 3 | 5/6 | 6/6 | 0/6 | 19/50 | 30/50 | 0/50 |
| 4 | 5/6 | 6/6 | 0/6 | 18/50 | 34/50 | 0/50 |

Per-seed Wilson intervals, falls and descriptive measures are in the [completed-budget rates](E2/interim_completed_budget_rates.csv), with [matched comparisons](E2/interim_completed_budget_pairs.csv). The zero-count intervals at the lower budgets are 0–0.390 for six ICs and 0–0.071 for fifty ICs. These results concern the submitted collection/fitting/controller protocol, not a lower bound for a redesigned method. The 10,000-episode evaluations remain in progress; no final data-sufficiency threshold is established yet.

### Local-fit stability diagnostic

Every one of the 30 identified DARE gains stabilizes its fitted linear model (spectral radius 0.9765–0.9984), but that does not ensure stability on the true plant. At 35, 100 and 350 episodes, all five gains give a true upright closed-loop Jacobian with spectral radius greater than one. This diagnoses a limitation of fitting a broadly selected neighborhood as a local linear model. Larger budgets reduce the discrepancy substantially, but the stability result is not monotone with data size.

The table uses the exact derivative of the submitted sample-and-hold RK4 map with input u=−Kx, before saturation. Independent central differences agree to at most 6.4×10⁻¹²; a gain designed on that true derivative gives radius 0.998382. These are post hoc diagnostics of the recorded gains, with no controller modification or gain selection. A radius above one concerns the full-state upright equilibrium; slowly growing cart modes can still pass a finite-horizon angle endpoint. A radius below one does not establish a capture region for the finite ICs. Sources: [per-model spectral radii](E2/local_stability.csv), [figure, including a zoom near one](E2/local_stability.png), [independent checks](E2/local_stability_checks.json), [calculation](../experiments/local_stability.py).

| Episodes | True local Jacobian radius, min–max across five seeds | Seeds above one |
|---|---|---|
| 35 | 1.291198–9.687442 | 5/5 |
| 100 | 1.123959–5.703622 | 5/5 |
| 350 | 1.420435–16.854472 | 5/5 |
| 1000 | 0.999522–1.003937 | 4/5 |
| 3500 | 0.986200–1.000032 | 1/5 |
| 10000 | 0.990098–1.001227 | 3/5 |


<!-- E2_AUTO_START -->

### E2 completed measurements

Sources: [per-seed counts and Wilson intervals](E2/rates.csv), [matched comparisons](E2/paired_comparisons.csv), [case metrics](E2/summary.csv), [figure](E2/success_rates.png), [manifest](E2/manifest.json). Raw-array paths are recorded in the case metrics.

Counts below are ranges across whole-run seeds; seeds are not pooled. The common cart-pole endpoint remains angle stabilization with no fall. Novel-plant endpoints are stated in their manifests.

| Episodes | IC condition | Seconds | Arm | Success-count range / n |
|---|---|---|---|---|
| 100 | A | 14.0 | H | 0/6 |
| 100 | A | 14.0 | L | 0/6 |
| 100 | A | 14.0 | M1 | 0/6 |
| 100 | C50 | 14.0 | H | 0/50 |
| 100 | C50 | 14.0 | L | 0/50 |
| 100 | C50 | 14.0 | M1 | 0/50 |
| 1000 | A | 14.0 | H | 0–4/6 |
| 1000 | A | 14.0 | L | 0–6/6 |
| 1000 | A | 14.0 | M1 | 0/6 |
| 1000 | C50 | 14.0 | H | 0–16/50 |
| 1000 | C50 | 14.0 | L | 0–37/50 |
| 1000 | C50 | 14.0 | M1 | 0/50 |
| 10000 | A | 14.0 | H | 4–5/6 |
| 10000 | A | 14.0 | L | 6/6 |
| 10000 | A | 14.0 | M1 | 0/6 |
| 10000 | C50 | 14.0 | H | 5–20/50 |
| 10000 | C50 | 14.0 | L | 29–34/50 |
| 10000 | C50 | 14.0 | M1 | 0/50 |
| 35 | A | 14.0 | H | 0/6 |
| 35 | A | 14.0 | L | 0/6 |
| 35 | A | 14.0 | M1 | 0/6 |
| 35 | C50 | 14.0 | H | 0/50 |
| 35 | C50 | 14.0 | L | 0/50 |
| 35 | C50 | 14.0 | M1 | 0/50 |
| 350 | A | 14.0 | H | 0/6 |
| 350 | A | 14.0 | L | 0/6 |
| 350 | A | 14.0 | M1 | 0/6 |
| 350 | C50 | 14.0 | H | 0/50 |
| 350 | C50 | 14.0 | L | 0/50 |
| 350 | C50 | 14.0 | M1 | 0/50 |
| 3500 | A | 14.0 | H | 4–5/6 |
| 3500 | A | 14.0 | L | 6/6 |
| 3500 | A | 14.0 | M1 | 0/6 |
| 3500 | C50 | 14.0 | H | 6–19/50 |
| 3500 | C50 | 14.0 | L | 30–36/50 |
| 3500 | C50 | 14.0 | M1 | 0/50 |

<!-- E2_AUTO_END -->

## Q2 — Errors in the implementation note

The reviewed transcription is [trajectory_manifold_implementation.tex](../../../trajectory_manifold_implementation.tex), stored in the parent directory.

**Review complete: no equation/code contradiction found, but the note needs 27 qualifications and clarifications.** The [v3 annotated copy](reviews/implementation_tex_v3/trajectory_manifold_implementation_reviewed_v3.tex) preserves the labmate's text and flags notation overloads, implicit normalized costs, collection details, switch geometry, numerical fallback and evaluation omissions. The [comment ledger](reviews/implementation_tex_v3/annotations.json) records each issue. The selected local-fit neighborhood is substantially broader than the handoff estimate; see the measured E0 diagnostics above.

## Q3 — Comparison and contribution

**H is a full-state learned-dynamics shooting controller. E0–E1 favor its local-feedback ablation over H; matched ensemble, DeePC and SMC comparisons remain unresolved pending E4 and E6.** It combines one-step transition regression, CEM planning, five-step queues and a ridge/DARE local catch. E0–E1 support the catch's importance on the authored and broader tested sets. The graph theorem supplies explicit trajectory coordinates but does not constrain prediction error or guarantee closed-loop success; see the [proof audit](reviews/paper_review_v1/proof_audit.md).

| Comparison | What differs from H | What this campaign can establish |
|---|---|---|
| DeePC | Input/output trajectory spanning and a convex QP; exact LTI representation requires excitation and observability assumptions. | Matched nonlinear cart-pole performance and regularization sensitivity; the LTI theorem does not guarantee this nonlinear baseline. |
| Lusch Deep Koopman | Autonomous state embedding trained with multi-step and linearity losses; no implemented controller. | Prose/model comparison, not an invented Lusch control baseline. |
| DFINE | Unforced noisy partial observations, latent factors and linear dynamics with Kalman inference. | Observation and uncertainty differences; no DFINE controller is constructed. |
| Baek deep RL | Model-free SAC, symmetry augmentation, triple pendulum and hardware demonstrations. | Qualitative budget/task comparison; its 0.8–2 million steps cannot establish a matched sample-efficiency advantage for H. |
| Same-data SMC | Designed scalar sliding variable and a sampled boundary-layer reaching law, using the saved local fit. | E6 compares SMC with its zero-correction control, L and H; nonlinear-plant robustness remains empirical. |

The [verified primary-source matrix](reviews/paper_review_v1/literature_matrix.csv) supports these distinctions. Nagabandi neural shooting MPC and single-model variants of PETS are the closest algorithmic antecedents. H's training and online controller do not use plant equations. Its limitations include full-state measurement, deterministic one-step learning, no uncertainty estimate, open-loop queues, a hard switch and no stability guarantee. So far, the identified local law supplies the strongest control-performance evidence. Data efficiency, robustness, novel systems and the remaining matched baselines are pending E2–E4.

Classical SMC designs desired motion on a state/error surface and a law that attracts and maintains that motion. Its ideal rejection result concerns uncertainty entering through the control channel; smoothing changes exact invariance to practical accuracy. H's quadratic ellipsoid selects the CEM or LQR branch, and its [source](../../scripts/run_inverted_pendulum_mc.py) supplies no sliding-variable reaching law. Mode switching alone does not establish a sliding-mode design. Source: [Spurgeon's tutorial](https://kar.kent.ac.uk/41730/1/sliding_mode.pdf).

A relevant planning/feedback connection is [Dynamic Tube MPC](https://arxiv.org/abs/1907.06553), which combines nominal trajectory optimization with boundary-layer feedback and an uncertainty-dependent tube. H's origin-centered catch does not implement that construction. The approved [E6 specification](smc_research.md) instead tests one same-data discrete feedback design. Its boundary width comes from validation residuals, not a certified uncertainty bound; its fitted linear-model guarantee does not extend automatically to the nonlinear plant, clipping, measurement noise or delay. Comparing with the zero-correction control separates the added nonlinear correction from the different underlying feedback gain. Fifteen [scientific wiring and numerical checks](E6_SMC/smc_tests.txt) passed during preparation; empirical outcomes remain unresolved until E6 runs after E4.

The fresh authored run records 216 CEM solves: median 30.2 ms, 95th percentile 31.0 ms, maximum 130.2 ms. All exceed the 20 ms physical sample interval; one exceeds the nominal five-step queue interval of 100 ms. The plant simulator advances a fixed step after the controller returns, so this demonstration does not include computation delay. The measured timing matters for a hardware implementation and for comparing online shooting with direct policy evaluation. Sources: [timing summary](E0/authored_solve_timing.json), [original timings](E0/authored_summary.json), [submitted simulator](../../scripts/run_inverted_pendulum_mc.py).

A plausible research gap is nonlinear input/output data-enabled prediction with explicit identifiability, noise and approximation conditions. Part I flags the output injectivity issue and defers learning/control; the current full-state implementation does not exercise that extension. The [15-row claim ledger](reviews/paper_review_v1/claim_ledger.csv) separates correct graph/rank results from these unsupported extensions.

## Protocol and resume

Operating instructions: [superseding handoff](../codex_handoff_from_claude_v1.md). Jobs longer than two minutes run detached; each experiment's `JOB` records its PID and command. New evaluations save per-case outputs and skip completed cases. The usage supervisor handles account limits; this campaign does not track them.

Time-to-angle-settle is the earliest sample after which both angular limits hold through the end, requiring at least 3 s of remaining observation. Time-to-full-settle also requires physical cart speed ≤0.05 m/s and position change ≤0.01 m/s over rolling 1 s windows. These are descriptive times, not additional primary pass/fail criteria. Prediction RMSE is reported in stored normalized state units; physical velocities multiply by √9.81.

The runner's seed jointly controls data collection, episode splitting, network initialization and CEM draws. Seeds 0–9 therefore test whole-run reproducibility, not training randomness alone. Arms share data, fitted gain and initial conditions within each seed; every case restarts the controller RNG as the authored runner does. H* copies the learned state scaling into the existing true-RK4 transition so its exploration scaling stays matched.

Wilson 95% intervals are reported separately per seed/condition. Fixed authored cases and grids describe tested conditions; their intervals do not imply random population coverage over an untested operating domain. Exact McNemar comparisons use matched ICs within seed. Settling times are right-censored when the observation is too short; omitted settling times are counted in distribution tables.

The bounded [paper ledger and proof audit](reviews/paper_review_v1/proof_audit.md) and [primary-source literature matrix](reviews/paper_review_v1/literature_matrix.csv) are available. The graph and rank results are correct under their assumptions. The process-noise extension needs qualification if noise is not retained in the behavior vector. The [v3 annotated implementation copy](reviews/implementation_tex_v3/trajectory_manifold_implementation_reviewed_v3.tex) contains 27 comments and preserves source prose/equations. No equation/code contradiction was identified; normalization, notation, evaluation and collection omissions remain real specification issues.

## Deviations

2026-10-11 (user request): Queue the same-data SMC and zero-correction feedback baselines automatically after required E2–E4 measurements. The [request](E6_SMC/REQUEST.json) and [research specification](smc_research.md) define CPU-only evaluation on saved models, matched ICs and affordable E3 conditions, with a one-hour CPU/wall allowance inside the original cap. The future E4 entry point and resumed continuation invoke `experiments/queued_smc.py`; active jobs and supervisor guards are unchanged. Inspect `E6_SMC/JOB` and its manifest on resume, and keep STATUS RUNNING until the extension finishes or records a cap deferral. Summarize its [results](E6_SMC/RESULTS.md), paired feedback comparisons and figures before the final verdict; no SMC outcome exists yet.

2026-10-09: E0 was restarted with eight independent GPU workers after measuring 9% serial GPU utilization. Saved cases/models were reused. Only progress-bar display was suppressed; each case still calls the submitted runner/class implementation. E1 runtime projection will be checked against the cap using measured parallel throughput.

E1 long-horizon subsets are fixed before evaluation: IC-B −60°, 0°, +60°; IC-C indices 0–2; IC-D indices 20, 840, 1660 (θ0=−85°, 0°, +85°, zero angular velocity). All five arms are evaluated for 60 s on these subsets.

Prepared for the post-checkpoint stages: `parallel_campaign.py E2 --workers 8`, then `e2_distribution.py`, then `e3_robustness.py`. The E2 collection adapters and E3 nominal/length/noise wiring have passed three collection checks and six robustness checks ([collection](collection_variant_tests.txt), [robustness](robustness_wiring_tests.txt)). They have not yet run as scientific experiments. E3 robustness uses H and L. Novel-system code is now prepared for a four-state LTI and a uniform-rod acrobot; ten [scientific adapter, dynamics, endpoint and optimal-cost checks](novel_plant_tests.txt) pass. These runs remain pending E2. The DeePC preparation and checks are recorded below.

<!-- E0_AUTO_START -->

### E0 measured results

Sources: [E0 rates](E0/rates.csv), [matched comparisons](E0/paired_comparisons.csv), [case metrics](E0/summary.csv), [distributions](E0/distributions.csv), [success figure](E0/success_rates.png), [manifest](E0/manifest.json).

| Seed | IC set | Arm | Successes/n (14 s, no fall) | Wilson 95% CI |
|---|---|---|---|---|
| 0 | A | H | 4/6 | 0.300–0.903 |
| 0 | A | Hstar | 4/6 | 0.300–0.903 |
| 0 | A | L | 6/6 | 0.610–1.000 |
| 0 | A | M1 | 0/6 | 0.000–0.390 |
| 0 | A | M | 0/6 | 0.000–0.390 |
| 1 | A | H | 4/6 | 0.300–0.903 |
| 1 | A | Hstar | 4/6 | 0.300–0.903 |
| 1 | A | L | 6/6 | 0.610–1.000 |
| 1 | A | M1 | 0/6 | 0.000–0.390 |
| 1 | A | M | 0/6 | 0.000–0.390 |
| 2 | A | H | 4/6 | 0.300–0.903 |
| 2 | A | Hstar | 4/6 | 0.300–0.903 |
| 2 | A | L | 6/6 | 0.610–1.000 |
| 2 | A | M1 | 0/6 | 0.000–0.390 |
| 2 | A | M | 0/6 | 0.000–0.390 |
| 3 | A | H | 5/6 | 0.436–0.970 |
| 3 | A | Hstar | 4/6 | 0.300–0.903 |
| 3 | A | L | 6/6 | 0.610–1.000 |
| 3 | A | M1 | 0/6 | 0.000–0.390 |
| 3 | A | M | 0/6 | 0.000–0.390 |
| 4 | A | H | 5/6 | 0.436–0.970 |
| 4 | A | Hstar | 4/6 | 0.300–0.903 |
| 4 | A | L | 6/6 | 0.610–1.000 |
| 4 | A | M1 | 0/6 | 0.000–0.390 |
| 4 | A | M | 0/6 | 0.000–0.390 |
| 5 | A | H | 4/6 | 0.300–0.903 |
| 5 | A | Hstar | 4/6 | 0.300–0.903 |
| 5 | A | L | 6/6 | 0.610–1.000 |
| 5 | A | M1 | 0/6 | 0.000–0.390 |
| 5 | A | M | 0/6 | 0.000–0.390 |
| 6 | A | H | 4/6 | 0.300–0.903 |
| 6 | A | Hstar | 4/6 | 0.300–0.903 |
| 6 | A | L | 6/6 | 0.610–1.000 |
| 6 | A | M1 | 0/6 | 0.000–0.390 |
| 6 | A | M | 0/6 | 0.000–0.390 |
| 7 | A | H | 5/6 | 0.436–0.970 |
| 7 | A | Hstar | 4/6 | 0.300–0.903 |
| 7 | A | L | 6/6 | 0.610–1.000 |
| 7 | A | M1 | 0/6 | 0.000–0.390 |
| 7 | A | M | 0/6 | 0.000–0.390 |
| 8 | A | H | 4/6 | 0.300–0.903 |
| 8 | A | Hstar | 4/6 | 0.300–0.903 |
| 8 | A | L | 6/6 | 0.610–1.000 |
| 8 | A | M1 | 0/6 | 0.000–0.390 |
| 8 | A | M | 0/6 | 0.000–0.390 |
| 9 | A | H | 5/6 | 0.436–0.970 |
| 9 | A | Hstar | 4/6 | 0.300–0.903 |
| 9 | A | L | 6/6 | 0.610–1.000 |
| 9 | A | M1 | 0/6 | 0.000–0.390 |
| 9 | A | M | 0/6 | 0.000–0.390 |

Settling, excursion, input saturation and local fractions are distributions, not extra acceptance criteria. The 60 s runs remain separate in the CSVs.

Across the ten whole-run seeds, authored-set primary successes per six cases: H 4–5/6, Hstar 4–4/6, L 6–6/6, M 0–0/6, M1 0–0/6. These are repeated fixed ICs; the per-seed counts, not a pooled population interval, are the reproducibility evidence.

<!-- E0_AUTO_END -->

Resolved launcher metadata issue: the host Python lacks `shlex.join`; command quoting now uses `shlex.quote`. E1 was already running and was reattached without duplication.

The detached E1 job is PID 679000; its completion watcher is recorded in `ADVANCE_JOB`. The watcher metadata error after initial launch was repaired without rerunning E0 or launching another E1.

E0 horizon follow-up uses saved 60 s trajectories: score a 20 s prefix only for H/H* cases that were not balanced but had no fall at 14 s. This is a post hoc horizon-limitation diagnostic, not a new random sample ([selection and counts](E0/horizon_followup.json), [20 s prefix endpoints](E0/horizon_limited_20s.csv)). No extra simulation or altered controller was used.

The 20 s prefix check finds 2 successes among 11 selected H cases and 0 among 20 selected H* cases. Most misses are not resolved by adding only six seconds. This selection excludes cases that already succeeded or fell at 14 s and is not a population success estimate.

Prepared novel-system choices: the LTI has discrete A=diag(1.06, 1.02, 0.96, 0.90), B=(0.2, 0.2, 0.1, 0.1)^T and identity Q. Acrobot uses two unit uniform rods, normalized elbow torque limited to ±1, and the authored 100-per-angle/10-per-rate weight rule. All arms share costs and data within a plant. The [harness](../experiments/e3_novel.py) records units, dynamics, integration, ICs and plant-specific endpoints in each manifest. No novel-system performance claim is available yet.

2026-10-09: Prepared E4 compares H, L, M1 and full-state/output-only regularized DeePC on IC-A plus the same fixed 50 IC-C subset used in E2, with three seeds. To fit a two-sample observed past and N=30 future without joining episodes, collection uses 3281 × 32 steps (104992 transitions, compared with authored 105000). All E4 arms share this new collection/split; H keeps N=30 and the submitted training/control algorithm. IO DeePC receives only (p, theta), starts with two actual zero-input samples, and uses finite-difference velocities in its cost; this cost difference is disclosed. Three prespecified coefficient penalties are all reported, without selection on test ICs. SVD compression retains directions above 1e−8 times the largest singular value after row-standard-deviation scaling; coefficient-norm equivalence applies within that retained subspace. The [baseline code](../experiments/e4_deepc.py) records those choices and QP residual/failure diagnostics. Six [hand-computed DeePC checks](deepc_tests.txt) and ten [endpoint/arm/paired-statistic checks](reported_number_tests.txt) pass. Scientific E4 runs remain pending E0–E3.

E1 plotting now explicitly excludes the 60 s subsets from 14 s maps and includes descriptive cart-excursion maps and overlapping failure-category counts. Switch-toggle counts are reported without inventing a chatter cutoff. The completion watcher regenerated figures using this corrected plotting code before the E0+E1 checkpoint.

Optional E5 is now prepared inside E4 on the same base data: five independent training-episode bootstraps, mean of member trajectory costs, one-step replanning and no local switch. This is PETS-lite; it has neither learned aleatoric variance nor particle propagation. Three [ensemble checks](ensemble_tests.txt) pass. Two [recorded-data QP checks](deepc_data_wiring_tests.txt) pass without closed-loop baseline execution. `e4_deepc.py --skip-ensemble` omits E5 if required by the stated cut order.

E3 now derives applied-delay inputs and actual measurement-coordinate switch membership from saved arrays, so a pause between a case save and diagnostic augmentation is recoverable. Per-seed nominal state standard deviations and actual noise amplitudes will be recorded. No E2–E4 scientific result is implied by these preparation checks.

E1's completion postprocessing now updates the leading Q1.1 answer and saves exact tested from-rest success intervals without bridging failed or missing grid points. E3 also records final-three-second angular RMS and peak values in physical velocity units, alongside the unchanged primary endpoint, to distinguish noisy fluctuations from loss of the operating region.

<!-- E1_AUTO_START -->

### E1 measured results

Sources: [E1 rates](E1/rates.csv), [matched comparisons](E1/paired_comparisons.csv), [case metrics](E1/summary.csv), [distributions](E1/distributions.csv), [success figure](E1/success_rates.png), [manifest](E1/manifest.json).

| Seed | IC set | Arm | Successes/n (14 s, no fall) | Wilson 95% CI |
|---|---|---|---|---|
| 0 | B | H | 83/171 | 0.412–0.560 |
| 0 | C | H | 48/200 | 0.186–0.304 |
| 0 | D | H | 412/1681 | 0.225–0.266 |
| 0 | B | Hstar | 85/171 | 0.423–0.571 |
| 0 | C | Hstar | 42/200 | 0.159–0.272 |
| 0 | D | Hstar | 454/1681 | 0.249–0.292 |
| 0 | B | L | 137/171 | 0.735–0.854 |
| 0 | C | L | 122/200 | 0.541–0.675 |
| 0 | D | L | 1073/1681 | 0.615–0.661 |
| 0 | B | M1 | 0/171 | 0.000–0.022 |
| 0 | C | M1 | 0/200 | 0.000–0.019 |
| 0 | D | M1 | 0/1681 | 0.000–0.002 |
| 0 | B | M | 0/171 | 0.000–0.022 |
| 0 | C | M | 0/200 | 0.000–0.019 |
| 0 | D | M | 0/1681 | 0.000–0.002 |
| 1 | B | H | 72/171 | 0.350–0.496 |
| 1 | C | H | 31/200 | 0.111–0.212 |
| 1 | D | H | 356/1681 | 0.193–0.232 |
| 1 | B | Hstar | 95/171 | 0.481–0.628 |
| 1 | C | Hstar | 53/200 | 0.209–0.330 |
| 1 | D | Hstar | 548/1681 | 0.304–0.349 |
| 1 | B | L | 141/171 | 0.761–0.874 |
| 1 | C | L | 130/200 | 0.582–0.713 |
| 1 | D | L | 1089/1681 | 0.625–0.670 |
| 1 | B | M1 | 0/171 | 0.000–0.022 |
| 1 | C | M1 | 0/200 | 0.000–0.019 |
| 1 | D | M1 | 0/1681 | 0.000–0.002 |
| 1 | B | M | 0/171 | 0.000–0.022 |
| 1 | C | M | 0/200 | 0.000–0.019 |
| 1 | D | M | 0/1681 | 0.000–0.002 |
| 2 | B | H | 107/171 | 0.551–0.695 |
| 2 | C | H | 78/200 | 0.325–0.459 |
| 2 | D | H | 717/1681 | 0.403–0.450 |
| 2 | B | Hstar | 97/171 | 0.492–0.639 |
| 2 | C | Hstar | 51/200 | 0.200–0.320 |
| 2 | D | Hstar | 545/1681 | 0.302–0.347 |
| 2 | B | L | 141/171 | 0.761–0.874 |
| 2 | C | L | 128/200 | 0.571–0.703 |
| 2 | D | L | 1085/1681 | 0.622–0.668 |
| 2 | B | M1 | 0/171 | 0.000–0.022 |
| 2 | C | M1 | 0/200 | 0.000–0.019 |
| 2 | D | M1 | 0/1681 | 0.000–0.002 |
| 2 | B | M | 0/171 | 0.000–0.022 |
| 2 | C | M | 0/200 | 0.000–0.019 |
| 2 | D | M | 0/1681 | 0.000–0.002 |

Settling, excursion, input saturation and local fractions are distributions, not extra acceptance criteria. The 60 s runs remain separate in the CSVs.

<!-- E1_AUTO_END -->
