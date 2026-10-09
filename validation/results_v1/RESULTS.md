# Validation of the submitted trajectory-manifold controller

## Q1 — Does it work as claimed?

**The authored seed-234 run reproduces all six angle/no-fall successes, but the hybrid is sensitive to seed and its local-feedback ablation is stronger on the authored cases.** The fresh unchanged `--profile dev` run is saved in [E0/authored_summary.json](E0/authored_summary.json); expanded endpoints and ten-seed ablations are complete. Nine focused [metric and arm tests](E0/metric_arm_tests.txt) passed. The authored notebook already contains a development-profile demonstration: all six initial angles (±15°, ±30°, ±60° from rest, 14 s) have `balanced=True`; three also have `cart_rest=True`. Local feedback supplies 50–97% of steps, with 6–70 CEM solves per case. The notebook is the authored baseline; the fresh independent reproduction is reported below. Source: [notebook](../../manifold_control.ipynb), cells 10 and 12.

The primary endpoint is the runner's `balanced` metric (|θ|≤0.02 rad and physical |θ̇|≤0.05 rad/s in the last 3 s) **and no fall** (|θ|<π/2 throughout). Cart excursion, drift, settling, final position, input saturation and local feedback use are reported separately. There is no rail limit, recentering deadline or acceptance gate.

The fresh authored reproduction gives **6/6 primary successes** (Wilson 95% interval 0.610–1.000), with cart rest in 4/6 rather than the notebook's 3/6. Local feedback supplies 43.1–96.7% of steps and CEM solves range from 6 to 81. Maximum cart excursions span 0.413–7.858 m. These differences from the saved notebook should be retained as reproduction variability, not overwritten. The authored local-fit selection reaches |θ|=1.775 rad (101.7°), with standardized radius 0.809 and inputs spanning the full ±10 limit. Sources: [authored endpoints](E0/authored_cases.csv), [angle/cart/input figure](E0/authored_excursions.png), [model diagnostics](E0/authored_model_diagnostics.json).

On IC-A at 14 s, H succeeds in 4–5/6 cases per seed, while L succeeds in 6/6 for every seed. M and M1 each have 0/6 for every seed; H* has 4/6 for every seed. At 60 s, H improves to 4–6/6, L remains 6/6, and H* gives 4–5/6. This supports the local catch's importance; it does not yet determine whether H adds reach on harder ICs. The true-model failures show that learned-model error is not the only limitation. Sources: [per-seed rates](E0/rates.csv), [paired comparisons](E0/paired_comparisons.csv), [figures](E0/success_rates.png).

M and M1 can approach upright while missing the authored settling limits. In the illustrative seed-0 −15° case, M1's final-three-second maximum angle is 0.0388 rad and maximum physical angular speed is 1.047 rad/s; L gives 0.000275 rad and 0.000102 rad/s. The [terminal-window figure](E0/terminal_behavior.png) and [all 300 seeded 14 s terminal summaries](E0/terminal_behavior.csv) show the distinction between approaching upright and meeting the authored settling limits.

A [feasible-alternative diagnostic](E0/feasible_alternative_diagnostics.json) examines the last saved plans of three illustrative seed-0 misses. It found no lower internal-model cost among the local-law input sequences tried; that does not establish global optimality. Some older E0 arrays lack full planned inputs, so their chosen true-plan costs are left unavailable. E1 saves full plans for matched-input replays. Five [objective, missing-data, interval and matched-selection checks](failure_diagnostic_tests.txt) pass.

## Q1.1 — Failure region, data needs, novel systems

**E1 is running; data efficiency and robustness remain unresolved pending E2–E3.** The previous true-model wide-box diagnostic has 147/200 angle-balanced runs, 67/200 also reaching cart rest, and 103/200 crossing π/2. The first two counts omit the newly required no-fall condition; they must not be presented as primary successes. Initial speeds reach 7.8 m/s and angles reach 74°, unlike the authored demonstration. Source: [independent audit](../overnight_runs/20261007T054549.695689Z/calibration/independent_audit.json); recomputation from saved trajectories gives **65/200** angle-balanced with no fall (Wilson 95% interval 0.264–0.393). Cart-excursion median is 3.829 m, 95th percentile 14.196 m and maximum 20.009 m. This historical run used N=60 and 20 s, so it is not a matched comparison with the new N=30, 14 s arms. Sources: [recomputed summary](historical_oracle/endpoint_summary.json), [all cases](historical_oracle/summary.csv), [manifest](historical_oracle/manifest.json).

`run_inverted_pendulum_mc.log`, `inverted_pendulum_mc_debug.ipynb`, and `linear_system_mc.ipynb` contain historical latent-MPC outputs. They are not evidence about the active CEM controller.

## Q2 — Errors in the implementation note

The reviewed transcription is [trajectory_manifold_implementation.tex](../../../trajectory_manifold_implementation.tex), stored in the parent directory.

**Review complete: no equation/code contradiction found, but the note needs 27 qualifications and clarifications.** The [v3 annotated copy](reviews/implementation_tex_v3/trajectory_manifold_implementation_reviewed_v3.tex) preserves the labmate's text and flags notation overloads, implicit normalized costs, collection details, switch geometry, numerical fallback and evaluation omissions. The [comment ledger](reviews/implementation_tex_v3/annotations.json) records each issue. The selected local-fit neighborhood is substantially broader than the handoff estimate; see the measured E0 diagnostics above.

## Q3 — Comparison and contribution

**H is a full-state learned-dynamics shooting controller; its empirical advantage over the comparison methods remains unresolved.** It combines one-step transition regression, CEM planning, five-step queues and a ridge/DARE local catch. E0 supports the catch's importance on the authored cases. The graph theorem supplies explicit trajectory coordinates but does not constrain prediction error or guarantee closed-loop success; see the [proof audit](reviews/paper_review_v1/proof_audit.md).

| Comparison | What differs from H | What this campaign can establish |
|---|---|---|
| DeePC | Input/output trajectory spanning and a convex QP; exact LTI representation requires excitation and observability assumptions. | Matched nonlinear cart-pole performance and regularization sensitivity; the LTI theorem does not guarantee this nonlinear baseline. |
| Lusch Deep Koopman | Autonomous state embedding trained with multi-step and linearity losses; no implemented controller. | Prose/model comparison, not an invented Lusch control baseline. |
| DFINE | Unforced noisy partial observations, latent factors and linear dynamics with Kalman inference. | Observation and uncertainty differences; no DFINE controller is constructed. |
| Baek deep RL | Model-free SAC, symmetry augmentation, triple pendulum and hardware demonstrations. | Qualitative budget/task comparison; its 0.8–2 million steps cannot establish a matched sample-efficiency advantage for H. |

The [verified primary-source matrix](reviews/paper_review_v1/literature_matrix.csv) supports these distinctions. Nagabandi neural shooting MPC and single-model variants of PETS are the closest algorithmic antecedents. H's training and online controller do not use plant equations. Its limitations include full-state measurement, deterministic one-step learning, no uncertainty estimate, open-loop queues, a hard switch and no stability guarantee. Benefits beyond the measured E0 local catch remain pending E1–E4.

The fresh authored run records 216 CEM solves: median 30.2 ms, 95th percentile 31.0 ms, maximum 130.2 ms. All exceed the 20 ms physical sample interval; one exceeds the nominal five-step queue interval of 100 ms. The plant simulator advances a fixed step after the controller returns, so this demonstration does not include computation delay. The measured timing matters for a hardware implementation and for comparing online shooting with direct policy evaluation. Sources: [timing summary](E0/authored_solve_timing.json), [original timings](E0/authored_summary.json), [submitted simulator](../../scripts/run_inverted_pendulum_mc.py).

A plausible research gap is nonlinear input/output data-enabled prediction with explicit identifiability, noise and approximation conditions. Part I flags the output injectivity issue and defers learning/control; the current full-state implementation does not exercise that extension. The [15-row claim ledger](reviews/paper_review_v1/claim_ledger.csv) separates correct graph/rank results from these unsupported extensions.

## Protocol and resume

Operating instructions: [superseding handoff](../codex_handoff_from_claude_v1.md). Jobs longer than two minutes run detached; each experiment's `JOB` records its PID and command. New evaluations save per-case outputs and skip completed cases. The usage supervisor handles account limits; this campaign does not track them.

Time-to-angle-settle is the earliest sample after which both angular limits hold through the end, requiring at least 3 s of remaining observation. Time-to-full-settle also requires physical cart speed ≤0.05 m/s and position change ≤0.01 m/s over rolling 1 s windows. These are descriptive times, not additional primary pass/fail criteria. Prediction RMSE is reported in stored normalized state units; physical velocities multiply by √9.81.

The runner's seed jointly controls data collection, episode splitting, network initialization and CEM draws. Seeds 0–9 therefore test whole-run reproducibility, not training randomness alone. Arms share data, fitted gain and initial conditions within each seed; every case restarts the controller RNG as the authored runner does. H* copies the learned state scaling into the existing true-RK4 transition so its exploration scaling stays matched.

Wilson 95% intervals are reported separately per seed/condition. Fixed authored cases and grids describe tested conditions; their intervals do not imply random population coverage over an untested operating domain. Exact McNemar comparisons use matched ICs within seed. Settling times are right-censored when the observation is too short; omitted settling times are counted in distribution tables.

The bounded [paper ledger and proof audit](reviews/paper_review_v1/proof_audit.md) and [primary-source literature matrix](reviews/paper_review_v1/literature_matrix.csv) are available. The graph and rank results are correct under their assumptions. The process-noise extension needs qualification if noise is not retained in the behavior vector. The [v3 annotated implementation copy](reviews/implementation_tex_v3/trajectory_manifold_implementation_reviewed_v3.tex) contains 27 comments and preserves source prose/equations. No equation/code contradiction was identified; normalization, notation, evaluation and collection omissions remain real specification issues.

## Deviations

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

2026-10-09: Prepared E4 compares H, L, M1 and full-state/output-only regularized DeePC on IC-A plus the same fixed 50 IC-C subset used in E2, with three seeds. To fit a two-sample observed past and N=30 future without joining episodes, collection uses 3281 × 32 steps (104992 transitions, compared with authored 105000). All E4 arms share this new collection/split; H keeps N=30 and the submitted training/control algorithm. IO DeePC receives only (p, theta), starts with two actual zero-input samples, and uses finite-difference velocities in its cost; this cost difference is disclosed. Three prespecified coefficient penalties are all reported, without selection on test ICs. The [baseline code](../experiments/e4_deepc.py) records those choices and QP residual/failure diagnostics. Six [hand-computed DeePC checks](deepc_tests.txt) and ten [endpoint/arm/paired-statistic checks](reported_number_tests.txt) pass. Scientific E4 runs remain pending E0–E3.

E1 plotting now explicitly excludes the 60 s subsets from 14 s maps and includes descriptive cart-excursion maps and overlapping failure-category counts. Switch-toggle counts are reported without inventing a chatter cutoff. The completion watcher will regenerate figures using this corrected plotting code before the E0+E1 checkpoint.

Optional E5 is now prepared inside E4 on the same base data: five independent training-episode bootstraps, mean of member trajectory costs, one-step replanning and no local switch. This is PETS-lite; it has neither learned aleatoric variance nor particle propagation. Three [ensemble checks](ensemble_tests.txt) pass. Two [recorded-data QP checks](deepc_data_wiring_tests.txt) pass without closed-loop baseline execution. `e4_deepc.py --skip-ensemble` omits E5 if required by the stated cut order.

E3 now derives applied-delay inputs and actual measurement-coordinate switch membership from saved arrays, so a pause between a case save and diagnostic augmentation is recoverable. Per-seed nominal state standard deviations and actual noise amplitudes will be recorded. No E2–E4 scientific result is implied by these preparation checks.

E1's completion postprocessing now updates the leading Q1.1 answer and saves exact tested from-rest success intervals without bridging failed or missing grid points. E3 also records final-three-second angular RMS and peak values in physical velocity units, alongside the unchanged primary endpoint, to distinguish noisy fluctuations from loss of the operating region.
