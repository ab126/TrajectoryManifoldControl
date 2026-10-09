# Validation of the submitted trajectory-manifold controller

## Q1 — Does it work as claimed?

**The authored seed-234 run reproduces angle balance in all six cases.** The fresh unchanged `--profile dev` run is saved in [E0/authored_summary.json](E0/authored_summary.json); expanded endpoints and multi-seed ablations are running. Nine focused [metric and arm tests](E0/metric_arm_tests.txt) passed. The authored notebook already contains a development-profile demonstration: all six initial angles (±15°, ±30°, ±60° from rest, 14 s) have `balanced=True`; three also have `cart_rest=True`. Local feedback supplies 50–97% of steps, with 6–70 CEM solves per case. This is authored evidence, not yet an independent reproduction. Source: [notebook](../../manifold_control.ipynb), cells 10 and 12.

The primary endpoint is the runner's `balanced` metric (|θ|≤0.02 rad and physical |θ̇|≤0.05 rad/s in the last 3 s) **and no fall** (|θ|<π/2 throughout). Cart excursion, drift, settling, final position, input saturation and local feedback use are reported separately. There is no rail limit, recentering deadline or acceptance gate.

The fresh authored reproduction gives **6/6 primary successes** (Wilson 95% interval 0.610–1.000), with cart rest in 4/6 rather than the notebook's 3/6. Local feedback supplies 43.1–96.7% of steps and CEM solves range from 6 to 81. Maximum cart excursions span 0.413–7.858 m. These differences from the saved notebook should be retained as reproduction variability, not overwritten. The authored local-fit selection reaches |θ|=1.775 rad (101.7°), with standardized radius 0.809 and inputs spanning the full ±10 limit. Sources: [authored endpoints](E0/authored_cases.csv), [model diagnostics](E0/authored_model_diagnostics.json).

## Q1.1 — Failure region, data needs, novel systems

**Unresolved pending E1–E3.** The previous true-model wide-box diagnostic has 147/200 angle-balanced runs, 67/200 also reaching cart rest, and 103/200 crossing π/2. The first two counts omit the newly required no-fall condition; they must not be presented as primary successes. Initial speeds reach 7.8 m/s and angles reach 74°, unlike the authored demonstration. Source: [independent audit](../overnight_runs/20261007T054549.695689Z/calibration/independent_audit.json); recomputation from saved trajectories gives **65/200** angle-balanced with no fall (Wilson 95% interval 0.264–0.393). Cart-excursion median is 3.829 m, 95th percentile 14.196 m and maximum 20.009 m. This historical run used N=60 and 20 s, so it is not a matched comparison with the new N=30, 14 s arms. Sources: [recomputed summary](historical_oracle/endpoint_summary.json), [all cases](historical_oracle/summary.csv), [manifest](historical_oracle/manifest.json).

`run_inverted_pendulum_mc.log`, `inverted_pendulum_mc_debug.ipynb`, and `linear_system_mc.ipynb` contain historical latent-MPC outputs. They are not evidence about the active CEM controller.

## Q2 — Errors in the implementation note

The reviewed transcription is [trajectory_manifold_implementation.tex](../../../trajectory_manifold_implementation.tex), stored in the parent directory.

**Review in progress.** The prior [v2 annotations](../reviews/implementation_tex_v2/annotations.json) checked equation/code correspondence but leave notation, normalization, collection details and evaluation omissions. The superseding handoff identifies these; v3 will annotate them without changing the labmate's source text.

## Q3 — Comparison and contribution

**Empirical verdict unresolved.** H combines a full-state one-step transition network, CEM shooting MPC, five-step queues and a data-identified local LQR catch. Model, optimizer and switch contributions will be separated by matched initial conditions and ablations. Paper and literature deliverables will distinguish the full-state rollout graph theorem from the unexplored output-only/noisy extension.

## Protocol and resume

Operating instructions: [superseding handoff](../codex_handoff_from_claude_v1.md). Jobs longer than two minutes run detached; each experiment's `JOB` records its PID and command. New evaluations save per-case outputs and skip completed cases. The usage supervisor handles account limits; this campaign does not track them.

Time-to-angle-settle is the earliest sample after which both angular limits hold through the end, requiring at least 3 s of remaining observation. Time-to-full-settle also requires physical cart speed ≤0.05 m/s and position change ≤0.01 m/s over rolling 1 s windows. These are descriptive times, not additional primary pass/fail criteria. Prediction RMSE is reported in stored normalized state units; physical velocities multiply by √9.81.

The runner's seed jointly controls data collection, episode splitting, network initialization and CEM draws. Seeds 0–9 therefore test whole-run reproducibility, not training randomness alone. Arms share data, fitted gain and initial conditions within each seed; every case restarts the controller RNG as the authored runner does. H* copies the learned state scaling into the existing true-RK4 transition so its exploration scaling stays matched.

Wilson 95% intervals are reported separately per seed/condition. Fixed authored cases and grids describe tested conditions; their intervals do not imply random population coverage over an untested operating domain. Exact McNemar comparisons use matched ICs within seed. Settling times are right-censored when the observation is too short; omitted settling times are counted in distribution tables.

The bounded [paper ledger and proof audit](reviews/paper_review_v1/proof_audit.md) and [primary-source literature matrix](reviews/paper_review_v1/literature_matrix.csv) are available. The graph and rank results are correct under their assumptions. The process-noise extension needs qualification if noise is not retained in the behavior vector. The [v3 annotated implementation copy](reviews/implementation_tex_v3/trajectory_manifold_implementation_reviewed_v3.tex) contains 27 comments and preserves source prose/equations. No equation/code contradiction was identified; normalization, notation, evaluation and collection omissions remain real specification issues.

## Deviations

2026-10-09: E0 was restarted with eight independent GPU workers after measuring 9% serial GPU utilization. Saved cases/models were reused. Only progress-bar display was suppressed; each case still calls the submitted runner/class implementation. E1 runtime projection will be checked against the cap using measured parallel throughput.

E1 long-horizon subsets are fixed before evaluation: IC-B −60°, 0°, +60°; IC-C indices 0–2; IC-D indices 20, 840, 1660 (θ0=−85°, 0°, +85°, zero angular velocity). All five arms are evaluated for 60 s on these subsets.
