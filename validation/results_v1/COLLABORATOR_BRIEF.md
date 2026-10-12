# Collaborator briefing: what works, where, and what remains uncertain

October 12, 2026 · Original validation completed; literature comparisons in progress.

Headline: **The original demonstration reproduces, but simple feedback usually beats the complete hybrid.** These are finite-time simulation results, not a general stability guarantee.

**Controllers and success.** The trajectory manifold control method, denoted **H** as it is a hybrid approach, combines neural-network prediction, sampled input planning and a switch to linear feedback. The linear feedback controller, denoted **L** runs that feedback alone: fit `x_next = Ax + Bu` to the 6,000 training transitions nearest upright in standardized coordinates, compute a **linear-quadratic regulator** (feedback minimizing a quadratic state/input cost), and apply clipped `u = −Kx`. State `x` contains cart position/velocity and pendulum angle/angular velocity; `K` is the feedback *gain*. [The fit](../../scripts/run_inverted_pendulum_mc.py) uses measured transitions, not exact plant equations or derivatives. Nominal tests provide noiseless full state every 20 ms and allow ±98.1 N. Success requires angle ≤0.02 radians and angular speed ≤0.05 radians/second in magnitude throughout the final three seconds, with no earlier fall. Runs last 14 seconds.

**Reproduction and coverage.** The original demonstration succeeds at all six starting angles (±15°, ±30°, ±60°, from rest). Across ten additional seeds, H succeeds at **four or five of six**; L at **all six**. On the same 200 random states, H succeeds at **48, 31 and 78 of 200** across seeds 0, 1 and 2; L at **122, 130 and 128 of 200**, respectively. Seeds change collection, fitting and planning randomness; test states stay fixed.

**Matched examples:** the [H-versus-L notebook](../notebooks/collaborator_hybrid_linear_examples.ipynb) shows plots and playback for both succeeding, either alone succeeding, and both failing, using the same starting state and seed.

**Operating regions.** On the 1° from-rest grid, with cart position and both velocities zero, L succeeds from −68° to +68° in all three seeds; H and L both fail beyond ±70° through ±85°. H's contiguous all-seed success interval is −36°…+30°, with additional successes elsewhere. [The H/L figure](briefing_figures/initial_state_regimes.png) shows tested angle/angular-speed combinations and fractions of three seeds succeeding: descriptive frequencies, not calibrated probabilities or guarantees between points. No universal cart-speed or force failure threshold was established.

**Controller comparisons.** Entries give successes for seeds 0, 1, 2. **E4**, the matched-baseline study, recollected longer training episodes; **E6**, the feedback study, reused the original collection. Compare within each study. Dashes mean results pending; the supervised follow-up targets the original 200 states within its compute allowance, with training cohorts kept separate.

| Controller | Original collection: 200 states | E4: matched baselines | E6: feedback comparison |
|---|---:|---:|---:|
| H: hybrid | 48, 31, 78 / 200 | 21, 4, 33 / 50 | 11, 6, 16 / 50 |
| L: fitted linear feedback | 122, 130, 128 / 200 | 35, 36, 33 / 50 | 33, 36, 36 / 50 |
| Data-enabled Predictive Control (DeePC), each of six settings | — | 0, 0, 0 / 50 | — |
| Sliding mode control (SMC): feedback with a surface-attracting correction | — | — | 35, 36, 36 / 50 |
| Deep Koopman Control (DKC): Koopman-based feedback with a surface-attracting correction | — | — | - |
| Dynamical flexible inference of nonlinear latent factors and structures in neural population activity (DFINE) + control | — | — | - |
| Sparse identification of nonlinear dynamics (SINDy) + control | — | — | - |
| Proportional-derivative-integral (PID) | — | — | - |

<!-- The [literature campaign](../results_literature/STATUS) has fitted Koopman and SINDy (sparse identification of nonlinear dynamics) predictors; control results are pending. Prediction-only models have separate forecast metrics. Further controllers follow the [comparison plan](../LITERATURE_COMPARISON_PLAN.md). -->

[Baseline figure](briefing_figures/matched_baseline_regimes.png): individual starting states, including cart position/speed. SMC has no statistically resolved nominal advantage over L. Its correction-free ablation and delay results remain in the [full report](RESULTS.md).

**Why DeePC fails here.** **Data-enabled Predictive Control** (DeePC) optimizes linear combinations of nonlinear trajectories that need not obey plant dynamics. All 866,538 optimization calls passed acceptance checks, yet every tested DeePC run fell. In [six saved-plan replays](E4/deepc_matched_plan_replays.json), predictions ended within 0.00063 radians of upright; the same inputs on the plant ended at −1.58…−1.74 radians and crossed the fall boundary. L imposes a fitted transition equation and corrects from measured state every step. This diagnoses invalid predictions despite feasible optimization, not failure of nonlinear DeePC generally. The ensemble also simplified a published method.

** Recommended next steps.** Identify problems where the manifold trajectory approach offers clear benefits over existing methods, while simultaneously developing guarantees and robustness regarding input and output data, constraints, and noise.

<!-- **Limitations.** The implementation [note](../../../trajectory_manifold_implementation.tex) calls the fit local; [selected samples](E0/authored_model_diagnostics.json) extend to 101.7°, not a successful starting-angle bound. The [data-size study](E2/local_stability.csv) finds unstable gains in 1/5 seeds at 3,500 episodes and 3/5 at 10,000. The [E4 check](E4/local_stability.csv) finds seed 1 unstable on the true plant despite stabilizing its fitted model; a [14/60-second witness](E4/long_horizon_prefix_check.json) shows short-run success later failing. The [review](DOCUMENTATION_REVIEW.md) details planner defects; the [note audit](reviews/implementation_tex_v3/annotations.json) found no equation/code contradiction. A [guarantee handoff](STABILITY_GUARANTEE_HANDOFF.md) gives a conditional local result; no numerical region is certified.

Noise endpoint failures can include upright runs; computation delay is omitted; long-run samples are small; raw replay artifacts remain local. [Full findings](RESULTS.md), [review limitations](DOCUMENTATION_REVIEW.md), [proof audit](reviews/paper_review_v1/proof_audit.md), [completion checks](completed_campaign_checks.json), and local [supervisor](supervisor.log)/[execution](codex_exec.log) logs provide the evidence trail. -->
