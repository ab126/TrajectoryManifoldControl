# Collaborator briefing: what works, where, and what remains uncertain

October 12, 2026 · Experiments completed; figures use saved results only.

**The original demonstration reproduces, but simple feedback usually beats the complete hybrid.** These are finite-time simulation results, not a general stability guarantee.

**Controllers and success.** The collaborator's hybrid **H** combines neural-network prediction, sampled input planning and a switch to linear feedback. **L** runs that feedback alone: fit a linear state-transition model to selected training samples, compute a **linear-quadratic regulator** (feedback minimizing a quadratic state/input cost for that fitted model), and apply `u = −Kx`, clipped to the input limit. Here `x` contains cart position/velocity and pendulum angle/angular velocity; `K` is the fitted feedback coefficient vector, or *gain*. Success requires angle ≤0.02 radians in magnitude and angular speed ≤0.05 radians/second throughout the final three seconds, with no earlier fall. Runs last 14 seconds unless stated otherwise.

**Counts, not indices.** The authored seed succeeds at all six starting angles (±15°, ±30°, ±60°, from rest). Across ten additional seeds, H succeeds at **four or five of those six**; L at **all six**. On the same 200 random starting states, H succeeds at **48, 31 and 78 of 200** across seeds 0, 1 and 2; L succeeds at **122, 130 and 128 of 200**, respectively. Seeds change collection, fitting and planning randomness—not the test-state indices.

**Watch matched examples:** the [H-versus-L notebook](../notebooks/collaborator_hybrid_linear_examples.ipynb) includes plots and side-by-side playback for both succeeding, L alone succeeding, H alone succeeding, and both failing. Examples share the same starting state and seed; they illustrate outcomes rather than general boundaries.

**Where success changes.** On the 1° from-rest grid, with cart position and both velocities initially zero, L succeeds at every tested angle from −68° to +68° in all three seeds; both H and L fail at every tested angle beyond ±70° (up to ±85°). H has gaps: its continuous success interval shared by all three seeds is only −36°…+30°, with additional successes elsewhere. These statements concern tested grid points only. [The initial-state figure](briefing_figures/initial_state_regimes.png) shows all tested angle/angular-speed combinations and the fraction of three seeds succeeding. It is descriptive frequency, not a calibrated probability or a guarantee between grid points. No universal cart-speed or force threshold was established; force was capped at ±98.1 N throughout.

**How the baselines compare.** Counts below are successes out of 50 starting states, in seed order 0, 1, 2. The matched-baseline study **E4** recollected slightly longer training episodes; the feedback study **E6** reused the original collection. Compare within each study; they are not a paired cross-study ranking.

| Controller | E4: matched baselines | E6: feedback comparison |
|---|---:|---:|
| H: hybrid | 21, 4, 33 / 50 | 11, 6, 16 / 50 |
| L: fitted linear feedback | 35, 36, 33 / 50 | 33, 36, 36 / 50 |
| Neural planner, replanning every step | 0, 0, 0 / 50 | — |
| Five-network mean-cost ensemble | 0, 0, 0 / 50 | — |
| Data-enabled Predictive Control (DeePC), each of six settings | 0, 0, 0 / 50 | — |
| Sliding mode control (SMC): feedback with a surface-attracting correction | — | 35, 36, 36 / 50 |
| Same SMC design with correction removed | — | 35, 37, 37 / 50 |

Koopman and SINDy were not run; DFINE was a literature comparison only. Their proposed implementations are in the [ranked comparison plan](../LITERATURE_COMPARISON_PLAN.md).

[Baseline figure](briefing_figures/matched_baseline_regimes.png): individual starting states, including cart position/speed. DeePC used linear trajectory combinations on nonlinear data; the ensemble simplified a published method. Their failures do not rank entire algorithm families. SMC has no statistically resolved nominal advantage over L; with a 20 ms delay, removing its correction gives 33–34/50 successes versus 0/50 for H, L and SMC.

**Implementation concerns and validation limits.** “Near the origin” comes from the collaborator's [note](../../../trajectory_manifold_implementation.tex); our [measurement](E0/authored_model_diagnostics.json) finds the selected fitting samples extend to 101.7°. These are fitting samples, not successful starting angles. The [data-size study](E2/local_stability.csv) finds unstable gains in 1/5 seeds at 3,500 episodes and 3/5 at 10,000. The [E4 gain check](E4/local_stability.csv) identifies seed 1 as unstable near upright on the true plant despite stabilizing its fitted model; [14/60-second evidence](E4/long_horizon_prefix_check.json) confirms a short-run success later fails. Planner defects are detailed in the [review](DOCUMENTATION_REVIEW.md). [The 27 annotations](reviews/implementation_tex_v3/annotations.json) found no equation/code contradiction. [A separate handoff](STABILITY_GUARANTEE_HANDOFF.md) develops a conditional local guarantee for specific stable gains; no numerical region has been certified.

Noise endpoint failures can include upright runs; computation delay is omitted; long-run samples are small; raw replay artifacts remain local. [Full findings](RESULTS.md), [review limitations](DOCUMENTATION_REVIEW.md), [proof audit](reviews/paper_review_v1/proof_audit.md), [completion checks](completed_campaign_checks.json), and local [supervisor](supervisor.log)/[execution](codex_exec.log) logs provide the evidence trail.
