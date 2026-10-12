Sliding mode control and the trajectory-manifold validation

This document records pre-run research. The completed empirical outcomes are in [RESULTS.md](RESULTS.md#same-data-sliding-feedback-and-zero-correction-comparison).

Research date: 2026-10-11. Three agents independently examined primary literature, the active implementation and a feasible baseline. The user subsequently approved automatic execution after E2–E4; the durable request is E6_SMC/REQUEST.json. No SMC experiment was launched during interactive setup. Development took place in /tmp because /scratch/ap60 returned an I/O error; this copy accompanies the queued reproducible code.

The recommendation is to add a small optional comparison using discrete quasi-sliding-mode feedback derived from the same fitted linear model used by the existing LQR arm. It addresses whether a different feedback design improves robustness or capture under equal data access. It should follow the core E2–E4 work and remain within the existing cap. It is not necessary to establish that the submitted controller is, or is not, SMC: the source already resolves that question.

**What SMC means and what carries over**

Classical SMC selects a state/error constraint s(x)=0 and a control that attracts trajectories to it and maintains sliding. The dynamics along the surface must also be stable. Ideal rejection applies to a specified class of disturbances entering through the control channel; sufficient gain and actuation are required. Smoothing the discontinuity trades exact invariance for practical accuracy. These conditions distinguish designed sliding from generic switching. Source: [Spurgeon, Sliding mode control: a tutorial, ECC 2014](https://kar.kent.ac.uk/41730/1/sliding_mode.pdf), especially sections II–IV.

Higher-order SMC offers alternatives to a discontinuous applied input, but still assumes suitable relative degree, nonvanishing input effectiveness and bounded uncertainties. Its output regulation does not itself stabilize the remaining internal dynamics. Source: [Levant, Higher-order sliding modes, differentiation and output-feedback control, 2003](https://www.tau.ac.il/~levant/ijc_hosm_2003.pdf), section 3, equations (4)–(5).

Digital sampling, sensor noise and actuator saturation matter. In super-twisting control, the dynamic integral state can wind up during saturation; modern conditioned implicit discretizations address that issue for their stated plant class. Source: [Seeber and Andritsch, Proper Implicit Discretization of the Super-Twisting Controller—without and with Actuator Saturation, Automatica 2025](https://arxiv.org/html/2406.16094v2). A cart-pole application must still supply a valid sliding variable and stable internal dynamics.

**Relation to the collaborator's actual implementation**

| Feature | Active collaborator implementation | Sliding-mode design |
|---|---|---|
| Geometric object | Graph of finite-horizon state/input rollouts | Designed constraint in state/error space |
| Purpose | Parameterize model-consistent candidate trajectories | Specify desired reduced closed-loop behavior |
| Feedback law | CEM shooting outside a threshold; clipped identified LQR inside | A reaching and maintenance law for the sliding variable |
| Switching geometry | xᵀQx<0.75 selects the feedback branch | s(x)=0 defines the desired sliding motion |
| Evidence required for stability | Actual learned, sampled, saturated hybrid must be assessed | Reaching, stable surface dynamics and uncertainty/actuation assumptions must be established |

Source locations, relative to /home/ap60/github/data-driven-control/TrajectoryManifoldControl:

- scripts/run_inverted_pendulum_mc.py:180: ridge identification of discrete A,B and DARE gain K.
- scripts/run_inverted_pendulum_mc.py:199: quadratic planning cost and CEM solver.
- scripts/run_inverted_pendulum_mc.py:217: threshold selects clipped −Kx; otherwise the controller executes queued planned inputs or replans.
- src/manifold_control.py:78: canonical rollout decoder.
- src/manifold_control.py:270: CEM trajectory-cost optimization.

There is no designed s(x), reaching inequality or equivalent-control construction in this active path. The ellipsoid boundary is used to choose the controller, not as a target on which motion should remain. Incidental sliding-like behavior in an idealized discontinuous hybrid would not establish that SMC was designed or certified here.

The rollout graph has 34 free coordinates in a 154-dimensional ambient trajectory space at the development horizon; fixing the current state leaves 30 input coordinates. A regular scalar sliding constraint for the four-state plant instead normally describes a three-dimensional state-space surface. These dimensions concern different objects.

An actual connection exists in robust tube MPC: [Lopez, Slotine and How, Dynamic Tube MPC for Nonlinear Systems, 2019](https://arxiv.org/abs/1907.06553) uses boundary-layer sliding feedback to maintain a tube around planned trajectories under stated uncertainty assumptions. Such feedback acts during trajectory execution. The collaborator's origin-centered LQR catch and five-step input queue do not implement that construction. This is a literature antecedent and possible future improvement, not a baseline needed for the present validation.

**Effect on our feedback**

The main conclusions remain: the graph result supplies coordinates; it does not establish prediction accuracy, uncertainty rejection or nonlinear hybrid stability. The stronger observed performance of identified linear feedback alone remains evidence about component attribution. SMC performance has not been measured here and cannot retrospectively invalidate or explain those results.

Wording should be precise. DARE provides a meaningful guarantee for the fitted linear model under its assumptions. The unestablished guarantee concerns the actual nonlinear, sampled, saturated hybrid. Switch toggles are evidence of mode changes; they do not alone establish damaging chattering or causal failure. Avoid implying that all SMC requires known physics, that all SMC is bang-bang, or that every uncertainty is matched.

For this particular plant, angle-only regulation leaves an important ambiguity. With θ=θ̇=0, the second mass-matrix equation in src/inverted_pendulum.py:66–89 forces p̈=0; the first then gives u=0. Cart velocity can therefore persist. The surface s=θ̇+λθ may regulate angle while cart position drifts. This is a deduction from the submitted plant equations, not a new experiment. Keep both the primary no-fall angle endpoint and the descriptive cart/settling measures.

**Baseline to recommend**

Use the nominal development data and saved models for seeds 0,1,2. Reuse the same ridge-fitted A,B, state coordinates and Q,R as the L arm. Reconstruct its DARE solution P, then define the full-state sliding row

    C = (BᵀPB)⁻¹ BᵀP,  s = Cx,  CB = 1.

Riccati-based surface construction is an established design family: [Wang and Zhang, Hyperplane design for discrete-time sliding mode control with event-trigger strategy and disturbance observer, 2020](https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/iet-cta.2019.0509), section 3.2, Theorem 1. The particular reuse of the submitted DARE P and the following reaching parameters are a proposed campaign adaptation, not a reproduction of that paper's complete observer/event-trigger controller.

The nominal equivalent feedback is u_eq=−CAx, with A_eq=(I−BC)A. There is a direct algebraic reason this is a sensible surface for the fitted model. Equivalent feedback minimizes the next-state P norm over inputs; its value is no larger than that produced by the DARE LQR input. Consequently,

    A_eqᵀ P A_eq ≤ (A−BK)ᵀP(A−BK)
                    = P−Q−KᵀRK < P.

For P>0 and Q>0 this gives Schur stability of A_eq. This argument concerns the exact fitted, unsaturated discrete model. It is not a nonlinear plant certificate. Check P symmetry/positivity, CB≈1, conditioning and the actual fitted equivalent eigenvalues before closed-loop evaluation.

An initial smooth discrete reaching law is

    s_next_target = ρ s − β sat(s/φ)
    u_raw = −CAx + ρ s − β sat(s/φ)
    u = clip(u_raw, −10, 10).

Here sat(v)=clip(v,−1,1), ρ=0.9 and β=0.45φ. On the exact fitted unsaturated model, the boundary-layer coefficient is 0.45; outside it the surface magnitude shrinks without a forced sign flip. This is practical quasi-SMC, not ideal continuous-time sliding. Sampling stays at the submitted physical 20 ms; no hidden faster inner control loop is allowed.

Set a central boundary width from validation data only: φ=max(10⁻³,4δ95), where δ95 is the 95th percentile of |C(x_next−Ax−Bu)| for validation transitions inside the training fit's standardized-state radius. Reuse the training scale and radius; report the number of selected validation transitions. δ95 is an empirical residual scale, not a certified uncertainty bound. If this selection is inadequate, disclose that before evaluation rather than choosing parameters from test outcomes. Any width sensitivity choices must be frozen before evaluating the fixed test ICs and all reported.

Include β=0 at the same ρ as a cheap linear-reaching comparison. Its underlying linear feedback differs from LQR. Therefore SMC-versus-LQR alone cannot attribute a benefit specifically to the nonlinear correction. Compare against both the zero-correction control and L; retain H for the method-level comparison.

**Proposed queue scope**

1. Finish the core E2–E4 scientific work. Introduce this as optional additional feedback comparison, with a dated deviation in the living results document if adopted; no new versioned implementation plan is needed.
2. Implement an isolated controller adapter and meaningful numerical checks under the usage supervisor. Develop temporary files under accessible scratch. Preserve the submitted controller and existing case records.
3. Start with the same six authored ICs and fixed 50 random-box ICs, three nominal development seeds, and existing 60-second authored subsets. A 171-point from-rest map is an optional extension; a second large 2-D campaign is unnecessary.
4. Reuse E3's existing noise, mismatch and delay protocols and matched disturbance arrays when affordable. A new cart-force-disturbance test is optional and must be evaluated for all comparison arms. Mass/length mismatch, arbitrary state noise and delay must not be labeled collectively as matched disturbances.
5. Retain success/no-fall, Wilson intervals and paired comparisons, settling times, final position, excursion and saturation. Add input total variation, solve time and surface residual distributions to expose the accuracy/effort tradeoff.
6. Reuse data and fitted models; no neural retraining or GPU planning is required for the new feedback arms. Budget at most one additional CPU-hour within the existing overall cap, subject to a measured supervised pilot and the existing three-times-estimate stop rule. The one-hour allowance is a proposed bound, not a measured runtime prediction. If the core work exhausts the cap, defer this optional arm.

The result would support an empirical statement about a particular same-data quasi-SMC design. It would not establish superiority over all SMC, reproduce a nonlinear cart-pole theorem, or certify robustness outside measured conditions.

**Published nonlinear SMC comparator considered**

[Aguilar-Ibáñez, Mendoza-Mendoza and Dávila, Stabilization of the cart pole system: by sliding mode control, 2014](https://link.springer.com/article/10.1007/s11071-014-1624-6) combines PD and twisting-like control and targets all states. It assumes known parameters/full state and an initial angle in the upper-half plane. Its tan/sec transformation is singular near horizontal. Its normalized mass matrix differs from the submitted uniform-rod matrix, and actuator clipping/digital implementation cannot inherit the ideal proof automatically. This is a model-informed contextual reference; adapting and verifying it would add substantial scope. Defer it for this campaign.

Agent work completed as read-only research. The user then approved the post-E4 addition. The active core jobs are unchanged; the future E4 entry point and resumed continuation now schedule E6_SMC automatically. No SMC performance claims are available.
