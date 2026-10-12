# Literature comparison: methods and current evidence

This comparison tests whether learned nonlinear models improve forced cart-pole
control relative to H (neural planning with fitted local feedback) and L (that
feedback alone). L is fitted from measured transitions. Simulator equations are
used for evaluation, not supplied to identification or controller decisions.

Current evidence, October 12, 2026, approximately 16:32 Central:

| Stage | Method and purpose | Evidence and remaining work |
|---|---|---|
| 1 | EDMD: nonlinear features, linear lifted prediction, bounded QP control | Three seeds fitted; saved correctness checks and pilot complete; native/common-L minimum matrix running |
| 2 | SINDYc: sparse measured state/input increment model, sampled or nonlinear MPC | Three seeds fitted; sparse recovery and compiled evaluation parity checks saved; four-arm pilot running |
| 3 | PETS: five probabilistic models, persistent particle/member assignments, sampled MPC | Checks and 100-epoch fits complete for all three seeds; controller timing/evaluation pending |
| 4 | Source-faithful kernelized DeePC | Source/solver verification and implementation pending |
| 5 | Autonomous Deep Koopman | Dedicated environment prepared; passive dataset, reproduction checks and forecasts pending |
| 6 | DFINE causal inference/prediction | Dedicated environment prepared; reproduction, noise/missing-data and causal forecast checks pending |
| 7 | Controlled Deep Koopman plus MPC | Input-aware adaptation and native/common-L comparisons pending |
| 8 | Controlled DFINE plus MPC | Control-aware filtering, native/common-L and filtered-L comparisons pending |

Saved standardized full-rollout test MSE is 0.16045 / 0.14478 / 0.15880 for
EDMD and 0.06862 / 0.07103 / 0.07185 for SINDYc across seeds 0–2. These forecast
scores do not establish controller superiority. Partial closed-loop case records
are not final cohort scores; report exact denominators at stage closure.

The final controller table uses the same 200 saved starting states, ordered
identically, for each fitting seed. It contains one prespecified corrected SMC
design (rho 0.9, beta 0.45 phi). SMC_linear remains an ablation. The fresh holdout
is separate. E4 uses different training data and requires its own matched H/L
references. Missing or cap-limited cells remain missing. Passive prediction and
causal-inference results belong in separate forecast tables.

No stage is yet certified complete as an end-to-end literature comparison.
Static review supports the inspected classical scaling, input and cost wiring;
independent endpoint replay, complete inventories and source-faithfulness closure
remain required. PETS epoch/configuration selection and artifact reuse identity
checks are tracked in [the focused review](reviews/coordinator_classical_review.md).

Execution continues in the existing supervised host session. Usage interruptions
are operational pauses, not scientific failures. The recorded ledgers are below
the 30 GPU-hour / 84 aggregate CPU core-hour ceilings, but interruption accounting
must be reconciled before final totals. The 85%/90% usage thresholds and credit
guard remain unchanged. See [the protocol](FOLLOWUP_COMPARISON_MATRIX.md),
[execution status](STATUS), and [coordination packet](reviews/COORDINATION.md).
