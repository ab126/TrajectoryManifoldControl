# Collaborator memo — validation in progress

October 8, 2026. **The validation is incomplete. The original true-model
reference failed; dependent experiments remain stopped.** The submitted
controller and its original uniform-rod plant have been preserved.

The largest concerns, in order:

1. **We have not yet established a suitable safe-recovery reference task.**
   Gate A achieved only **3/200 strict successes**. This used known plant
   dynamics; full-budget evaluations of the submitted neural hybrid (H) and
   ensemble baseline (E) have not been run. Whether the full initial-state
   box is physically recoverable under the imposed safety limits remains open.
2. **The scored task and original policy are poorly aligned.** Rail safety
   (±2.4 m) and cart recentering were added by the reviewer. The active planner
   and local feedback enforce input bounds but no hard state constraints.
   Ignoring earlier safety violations, **147/200** settled angle/angular speed,
   **67/200** also settled cart speed, and **12/200** met all final tolerances.
   Angular settling and strict safe recovery must be reported separately.
3. **Proposed fixes have not been validated together.** A fixed stronger
   position-weighted local gain passed seven selected probes where the original
   gain passed none. Five-start search lowered cost in **22/40** selected pairs,
   but combined predicted safety stayed **18/40**. These are local probes and
   1.2-s open-loop predictions—not broad-box success, global optimality or
   safety certificates. The proposed OA-1 gain/cost/switch/safety/queue package
   is still unadopted and unimplemented.
4. **Compute and independent review are unresolved.** Stage 0 used
   **1.385377/15 worker-hours**; the unchanged CUDA reservation exceeds the
   remaining **13.614623 h**. A 20-h total cap is proposed, conditional on a new
   amended reservation; it is not approved. Existing equivalence/reviews used
   separate implementations by the same author, not independent-author certification.

**Established on the checked evidence:** CPU/CUDA equivalence records and
29 existing tests passed, with a disclosed workaround omitting two unused
CVXPY imports. Independent saved-array rescoring, objective arithmetic,
transition residuals and hashes corroborate the reported Gate A and diagnostic
results. Submitted source hashes still match the originally evaluated commit.
These establish tested numerical consistency, not general stability or safety.

**Still open:** nonlinear safe-entry behavior and coupled recoverability;
OA-1 implementation/audit/timing/calibration; real full-budget H/L/E comparisons
(L is the data-identified LQR baseline);
learned-model/local-fit quality; ensemble uncertainty calibration; statistical
power/type-I calibration; and the exact-source proof/novelty review. There is
no established superiority, data-efficiency or novelty verdict. The proposed
timing-agnostic primary endpoint remains pending. BOTH one-sided 95% H−L and
H−E success-probability difference bounds must exceed .05 for the primary test;
retain Holm for the secondary family.

**Next practical step:** build strict configuration/provenance support, then
cost/ranking checks, the explicit oracle state machine, an independently
reviewed auditor and a capped launcher. Validate the new entry guards and
allocation before another Gate A. An independent collaborator review of the
specification and auditor is particularly useful now. Do not rerun either
completed diagnosis or silently change H, plant, endpoint, box, device or precision.

The [implementation plan](implementation_plan_v1.md) gives the sequence and
acceptance gates; the [evidence assessment](oracle_review_assessment_v1.md) and
[OA-1 proposal](oracle_amendment_proposal_v1.md) contain details. HEAD is
`4c9de6d`, with uncommitted validation documentation/review work. Raw arrays,
checkpoints, media and environments require the separate lab archive; missing
raw files mean unavailable evidence, not failed tests.
