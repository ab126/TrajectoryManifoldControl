# Evidence assessment for oracle amendment OA-1

October 8, 2026. Review/specification only: no new calibration, training, tuning,
benchmark, sweep, transfer or subagent. Both completed diagnoses were preserved.
The next concrete deliverable is [OA-1](oracle_amendment_proposal_v1.md), with
[proposed configuration](oa1_config_proposed.json) and explicit adoption,
implementation, independent audit and allocation prerequisites.

## Repository and evidence authority

Actual HEAD is `4c9de6db91d1c2306c0cb1318c8b309f4da390df` (`updated validation notebook`),
one commit after latest reviewed `17d42cd83dea810af5b2979e657f0074244f2cc4`.
The working tree was clean at review entry. All 39 changed paths since that
review are under `validation/`; the [new manifest](reviews/oracle_amendment_v1/evidence_final.json)
lists them. They include the CUDA run/config/results/snapshots, CEM review code
and findings, local/endpoint reports, usage ledger, handoff/docs and angle media/
output-free notebook work. There is no new SUT change. The handoff and its
manifest still name `17d42cd` and its then-dirty status; those are historical
provenance, not the actual HEAD/status. Historical plan statements that no
scientific campaign ran also predate the failed Gate A.

Source SHA256 values match BOTH the readiness manifest and the exact submitted
`325beae718ab093d3ca48b18a59ae61fd93798a3` Git blobs:

| SUT file | SHA256 |
|---|---|
| `scripts/run_inverted_pendulum_mc.py` | `b96f4871c282db5118cc7c08dd5f2d3a7d6a78770dbb4fc94d3efd4a05f1d2e4` |
| `src/manifold_control.py` | `df3d9680fce60538933fece012082a34a989b3febdc389b52de3ccb2cf5df0e4` |
| `src/inverted_pendulum.py` | `c2ce099a7ff020f3e0c75263035f14563e6364dfdeb018339c301be1da2a58d4` |

No applicable `AGENTS.md` was found in the repository or ancestor directories.
Read [README](README.md), [handoff](validation_handoff.md), [CEM findings](diagnostics/cem_findings.md),
[cem_review.json](diagnostics/cem_review.json), the earlier proposal, and all
three full [plan snapshots](handoff/plans/). The current plan differs from the
original principally by its GPU/allocation amendment; the IUT remains 95%, not
97.5%. `inverted_pendulum/` candidates and `external_repos/` were not used as
audited controllers/plants. The original supplied mathematics PDFs remain an
unresolved source requirement for a page-by-page proof/novelty verdict.

Raw Gate A arrays, fourteen local traces and all 200 CEM arrays are present
in this workspace. [review_oracle_evidence_v1.py](review_oracle_evidence_v1.py)
imports no SUT/controller/audit, recomputes endpoint and objective arithmetic,
checks hashes/provenance, and evaluates independent pointwise RK4 residuals
at saved states. It generates no new trajectory or optimizer result. Fixed
Riccati algebra verifies the predeclared gains; it is not gain tuning.
This remains a same-author review, not external-author certification.

The [first arithmetic pass](reviews/oracle_amendment_v1/evidence.json) and
[final provenance pass](reviews/oracle_amendment_v1/evidence_final.json)
are new outputs outside historical runs. The latter adds independent case-seed
regeneration, selected-state provenance, runtime record and selected pure-position
plan checks. The intermediate `evidence_verified.json` is retained too. All 212 original run checksums and 222 inputs in the CEM review
hash manifest matched; each diagnostic job's three recorded execution-source
hashes matched its snapshots. No referenced input in these checks was missing.
This is not an inventory/certification of every media/checkpoint/environment in
the lab. A clone will lack ignored raw artifacts: mark them unavailable, never
infer test failure or regenerate them. The notebook source has no cell outputs
or execution counts; this review did not render it or regenerate media.

## Independently checked results and their limits

| Evidence | Result checked here | Valid interpretation |
|---|---|---|
| [Plant checks](artifacts/plant_checks.json) and [source derivation](plant_review.md) | Recorded RHS error 1.42e−14, energy drift 1.675e−5, RK4 refinement errors 4.402e−6/1.972e−7/9.800e−9; mass determinant ≥.75 follows algebraically | Correct uniform-rod equations and numerical checks on tested states, not a global accuracy/safety bound |
| [CPU](artifacts/equivalence.json)/[CUDA](artifacts/equivalence_cuda.json) equivalence | Recorded 50×100 per device, 1020 solves per implementation, zero maximum discrepancies and identical flags/queues | Tested ordinary-shooting correspondence with synthetic shared networks; separate implementation by same author |
| [Existing tests](artifacts/existing_tests.json) and local raw log | Exit zero; log says 29 passed, one warning | Existing suite passed once; no rerun; top-level plotting experiment excluded |
| Original saved Gate A | All 200 finite; 3 strict successes, IDs 22/72/89; halves 0/94 and 3/106 | Failed true-model reference; NOT fully trained H or E performance |
| Nested settling, ignoring earlier safety | Angle/angular speed 147; also cart speed 67; all final tolerances 12; strict safe recovery 3 | Endpoint addition matters: nine fully settled cases had earlier safety violations |
| [Local probes](diagnostics/20261007T175948.296063Z/results.json) | Qp=.02:0/7; fixed Qp=2:7/7; all fourteen rail/angle-safe; saved controls match each gain | Selected local-only evidence supports gain amendment, not a safe region or new hybrid calibration |
| [CUDA diagnosis](diagnostics/20261008T190703.853434Z/results.json) | 200 finite saved solves; 11/20 cost improvements per warm group, 9/20 safe per group for one and best-five; no pair changed combined safety | Finite search lowered original cost without improving selected 1.2-s predicted combined safety |

The saved-array objective discrepancy in this independently ordered sum was
5.821e−11; the earlier review reports 1.019e−10. Both are roundoff-scale and
pass the declared tolerance; the original report is preserved. Independent
pointwise RK4 maximum residual was 1.776e−15 on both the Gate A saved steps and
CEM saved predictions. This checks logged transition arithmetic, not forward
error versus exact continuous-time physics. The original full plant replay's
zero-error report is historical and was not rerun.

At p=±1 with zero velocities/angle, the selected one-start/best-five **zero-warm**
plans contain zero input and fixed position: J=60*.02+12*.02=1.44. The selected
original local-warm plan costs 2.1055064445. Perturbed later attempts in the
zero-warm group can have nonzero inputs; the no-motion statement applies to
selected plans. This is a direct comparison of two tested plans, not proof that
zero is a global optimum. Warm-retention inequalities are verified against
reported warm costs; full warm trajectories for every perturbed attempt were
not saved, so those cost reports are not independently certified here. Historical
Gate A full plans are also absent; no exact historical CEM replay is possible.

## Control, mathematics, ML and statistical conclusions

**Objective mismatch:** the added rail and centering task does not follow from
the rod physics. [Active H](../scripts/run_inverted_pendulum_mc.py) uses
Q=diag(.02,30,100,10), R=.03, Qf=12Q and cost-only CEM selection; no hard state
constraints. The local switch x'Qx<.75 permits pure cart displacement up to
6.1237 m, even beyond the added rail. Soft position cost is not a centering
deadline and soft angle cost is not a fall constraint. The selected no-motion
cost comparison makes this mismatch concrete, without isolating every failure.

**Local recovery:** independently derived original linear spectral radius
0.9983820939 gives a 12.3517-s time constant; Qp=2 gives 0.9785039640 and
0.9204 s. The nonlinear selected probes corroborate faster centering. More CEM
starts cannot improve samples spent entirely in the original local branch
(120588/200000 samples). OA-1 uses this fixed candidate gain, but its changed
entry rule sends pure p=±1 to planning first; the seven local-only passes do
not test that new hybrid. Neither DARE nor a scalar entry cost certifies nonlinear
attraction under saturation and switching.

**Finite search:** best-five can only improve the minimum over its finite
common-prefix attempts. It does not bound the global minimum, and safety is
not its original optimization criterion. Local warm wins best-five cost on 8/20
states, zero warm on 12/20; neither dominates. All six pre-violation probes
remain predicted unsafe, but other inputs/horizons/controllers may work. More
starts alone has no evidenced safety benefit for this selected cohort; it also
does not repair objective or local-feedback mismatch.

**Safety versus recoverability:** 145 rail failures, 103 angle failures and
165 failures of either limit describe the old policy. The box permits physical
cart/angular speeds up to 7.830/10.962, with coupled accelerations and finite
force. A cart-only braking estimate ignores rod momentum and angle limits.
No safe-plan existence, unavoidable-failure or viability certificate is available.
An initially safe state can already lie outside the constrained viable set.
OA-1 candidate feasibility and one-step guards improve specification clarity,
but finite predictions and zero fallback do not prove safe closed-loop recovery.
The endpoint scores sampled states; between-sample crossings are a separate
integration/safety question, not a silently added continuous-time requirement.

**Mathematics:** in the active decoder q=(x0,u) is copied into its output, and
future states are a deterministic rollout graph. Projection gives identity,
injectivity and an identity Jacobian block irrespective of predictor fidelity.
For H N=30, full chart dimension is 34 and fixed-state online input dimension
30, matching ordinary shooting. These facts do not imply learning efficiency,
stability or global search performance. Smooth open-domain graph/rank arguments
need separate qualifications for constrained boxes, corners and safety sets;
the clipped hybrid policy is not globally smooth. Exact differentiable latent
dimension bounds cannot be transferred to approximate encodings or restricted
policy data. Noise enlarges observed behavior dimension only with suitable
injectivity/augmentation assumptions. No novelty verdict follows from failed
control or tested equivalence; the implementation explicitly defers guarantees.

**ML:** these oracle results remove learned dynamics error from this reference
comparison, not from H. There is no full-budget H/L/E study or trained-E nominal
success estimate. Repeated smoke-data training and four untrained ensemble
members establish workload timing only. Coverage, free-run error, local
nearest-6000 excitation/radius, equilibrium/derivative defects and trained-E
uncertainty calibration remain missing. Preserve episode splits, train-only
normalization/identification, mean member trajectory cost and no-switch E;
never supply oracle derivatives/gains to H/L training.

**Statistics:** seven selected local states and twenty selected diagnostic
states are not IID success-probability samples. Attempts/warm variants share
states and RNG; 200 solves are not 200 independent cases. Gate A is development
triage, with outcome-guided reference specification, not confirmatory evidence.
Even a ≥196/200 pass is an empirical acceptance rule, not a 98% guaranteed
success probability. Final evaluation must independently sample training bundles
and common cases, preserve pairing, exclude development data and validate crossed
bootstrap/type-I behavior. Keep BOTH one-sided 95% H−L/H−E lower difference
bounds >.05, the absolute target, fixed-sequence efficiency test, then Holm for
the six secondary tests. The timing-agnostic primary remains pending.

CVXPY was absent: [compat.py](compat.py) removes only two unused imports in memory
after AST checks. Submitted equations/source stay unchanged; legacy CVXPY paths
and an unqualified full-environment claim remain unavailable. CPU/CUDA results
are conditional on this workaround. No dependency install is attempted here.

## Prioritized remaining tests and visualizations

These are future review gates, not permission to spend the residual diagnostic
budget. Do not repeat CT01 or either diagnosis.

| Priority | Minimum next work | Decision / visualization |
|---|---|---|
| P0, saved evidence | OA-1 cost/gain/indexing/schema review and independently audited finite-ranking/queue implementation | Selected CEM one/best-five paired cost decomposition **beside** rail/angle margins; mark warm type, initial unsafe states and 1.2-s scope; retain all 40 pairs |
| P0, before adoption | Fixed OA-1 entry/exit/saturation and decrease checks on the unchanged nonlinear plant; no threshold sweep | p–p_dot and theta–theta_dot projections, c=.75/1 surfaces, saturation/decrease/guard failures and remaining margins; state that projections/sampling are not a 4-D invariant certificate |
| P0, implementation | Independent scorer/ranking/seed/gain/elapsed-queue/fallback rejection checks listed in OA-1; inspect original snapshot integrity | Hybrid timeline: branch, candidate feasibility/rank, local entry/decrease, warm age/shift, queue length, zero fallback, first violation; malformed/partial cases must not pass |
| P0, allocation | Separately scoped amended end-to-end timing and full reservation after config review | Worst branch timing including entry+planning, plant/logging/transfers; worker occupancy and startup/audit slots; no extrapolated old replan fraction |
| P0, only after adoption | One complete fresh 200-case OA-1 Gate A, independent trace/config audit, nominal task decision | Nested outcome table, sign halves, first-crossing times and cohort failure map; plot actual all-time safety and 17–20-s window, not just terminal points |
| P0 conditional | Coupled physical recoverability assessment only if amended failures require it | Saved early-failure force/momentum/phase traces and explicitly bounded reachable envelopes; separate constructive feasible witnesses from any formal impossibility bounds |
| P1 before pilot | Full H/L/E adapters, episode leakage/randomness checks, actual training and local-ID diagnostics under a separately feasible campaign | Coverage/radius/singular-value panels, actual seed points, predicted-vs-true 1/5/30/60-step errors; 60-step targets require separately declared longer data |
| P1 before lock | Crossed-effects power/type-I calibration, hidden final cases, endpoint/device decision and inference pipeline | Boundary and mixed-null error/power maps, censored learning curves and sensitivity; no fake bootstrap bands from oracle cases |
| P1 read-only | Claim C01–C12 ledger, exact PDF/proof-to-code audit and assumption-matched primary-source novelty review | Claim/equation/source/status matrix; distinguish verified graph identities, deferred guarantees and unresolved novelty |
| P2 only if informative | Declared switch/queue/model variants and the three planned robustness stresses | Paired actual-policy timelines/failure maps; preserve submitted H; no transfer allocation yet |

Use physical units/seconds, raw angle, input in u and newtons, tolerance/safety
bands, accessible line styles and complete-cohort summaries. Existing 200-case
gallery and angle section are already generated; inspect the archive instead
of regenerating it. Browser rendering is still untested. Save numerical exports
and PNG/PDF alongside any interactive view only under a declared export task.

## Decision

Do not launch amended Gate A now. Independent arithmetic supports the recorded
failure mechanisms and a concrete new reference candidate; it does not validate
OA-1 or prove the full box physically recoverable. The [versioned proposal](oracle_amendment_proposal_v1.md)
specifies exact costs/gain/switch/safety/search/fallback/queue rules and required
configuration/audit/launcher changes. Current Stage-0 usage is 1.385376803 h,
diagnosis 78.986003356 s; neither residual cap authorizes scope expansion.
Original CUDA reservation 14.279504607 h exceeds the remaining 13.614623197 h.
The proposed 20-h total cap is arithmetically feasible only under OA-1's explicit
new reservation condition and separate adoption; otherwise continue to hold.
All dependent stages remain stopped, even after any future passing Gate A until
the nominal task, endpoint, E recipe, inference and campaign allocation are reviewed.
