# Oracle amendment OA-1 — proposed, not adopted

October 8, 2026. Reviewed repository HEAD:
`4c9de6db91d1c2306c0cb1318c8b309f4da390df`.
This specification replaces the open choices in the [earlier proposal](oracle_amendment_proposal.md)
for review purposes. It authorizes no execution. The [assessment](oracle_review_assessment_v1.md)
and [independent arithmetic evidence](reviews/oracle_amendment_v1/evidence_final.json)
explain its motivation and limits. [oa1_config_proposed.json](oa1_config_proposed.json)
is a specification manifest, **not an input accepted by the current launcher**.

## 1. Fixed task and explicit reference changes

Preserve submitted H at `325beae718ab093d3ca48b18a59ae61fd93798a3`, including
its original objective, fitted LQR, switch and queue. OA-1 is a known-model
development reference under `validation/`, never a replacement H or baseline E.
Evaluation uses the original NumPy uniform-rod RK4, cart/rod masses 2/1 kg,
full rod length 1 m, g=9.81, normalized force u=F/(mg), |u|≤10 =98.1 N.
State x=(p/l, p_dot/sqrt(lg), theta, theta_dot/sqrt(g/l)). No angle wrapping.
Physical dt=0.02 s; normalized h=0.02/sqrt(1/9.81).

Retain the full normalized box half-widths (1.5,2.5,1.30,3.5), 200 cases,
1000 transitions/20 s, and the sampled true-state endpoint. All 1001 states
must be finite with |p|≤2.4 m and |theta|≤pi/2. All samples 850…1000,
including both 17 and 20 s, must satisfy |p|≤0.1 m, |p_dot|≤0.05 m/s,
|theta|≤0.02 rad, |theta_dot|≤0.05 rad/s. Exceptions fail; incomplete runs
cannot pass. Gate A requires ≥196/200 and successes≥ceil(0.95*n) in EACH
nonempty initial-angle-sign half. Report balancing and every nested endpoint too.

Prediction stays float64 CUDA on the host L4, one serial worker and one CPU
thread; original plant and local gain arithmetic stay float64 CPU. Pin GPU UUID,
driver, wheel/runtime/native hashes, deterministic algorithms and TF32 off.
Existing runtime UUID is `GPU-ab0ee8e5-cf5f-5b7b-82dc-1e3f054fa136`;
a different device requires an explicit manifest amendment. No CPU fallback.
Oracle timing remains exempt. The timing-agnostic H/L/E primary proposal is
separate and pending; current H/L/E deadline failures remain failures.

OA-1 changes **only the reference**: local/planning Qp .02→2, position-aware
local entry/exit, local decrease and prediction checks, finite-sample safety
ranking, deterministic candidate retention, and queue/warm/fallback rules.
These guards are design choices, not measured improvements. No box shrink,
horizon/start sweep or terminal-P substitution is included.

Use a new development case stream, declared before execution:
`indexed_seed('gate_a_oracle_v1_cases', 1.0)` with root seed 20261006 and the
exact SHA256/compact-JSON/little-endian/modulo contract in `common.py`.
Draw one NumPy uniform(-1,1,(200,4)) array and multiply by the fixed half-widths.
This deliberately changes the development draws, not their distribution. Save
the integer seed and the array hash; no redraws or sign balancing. Preserve the
old 200 cases as historical development evidence, excluded from future final data.

## 2. Exact objective and local gain

Reference Q=diag(2,30,100,10), R=[0.03], Qf=12Q=diag(24,360,1200,120).
Zero state/input references. At N=60 (1.2 physical seconds), minimize

```
J = sum(j=0…59)[x_j' Q x_j + 0.03*u_j^2]
    + x_60' (12Q) x_60
    + 0.002*sum(j=1…59)(u_j-u_(j-1))^2.
```

No dt factor, wrap, barrier, hidden slack, or first-input/last-applied slew term.
This is a planning objective, not the common H/L/E evaluation cost; comparisons
must use the separately locked common cost. For pure p=±1 and zero motion,
OA-1's no-motion J is exactly 144. That is algebra, not evidence a better
OA-1 plan will be found. Keep Qf=12Q to isolate the position-weight change;
it remains a heuristic terminal objective, not a stability argument.

Local matrices follow the exact derivative of the held-input RK4 at upright:

```
Ac = [[0,1,0,0], [0,0,-1/3,0], [0,0,0,1], [0,0,2,0]]
Bc = [0,4/9,0,-2/3]'
A  = I+h*Ac+h^2*Ac^2/2+h^3*Ac^3/6+h^4*Ac^4/24
B  = (h*I+h^2*Ac/2+h^3*Ac^2/6+h^4*Ac^3/24)*Bc
P  = stabilizing DARE(A,B,Q,R)
K  = solve(R+B'PB, B'PA)
u_local(x) = clip(-K*x,-10,10)
```

K≈[-4.267101768904,-26.540276191105,-53.533584744835,-36.645634211809].
The formula defines K; serialize full float64 A,B,P,K and verify against the
recorded candidate gain within atol=rtol=1e-10. DARE failure or unstable linear
A−BK aborts launch, without changing Q/ridge/gain. P is used only for the
local diagnostic V(x)=x'Px, never as Qf. Its checked spectral radius is
0.9785039640 (linear slow-mode time constant ≈0.9204 s); saturation and nonlinear
switching invalidate an inference of global stability.

## 3. Local switch and entry predicate

Define c(x)=x'Qx and S(x): finite, |x[0]|≤2.4 and |x[2]|≤pi/2.
Start in planning mode. Evaluate local entry before consuming any queued input.
From nonlocal mode, enter local only if S(x), c(x)≤0.75 and a float64
60-step known-model rollout of the clipped local law meets S at every node and
the following decrease check at every transition:

```
V(z_next)-V(z) <= -0.5*(z'Qz + 0.03*u_local(z)^2)
                       + 1e-10*max(1,V(z)).
```

The tolerance is numerical; it permits a tiny residual near zero. Entry saves
all guard values, maximum saturation and minimum safety margin. Entering clears
both queue and previous plan. OA-1's entry cost bounds pure |p| by sqrt(.75/2)
≈0.6124 m, versus the old sqrt(.75/.02)≈6.1237 m. Thus the p=±1 local-only probes
do **not** demonstrate the complete OA-1 hybrid: those states must first use MPC.

In local mode, stay only when S(x), c(x)≤1.0, and the next state under the
clipped local input satisfies S and the same decrease check. Apply one input,
recheck on the next observation. Otherwise exit local, clear previous/queue,
and plan on this sample. Hysteresis is exactly 0.75 entry / 1.0 stay, with ≤
at both boundaries. Any failed safety/decrease check overrides hysteresis.
The 60-step check occurs at entry only; the stay check is one step.

This defines an executable entry predicate without claiming an invariant
ellipsoid. Its threshold/decrease factor have not been calibrated. Independent
nonlinear entry/boundary/saturation checks are mandatory before adoption; a
failed prereview check returns the recipe for a new version, not silent retuning.

## 4. Finite search and safety selection

One CEM start per planning call; population 256, elites 32, iterations 3,
smoothing 0.15, normalized transition state scale all ones. Initial mean is the
elapsed-step shifted previous plan, or zero if none. Initial standard deviation
is 20*(0.05+0.45*clamp(2*sqrt(mean(x^2)),0,1)); elite std is unbiased, floored
at 0.02; mean/std updates retain 0.15 old plus 0.85 elite estimate.

Use the existing indexed float32 CPU standard-normal tensors cast to float64
on CUDA: case key `['oracle_v1',case,0]`, physical sample k, horizon 60, three
iterations. Consume the full draws even when deterministic slots replace noise.
At every iteration candidate slots are: 0 current clipped mean, 1 all-zero plan,
2 fresh 60-step clipped Qp=2 local-feedback plan, 3 the original shifted warm
plan (zero if absent), 4…255 sampled/clipped plans. The local warm rollout is
constructed once per planning call from the current state; it is available
outside the local entry predicate as a candidate only. No extra CEM starts.
Retain the best candidate across all iterations and deterministic slots.

For a finite predicted trajectory define, at nodes j=0…60,
r_j=max(0,|p_j|/2.4−1), a_j=max(0,|theta_j|/(pi/2)−1),
D=max_j(r_j,a_j), and W=sum_j(r_j^2+a_j^2).
A candidate is feasible iff its entire state/input arrays are finite, all
|u|≤10, D=0. No constraint slack; safety is sampled, not continuous-time.

Rank finite bounded candidates lexicographically by
`(infeasible_flag,D,W,J,first_seen_candidate_index)`; for feasible candidates
D=W=0, so selection minimizes J. Use this SAME ranking for elites and the
returned incumbent. Nonfinite/out-of-input-bound candidates rank after all
finite bounded candidates and never contaminate moments. If fewer than 32
finite bounded candidates exist, return search-error fallback without an elite
update and latch exception failure; apply the one-sample zero fallback in section 5.
Save every candidate's finite flag, D,W,J and feasible count per
iteration. Save full selected/warm plans and predictions, indexed RNG keys,
and selected component costs. A feasible incumbent cannot be displaced by a
cheaper unsafe plan. The old warm-*cost* inequality is replaced by warm-*rank*
retention; audit both appropriately rather than requiring an obsolete inequality.

These are hard acceptance checks on sampled candidates. They neither solve a
global constrained optimum nor guarantee recursive feasibility or recovery.
A failure to find a feasible trajectory is a search outcome, not nonviability.

## 5. Queue, action checks and fallbacks

After finding a feasible plan, queue its first five controls, apply the first,
and save all 61 predicted states with planning sample k_plan. Before EACH
queued input, try local entry, then compare actual state to its saved predicted
node (atol=rtol=1e-10), and independently check the next state under that input
against S. A mismatch or failed next-state check clears the queue and triggers
one planning call on that same sample. At most one CEM call is allowed per
physical sample. A guard failure immediately after that new solve is an oracle
implementation error, followed by the exception fallback below, not another solve.

For subsequent planning, shift by the ACTUAL number d=k−k_plan of applied
samples; append d copies of the final input for 0<d<60, all final input when
d≥60, unchanged when d=0. A local interval, failure or interruption invalidates
the previous plan. Never reuse a plan across local feedback. Queue clear/age/
shift/reason are logged. s=5 is an upper queue length, not a safety exemption.

If finite search finds no full-horizon feasible candidate, discard the selected
unsafe plan, clear previous/queue and apply normalized zero input for ONE
sample, then retry from the next observed state. Log the zero-input one-step
safety prediction even when it is unsafe. Zero is a fully specified finite
fallback, not a safety certificate or privileged rescue. Infeasibility alone
does not alter the physical endpoint; actual rail/angle violations latch failure.
Continue finite trajectories for all 20 s, as the original simulator did.

On solver/local exception or malformed/nonfinite prediction, clear memory,
apply zero for one sample when the current plant state is finite, latch exception
failure and log cause. A nonfinite plant state ends execution with a marked
nonfinite suffix. Never erase a prior safety/exception failure because a case
later settles. Cap/termination checkpoint preserves the prefix and marks the
entire Gate A incomplete. Timing exemption entails no artificial hold-for-lateness
in this oracle; H/L/E's current five-sample planning/one-sample local deadline
holds remain their separate operational specification.

## 6. Required implementation and independent audit before Gate A

1. Add an OA-1 adapter/config loader under `validation/`. Preserve legacy
   `gate_a.py`, old runs and snapshots; do not make OA-1 a default. Validate a
   strict schema and reject unknown fields, inconsistent gains/units/seeds,
   wrong device/precision/box/endpoint, or an unadopted config. Hash the reviewed
   config separately from run-produced config; freeze replacement execution HEAD.
2. Generalize the audit from `overnight_audit.py`'s hardcoded seed/full-box/
   one-start assumptions. An independently authored/audited implementation
   should take the externally reviewed immutable config as authority, recompute
   case states, plant steps, cost, feasible ranking, local decrease predicates,
   branch/queue/warm ages, fallbacks, endpoint and sign halves. It must not import
   the OA-1 policy/scorer or trust its outcome flags. Missing archive data means
   unavailable review; mismatches mean failed checks. Existing same-author
   equivalence is not an independent-author approval of OA-1.
3. Meaningful rejection checks: wrong seed/box/Q/Qf/gain/reference ID/device;
   cheaper unsafe versus dearer feasible candidate; all-infeasible and NaN
   populations; too few finite elites; local entry then exit with stale plan;
   actual elapsed shifts 1/5/>60; unsafe queue guard; exception/nonfinite prefix;
   inclusive endpoint boundaries/final-window counts; interrupted/partial cases.
   Verify the declared entry guards with the unchanged nonlinear plant near
   entry/exit surfaces and saturation. Finite sampling yields an empirical map,
   not a certified region. These future checks need explicit bounded scope.
4. Replace the queue's readiness-HEAD equality, fixed original CLI, fixed slots
   and ≤15 h allocation validator with reviewed configuration-specific controls.
   Retain immutable submitted-source hashes, locks, cumulative ledger, no
   overwrite, heartbeat/grace/checkpoint, auditable process accounting and no
   downstream jobs. Refuse old benchmark timing as an OA-1 timing estimate.
   Reserve full-control-step overhead, entry/warm prediction, plant, logging,
   startup, failures, audit, scoring and archival before execution. Lock raw-array
   archive destination and checksums; Git retains code/compact manifests only.

## 7. Allocation decision and unresolved risks

**Now: zero new experimental allocation; hold amended Gate A.** Usage is
1.385376803/15 h; remainder 13.614623197 h. Diagnosis 78.986003356/900 s is
complete and its unused seconds are not a reserve for OA-1. The unchanged CUDA
one-start floor plus 20% is 14.279504607 h, already 0.664881410 h too large,
before OA-1 guards and queue/audit overhead. Parallelism does not fix worker-hours.

Propose a separately adopted **20 h TOTAL cumulative Stage-0 cap**, keeping
the L4, precision, full box and endpoint. At current usage this leaves
18.614623197 h. Prospective slots: ≤1 h configuration/independent entry and
implementation verification plus a bounded amended timing measurement; ≤17 h
one-worker Gate-A-only queue including ≤1/3 h runtime/startup/shutdown/audit;
≤0.5 h archive/review; ≥0.114623197 h unassigned reserve. No slot authorizes
training, tuning, sweeps, extra starts or dependent stages. Unspent support
time does not expand its scope. This is a proposed cap revision, not current permission.

Accept preflight only if an independently reviewed OA-1 end-to-end sample-cost
reservation, taking worst branches including entry plus planning, obeys
`200*1000*tau_amended*1.2/3600 + 1/3 <= 17 h` (tau≤0.25 s), with every extra
cost accounted inside the slots. Means plus 20% remain a budgeting model, not
a hard runtime bound; a supervisor enforces the actual slot and checkpoints
incomplete rather than claiming a partial gate. If OA-1 exceeds this criterion,
hold for another explicit allocation decision. The present original timing
0.21419 s only shows arithmetic headroom for this conditional decision, not
OA-1 feasibility. No new measurement is made in this review session.

The gain/entry/search/safety/queue package has never been tested together.
Planning Qp=2 can trade recentering against angle recovery; N=60 may remain
too short. A zero fallback may cross safety limits; linear P checks do not
certify nonlinear viability. Broad-box coupled velocity recovery remains
unresolved. Sequential half-box then quarter-box calibration would require a
new reviewed version/allocation after failure, retain thresholds and broad-box
stress evidence, and never occur automatically. A passing Gate A still needs
review and nominal-task lock; it alone does not authorize Stage 1 or certify H/E.

Retain primary IUT: BOTH one-sided 95% lower difference bounds H−L and H−E
must exceed .05. Preserve the absolute H target, gated acquisition-ratio test
and then Holm for six secondary tests. Nothing in OA-1 changes inference,
final-case separation, required L/E arms, or the pending timing endpoint decision.
