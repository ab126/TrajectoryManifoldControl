# Gate A review — October 7, 2026

**Gate A failed. Keep Stage 1, tuning, confirmation and transfer stopped.**
The true-model controller passed **3/200 (1.5%)**, against ≥98% (196/200)
and ≥95% in each angle-sign half. This failed development reference does not
prove that the plant/task is physically unattainable or that H has no contribution.

Original evidence: `overnight_runs/20261007T054549.695689Z/`. It was not edited.
Detailed analysis: `reviews/20261007T054549.695689Z/trace_review.json`.

## Execution and evidence

All 200 cases completed 1000 samples: **200000 original-plant steps**.
Successful IDs: 22, 72, 89. Negative-angle half: **0/94**; positive: **3/106**.
No nonfinite states or solver-fallback branches were recorded. The saved audit
passed physical scoring and original NumPy plant replay with **zero maximum
absolute error**. This review checked **212 saved checksums**, submitted-source
hashes and the validation-source snapshot: no mismatch at review time. HEAD is
`325beae718ab093d3ca48b18a59ae61fd93798a3`; tracked controller files are unchanged.
These checks establish execution consistency, not optimizer optimality.

Runtime check/preflight/audit exited 0; calibration exited **3**, its intentional
failed-Gate-A code, without timeout/cancellation. The morning report's
`calibration_finished` is intermediate; final `run.json` says
`stopped_awaiting_review`. Recorded L4 CUDA allocation/deterministic arithmetic
and float64 oracle checks passed. The disclosed unused-CVXPY-import workaround
and lack of a fresh-context independent equivalence author remain limitations.

| Failed condition | Cases |
|---|---:|
| Rail ±2.4 m | 145 |
| Angle ±π/2 | 103 |
| Final-three-second position | 188 |
| Final cart speed | 133 |
| Final angle | 52 |
| Final angular speed | 53 |

Counts overlap. Only **35** cases stayed within both safety limits throughout:
3 passed, **27 failed only position**, and 5 failed position and cart speed.
Among violating cases, median first rail crossing was **0.52 s** and first
angle crossing **0.22 s**: failures often occur during initial recovery.

## Diagnosis

**Local position recovery conflicts with the strict endpoint.** Gate A uses
`Q=diag(0.02,30,100,10)` and switches to its known-model LQR below cost 0.75
(`gate_a.py:70,98`). A pure position displacement anywhere within the rail
satisfies that switch: even p=2.4 m costs only 0.1152. Local entry therefore
does not imply proximity to the ±0.1 m position target.

The exact upright RK4 linearization with the saved gain is stable, so there
is no linear feedback sign error. Its slow pole **0.998382/sample** corresponds
to **12.35 s** decay. Starting from (p=1 m, other coordinates zero), the linear
diagnostic predicts **0.270 m at 17 s** and **0.212 m at 20 s**, with cost below
0.75 throughout. This is linear algebra, not a new nonlinear test. Saved nonlinear
traces support the concern: case 1 remains safe but its terminal-window maximum
|p| is **0.505 m**, while meeting all other endpoint tolerances. More CEM starts
cannot directly fix time spent exclusively in LQR.

**Safety and search require separate diagnosis.** The cost in
`src/manifold_control.py:270` penalizes quadratic tracking, input and input rate;
it has no explicit rail/angle constraints. Gate A uses a 1.2 s horizon,
256 candidates, 32 elites and 3 iterations (`gate_a.py:73`), with one start.
The box allows initial physical speeds up to ±7.83 m/s and ±10.96 rad/s.
Failures may reflect recoverability, objective mismatch, finite-horizon effects
or inadequate search. The saved traces do not distinguish these explanations.
Missing safety constraints are a design limitation, not an identified code bug.

## Usage and timing

Final `run.json`: **0.886830 overnight worker-hours**, plus **0.476607 readiness**,
total **1.363436 of 15 Stage-0 hours**. Calibration itself took 0.873111 h.
The one-hour readiness budget remains separate; the user launched calibration
with an explicit Stage-0 allocation override.

There were **16128 replans (8.064%)**, **60.294% local** and **31.642% queued**
samples. Solve mean/median/p95/p99: **193.47/192.27/203.46/231.42 ms**. These
timestamps bracket solves, not the complete per-control-step latency endpoint.
Gate A is timing-exempt: deadline misses did not cause its physical failures.
Do not replace H/E replan forecasts with this oracle rate or assume it persists
after amendments.

The one-start reservation including 20% overhead was **14.28 h**. Actual work
was smaller because of local/queued execution. A full five-start worst-case
reservation is **71.40 h**, before prior Stage-0 work, exceeding the remaining
13.64 h. The short failed run does not justify removing reservation checks.

## Recommended next steps

1. **Bounded oracle diagnosis first**, suggested ceiling 0.25 worker-hours.
   Compare one versus five CEM starts on roughly 20 saved initial/pre-violation
   states, replaying common RNG prefixes. Inspect cost components, predicted
   safety violations and warm-start retention. Include position-only cases
   1, 2, 9, pure position states and both angle signs. Separately check exact
   nonlinear local recovery. This is diagnosis, not tuning or a replacement
   200-case calibration.
2. **Specify an oracle amendment before execution.** Investigate position-aware
   oracle local feedback/switching and safety-aware oracle planning; label each
   change to the reference. Preserve submitted H and the plant. If campaign
   costs/recipes change, declare and freeze that comparison separately; never
   silently improve H. More starts alone miss the observed local-feedback issue.
   Retain strict thresholds during diagnosis; relaxing them changes the claim.
3. **Then recalibrate the explicitly specified nominal task.** Following optimizer/
   objective diagnosis, the plan's fallback is box scales **0.5 then 0.25**, in
   sequence, retaining strict thresholds and original broad-box stress reporting.
   Box shrink alone is not assured to fix position recovery. Save new config,
   seeds/source hashes and a feasible allocation preflight in a fresh directory.
   Require ≥98% overall/≥95% per angle-sign half before freezing the box and
   authorizing Stage 1. No fallback or experimental rerun was launched here.

Before a pilot, also resolve E's recipe, timing-agnostic primary proposal,
independent-author equivalence limitation and infeasible worst-case stage
allocations. Retain the primary IUT rule (both one-sided 95% component bounds
above 0.05) and Holm for the secondary family.

Reproduce this saved-data review from the workspace root:

```bash
validation_dir="$PWD/TrajectoryManifoldControl/validation"
LD_LIBRARY_PATH="$validation_dir/.venv/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
LD_PRELOAD="$validation_dir/.venv/lib/libstdc++.so.6" \
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
"$validation_dir/.venv/bin/python" "$validation_dir/review_gate_a.py" \
  "$validation_dir/overnight_runs/20261007T054549.695689Z"
```

No new solves, training or plant acquisition were performed during review.
Historical readiness manifests describe that earlier phase; the overnight
`run.json` and this review describe the subsequent development calibration.
