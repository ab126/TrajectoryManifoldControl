# TrajectoryManifoldControl server readiness — 2026-10-06, follow-ups through October 8

**October 7 follow-up:** user-launched CUDA Gate A completed 200 cases and failed
(3/200 strict successes); its original-plant replay audit passed. Dependent stages
remain stopped. See [gate_a_review.md](gate_a_review.md) for the position-recovery
diagnosis, safety failures and recommended bounded diagnostics. Total Stage-0
usage at that review was 1.363436 worker-hours, including 0.476607 readiness. The measurements
and “unexecuted” statements below describe the earlier readiness snapshot, before
calibration. The submitted controller remains unchanged.

The bounded local-only diagnosis subsequently reproduced position settling
failures on all seven selected probes; a fixed oracle-only position-weight
candidate passed seven, with both recipes safe. See
[local_findings.md](diagnostics/local_findings.md). This is selected-state
diagnosis, not Gate A success or an adopted H change. Additional experimental
usage is 7.077 worker-seconds (14000 diagnostic plant transitions).

**October 8 follow-up:** host CUDA diagnosis completed all 200 finite solves in
71.909 worker-seconds. Saved-array objective/safety/source checks passed; see
[cem_findings.md](diagnostics/cem_findings.md). Best-five improved cost in 22/40
pairs, while combined predicted safety remained 18/40. Both diagnostic phases
are complete; do not repeat them. Total Stage-0 occupancy is now **1.385377 h**
of the user-run 15 h allocation; shared diagnostic usage is 78.986003/900 s.
Gate A remains failed. Next specify an oracle amendment and configurable audit;
no reference/H/endpoint change was adopted. Current repository HEAD is
`17d42cd83dea810af5b2979e657f0074244f2cc4`, a validation-only commit; the submitted
source still matches the original evaluated commit. The current handoff is
[validation_handoff.md](validation_handoff.md).

**CPU and CUDA correctness gates passed with the disclosed offline import
workaround. GPU training is faster; submitted MPC is slower on the L4 in these
measurements. Forecast campaign reservations exceed caps. The original CUDA
Gate A failed under the user's allocation override; dependent stages remain
stopped. No attainability or trained-method superiority claim is made.**

## Snapshot and environment

Readiness HEAD: `325beae718ab093d3ca48b18a59ae61fd93798a3`; first inspection clean, final
status `?? validation/`. Tracked controller diff is empty; submitted source hashes match
before/after. The stress-test plan was subsequently amended for CUDA at the
user's explicit request; its original text, amendment diff and before/after
hashes are saved. The critique plan remains unchanged. The parent is not a Git repository. Six recent commits and
statuses are saved; the first saved status already includes our new validation/
directory. No applicable AGENTS.md or pre-existing controller changes were found.

**Version discrepancies:** the plan actually on disk contains this same valid
40-hex HEAD, not the 48-character invalid pin described in the request. It also
already uses the corrected 95% IUT component bounds. Neither plan was edited during the initial readiness campaign;
the requested invalid-pin/97.5% warnings apply to another plan version. The
subsequent authorized amendment changes GPU execution, not those statistical rules.

Private Python 3.11.9: CPU PyTorch 2.5.1, NumPy 2.2.6, SciPy 1.16.0,
Matplotlib 3.10.8, pytest 9.0.2. Lock/freeze and native fingerprints are saved;
pip check passes for installed distributions. **CVXPY is absent; package-index
DNS is blocked.** compat.py omits two unused plant imports in memory and refuses
if their names are referenced. Functions/files are unchanged. This supports the
active path, not the full environment.yml or legacy CVXPY paths. Install genuine
CVXPY before an unqualified reproduction. Native ABI/graphics import repairs
and failed attempts are recorded in preflight_commands.md.

Determinism: root seed 20261006; original smoke seed 234; indexed SHA256 seeds
for case/physical-sample CEM tensors, independent of method branching; separate
dataset/training/member/noise streams. Python hash seed fixed; deterministic
PyTorch algorithms enabled; cuDNN benchmark off; both TF32 flags off. Learned
models use float32, plant/reference/oracle float64; OMP/MKL/OpenBLAS/PyTorch and
interop threads=1. See `artifacts/determinism.json` and `seed_manifest.json`.

Terra is unavailable; no override/subagent was used. Inherited model/effort and
LLM usage telemetry are unexposed. Shooting code is separate but locally authored:
**fresh-context independence was not achieved**. No baseline/reference code was reused.

## Hardware and measured timing

Two AMD EPYC 7543 sockets: **64 physical / 128 logical cores** exposed;
1.0 TiB RAM, initially 830 GiB available. Host load 35–39; no exclusive
reservation. No scheduler tools/environment/instructions were discoverable;
confirm lab policy before larger allocations.

**NVIDIA L4 verified by the user's host execution**, GPU UUID
`GPU-ab0ee8e5-cf5f-5b7b-82dc-1e3f054fa136`, compute capability 8.9,
driver **610.43.02**, driver CUDA UMD **13.3**, installed nvcc **13.3**.
The private CUDA environment uses PyTorch **2.5.1+cu121**, bundled CUDA **12.1**;
imports/pip check pass, 43 native fingerprints and lock/freeze are saved.
Host probe reports **23034 MiB total, 3335 MiB used, 19258 MiB free** at its
snapshot (the earlier user estimate was 24 GiB/~3 GiB). Free memory is time-specific.
CUDA allocation, synchronized deterministic matmul and float64 oracle RK4
checks passed; oracle max error **4.44e-16**. This agent's managed session still
has no NVIDIA nodes, so the successful measurements came from the user's host
terminal running the prepared harness. The initial unavailable probe is historical.

N=30, population=256, elites=32, iterations=3; dev 192/192/192 SiLU architecture.
Ten warm-ups/120 calls per branch; oracle N=60: three warm-ups/12 calls.
End-to-end timing includes switch, warm start, transfers, queue and first action;
CPU/CUDA measurements synchronize where applicable. Empirical quantiles are not deadline guarantees.

| CPU workload | Median / p95 / p99 planning (ms) | Mean tau (ms) | Local p99 (ms) | Peak RSS (MiB) | Deadlines: planning / local |
|---|---:|---:|---:|---:|---|
| Single NN / H | 61.33 / 64.06 / 69.78 | 61.848 | 0.072 | 287.5 | pass / pass |
| Five-member mean-cost E workload | 338.76 / 367.62 / 370.00 | 327.245 | 0.083 | 289.3 | fail / pass* |
| True-model oracle, one start | 111.88 / 125.72 / 126.65 | 114.787 | 0.062 | 292.2 | fail / pass; Gate A timing exempt |

*The original local branch is measured for the ensemble workload; scientific E
has no switch. CPU planning deadline misses: H 0/120, E 120/120. Queue p99:
H 0.070 ms, E 0.124 ms. GPU allocator measurements follow below.

Training: T_train=84000 / T_validation=21000, three epochs; single **1.396
epochs/s**, five serial members **0.297 ensemble epochs/s** (1.483 member epochs/s),
including validation/checkpointing. Peak RSS 399.3/406.8 MiB. Repeated smoke data
and four untrained ensemble control members make these **performance workloads only**.

| CPU workers, threads=1 | Aggregate H solves/s | Aggregate E solves/s |
|---:|---:|---:|
| 1 | 16.45 | 3.42 |
| 2 | 32.23 | 6.74 |
| 4 | 65.18 | 13.32 |
| 8 | 128.53 | 26.44 |

Recommend **8 persistent CPU workers**, distinct physical cores, threads=1;
larger counts unmeasured. Summed peak RSS ≈2316 MiB per arm. Scaling excludes
startup from throughput but charges it to usage; it times solves without the
small switch overhead. Projected Stage 2 eight-worker arm batches take
`4.8e6/128.53 + 4.8e6/26.44` ≈ **60.8 wall-hours**, plus overhead.
Concurrency does not reduce charged worker-hours.

## Verified host CUDA readiness

The complete host readiness sequence passed: probe, **50×100 equivalence**,
tiny original-pipeline smoke, H/E/oracle timing, full-size training throughput,
and 1/2/4-worker scaling. GPU equivalence retains **zero errors**, identical
branch flags/queue lengths, 1020 solves/implementation and the same independence
limitation. CUDA smoke: **120/600 replans=0.20**, zero local steps; it remains
smoke only and is not a measured 20-second nominal-case rate.

| L4 workload | Planning median / p95 / p99 (ms) | Mean tau (ms) | Local p99 (ms) | Peak RSS / allocated / reserved VRAM (MiB) | Planning / local deadline |
|---|---:|---:|---:|---:|---|
| Single NN / H | 66.54 / 93.98 / 101.33 | 74.332 | 0.166 | 887.4 / 34.3 / 36 | fail / pass |
| Five-member mean-cost E workload | 395.96 / 431.65 / 437.33 | 399.123 | 0.169 | 887.8 / 35.6 / 36 | fail / pass* |
| Float64 oracle N=60, one start | 214.24 / 214.83 / 214.86 | 214.193 | 0.101 | 876.0 / 34.7 / 36 | fail / pass; calibration timing exempt |

H misses **2/120** planning deadlines; E misses **120/120**. H/E queue p99:
**2.344/2.360 ms**. GPU allocator peaks exclude CUDA context/driver memory and
other users; no continuous NVML peak sampling was performed. E's local branch
is a computational comparator; its scientific recipe has no switch.

Training at 84000 training / 21000 validation transitions: **7.025 H epochs/s**,
**7.098 ensemble member epochs/s**, **1.420 ensemble epochs/s** with serial
members. H/E training RSS **949.2/953.3 MiB**, allocated VRAM **104.3/104.3 MiB**,
reserved **106/122 MiB**. These replay smoke data; four control ensemble members
are untrained. All results here are **performance measurements only**.

| Workers on the same L4, CPU threads=1 | H solves/s | E solves/s |
|---:|---:|---:|
| 1 | 14.89 | 3.86 |
| 2 | 29.36 | 7.62 |
| 4 | 56.89 | 14.81 |

**Four persistent GPU workers** are the measured throughput choice for offline
performance work; eight are unmeasured. Four-worker summed peak RSS is
**3540/3541 MiB**, allocated VRAM **135.9/141.1 MiB**, reserved **144/144 MiB**
for H/E, excluding context/driver memory. Keep one worker for the timing-sensitive
endpoint until contended end-to-end deadline checks are available. Concurrency
runs measure fixed-state solver calls, not the randomized end-to-end workload;
E throughput therefore cannot replace its end-to-end tau estimate. Shared-device
load varied and there was no exclusive reservation.

GPU training is approximately 5× faster, but end-to-end mean H/E planning is
approximately 1.20/1.22× slower than CPU; float64 oracle is 1.87× slower. Static
source inspection suggests many sequential small neural/RK4 kernels and scalar
synchronizations may limit acceleration; this is an inference, not a measured
profiler diagnosis. Keep submitted code fixed. Projected Stage-2 four-worker
GPU arm batches take `4.8e6/56.89 + 4.8e6/14.81` ≈ **113.4 wall-hours**, plus
overhead; charged worker-hours do not shrink with parallelism.

## Correctness and smoke

**29 existing tests passed once**, one torch.load warning. test_lqr_dd.py is a
100-episode plotting experiment and was excluded from the unit suite.

Plant checks pass: uniform rod I_COM=ml²/12, I_pivot=ml²/3; positive definite
mass matrix; correct force/gravity signs and upright/downward equilibria.
Independent physical RHS max error **1.42e-14**; force 10 = **98.1 N**;
normalized dt **0.0626418391** corresponds to physical **0.02 s**. Zero-force
20-second energy drift **1.675e-5** normalized (threshold 1e-4). RK4 errors
versus DOP853 at 50/100/200 substeps over one physical second:
4.402e-6 / 1.972e-7 / 9.800e-9; convergence ratios 22.33 / 20.12.
Oracle vectorization matches the submitted RK4 to **4.44e-16** over 100 draws.
Equations and exact source locations are in `plant_review.md`.

**50 cases ×100 samples** equivalence passes at atol=rtol=1e-5: **1020 solves
per implementation**, identical flags/queue lengths; action/plan/queue/rollout/
candidate-cost/loss/warm-start errors **zero**. Branches: 862 local, 990 finite
plans, 3118 queued, 30 nonfinite fallbacks. Tests cover saturation, switch
crossings, interrupted queues, stale warm starts and NaN recovery. Weights are
shared synthetic dev weights; nonfinites have explicit branch checks. Agreement
is input-limited, not fidelity, causal attribution or independent certification.
Remove the comparator from sweeps.

**Smoke only:** original dev functions, 120×30 transitions, 3 epochs, original
modulo collection/84–18–18 split, six angles×2 s. **120/600 replans=0.20**, no
local steps. All six balance metrics false: no scientific verdict from tiny
training. A plotting failure was repaired and charged. Endpoint/nonfinite,
IUT/Holm, unit round-trip and indexed RNG contracts also passed.

## Runtime projection and caps

Use measured mean tau_H=0.061848 s, tau_E=0.327245 s,
tau_oracle=0.114787 s/start. Worst case reserves 1000 solves/episode; projection
uses observed H rate 0.20 and E's fixed 200 solves/episode. The **smoke-based
projection is a weak realistic proxy**, not a measured 20-second nominal rate.
Stage 0 uses that same rate only as an oracle proxy and assumes five starts.
Stage 2 uses the plan's provisional 12×200×10 design; power has not been sized.

| Stage | s=1 reservation model (worker-h) | Smoke-rate projection (worker-h) | Cap | s=1 exceeds cap? |
|---|---:|---:|---:|---|
| 0: readiness completed + future oracle, up to 5 starts | 32.31 | 6.81 | 8 | yes |
| 1: three bundles/cases/budgets + mandatory E tuning model | 49.35 | 10.44 | 12 | yes |
| 2: default 12×200×10, H/L/E | **2601.46** | **526.30** | **48** | **yes** |
| 3: three stresses | 48.64 | 9.73 | 8 | yes |
| 4 | no committed workload | no committed workload | 0 | not authorized |

Totals include modeled training/collection but are **lower bounds**: LQR,
plant/scoring/logging and statistics need reservations. Training scales with
transitions; width² tuning scaling is unmeasured. s=1 bounds solve count, not
runtime at mean tau. Stage 2 planning alone: 2593.95/518.79 h; Stage 3 projection
also exceeds cap. Stage 1's projected fit is insufficient launch authorization.

## Historical CUDA runtime projection and pre-calibration allocation decision

Use measured **tau_H=0.074332 s**, **tau_E=0.399123 s**,
**tau_oracle=0.214193 s/start**, CUDA smoke rate **0.20**. The same plan formulas,
training model and unmeasured-overhead qualifications apply to these GPU rows:

| Stage | s=1 model (worker-h) | Smoke-rate projection (worker-h) | Cap | s=1 / projection exceeds cap? |
|---|---:|---:|---:|---|
| 0: actual readiness + future oracle, up to 5 starts | 59.93 | 12.33 | 8 | yes / yes |
| 1: mandatory H/L/E pilot and E tuning model | 59.34 | 11.99 | 12 | yes / nominally no, essentially no overhead allowance |
| 2: default 12×200×10 | **3158.06** | **632.97** | **48** | **yes / yes** |
| 3: three stresses | 59.18 | 11.84 | 8 | yes / yes |
| 4 | no committed workload | no committed workload | 0 | not authorized |

Even **one-start CUDA Gate A** needs **11.90 planning worker-hours** under s=1,
or **14.28 h with 20% overhead**, exceeding both the proposed 7 h allocation and
the entire 8 h Stage-0 cap. The CUDA preflight was run **without --execute**;
calibration remains unlaunched. The 0.20-rate one-start projection is 2.38 h
before overhead, but the plan requires s=1 reservation. Do not launch based on
that projection. Either explicitly revise the oracle allocation for CUDA, or
amend the device rule to use the faster CPU oracle (one-start CPU planning
6.38 h; with 20% overhead 7.65 h, slightly above the 7.57 h Stage-0 remainder).
That CPU option also needs a small allocation revision or an evidenced overhead
reservation. Do not silently switch devices, shrink cases/box or reduce dtype.

## Remaining issues and proposed plan amendments

1. **Primary multiplicity:** require BOTH pH−pL>0.05 and pH−pE>0.05. This is an
   intersection-union test: under the union null, at least one component null
   is true, and rejecting both requires rejecting that true null. Valid
   one-sided alpha=0.05 component tests already control primary type-I error
   at 0.05. Replace any 97.5% Bonferroni component bounds with **one-sided 95%
   lower bounds for both differences**; splitting alpha only costs power.
   They are not simultaneous 95% coverage bounds. This rule is already in the
   inspected plan and is used by power_rules.py. Retain gated **Holm** for the
   secondary family. Crossed-effects power and boundary/mixed-null bootstrap
   calibration remain required; the scaffold's passing logic checks are not
   type-I calibration or power evidence.
2. **Timing-agnostic primary physical success is recommended**, with wall-clock
   latency and deadline-enforced operational success as secondary/descriptive
   endpoints. Currently misses are strict failures: CPU E's measured timing
   alone can fail episodes while H passes this timing check. Device choice can
   therefore change primary outcomes. Keep the existing endpoint until this
   amendment is accepted and locked; no deadline/scoring rule was silently
   changed for H/L/E validation.
3. Distinguish physical dt=0.02 from normalized RK4 dt=0.06264. Install genuine
   CVXPY, retain verified host CUDA execution, measure contended
   end-to-end deadlines and total-device VRAM peaks, and verify
   lab allocation policy before larger runs. Preserve submitted controller.
4. Keep the stale five-step warm shift, lack of switch hysteresis, and
   nearest-6000 local fit as submitted, with diagnostics. Their tested agreement
   does not certify nonlinear stability/safety; small-data local ID can be broad.
   No baseline repository has yet been audited. E recipe freezing, full dev
   reproduction, amended calibration, hidden final cases and complete deadline fallback
   logging for future H/L/E sweeps remain outstanding.
5. Both CPU and CUDA full-campaign reservations exceed caps. CUDA training
   helps training cost but does not fix planning cost. The earlier one-start
   CPU Gate A proposal was conditional on readiness usage; after the host run
   it needs an allocation/overhead review too. Resolve the CUDA Gate A decision
   above before execution and retain the unchanged 200 cases/box/thresholds.

## Readiness-phase usage, artifacts and historical next command

Charge **1715.78 seconds = 0.47661 worker-hours**, below 1 h: **1535.69 metered**
plus **180.09 conservatively charged** for an aborted CPU scaling attempt.
Includes warm-up/imports/failures/startup/supervision. Installation, inspection,
coding/hashing and LLM work excluded; LLM usage unexposed. No jobs remain active.

Commands, environment/config/seed/split manifests, source/data/checkpoint/tensor
hashes, raw timings, worker occupancy, test logs and smoke arrays/checkpoint/
figure are under validation/. `artifacts/stage_estimates.json` and
`estimate_stages.py` make the arithmetic reproducible. Host GPU results and allocator VRAM peaks
are in device-specific JSON files; continuous total-device VRAM peaks are unmeasured. No full calibration, tuning, power simulation,
confirmation, Gymnasium transfer or baseline experiment was launched.

At readiness completion, the proposed next-stage command was **preflight only**
(historical; not the next command after failed calibration and completed diagnosis):

```bash
bash TrajectoryManifoldControl/validation/run_gpu.sh gate_a \
  --cases 200 --starts 1 --box-scale 1 --max-worker-hours 7
```

This command was inspected without executing calibration. It reports a CUDA
planning reservation of 11.90 h, so the allocation is infeasible. Do not add
--execute until the device/allocation amendment is resolved. Gate A retains
original NumPy plant evaluation, validated float64 RK4 predictions, known-model
LQR, physical scoring and timing exemption; require ≥98% overall and ≥95% in
both angle-sign halves. Calibration was unexecuted at that snapshot; the later
one-start run completed and failed as described at the top of this report.

## Prepared overnight queue

At the user's request, `overnight.sh` queues host runtime/source/native checks,
200-case one-start CUDA Gate A, independent physical-score/plant replay audit,
and a checksum/provenance morning report. It launches no dependent experimental
stage. Default Stage-0 cap remains 8 h; the offered command explicitly requests
a 15 h total allocation, including prior usage. Calibration's internal cap is
14.4 h and its process slot is 14.42 h; complete reserved task slots total
14.51167 h before prior usage. The 15 h queue preflight fits the current ledger.
Nothing was launched while preparing it. Four synthetic-fixture regression tests
passed (checkpoint-on-stop, overwrite refusal, failed-child handling, endpoint
units/plant replay). Their usage is charged to readiness; existing tests were
not repeated. Partial runs cannot pass Gate A. Code/source/GPU native hashes,
locks, progress/checkpoints and prior occupancy protect the overnight audit trail.
See `overnight.md` for the copyable nohup command and morning file locations.
