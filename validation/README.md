Readiness harness for the active submitted controller. See `server_readiness.md`
for measured results, limitations, and next-stage allocation. Performance outputs
and tiny smoke results are not scientific validation.

**October 8 handoff:** [validation_handoff.md](validation_handoff.md) consolidates
the original/current plans, completed evidence, remaining control/mathematical/ML
tests and visualizations, implementation gaps, compute constraints, and completed
bounded CPU/GPU diagnostics. Full plan snapshots and provenance are in `handoff/`.

Git keeps validation source, reports, plan snapshots, dependency locks/freezes,
compact result/config JSON, usage ledgers, and hashes. Environments, raw
trajectories/checkpoints, generated media, runtime caches, and most logs stay
local. The notebook's output-free source companion is commit-ready; its rendered
version is preserved locally. Keep a separate run archive for full replay:
the committed manifests describe artifacts that are intentionally absent from
a fresh Git clone. See [.gitignore](.gitignore) and
[notebook instructions](notebooks/README.md).

**October 7:** the user-launched 200-case Gate A failed, with a passing saved-trace
audit. See [gate_a_review.md](gate_a_review.md); dependent stages remain stopped.
Readiness-era calibration status below is historical. `review_gate_a.py` reviews
existing traces without launching experimental work.

The [animation notebook](notebooks/gate_a_examples.ipynb) and
[figure gallery](figures/20261007T054549.695689Z/index.html) now cover every saved
case. See [notebook instructions](notebooks/README.md),
[local diagnostic findings](diagnostics/local_findings.md), and the completed
[CUDA CEM diagnosis](diagnostics/cem_findings.md). All 200 diagnostic solves
completed; best-five lowered cost in 22/40 state/warm pairs but combined predicted
safety stayed 18/40. **Do not rerun the completed phases.** Next prepare a versioned
oracle-amendment specification and reviewed configuration/audit support; no
amendment is adopted and no dependent stage is authorized by these results.

The notebook now also has an **angle-nonsettling** section: all 52 angle failures,
three examples (063/139/028), final-window angle/angular-speed plots, and
animations starting at 17 s. The output-free source companion includes it too.

The 3/200 strict Gate A successes include added all-time rail/angle safety and
cart recentering. Ignoring prior safety, 147 settled the pendulum, 67 also settled
cart speed, and 12 met all final tolerances. These were true-model reference
tests. Rail/position requirements were reviewer-added; active MPC/LQR only
enforce force bounds, with no hard state constraints.

Run from `/home/ap60/github/data-driven-control`:

```bash
bash TrajectoryManifoldControl/validation/run.sh plant_checks
bash TrajectoryManifoldControl/validation/run.sh existing_tests
bash TrajectoryManifoldControl/validation/run.sh equivalence --cases 50 --samples 100
bash TrajectoryManifoldControl/validation/run.sh smoke
bash TrajectoryManifoldControl/validation/run.sh benchmark --kind control --members 1
bash TrajectoryManifoldControl/validation/run.sh benchmark --kind control --members 5
bash TrajectoryManifoldControl/validation/run.sh benchmark --kind training
bash TrajectoryManifoldControl/validation/run.sh benchmark --kind training --members 5
bash TrajectoryManifoldControl/validation/run.sh benchmark --kind oracle
bash TrajectoryManifoldControl/validation/run.sh concurrency
bash TrajectoryManifoldControl/validation/run.sh harness_checks
TrajectoryManifoldControl/validation/.venv/bin/python TrajectoryManifoldControl/validation/estimate_stages.py
```

Existing tests were run once in this campaign; the command above documents how
to reproduce them. Do not execute the whole list again without a new allocation.
The launcher records process/import overhead, checks the cumulative one-hour
readiness limit, and kills individual tasks after at most 900 wall seconds.
Concurrency additionally checks worst-case reservations and logs summed worker
occupancy; it reports steady aggregate throughput separately from process startup.

The private `.venv` contains copied offline Python distributions and native
libraries, without system-site-packages. Its interpreter uses the installed
Python 3.11.9 standard library. `setup_offline.py`, `requirements.lock`, raw
`pip-freeze.txt`, and `artifacts/offline_environment.json` record provenance.
`run.sh` preloads the copied C++ runtime to satisfy the PyTorch ABI.

**CVXPY is absent and package-index DNS is blocked.** `compat.py` omits only the
two unused CVXPY imports from the original plant AST in memory. It asserts the
names are unused before doing so; all function definitions and submitted files
are untouched. This is an active-path environment workaround, not a full
`environment.yml` installation. `src/mpc.py` and legacy CVXPY paths are unavailable.
Install genuine CVXPY when package access is restored; the hook then disables
itself. Numerical results remain conditional on this disclosed workaround.

`independent_equivalence.py` imports no submitted decoder or solver. It shares
only the transition/weights, PyTorch/NumPy, numerical cost/gain specification,
and supplied CEM tensors. It is a separate implementation by the same
orchestrator, not the requested fresh-context Terra implementation; that model
was unavailable. Do not overstate reviewer independence.

The readiness ensemble benchmark used five dev-architecture networks and averaged member
trajectory costs. Four members are initialized rather than trained, so this is
only a computational workload. Training throughput repeats smoke triples to
84000 training and 21000 validation transitions. No ensemble recipe tuning,
confirmatory sweep, or transfer was run. The subsequent one-start Gate A
calibration and both bounded diagnoses completed as described above.

The following is a **historical original-reference preflight**, not the next
action after failed Gate A; its reservation does not fit this eight-hour cap:

```bash
bash TrajectoryManifoldControl/validation/run.sh gate_a --cases 200 --starts 5 --max-worker-hours 8
```

`--execute` runs Gate A only after its own conservative reservation fits the
future allocation. Read the report before executing. Oracle uses a validated
vectorization of the submitted rod RK4 for predictions and the original NumPy
plant for every evaluation step. It saves true states/actions/branches, scores
20 physical seconds including the final 3 seconds, and does not automatically
shrink the nominal box on failure. Timing is exempt only for this calibration.
Boundary/mixed-null calibration and power simulations remain future work;
`power_rules.py` uses 95% component lower bounds for the primary IUT and Holm
for secondary p-values, with no power calculation claimed here.

GPU campaign (user-requested plan amendment): run these from the host terminal
where `nvidia-smi` succeeds, in the workspace root:

```bash
bash TrajectoryManifoldControl/validation/run_gpu.sh probe
bash TrajectoryManifoldControl/validation/run_gpu.sh readiness
```

The private `.venv-gpu` copies CUDA-enabled PyTorch 2.5.1+cu121 and its bundled
CUDA 12.1 libraries from the installed CUDA environment, without inheriting
system site packages. `setup_gpu.py`, `gpu-requirements.lock`, `gpu-pip-freeze.txt`
and `artifacts/gpu_environment_setup.json` record it. The host driver need not
match the installed nvcc 13.3 toolkit version; this launcher uses the wheel's
bundled runtime and performs a real allocation/synchronized matmul check.

This managed session's private `/dev` lacks NVIDIA device nodes. Installing
another CUDA wheel does not supply device access. GPU execution must use a
host terminal that exposes those nodes or an administrator-provided GPU
execution context. No sandbox bypass or silent CPU fallback is attempted.
The launcher respects an existing `CUDA_VISIBLE_DEVICES`, defaulting to GPU 0.

`readiness` uses the existing cumulative one-hour ledger and stops at a failed
probe/equivalence check. It runs full GPU equivalence, tiny original-pipeline
smoke, H/E/oracle performance timings, training throughput, and 1/2/4-worker
GPU scaling only as reservations permit. GPU JSON/timing/smoke/scaling outputs
are separate from CPU results. It does not start Gate A or any scientific sweep.
The tiny CUDA smoke supplies a device-specific replan-rate proxy; training
performance still replays the fixed CPU smoke data for a common workload.
A GPU probe success is required even when CUDA imports successfully.

Historical CUDA Gate A preflight (before the completed failed calibration):

```bash
bash TrajectoryManifoldControl/validation/run_gpu.sh gate_a \
  --cases 200 --starts 1 --box-scale 1 --max-worker-hours 7
```

The default is preflight only; `--execute` belongs to a separately authorized
allocation. CUDA Gate A requires its own measured oracle timing and correctness
artifacts; the original NumPy plant still evaluates every transition. The
7-hour cap is a proposed budget, subject to the measured reservation and actual
Stage-0 usage. No GPU speedup or deadline feasibility is claimed before measurement.

The original unattended Gate-A-only queue completed under the user's explicit
15-hour Stage-0 allocation; see [overnight.md](overnight.md) for its historical
command. It stopped after failed Gate A and did not launch Stage 1. Do not reuse
it for an amended reference without generalizing its configuration and audit.
