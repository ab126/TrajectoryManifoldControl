# Fixed oracle diagnosis specification — October 7, 2026

**Execution update, October 8:** both fixed phases are complete. Local usage
7.076882 s; CUDA usage 71.909122 s; total 78.986003/900 s. The commands below
record the original execution protocol and must not be rerun. See
[cem_findings.md](diagnostics/cem_findings.md) for the completed CUDA review.

Authorized after the Gate A review; development only. Shared cap **0.25 worker-hours**
includes imports, failures and sequential CPU-local/CUDA-CEM jobs. No tuning,
200-case recalibration, box reduction or dependent stage is launched here.
Supervisor occupancy contributes to the existing 15-hour Stage-0 allocation.

Local phase, original CPU NumPy plant: four pure-position starts p=±1,±0.5 and
saved initial states 1,2,9, for 20 physical seconds. Compare exactly two gains:
the original known-model discrete LQR with Qp=0.02, and a **predeclared Qp=2**
candidate (all other Q/R entries unchanged). The candidate is an oracle-only
diagnostic, not a submitted H modification or selected optimum. Local feedback
is applied throughout, including outside the original switch region, to isolate
the gain; report clipping/safety and strict endpoint results. This creates
14000 diagnostic plant transitions, separate from calibration/acquisition data.

CUDA CEM phase: 20 fixed states: 12 saved initial states including cases 1,2,9,
0 and all successes; six immediately-before-first-violation saved states ranked
by earliest crossing; two pure position states p=±1. At each state compare zero
warm start and a known-model local-feedback rollout. Execute five independent
starts per warm start and compare start zero with the best of its five-start
prefix: **200 total CEM solves**, current architecture/settings N=60,256/32/3.
Keep submitted cost, clipping, smoothing, original float32 noise draws cast to
float64, and SHA256 seeds. Verify objective decomposition and warm-candidate
retention. Record predicted rail/angle violations, losses and first actions.

This open-loop phase does not demonstrate 20-second closed-loop success.
Historical full warm-start plans were not stored, so this is a controlled
comparison with explicit warm starts, not exact reproduction of historical
pre-violation solves. No hindsight future control samples are used as warm starts.
No CPU fallback for CEM. Actual solves remain CUDA-only as in the device plan.
Before solving, repeat the established host runtime check: frozen L4 UUID/driver,
private CUDA native hashes, actual allocation, deterministic GEMM and oracle RK4
agreement. Its process/import occupancy is included in the diagnostic reservation.

Historical execution commands from workspace root (completed):

```bash
# Local phase (agent executes this once).
bash TrajectoryManifoldControl/validation/run_diagnostics.sh local --execute
# Host terminal with working CUDA, after the local phase:
bash TrajectoryManifoldControl/validation/run_diagnostics.sh cem --execute
```

Default without --execute is reservation preflight. No additional allocation
or broad calibration is inferred from a diagnostic result. Before recalibration,
review the evidence and explicitly freeze the oracle feedback/switch/cost/safety
amendment and nominal box. The Qp=2 candidate may fail and is not silently adopted.
