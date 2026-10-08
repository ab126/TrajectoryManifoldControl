# Proposed oracle amendment — after completed CUDA diagnostic

The existing local-only probes support correcting the reference's slow position
recovery. They do not establish broad-box attainability or justify changing H.
The submitted controller/plant and primary statistical rules stay unchanged.

The October 8 CUDA diagnostic completed all 200 solves; see
[cem_findings.md](diagnostics/cem_findings.md). Best-five improved cost in 22/40
state/warm pairs, but combined predicted safety stayed 18/40. Zero and original
local-feedback warm starts each had nine safe states; neither dominated cost
across the cohort. Pure-position zero-control plans cost 1.44 versus 2.105506 for
the tested local-feedback plan. This supports objective/recentering mismatch;
it does not establish irrecoverability or the efficacy of an amended objective.

The candidate for a **new, versioned reference calibration**, still not adopted,
is:

- Known-model oracle local feedback with Qp=2 (other weights and R unchanged),
  as predeclared in the local diagnostic. Require a rail/angle-safe state at
  local entry. Verify its nonlinear safe entry region before accepting a switch.
- Position-aware oracle planning, explicitly documenting any change from the
  submitted Qp=0.02 to Qp=2, and predicted rail/angle feasibility. The completed
  fixed-state results do not justify choosing more starts alone as the remedy.
  Specify safety handling and any infeasibility fallback before execution;
  no penalties/constraints have silently been added. A position-aware planning
  objective has not yet been tested.
- Retain N=60, original input limits/plant, float64 oracle, physical strict
  thresholds, 20 s scoring and oracle timing exemption. Start-count changes need
  a feasible conservative allocation; full five-start worst-case work does not
  fit the current Stage-0 remainder.

The CUDA evidence has now been reviewed. Before adoption or recalibration,
finish a concrete recipe: exact weights, terminal objective, safe-entry/switch
conditions, planning constraints or penalties, and fallback/queue behavior.
Being inside the rail/angle limits at local entry alone does not certify a safe
local region. Validate the configuration/audit implementation and reserve
compute. Keep H as submitted; an amended H/cost comparison would be a separately
declared development variant. The ±2.4 m rail and recentering requirements are
reviewer-added; preserve separate balancing results and explicitly adopt any
endpoint amendment rather than silently removing those requirements.

If that reference still fails after diagnosing optimizer/objective defects,
consider the plan's sequential half-box then quarter-box fallback. Retain broad-
box failures as stress results; strict thresholds remain fixed. Before that run,
generalize the saved-trace audit from its current explicit full-box/one-start
assumptions to the reviewed configuration, with seed/box checks. The current
overnight launcher intentionally runs only the original box and must not be
reused as an amended calibration without that change. Freeze a passing nominal
task before Stage 1. No proposed reference change is currently adopted.
