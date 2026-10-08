# Proposed oracle amendment — pending CUDA diagnostic

The existing local-only probes support correcting the reference's slow position
recovery. They do not establish broad-box attainability or justify changing H.
The submitted controller/plant and primary statistical rules stay unchanged.

After examining the fixed CUDA diagnostic, the concrete candidate for a **new,
versioned reference calibration** is:

- Known-model oracle local feedback with Qp=2 (other weights and R unchanged),
  as predeclared in the local diagnostic. Require a rail/angle-safe state at
  local entry. Verify its nonlinear safe entry region before accepting a switch.
- Position-aware oracle planning, explicitly documenting any change from the
  submitted Qp=0.02 to Qp=2, and predicted rail/angle feasibility. Use the pending
  fixed-state results to distinguish a cost mismatch from failed search before
  choosing safety handling. No penalties/constraints have silently been added.
- Retain N=60, original input limits/plant, float64 oracle, physical strict
  thresholds, 20 s scoring and oracle timing exemption. Start-count changes need
  a feasible conservative allocation; full five-start worst-case work does not
  fit the current Stage-0 remainder.

Do not adopt the full candidate or launch recalibration until the CUDA loss,
safety and warm-start evidence is reviewed. Any safety constraint/infeasibility
fallback must be specified before execution. Keep H as submitted; an amended
H/cost comparison would be a separately declared development variant.

If that reference still fails after diagnosing optimizer/objective defects,
consider the plan's sequential half-box then quarter-box fallback. Retain broad-
box failures as stress results; strict thresholds remain fixed. Before that run,
generalize the saved-trace audit from its current explicit full-box/one-start
assumptions to the reviewed configuration, with seed/box checks. The current
overnight launcher intentionally runs only the original box and must not be
reused as an amended calibration without that change. Freeze a passing nominal
task before Stage 1. No proposed reference change is currently adopted.
