# Bounded local-feedback diagnostic

Original plant, known-model local feedback throughout, 20 s, seven fixed probes.
Original Qp=0.02: **0/7 strict successes**.
Predeclared oracle-only Qp=2: **7/7**.
All 14 trajectories remained within rail and angle safety limits. The original
gain failed only position on every probe. These are selected diagnostic states,
not IID calibration outcomes; the local law was applied outside the original
switch region to isolate it. The candidate is not adopted in H or Gate A.

For p=±1 m, the original final-window maximum |p| was 0.270384 m, versus about
1.03e-7 m for the fixed candidate. This confirms the nonlinear slow-recentering
problem on those probes. Early broad-box safety/search failures remain unresolved.

Actual supervised worker usage: 7.076882 s (0.001966 h); 14000 diagnostic
plant transitions, separate from training/acquisition and the 200-case calibration.
The shared diagnostic cap is 900 s, including imports, failure and future CEM work.

GPU CEM comparison is **prepared, not executed**: 20 fixed states, five common-
prefix starts, zero/local-rollout warm starts, 200 solves. This session has no
GPU nodes. Run in the GPU-enabled host terminal:

```bash
bash TrajectoryManifoldControl/validation/run_diagnostics.sh cem --execute
```

See `validation/oracle_diagnostic_spec.md` for its fixed scope and limitations.
After reviewing that result, specify and freeze an oracle amendment before a
new calibration. No box shrink, tuning or dependent stage has been launched.
