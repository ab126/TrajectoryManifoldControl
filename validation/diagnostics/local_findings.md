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
The shared diagnostic cap is 900 s, including imports, failures and both fixed phases.

**October 8 follow-up:** the user completed the GPU CEM comparison: 20 fixed
states, five common-prefix starts, zero/local-rollout warm starts, 200 solves.
See [cem_findings.md](cem_findings.md). More starts improved cost in 22/40 pairs
but combined predicted safety stayed 18/40; both phases are complete and should
not be rerun. Total diagnostic occupancy is 78.986003 s of the shared 900 s cap.

Next specify a versioned oracle amendment and reviewed configuration/audit
support before a new calibration. No box shrink, tuning, reference amendment or
dependent stage has been launched.
