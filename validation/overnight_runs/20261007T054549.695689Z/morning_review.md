# Overnight morning review

Status: calibration_finished. Gate A passed: False.

- Completed cases: 200/200.
- Physical success rate among completed cases: 0.015 (descriptive only if incomplete).
- Angle-sign half rates: [0.0, 0.02830188679245283].
- Independent endpoint/plant replay audit: True.
- Incomplete: False.
- Failure categories: {'rail': 145, 'terminal_position': 188, 'terminal_cart_speed': 133, 'terminal_angle': 52, 'terminal_angular_speed': 53, 'angle': 103}.
- Replay maximum absolute error: 0.0.
- Overnight charged worker-hours: 0.88675.
- Prior Stage-0 worker-hours: 0.47661.
- Total Stage-0 worker-hours: 1.36335.
- Explicit Stage-0 allocation: 15.0 h.

Only development oracle calibration was run. No Stage 1, tuning, confirmation,
transfer, extra starts, or automatic box reduction. Timing is exempt for Gate A.
If incomplete or failed, review the cause before any task/optimizer amendment.
If passed, freeze the box and review the still-unresolved Stage-1 recipe,
deadline endpoint and worst-case allocation before authorizing dependent work.

Logs and traces are local to this directory. `run.json` has commands, budgets,
exit codes, wall times and hashes. `calibration/independent_audit.json` has per-case
diagnostics; `calibration/progress.json` records completed-case checkpoints.
