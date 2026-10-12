# Literature extension: implementation and results

User-approved stages1–8 follow [the plan](../LITERATURE_COMPARISON_PLAN.md).
Compute ceilings:30 GPU-hours and84 aggregate CPU core-hours. Numerical processes
must remain on CPUs80–127. This is a new campaign; the original results remain DONE.

## Current state

The isolated supervisor adapter and host launcher are prepared. No scientific
experiment or new model session has started in the interactive container.
The original usage guard remains unchanged. The host must launch the task because
this container cannot write the Codex session directory or see NVIDIA device nodes.

Independent review routing to the user-named `overseer` is pending; see TASK.md and
reviews/ROUTING.json. The first scientific stages are EDMD and SINDy implementation,
correctness checks, timed pilots and matched evaluations.
