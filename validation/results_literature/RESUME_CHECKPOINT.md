# Supervised continuation checkpoint

**Execution hold — October 14, 2026.** The supervisor exited after its credit
guard detected a balance decrease. Read [STATUS](STATUS), [CREDIT_ALERT](CREDIT_ALERT)
and the [current summary](../results_v1/COLLABORATOR_BRIEF.md) first. Do not restart
the launcher or queue, relax the guard, or reset the credit baseline. Earlier
resumption instructions below do not clear this hold.


Primary execution handoff: [TASK.md](TASK.md). Approved research plan:
[LITERATURE_COMPARISON_PLAN.md](../LITERATURE_COMPARISON_PLAN.md). Additional
controller comparison requirements: [FOLLOWUP_COMPARISON_MATRIX.md](FOLLOWUP_COMPARISON_MATRIX.md).

The host launch succeeded. The supervisor interrupted its model session at the
85% five-hour usage threshold on October12 at14:42:18 Central and logged an
automatic resume at16:27:59. Recorded credits remained1758.873136.

Before relaunching any job, inspect current JOB records and preserved artifacts.
`edmd/JOB_matrix.json` and `pets/JOB_fit.json` have state `error`, exit1, but their
log tails show `RuntimeError: Usage pause: [85.0, 52.0]` from the usage gate.
These exits are usage interruptions, not demonstrated model/solver failures.
Distinguish an explicit usage-pause exception from other errors in future job
reporting. Check available checkpoints before resuming; do not assume an
incomplete epoch or trajectory has been saved. SINDy's pilot is marked paused.

The existing overseer remains unreachable from the interactive thread: repeated
messages to `overseer` return `live agent path /root/overseer not found`.
No replacement is authorized. Forward concise checkpoint packets to that existing
reviewer when reachable. The user explicitly permits overseer to delegate scoped
implementation to Sol/Terra, subject to tool/model availability and the same usage
guard. Delegation must not create unguarded model sessions or duplicate active jobs.
Parallel numerical workers may run across CPUs80–127 within the approved limits.

No research preference or protocol decision is awaiting user input. Reviewer
routing is the sole outstanding communication issue.
