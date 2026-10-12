# Approved literature extension: execution handoff

Additional approved comparator: read [DATA_DRIVEN_PID_SCOPE.md](DATA_DRIVEN_PID_SCOPE.md).
Include data-driven PID in the same matched experiments and controller tables,
within the existing total ceilings and the specified unspent-reserve suballocation.
Stages 1–8 remain required. Acknowledge the queued scope in reviews/WORKER_ACK.md.

The user approved [the plan](../LITERATURE_COMPARISON_PLAN.md), including stages7–8,
on October12,2026 and instructed implementation to begin. Accepted ceilings are
30 GPU-hours and84 aggregate CPU core-hours, with the stage limits in that plan.
Continue through the approved stages after reporting the EDMD/SINDy checkpoint.
All numerical commands must use `taskset -c 80-127`; the supervisor enforces inherited
affinity too. Use at most four single-thread CPU workers and one GPU job initially.
Parallel workers may use any cores in80–127 concurrently; this affinity is not a
requirement to serialize jobs. Keep combined worker accounting within the cap.

The original `codex_supervisor.py` is unchanged. `literature_supervisor.py` imports
its guard and changes only task routing/prompts, adds affinity, duplicate prevention
and startup checks. It preserves85% five-hour/90% weekly thresholds,20-second polls
and the credit-decrease guard. CPU affinity and those thresholds cannot be overridden
through this task entry point. Paid credits are not authorized. The guard is reactive
to recorded usage; missing startup telemetry refuses a launch.

This task owns `results_literature/STATUS`, logs, manifests and its own saved session.
Do not change `results_v1/STATUS` or resume its old session. Maintain `RESULTS.md` and
STATUS so the inherited no-progress check detects real work. Jobs over two minutes
must be detached with a command/PID in the stage JOB file and resume by matching
configuration/data/model identity. Record elapsed time separately from aggregate
CPU/GPU allocations. A cap-limited partial result is not a controller failure.

Use public implementations in isolated `/tmp/trajectory-literature-vendor` and
`/tmp/trajectory-literature-envs`; preserve existing external reference checkouts,
submitted runner/classes and shared environments. Record the baseline working-tree
state before edits. Durable adapters belong under `experiments/literature/`.

The user explicitly requires project-specific isolated environments and forbids
borrowing environments from other projects, including `overseeg`. The supervisor
uses `/tmp/trajectory-literature-envs/supervisor`, a stdlib-only venv with user and
system site-packages disabled. Create separate named venvs (or dedicated conda envs
under that same scratch directory) for numerical methods; invoke their interpreters
by absolute path. Never use ambient `python`/`pip` or an activated environment from
another project. Clear inherited `PYTHONPATH`/`PYTHONHOME` and conda activation
variables before setup or execution. Record each environment's interpreter, prefix,
package freeze/lock, and reconstruction command in durable results. Base interpreter
and standard-library paths may reference the installation used to create a venv;
its third-party packages must not be inherited. Recreate scratch environments from
the recorded specification after `/tmp` cleanup. Preserve the original project
`.venv`/`.venv-gpu`; the launcher reads the former only to create the supervisor venv.

## Independent review

Current coordination instruction (October 12, 2026): the user has assigned the
existing interactive `/root` agent coordination and independent review. Read
`reviews/COORDINATION.md` and `reviews/coordinator_classical_review.md` at the next
checkpoint and acknowledge in `reviews/WORKER_ACK.md`. The host supervised worker
retains sole ownership of numerical launches and implementation. Shared review
files are the cross-session communication route; `/root` names in separate sessions
are not globally addressable. This assignment supersedes the unresolved routing
requirement below; no replacement agent or unguarded session is needed.

The user requested an existing Astra agent named `overseer` for token-efficient
independent review when uncertainty arises or results arrive. An initial message
could not be delivered: that name is not visible in this interactive thread.
The user confirmed on October12 that the existing reviewer must be used. Do not
create a replacement. A second delivery attempt also found no live agent at that
name; routing remains pending until the existing agent becomes reachable.
Read `reviews/ROUTING.json` for updates. Do not pretend another agent is the same
reviewer or spend Astra tokens on broad duplicate audits without need.

At uncertainties and result checkpoints, save a concise packet under `reviews/`
containing the exact question, implementation/source/config paths, a small result
table and proposed interpretation; request a bounded read-only review from overseer
when reachable. Review corrections to scientific wiring/claims must be resolved
before expanding that method. Independent work may continue while a review is
pending. If the reviewer remains unreachable, record review as pending and report
the routing obstacle; do not claim independent approval.

## Immediate next actions

Read [RESUME_CHECKPOINT.md](RESUME_CHECKPOINT.md) for usage-interrupted job states
and the existing reviewer's delegation authorization before restarting workers.

Read [the matched 200-state comparison specification](FOLLOWUP_COMPARISON_MATRIX.md)
at the next checkpoint. It defines the single SMC reference, shared state identities,
controller-only summary table and separate E4 cohort. Respect existing running jobs
and compute ceilings; no interactive scientific launches are authorized.

1. Read the approved plan and manifest; inspect current status and any active jobs.
2. Capture baseline source/environment/working-tree state; inventory saved data and
   public source revisions without importing upstream batch runners.
3. Implement EDMD's isolated fit/prediction/QP adapter and bounded correctness checks.
4. Implement SINDYc's isolated adapter using public PySINDy where compatible.
5. Run the planned budgeted pilots and minimum matched matrices under supervision;
   submit the first compact result packet to overseer and continue the accepted plan.

The current interactive container cannot write `~/.codex` and exposes no NVIDIA
device nodes. The prepared host launcher must start the supervised model in a host
terminal. Do not work around the filesystem sandbox or imply it is already active.
