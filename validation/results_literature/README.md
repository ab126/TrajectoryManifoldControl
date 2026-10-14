# Literature campaign: resume and artifact guide

The approved extension is **stopped after its credit guard fired**, with incomplete work preserved. Its previous host supervisor exited on October 14; older queue/JOB activity labels are not proof of current execution. No restart is authorized. Read [TASK](TASK.md), [STATUS](STATUS), [manifest](manifest.json), [RESULTS](RESULTS.md), [QUEUE](QUEUE.json) and all stage JOB files before resuming. [Latest host inventory](inventory/latest_resume_jobs.json) records verified workers, commands, affinity, usage and provisional compute. Do not launch another supervisor or queue while an existing one is alive.

Monitor saved state without importing a scientific runner:

```bash
taskset -c 80-127 cat validation/results_literature/STATUS
taskset -c 80-127 cat validation/results_literature/QUEUE.json
taskset -c 80-127 tail -n 30 validation/results_literature/RESULTS.md
```

Verify host PID liveness and exact commands before any restart; sandbox process visibility can be incomplete. A usage-paused JOB is resumable only after telemetry clears under the unchanged guard and saved data/configuration/model identities match. Preserve prior JOB/log records and charged costs. The existing queue handles dependency ordering, duplicate prevention, at most four single-thread CPU workers and one GPU lease. It can run independent CPU work concurrently. Do not run upstream batch scripts or interactive scientific jobs. Jobs longer than two minutes remain detached with stage JOB command/PID records.

All execution stays on CPUs80–127 with OMP/MKL/OpenBLAS/NumExpr threads1. Invoke explicit dedicated method interpreters under `/tmp/trajectory-literature-envs`; environment reconstruction and package isolation records are in [environments](environments). Never borrow overseeg, other project environments or the preserved project venvs. Public checkouts are isolated under `/tmp/trajectory-literature-vendor`; durable adapters are under `validation/experiments/literature`.

The original85%five-hour/90%weekly usage guard,20second polling and credit-decrease stop remain unchanged. Ceilings are30GPU-hours/84aggregate CPU core-hours plus the approved per-stage caps. Missing telemetry refuses a launch. [Accounting check](inventory/accounting_reconciliation_checkpoint.json) gives retained-attempt lower bounds; totals remain provisional while workers run and older records are incomplete. Usage pauses and cap-limited partial work are not scientific controller failures.

Only this campaign owns its state/logs; never resume completed results_v1. The current CREDIT_ALERT takes precedence over the continuation instructions below. Write STATUS as ASCII text by temporary-file replacement: scientific workers may use an ASCII locale, and replacement prevents an empty-file guard race. Continue authorized stages without per-stage approval, while keeping untested, cap-limited and blocked work distinct from unsuccessful controller results.

Use the single [collaborator summary](../results_v1/COLLABORATOR_BRIEF.md) for current scientific findings and [transfer index](inventory/transfer_index.json) for artifact locations. Tables and figures are saved snapshots; partial results are labeled in review packets. A usage-window `--check` result does not clear CREDIT_ALERT or override the unchanged startup credit baseline. Primary no-horizontal-crossing success and terminal recovery remain separate. No model fitting or undeclared feedback is added after seeing test outcomes.

[Review routing](reviews/ROUTING.json) uses the existing reviewer and shared files. Save concise packets and acknowledge them in [WORKER_ACK](reviews/WORKER_ACK.md); do not create a replacement or claim independent approval while review is pending. The [reviewable code diff](reviews/reviewable_code.patch) excludes pre-existing edits; Git writes remain restricted, and no commit is claimed. DD-PID remains excluded following the unresolved automatic approval rejection; stages1–8 are unaffected.
