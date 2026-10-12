# I05 launch and archive contract

This is an **unadopted engineering contract for the preserved strict OA-1
reference**. It authorizes no execution. A balancing-aligned task needs a new
scientific config/schema/proposal; I05 refuses to relabel the old config.
Independent review, original-plant/CUDA validation, nonlinear guards and measured
full-step/audit timing remain pending. No experimental invocation is ready.

## Interfaces and trust

`oracle_queue_v1.preflight(root,config_path,metadata_path,request_path=...,
request_sha256=...)` reads only. It creates no directories, archives, source
snapshots or processes that perform scientific work. A runtime observation may
import Torch and query `nvidia-smi`; it allocates no CUDA tensors, changes no
settings and runs no numerical probe. Proposed metadata is rejected.
`fixture=True` alone allows injected runtime/disk/HEAD observations and marks
the result synthetic; production overrides are rejected.

The user supplies the trusted request's **byte SHA256 independently of its
contents**. Required exact fields: `request_version:1`, `evidence_kind`,
`release_sha256`, `scientific_sha256`, `adoption_sha256`, `allocation_sha256`,
`task_decision:{path,sha256}`, `ledger_sha256`, and
`output_parent:"validation/oracle_runs"`. It is not generated/adopted by preflight.
Release and scientific hashes follow I01; runtime/HEAD/source/environment bytes
are freshly observed and compared with externally reviewed pins.

The externally pinned task record has exactly `decision_version:1`,
`release_sha256`, `scientific_sha256`, `reference_role`, `endpoint_id`,
`approved_by`, `allowed_jobs:["oa1_gate_a"]`, `execution_authorized:true`.
For this old profile only `separate_strict_reference` plus
`historical-strict-reference-v1` is compatible. No such decision exists yet.

Adoption record exact fields: `decision_version:1`, release/scientific hashes,
`approved_by`, allowed jobs, `tests_sha256`, `independent_audit_sha256`,
`allocation_sha256`, `task_decision_sha256`, `timing_sha256`,
`archive_manifest_sha256`, `execution_authorized:true`. These bind the reviewed
event artifacts and later timing/archive decisions to the trusted approval.
Allocation record is exactly version 1, release hash, `adopted:true`, allowed
jobs and the six I01 allocation fields. Self-declared events do not replace
trusted adoption bytes or independent-author review.

`CLOSURE` names mandatory imported helpers/contracts and plant dependencies;
metadata must also cover I01's six required executable files and its full
reviewed import closure. Python executes from checked source snapshots. Every
copied source is rehashed before/after copying. Original SUT bytes and the
documented unused-CVXPY-import workaround are preserved; importing the SUT is
restricted to the adopted worker, never lightweight preflight.

## Timing, allocation and ledger

Timing record exact fields: `timing_version:1`, release hash, `measured:true`,
`full_step_logging_included:true`, `audit_scope:"full_candidate_replay"`,
`amended_tau_s`, `startup_seconds`, `audit_seconds`, `shutdown_seconds`,
`projected_archive_bytes`. Scalars are finite/nonnegative (tau positive).
Fixed costs must fit the adopted fixed-overhead slot, separately from
`200000*tau*1.2`; the I01 cumulative reservation must fit as well. No old solve
timing or observed replan rate substitutes for amended measurements.

Occupancy sums the two historical JSONL ledgers, each overnight `run.json`,
and each new oracle `run.json` **once**. Diagnostic job JSON is checked for
active/stale status but not charged a second time. Its combined hash is SHA256
of the I01 canonical ordered list of `[relative_path,byte_sha256]` pairs:
readiness JSONL, diagnostic JSONL, then lexically sorted overnight runs, then
lexically sorted new oracle runs. An active/stale run,
missing ledger, changed ledger or invalid occupancy blocks execution. Existing
ledgers are never appended or rewritten; each new run owns its occupancy record.

## Archive protocol (proposed, not measured/adopted)

Archive plan fields: `archive_version:1`, release hash, `locator`,
`reserved_bytes`, `format:"npz-tree-chunks-v1"`, `chunk_actions:32`,
`status:"available"`. Locator must match the fresh local oracle-run parent.
Require projected bytes ≥11,571,200,000 (known array lower bound), reserve at
least 1.2× the measured full footprint, and check current free capacity. This
does not prove user quota or future availability; runtime I/O failure stops the
case/run and cannot pass. The user transfers to a checksummed lab archive later.

`oracle_trace_io_v1` stores a finite JSON typed tree and separate typed arrays
in uncompressed NPZ, no pickle/object arrays or JSON NaN. Float64/float32/int64/
bool arrays preserve dtype/shape/nonfinites; tuple/list identity is retained.
Chunk references and case/run indexes carry byte SHA256. Reader bounds header,
array count and uncompressed size; rejects duplicate references, traversal,
symlinks, malformed types and corrupt hashes. Shared serialization is disclosed;
I04 numerical replay remains separate from producer calculations.

Every 32 actions flushes/fsyncs a new chunk and atomically updates the current
case index. Cooperative interruption flushes its exact prefix; SIGKILL/crash
can leave an uncommitted tail, which remains incomplete, never padded. Completed
chunks and old runs are never overwritten. Raw NPZ remains ignored by Git;
compact indexes/manifests/source snapshots belong in Git. Archive throughput,
metadata/inode overhead and checkpoint latency remain unmeasured.

## Runner and capped queue

`gate_a_v1.run_case` is an injected, fixture-testable loop. Production `worker`
creates all 200 cases from the frozen stream, uses original NumPy RK4 for the
plant and float64 CUDA prediction, writes every action/plan and preserves failed
cases. It evaluates only the preserved strict reference; I04 independently
replays plant/actions/search and checks the aggregate. Primary IUT/Holm and
submitted H are untouched.

`Supervisor.run(check_fresh,output)` requires passing preflight, holds the
**same flock as the old queue**, rechecks under the lock, refuses overwrite,
and runs exactly one gate child then one audit child. The lease is inherited
by children; worker/auditor also verify parent identity, snapshot/request pins
and fresh preflight. Queue seals the completed run-index hash for audit. A zero
audit exit is insufficient: the queue requires a nonempty all-pass audit report,
the same run-index hash, 200-case coverage, matching scientific/fixture kind and
agreement between the independent endpoint result and the gate exit status.
No downstream job is enqueued, even after a pass.

After acquiring the lease, the final preflight recheck (including a rejected
recheck), snapshotting, runtime checks, logging, audit, interruption/termination
and teardown all count in actual queue wall occupancy. Audit retains its
overhead slot; termination starts two seconds before a deadline, then kills if
necessary. OS scheduling/I/O are not hard real-time guarantees: any actual
overrun is charged without truncation and prevents a pass. SIGINT/SIGTERM
cancels; stale active records stop future launch for review. Partial files remain.

Default CLI behavior is preflight only, exit 2 if blocked. Worker execution
requires explicit `--execute`, an inherited queue lease and trusted adopted
decisions. Scientific entry points are not exercised by engineering tests.
Synthetic supervisor tests use fake processes and clocks; they cannot certify
actual signal, CUDA, filesystem, latency or controller performance.
