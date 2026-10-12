# Bounded classical checkpoint review

Scope: static review of common.py, edmd.py, sindy.py, run_classical.py,
run_sindy_compiled.py, sindy_kernel.py, job.py, neural_common.py and saved checks.
No scientific computation was launched by the coordinator. This is not a source
reproduction certification or independent numerical rerun.

The training split in common.load_data matches scripts/run_inverted_pendulum_mc.py:
the seeded whole-episode permutation allocates 525 test, 525 validation and 2450
training episodes. EDMD retains physical-state reconstruction, fixes the constant
lift, explicitly includes input, and uses the shared state/input/terminal and
adjacent-input costs. The QP acceptance path records rejection and zero input.
SINDy uses measured increments and rescales fitted coefficients into raw units;
native and common-L arms remain explicit. Configuration selection is saved before
test forecasts. Saved checks report QP agreement 1.65e-7 and sparse-system rollout
error 1.60e-10. These are recorded worker checks, not reruns by this reviewer.

| Evidence | Current interpretation |
|---|---|
| EDMD test rollout MSE, seeds 0–2: 0.16045 / 0.14478 / 0.15880 | Prediction evidence only |
| SINDy test rollout MSE: 0.06862 / 0.07103 / 0.07185 | Lower matched forecast error does not establish better control |
| PETS seed 0–2 fit completion, 100 epochs each | Training complete; no controller conclusion |

Resolve or document at the next bounded checkpoint:

1. Classical case reuse checks selected configuration and adapter/data identity,
   but currently does not bind the selected pickle, common-L gain, simulator,
   endpoint implementation, or initial_conditions file by artifact digest.
   Verify these identities before C200 reuse and include them in the reuse
   manifest. Preserve valid existing cases with an explicit verified migration.
2. Saved kernel_parity.json reports all three seeds passing, maximum absolute
   difference 5.33e-15, no refit and no parallel kernel. The compiled wrapper adds
   backend hashes to case identities. Preserve this provenance; do not silently
   refit or mix source/compiled identities. This static evidence resolves the
   backend-parity question, subject to endpoint spot replay at stage closure.
3. PETS selects the saved best epoch by validation NLL, while the approved common
   selection rule specifies standardized 30-step rollout MSE with calibration
   separate. Clarify the distinction between training early stopping and final
   configuration selection before final control evaluation; document any protocol
   discrepancy and its resolution without selecting on test outcomes.
4. Reconcile interrupted CPU/GPU costs across attempts, including archived pilots
   and preparation. pause_accounting_correction.json already records a conservative
   GPU charge; verify no lost interval or double counting. Report ledger totals as
   provisional until reconciled. Enforce the three-times-pilot stopping rule and
   forecast all arms including long runs, not merely successful short cases.
5. Correct stale RESULTS.md/manifest runtime statements and complete per-seed,
   per-arm counts. Review approval here is limited to the inspected static wiring;
   endpoint replay and source-faithfulness checks remain pending.

Continue independent approved work under the supervisor. Reply in WORKER_ACK.md
with paths to the resolved evidence; send a small result packet when ready.
