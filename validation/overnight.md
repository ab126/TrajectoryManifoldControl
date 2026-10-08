# Overnight Gate A queue

Run in the host terminal where CUDA works, from the workspace root. This
proposed launch explicitly raises the total Stage-0 allocation from 8 to
**15 worker-hours**, including already logged readiness and prior queue runs.
The default remains 8; preparing/preflighting this queue does not approve or
start calibration. Running the following --execute command authorizes its
explicit 15-hour allocation and the one-start Gate A calibration only.

```bash
cd /home/ap60/github/data-driven-control
nohup bash TrajectoryManifoldControl/validation/overnight.sh \
  --execute --stage0-cap-hours 15 \
  >> TrajectoryManifoldControl/validation/overnight.log 2>&1 < /dev/null &
```

The serial queue is:

1. Verify HEAD/source hashes and prior correctness gates; record code/native
   manifests. Check actual CUDA allocation, L4 UUID/driver, core native binary
   hashes, deterministic GEMM, and float64 oracle/plant agreement.
2. Preflight, then run **200 development cases**, **one start**, **N=60**,
   **unchanged normalized box**, original NumPy plant, float64 predictor and
   known-model LQR. Calibration's internal cap is 14.4 h; its process slot is
   14.42 h including imports and checkpoint shutdown. Gate A timing is exempt.
3. Independently score saved physical trajectories, verify initial seed/box,
   clipping/branch counts, and replay every recorded finite step through the
   original plant. This diagnoses failures without additional plant acquisition.
4. Save a concise morning report, exit codes, usage and checksums. Stop for
   human review even if Gate A passes.

No Stage 1, tuning, confirmation, additional starts, box reduction or transfer
is queued. E recipe/adapters and worst-case Stage-1 reservations are not ready;
GPU timing remains a primary-endpoint issue. Failure or a cap hit preserves
partial evidence and stops experimental work; the saved-data audit still runs
when possible. Partial results are not attainment evidence. Ordinary shooting
is not included in sweeps.

`overnight_queue.py` reserves 14.51167 h across runtime check/preflight/calibration/
audit/report slots plus all already charged Stage-0 work. It rejects an
insufficient remaining allocation, permits only one queue, gives each launch a
fresh UTC directory and refuses to overwrite calibration output. Repeated
launches count prior queue occupancy rather than reset the budget. SIGTERM/SIGINT
requests a partial-case checkpoint; a stuck child is killed after bounded grace.
One experimental worker runs at a time; OMP/MKL/OpenBLAS/PyTorch threads=1.

For preflight only (already checked):

```bash
bash TrajectoryManifoldControl/validation/overnight.sh --stage0-cap-hours 15
```

Watch the host job:

```bash
tail -f TrajectoryManifoldControl/validation/overnight.log
```

Review files are under `validation/overnight_runs/<UTC>/morning_review.md`,
`run.json`, job logs, `checksums.json` and `calibration/`. `latest.json` identifies
the directory; `heartbeat.json` reports the current job and `progress.json`
reports completed cases. On an external power loss, a run may retain status
"running"; the next queue refuses to assume unlogged occupancy is zero. Review
that run and its allocation before restarting. No background calibration was
started by the agent while preparing this launcher.
