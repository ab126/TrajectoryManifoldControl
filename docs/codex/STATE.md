# Codex Project State

## Current objective
Commit and publish the minimal validation integration, then preserve Roy's
current fork history on a published branch and merge its ancestry into main
without replacing the local implementation or restoring redundant files.

Previous objective (completed):
Replace the redundant full validation submodule with the minimal root validation
snapshot, preserving scientific behavior, results and frozen-run continuity.
The research target (native recovery exceeding L on independent starts) remains
unachieved; this task changes integration layout only.

## Completed work
- Root validation/: 2.2 MB, 19 files. Removed the full submodule checkout,
  gitlink and .gitmodules. Retained its Git object store for recovery.
- Upstream pin e66a2b79f27f347286576f678f424625f60787b4; SOURCE.json SHA256
  7d97e6bd3a8f082b5b87c0c9c6372545e203e6cde06643c695ee347cbb913dea.
- Six unchanged runtime helpers, original/fresh saved starts, fitted diagnostics,
  four reports and implementation note. All16 copied files match upstream bytes.
  Packed1236 H/L reference objects losslessly; no duplicated src or notebooks.
- Adapter reuses local baseline source after upstream hash verification. Updated
  loaders, reference reader, path guards, docs and pytest discovery.
- Reviewed Roy's checked-in plans/reader notes; public commit/issue comment
  endpoints contained no additional comments.
- Old saved models, experiments, frozen bundle and EVIDENCE.json stay unchanged.
  New integration-only bundle and continuation reuse6 completed controller cases
  (two paired starts),6 raw arrays and summary with byte/hash checks.

## Verified results and tests
- Root-layout adapter contracts pass: import provenance, states, timing, units,
  episode separation. Nine L replays exactly match archived outcomes and compared
  final angle/cart position/maximum cart excursion (all differences0).
- 50 tests passed in5.83s, including snapshot corruption, packed H/L coverage
  and rejection of scientific changes during layout migration. Syntax and
  whitespace checks pass. See docs/validation/LAYOUT.md for commands.
- Frozen new-layout verification passes; no new native evaluation or training.
- Original baseline plant/controller, native optimizer and scoring source remain
  unchanged. Scientific frozen settings and evaluation protocol compare equal.

## Frozen continuation
Historical bundle saves/validation/frozen-native.json, SHA256
97a3d77c12d69a30a8ef799f8703c77f425f0dc94f762c88a0403c7a01ad9f41.
New bundle saves/validation/frozen-native-root-validation.json, SHA256
01e556c2f7100e95bb121482bbb76ff596b2bd04d854e340df6ef16e80e04767.
New output saves/validation/final-root-layout-seed0-C contains copied measurements
and LAYOUT_MIGRATION.json, authenticated against original source commit
f41056d670cdb60a03ca73e118be9517fff987cd. Old hash guards were not relaxed.

Current resume command (NOT run during layout work):
/tmp/tmc-validation-env/bin/python -u scripts/frozen_manifold_validation.py evaluate --bundle saves/validation/frozen-native-root-validation.json --arms L H native --set C --output saves/validation/final-root-layout-seed0-C --max-wall-seconds 180

## Scientific context retained
Original L reproduces122/130/128 of200 for seeds0/1/2. Original seed0 H agrees
on9 baseline cases. Tuning: native Adam2/2,M0/2,H1/2,L2/2; periodic1/2,window2/2.
Frozen nominal first2starts: native1/2,H1/2,L2/2; native0wins/1loss vsL.
Delay1 on authored-15deg: native1/1,H0/1,L0/1; mass+25% all1/1.
Native~0.2s/call misses20ms; full C200/fresh200/remaining seeds/native60s pending.
Never tune the frozen configuration using these observed final outcomes.

## Decisions, limits and blockers
- Minimal local copy is explicitly requested; imported helpers stay unchanged.
  No upstream project-level license found; preserve attribution, infer no new
  redistribution license. Detailed reports may link omitted upstream media.
- System Python lacks scientific libraries. Use /tmp/tmc-validation-env/bin/python
  or rebuild from docs/validation/requirements-cpu.txt.
- Sandbox bwrap still fails before execution; approved escalated commands work.
- Original upstream weights/raw arrays absent from Git; prior CPU reconstruction
  remains separately identified. No full final-validation success claimed.

## Uncommitted changes
On main at f41056d. Submodule/gitmodules deletions staged by git rm; snapshot,
import/migration tools, loader/path/test/docs changes uncommitted. No commit/push.
No unrelated edits discarded; no saved local experiment or model removed.

## Next actions
1. Review docs/validation/LAYOUT.md and SOURCE.json for this completed refactor.
2. Research continuation, only if requested: use the new frozen bundle/output
   command above; it skips preserved cases. Full native C200 costs~8CPU hours/seed.
3. Seed1/2 reconstruction, fresh200 and native60s remain separate future work;
   any new selection must use only training/validation and independent tuning.

## Last checkpoint
2026-10-18. User authorized committing/pushing cleanup, preserving Roy's current
fork branch, and a history-only merge retaining our tree. Rechecked 50 tests
(5.75s) and whitespace checks before committing. No new research experiments.
