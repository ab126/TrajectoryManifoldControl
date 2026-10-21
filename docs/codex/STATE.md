# Codex Project State

## Current objective
Completed reusable trajectory-manifold visualization and appended examples to
the existing simulation notebooks, without changing training or MPC logic.

## Visualization milestone (2026-10-21)
- Added src/trajectory_visualization.py: state-first full-horizon cloud, canonical
  exact conditioning, legacy latent feasibility search, explicit residual/bounds
  masks, fixed normalization/PCA/camera/cost scale, optional diagnosed UMAP fallback.
- Static and cached GIF/MP4 rendering; no decoder/optimizer/projection calls during
  rendering. Physical panel uses SI cart position, continuous angle, physical rate;
  selected and applied force are separately shown. Terminal measured state included.
- Added visualization-only driver and notebook-cell executor. Append-only sections
  in primary manifold_control.ipynb and inverted_pendulum_mc_debug.ipynb; original
  experiments unchanged. Canonical live model copied to preserve training flags.
- Output examples under saves/visualizations/trajectory_manifold_cli and
  trajectory_manifold_notebook: static.png, animation.gif, closed_loop.npz,
  trajectories.npz, metadata.json, executed.ipynb (notebook). No artifact committed.
- Final verification: 60 tests passed in10.46s; syntax/whitespace checks pass.
  CLI and both notebooks' appended code cells executed with existing seed0
  checkpoint; 31frames, .6s run,256background,24candidates/frame. Primary notebook
  executed artifact retained. All744 conditioned candidates pass; maximum initial
  residual1.415e-8. Both final GIFs contain31frames; cached behaviors dimension154.
- PCA variance fraction .72999, neighbor overlap .56396; UMAP unavailable and
  fallback verified. ffmpeg unavailable; GIF executed, MP4 dependency guard tested.
  Final temporal distortion .168267; radius normalization regression tested on
  an exactly embedded rank-three19D trajectory set. Earlier .883913 diagnostic
  used dimension-dependent normalization and is superseded by final metadata.
- No new training, benchmark scoring or scientific performance claims. Legacy
  notebook has obsolete earlier interfaces, unchanged; live latent integration
  cannot be exercised end-to-end without its missing checkpoint. Synthetic
  latent-search tests pass. Optional notebook tools installed in temporary env.
- See docs/validation/VISUALIZATION.md for exact commands and geometry caveats.

Previous history task completed: published the minimal validation integration
and preserved Roy's current fork history on roy-validation and in main's ancestry,
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
Cleanup committed/pushed as 7d495857bf0f5429f288f0d80b11218296081805.
Roy's branch roy-validation points to e66a2b79f27f347286576f678f424625f60787b4
and is published on origin. Local remote roy tracks his original fork.
History-only merge 75f0671222788b70a77f7074a3f5b787b175ac39 is pushed to main.
Its tree exactly equals cleanup commit's tree; Roy's original author/commit IDs
and ancestry are preserved. Handoff committed/pushed as dd6de61a.
Current uncommitted changes: visualization module, two scripts, tests, docs,
README links, appended cells in both notebooks, and STATE/LOG. No commit/push
authorized for the visualization task. No saved model or experiment removed.

## Next actions
1. Review docs/validation/VISUALIZATION.md and generated outputs. Optional future
   checks: real UMAP, installed ffmpeg, and a legacy trained autoencoder checkpoint.
2. Research continuation, only if requested: use the new frozen bundle/output
   command above; it skips preserved cases. Full native C200 costs~8CPU hours/seed.
3. Seed1/2 reconstruction, fresh200 and native60s remain separate future work;
   any new selection must use only training/validation and independent tuning.

## Last checkpoint
2026-10-21. Visualization complete; 60tests pass, both appended notebook sections
and CLI executed, final GIF/caches validated. Original16primary/62debug cells and
metadata unchanged. No controller/training source modified; no commit or push.
