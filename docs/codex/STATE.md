# Current checkpoint — 2026-11-16

## Current objective / completed work
- Audited misleading right plot; added illustrated docs/visualization/RIGHT_PLOT.md
  with element table, exact computations, provenance and3figures plus audit.json.
- Corrected GIF: saves/visualizations/trajectory_manifold_convergence_plane/
  animation.gif,301frames,1170x540,15FPS; original30s controller run reused.
- New src/trajectory_target_visualization.py and
  scripts/explain_trajectory_visualization.py preserve all earlier renderers/GIFs.
  .gitignore now permits tracked docs/visualization figures and explanations.

## Findings / important decisions
- Original gray background and colored nearby points are synthetic decoded
  trajectories, NOT training records or actual optimizer population.
- Original zero-state/zero-input full-horizon star includes zero cart position,
  whereas left target omits position. Old PCA star distance1.32->48.20 while
  displayed state stabilizes; background center distance1.80->48.33. Background
  x0 position sampled±.5, final actual cart4.40m; fixed PCA/limits compress cloud.
- Generic full-horizon PCA does not map target constraint to a2Dplane. New view
  explicitly uses2full-behavior PCs fitted on2450actual training episodes plus
  height=log1p(norm(SI x0 errors in angle/angular speed/cart speed / [.02,.05,.05])).
  Planezero matches left target exactly, cart position/futureinputs free. It is
  a target-constraint summary, not a decoder-feasible surface/terminal condition.
- Added256measured training episodes as filled gray squares, separate from old
  decoded hollow gray background and colored conditioned decoder candidates.
  Original seed0 train split and episodeIDs preserved; hashes verified.
- Displayed normalized error7.5->.0933; height2.140->.0892. Not monotonic.
  Final SI [position,velocity,angle,angular_velocity]:
  [4.4020925,.00010423,-.00029679,-.00460384]. Full-state convergence FALSE.

## Verified results / tests
- Focused tests/test_trajectory_visualization.py:20passed in71.40s.
  Includes exact target plane with arbitrary position/futureinput/futurestates,
  physical scales, separate training labels, and cached rendering.
- All301GIFframes decoded. Raw t/X/U identical to previous rollout; actual
  training rows exactly match dataset and training split; cached selected-plane
  heights match measured state error within2e-6. Three report figures inspected.
- py_compile and git diff --check passed. No controller, plant, model, training,
  or original visualization source functions changed. Source/docs ready to push.

## Other ongoing work / next actions
- Prior standalone TMC/L grid output saves/validation/grid-standalone-3seed-v1
  left untouched; no training/tuning launched. Its process status was not checked
  this task. Monitor completion before interpreting maps or starting tuning.
- No remaining requested visualization work or blockers. GIFs stay ignored;
  explanatory figures/audit/source/tests are tracked. Publish this checkpoint.
