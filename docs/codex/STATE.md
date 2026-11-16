# Current checkpoint — 2026-11-16

## Current objective / completed work
- Requested separate convergence GIF completed at
  saves/visualizations/trajectory_manifold_convergence/animation.gif.
- Same checkpoint/initial state/Adam20/seed234 as original short CLI GIF;
  30s cap,1500control steps,301frames,stride5,PCA,15FPS. Original GIF untouched.
- Left x/y/z now angle, angular velocity, cart velocity throughout paths,
  targets, predictions and fixed limits. Added --until-converged/--frame-stride.
- Optional canonical stop requires all four SI coordinates inside
  [.05m,.05m/s,.02rad,.05rad/s] for50samples (1s); --seconds is cap.

## Verified results / important limits
- Final SI [position,velocity,angle,angular_velocity]:
  [4.4020925,.00010423,-.00029679,-.00460384].
- Displayed three coordinates meet tolerances throughout final second.
  Full-state convergence FALSE: cart drifts4.40m. Do not claim return to origin.
- Pillow decoded final GIF frame and verified301frames,1170x540.
  Raw arrays verified1501states/1500actions. Static preview inspected.
- control311 focused visualization suite:17passed in34.35s; diff check passed.
  Notebook refresh test now explicitly selects PCA, independent of notebook's
  compare default (initial run13passed/1failed due stale test assumption).
- Artifacts under saves are ignored; code/docs/tests are committed separately.

## Other ongoing work (preserved)
- Standalone TMC/L grid PID1499380 verified live during this task; untouched.
  Output saves/validation/grid-standalone-3seed-v1, seeds0/1/2,10086cases.
  Exact launch/resume and historical evidence in LOG.md and docs/launch.md.
- Wait for grid completion before launching proposed tuning/training plan.
  No final core-TMC recovery conclusion; large candidate remains unverified.

## Uncommitted changes / blockers / next actions
- Visualization source,docs,tests and handoff finalized for publication.
- No artifact-generation blockers. Full cart-position convergence would need
  separate controller investigation; not changed for this visualization request.
- Next monitor active grid and inspect completeness before interpreting results.
