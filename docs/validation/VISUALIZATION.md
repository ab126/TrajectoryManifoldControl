# Trajectory Manifold Control Visualization

`src/trajectory_visualization.py` is an observational Matplotlib module. It does
not modify training, MPC objectives, solvers, benchmark endpoints or checkpoints.
The demonstration is not a new performance comparison.

For the maintained Python3.11 `control311` micromamba environment (including
UMAP, ffmpeg and notebook tools), use [current setup and commands](../visualization.md).
The `/tmp/tmc-validation-env` commands and unavailable-dependency notes below
describe the historical2026-10-21 verification environment, not the new setup.

## Interpretation

The left panel uses cart position [m], continuous/unwrapped angle [rad] and
physical angular speed [rad/s]; cart speed is shown in the status text. It shows
the measured state, upright target, realized path and selected predicted rollout.
The right panel projects **whole horizon behaviors**, not single states:
`w = [vec(x[0:H+1]), vec(u[0:H])]`. This visualization packing is state-first;
the canonical decoder's input-first packing is never confused with it.

Canonical free coordinates `(x0, future inputs)` yield the exact current-state
slice by fixing measured x0 and sampling bounded inputs. For the legacy latent
decoder, sample encoder coordinates, then run a visualization-only Adam search
on latent coordinates for x0 agreement and bounds. Search does not update model
weights or use simulator dynamics. These candidates are **not** a recording of
the optimizer's population; selected horizons come from actual solver results.

Feasibility uses maximum absolute initial-state error and maximum input-bound
violation, each compared to the exposed tolerance **in model units**. Finite
decoder membership is required. Orange crosses denote approximate/infeasible
latent candidates; red X marks an infeasible selected plan. Rejected nonfinite
points cannot be projected and are omitted; counts remain visible. Background
nonfinite/out-of-bound points are excluded and counted in metadata. Decoder
membership does not prove true-plant feasibility or cross-time physical accuracy.

Each run fits feature means/scales and one projection on the background only.
Camera, axis limits, normalization and color range stay fixed. PCA uses NumPy SVD;
optional `umap-learn` supports out-of-sample transforms, never per-frame refits.
`method='compare'` or `'umap'` compares k-neighbor overlap and relative consecutive
selected-plan step distortion against PCA. UMAP is used only when it is no worse
on these empirical checks; import/fit failures fall back to PCA. Small diagnostic
samples (at most256 points) are used for cost control. Continuity is unassessed
if fewer than two selected plans exist. No projection preserves exact manifold
distance/curvature, and a visible cloud is a sample, not a learned surface mesh.

Colors use the existing MPC cost, not a new visualization objective. The native
example calls `LearnedObjective` directly. The latent example uses its existing
tracking+terminal component, explicitly excluding augmented feasibility and latent
regularization terms (which are not comparable across latent multiplier states).
Selected `u*` and actually applied force are displayed separately, including
clipping/fallback differences. The terminal frame has no new action or plan.

## Notebook and command-line examples

Append-only sections: `manifold_control.ipynb` (primary results notebook) and
`inverted_pendulum_mc_debug.ipynb`, “Trajectory Manifold Control Visualization”.
The primary notebook reuses its live decoder and `summary['config']`.
Both can reuse a live trained decoder and configuration;
legacy usage additionally needs `single_step_solver`, `encoder`, `W_train`, and
the normalization variables from that notebook. It runs a separate short loop,
without overwriting any original experiment variables. The legacy solver is
shallow-copied to isolate bookkeeping; no linear dynamics penalty is allowed.

The old notebook has obsolete preflight/generation/runner calls; they are left
unchanged. To execute only the new section, select an explicit saved canonical
checkpoint in `VIZ_CHECKPOINT`. The default local example is the previously
reconstructed seed0 model; it is NOT bundled, downloaded or trained automatically.
Missing models give an error. This path reuses the checkpoint's horizon and
simulation configuration and the existing standalone native Adam solver
(20iterations,3starts, one-action feedback); it does not use LQR. Both paths use
the repository simulator directly, with explicit state/SI conversions.

```bash
PY=/tmp/tmc-validation-env/bin/python
env MPLBACKEND=Agg OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  "$PY" -m scripts.visualize_trajectory_manifold \
  --checkpoint saves/validation/models/reconstructed-seed0/transition.pt \
  --output saves/visualizations/trajectory_manifold_cli --projection compare
```

GIF uses existing Pillow. MP4 requires an external ffmpeg executable; request
`--format mp4`. Optional UMAP is not a required dependency. Notebook execution
alone needs `nbclient nbformat ipykernel` (install in your environment). Register
the environment's kernel and execute the appended cells:

```bash
"$PY" -m ipykernel install --prefix /tmp/tmc-validation-env --name tmc-viz
"$PY" -m scripts.execute_visualization_notebook --kernel tmc-viz
# Optional legacy notebook's new section only:
"$PY" -m scripts.execute_visualization_notebook --kernel tmc-viz \
  --notebook inverted_pendulum_mc_debug.ipynb
"$PY" -m pytest -q tests/test_trajectory_visualization.py
```

Outputs: `static.png`, `animation.gif` (or MP4), `closed_loop.npz`, `metadata.json`,
`trajectories.npz` (full background/candidate/selected behaviors, residuals,
costs and projection coordinates), and, for the notebook executor,
`executed.ipynb`, under the selected `saves/`
directory. The source notebook keeps its original outputs untouched. Rerunning
an export replaces only files in the explicitly chosen visualization directory.

## Reusable API

Construct `DecoderSampler` with a trained decoder, horizon, bounds and seed;
legacy samplers also need their solver and encoded latent sample pool. Construct
`TrajectoryManifoldVisualization` with time-major state/control arrays, actual
selected solution objects keyed by control step, and a batched objective callback
`objective(x,u,step)`. Supply SI `state_scale`/`force_scale` explicitly when model
coordinates are normalized, and an explicit background initial-state box for a
canonical sampler. The background box is a visualization distribution only,
not guaranteed training support. Inspect held-out forecast errors separately.

`save_static(path, step=...)`, `animation(fps=...)`, and `save_animation(path, fps=...)`
then operate only on cached data: no decoder, optimizer, projection fitting or
projection transforms during rendering. Default frame stride1 includes every
sample and terminal state; static steps must be cached. Increase stride or lower
samples/candidates to reduce memory and rendering cost for long runs. Playback
FPS controls presentation speed, not physical sampling frequency or controller
latency. Sampling, feasibility search, MPC and projection are preparation costs;
none establishes a20ms hardware deadline. Prepared memory scales with frames ×
candidates × full-horizon dimension. UMAP and live legacy checkpoint integration
need their optional dependencies/artifacts; PCA and synthetic latent-search tests
work without them.

## Verified local example (2026-10-21)

The full suite passes60tests. Both notebooks' appended cells were executed,
without running old training cells; the primary executed notebook is retained
under the output directory. CLI and notebook exports contain31GIF frames for a
0.6s run,256background behaviors and24candidates per frame. All744 conditioned
candidates passed the1e-4 model-unit tolerance (max x0 residual1.415e-8).
PCA retained72.999% variance; neighbor overlap0.564, normalized temporal-step
distortion0.168. These are visualization diagnostics, not plant accuracy or
control-performance evidence. Real UMAP comparison and actual MP4 export remain
unexecuted because UMAP/ffmpeg are unavailable; fallback/error paths are tested.
Checkpoint SHA256 and physical conversion scales are recorded in metadata.
Canonical live models are copied before constructing the existing solver, so
its parameter-freezing behavior cannot alter the notebook's training flags.

