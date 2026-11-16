# Trajectory manifold visualization

For the corrected target-plane view, element-by-element computations, training
versus decoded-point provenance, and measured before/after figures, see
[What the right plot shows](visualization/RIGHT_PLOT.md). The constant-reference
star view below is retained as a historical renderer; its distance is not a
state-convergence metric.

## Environment

From the repository root:

```bash
micromamba create -f environment.yml -y
micromamba activate control311
python -m ipykernel install --sys-prefix --name control311 --display-name "Python (control311)"
jupyter lab manifold_control.ipynb
```

Select **Python (control311)** in Jupyter. The environment uses Python3.11 and
includes the core control/scientific packages, notebook execution tools,
`umap-learn`, Pillow and ffmpeg. CPU PyTorch is intentional. Shell activation can
be replaced by `micromamba run -n control311` before each command below.

## Run the examples

The appended sections in `manifold_control.ipynb` and
`inverted_pendulum_mc_debug.ipynb` expose projection, sample count, feasibility
tolerance, FPS, run duration and output paths. They reuse live trained models
when available, or an explicitly selected saved canonical checkpoint. Existing
experiments are not rerun or changed. Checkpoints are not downloaded or trained
automatically; change `VIZ_CHECKPOINT` to a trained artifact you have.

Execute just the new primary notebook cells:

```bash
python -m scripts.execute_visualization_notebook --kernel control311
```

Or export directly using a locally available checkpoint:

```bash
python -m scripts.visualize_trajectory_manifold \
  --checkpoint saves/validation/models/reconstructed-seed0/transition.pt \
  --output saves/visualizations/control311-pca --projection pca

python -m scripts.visualize_trajectory_manifold \
  --checkpoint saves/validation/models/reconstructed-seed0/transition.pt \
  --output saves/visualizations/control311-compare --projection compare --format mp4

python -m pytest -q
```

For a longer GIF that stops after one second near the upright target:

```bash
python -m scripts.visualize_trajectory_manifold \
  --checkpoint saves/validation/models/reconstructed-seed0/transition.pt \
  --output saves/visualizations/trajectory_manifold_convergence \
  --seconds 30 --until-converged --frame-stride 5 --projection pca
```

Here `--seconds` caps the run. Convergence requires cart position within 0.05 m,
cart speed within 0.05 m/s, angle within 0.02 rad, and angular speed within
0.05 rad/s for 50 consecutive samples (1 s). Metadata records whether the
criterion was reached and the final SI state. The controller still runs at every
sample; frame stride only reduces rendered frames and always includes the final
state. The left plot uses angle, angular speed, and cart speed as x/y/z.

Outputs include `static.png`, `animation.gif` or `animation.mp4`, `metadata.json`,
`closed_loop.npz` and `trajectories.npz`; notebook execution also saves an
`executed.ipynb`. Keep separate output paths for comparisons; rerunning an export
replaces files only in its selected output directory. Longer runs use more
preparation time and memory. Playback FPS is independent of physical sample time.

## Geometric interpretation

Each projected point represents **one complete horizon trajectory**, not one
physical state. The current-state slice fixes measured x0 for canonical decoders;
legacy latent decoders use feasibility search and retain explicit residual checks.
Decoder feasibility does not establish true-plant feasibility. Projection,
normalization, background, camera and color scale remain fixed over the run;
rendering uses cached trajectories without new optimizer or decoder calls.

PCA is the default. `umap` displays UMAP even when its neighborhood or
continuity diagnostics are worse than PCA, with a quality warning. `compare`
computes both from the same trajectories: the notebook displays both static
views, and export writes each complete visualization under `pca/` and `umap/`.
The top-level view still selects using the empirical diagnostics. Metadata records
`projection_requested`, the actual `projection`, diagnostics and available comparisons.
If UMAP cannot be imported or fitted, both modes warn and fall back to PCA;
inspect `projection_diagnostics['fallback']` for the reason. Rerun the configuration,
creation/display, and export cells after changing `VIZ_PROJECTION`; existing notebook
outputs do not update automatically. Projected distances and
apparent curvature are not exact manifold geometry. See the
[full API, feasibility definitions and limitations](validation/VISUALIZATION.md).

After updating visualization code, rerun all three visualization code cells,
starting with the configuration cell. It reloads both visualization modules in
dependency order so an existing kernel uses the current classes. This resolves
stale-import errors such as a missing `comparisons` attribute without rerunning
training. Existing `viz` objects are replaced by the creation/display cell.

The optional legacy control dependencies do not repair obsolete notebook calls;
the historical debug notebook's old interfaces and missing autoencoder checkpoint
remain separate limitations. No performance improvement is inferred from these
visualization runs.

## Verified control311 setup

The installed environment is `/home/ab126/micromamba/envs/control311`, with
Python3.11.17, PyTorch2.14.1 (CPU), NumPy2.4.6, SciPy1.17.1, CVXPY1.9.3,
Matplotlib3.11.2, UMAP0.5.12 and ffmpeg9.0.2. CasADi3.7.2, do-mpc5.1.2 and
PyDeePC0.3.6 import successfully. PyDeePC is sourced from its
[official repository](https://github.com/rssalessio/PyDeePC) at pinned revision
`ab05d231e39c5dfe616f3edd60a317b4f9419665`; Git must be available during setup.
Unused do-mpc ONNX/OPC-UA extras are not required and may emit import warnings.

Verification: `pip check` found no broken requirements; all60tests passed.
The control311 kernel executed the three visualization cells, and the CLI
exported a31-frame MP4 under `saves/visualizations/control311-compare/`.
UMAP was actually evaluated: neighbor overlap0.594 versus PCA0.564, but temporal
distortion7.089 versus PCA0.168. The reliability check therefore retained PCA.
These results describe the short visualization example, not controller quality.

## Convergence GIF (2026-11-16)

`saves/visualizations/trajectory_manifold_convergence/animation.gif` covers 30 s
(1,500 controller steps; 301 rendered frames at stride5; 15 FPS). It reuses the
original CLI checkpoint, initial condition, Adam20 controller and seed234.
Final SI state in position/velocity/angle/angular-velocity order:
`[4.40209, 0.0001042, -0.0002968, -0.0046038]`.
The displayed angle/angular-speed/cart-speed coordinates remain within their
near-target tolerances throughout the final second. The full-state convergence
criterion is **not** reached because cart position drifts to 4.40 m. This export
shows stabilization, not return of the cart to its original position. The
original short GIF is preserved; generated artifacts remain Git-ignored.

## Target marker and replay

The right plot uses a blue diamond for the selected controller horizon (blue X
if infeasible) and a red star for the ideal reference: the target state repeated
across the horizon, with zero future input. The reference is projected with the
same fixed normalization and projection as the sampled/selected trajectories;
it is not asserted to belong to the learned decoder manifold. Arbitrary future
inputs generally change future states, so they do not define an equilibrium
target plane. A fixed initial target state with free inputs would instead define
a separate decoded slice, generally curved after projection.

Both source workflows remain available: `notebook_visualization` runs the
experiment, `TrajectoryManifoldVisualization.from_export` restores a PCA cache,
and `export` writes its static image, movie, raw arrays and metadata. Rendering
and saving functions remain in `src/trajectory_visualization.py`. Replay does not
rerun the controller, sample candidates, or fit the projection. Older UMAP
exports cannot replay because their fitted reducer was not serialized.

```bash
python -m scripts.visualize_trajectory_manifold \
  --replay saves/visualizations/trajectory_manifold_convergence \
  --output saves/visualizations/trajectory_manifold_convergence_target

python -m scripts.visualize_trajectory_manifold \
  --replay saves/visualizations/trajectory_manifold_cli \
  --output saves/visualizations/trajectory_manifold_cli_target
```

Original GIF directories remain untouched. Use `--manifold-style legacy` (or
`manifold_style='legacy'` in Python) to retain the previous selected red star and
omit the reference star. The new `reference` style is the default. Replay uses a
separate output directory to preserve its source.
