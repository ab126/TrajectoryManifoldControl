# Trajectory manifold visualization

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

PCA is the default. `compare`/`umap` attempts UMAP and checks neighborhood overlap
and temporal-continuity distortion against PCA; an installed UMAP is not
automatically selected if its diagnostics are worse. Projected distances and
apparent curvature are not exact manifold geometry. See the
[full API, feasibility definitions and limitations](validation/VISUALIZATION.md).

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
