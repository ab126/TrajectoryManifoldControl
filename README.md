# Data-Driven Trajectory-Manifold Control

This repository implements a nonlinear predictive controller learned directly from measured state-input trajectories. The primary workflow learns a finite-horizon behavior decoder, optimizes bounded control sequences through that decoder, and optionally hands off to a local terminal controller identified from the same trajectory data.

The learning and control code does not require analytical dynamics, a known linearization, or system-specific equations. A system model is only needed behind the simulation or data-collection interface when producing synthetic experiments.

For a horizon `N`, the implementation uses the canonical trajectory coordinates

```text
q = col(x0, u0, ..., uN-1)
w = col(u0, ..., uN-1, x0, ..., xN)
```

The learned decoder maps `q` to `w`, passes the measured initial state and proposed inputs through exactly, and predicts the future state sequence. The online controller uses batched cross-entropy optimization with inputs kept inside their configured bounds.

## Repository layout

- `src/manifold_control.py`: generic trajectory decoder, learned transition model, manifold solvers, and data-driven MPC implementations.
- `scripts/run_inverted_pendulum_mc.py`: end-to-end data collection, training, evaluation, and closed-loop simulation entry point.
- `manifold_control.ipynb`: primary notebook for inspecting data coverage, training, prediction errors, solver diagnostics, closed-loop results, and animations.
- `src/inverted_pendulum.py`: current simulated plant and numerical integration utilities.
- `tests/`: unit and regression tests for trajectory packing, decoder behavior, data splitting, bounds, units, and controller solvers.
- `saves/`: generated datasets, checkpoints, figures, and simulation results. This directory is intentionally ignored by Git.

## Installation

Python 3.11 is recommended. `environment.yml` selects the newest compatible
3.11 patch available when the environment is created. Historical results used
Python 3.11.4; changing environments does not reproduce or replace those results.
The verified `control311` environment resolves Python **3.11.17**.

Create the complete environment with micromamba:

```bash
micromamba create -f environment.yml -y
micromamba activate control311
```

If shell activation is not initialized, use `micromamba run -n control311`:

```bash
micromamba run -n control311 python --version
micromamba run -n control311 python -m pytest -q
micromamba run -n control311 jupyter lab manifold_control.ipynb
```

Confirm the installation:

```bash
python --version
python -c "import torch; print(torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

The environment includes PyTorch (CPU), NumPy, SciPy, Matplotlib, CVXPY, tqdm,
pytest, JupyterLab, notebook execution tools, Pillow, scikit-learn, UMAP and ffmpeg.
CasADi, do-mpc and PyDeePC are included for optional/legacy control experiments;
some historical notebooks still reference obsolete project interfaces.
PyDeePC is installed from its [official repository](https://github.com/rssalessio/PyDeePC)
at pinned commit `ab05d231e39c5dfe616f3edd60a317b4f9419665` (no PyPI release).
Conda or Mamba can also create the same `environment.yml` as `control311`.

CPU is the portable default, not a claim that controller calls meet a20ms
deadline. For GPU training, use a separate environment with a PyTorch CUDA build
matched to your hardware; do not mix CPU and CUDA packages in this environment.
Register a notebook kernel once (the environment-local prefix avoids changing
other kernels):

```bash
python -m ipykernel install --sys-prefix --name control311 --display-name "Python (control311)"
```

## Run the current simulation

Run commands from the repository root.

Start with the smoke profile to verify the environment and complete the entire pipeline on a small dataset:

```bash
python scripts/run_inverted_pendulum_mc.py --profile smoke
```

Run the reproducible development experiment used for the reported results:

```bash
python scripts/run_inverted_pendulum_mc.py --profile dev
```

The development profile:

- collects 3,500 bounded-input trajectory episodes;
- splits complete episodes into training, validation, and test sets;
- trains the data-driven transition and canonical behavior decoder for 100 epochs;
- evaluates untouched multi-step test trajectories;
- identifies a local terminal policy from training trajectories; and
- simulates initial conditions from `-60` to `+60` degrees for 14 seconds.

To repeat only evaluation using an existing dataset and checkpoint:

```bash
python scripts/run_inverted_pendulum_mc.py --profile dev --reuse-data --reuse-model
```

The main generated artifacts are:

```text
saves/datasets/canonical_trajectories_dev.npz
saves/saved_models/canonical_decoder_dev.pt
saves/simulation_results/canonical_manifold_dev.npz
saves/simulation_results/canonical_manifold_dev.json
saves/figures/canonical_manifold_dev.png
```

The script automatically uses CUDA when it is available. To select a device explicitly:

```bash
python scripts/run_inverted_pendulum_mc.py --profile dev --device cpu
```

## Inspect the experiment

Open the main notebook after running the development profile:

```bash
jupyter lab manifold_control.ipynb
```

The notebook is available at [manifold_control.ipynb](manifold_control.ipynb). It loads the saved development artifacts, so inspecting the results does not retrain the model. Run its cells from top to bottom to view:

- trajectory and input coverage;
- training and validation histories;
- held-out rollout errors;
- canonical decoder consistency checks;
- closed-loop state and input trajectories;
- optimizer timing and terminal-policy usage; and
- the existing system animation.

### Trajectory manifold visualization

The appended visualization sections in `manifold_control.ipynb` (primary) and
`inverted_pendulum_mc_debug.ipynb` show synchronized physical-state and full-horizon
manifold projections. They reuse trained models without changing earlier experiments.
Static PNG and GIF/optional MP4 exports use a fixed projection and camera.
See [visualization setup and commands](docs/visualization.md) and
[geometric limitations](docs/validation/VISUALIZATION.md).

## Tests

The minimal independent validation snapshot lives in root [`validation/`](validation/README.md),
with upstream commit and file hashes recorded in `validation/SOURCE.json`.
It reuses our existing source instead of duplicating the repository. See [the integration protocol](docs/validation/PLAN.md)
and [technical report and reproduction commands](docs/validation/REPORT.md).
Opt-in standalone experiments are in `src/manifold_experiments.py`; original
controller implementations remain unchanged. Baseline reproduction, tuning
screens, and final independent validation must be reported separately.

Run the full test suite with:

```bash
python -m pytest -q
```

Tests should be run before changing trajectory layouts, checkpoint formats, input bounds, normalization, or solver behavior.

## Adapting the method to another system

The reusable learning and controller components operate on generic state and input arrays. To connect another system:

1. Collect fixed-interval trajectories with shapes `(episodes, steps + 1, state_dim)` and `(episodes, steps, input_dim)`.
2. Split by complete episode before extracting transitions or windows.
3. Train `DataDrivenTransition` using measured `(x_k, u_k, x_{k+1})` samples.
4. Wrap it with `CanonicalBehaviorDecoder` for the desired horizon.
5. Configure state, terminal, and input costs for `DataDrivenCEMSolver`.
6. Evaluate prediction error and closed-loop performance on untouched episodes and initial conditions.

Keep plant-specific simulation or hardware access behind the collection and evaluation interface. The decoder and online controller should consume only measured states, candidate inputs, fitted scaling statistics, and learned model parameters.
