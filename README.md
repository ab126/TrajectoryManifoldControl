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

Python 3.11 is recommended. The results in this repository were produced with Python 3.11.4.

The easiest setup uses Conda or Mamba and the included environment file:

```bash
conda env create -f environment.yml
conda activate trajectory-manifold-control
```

Mamba can be used as a faster drop-in replacement:

```bash
mamba env create -f environment.yml
mamba activate trajectory-manifold-control
```

Confirm the installation:

```bash
python --version
python -c "import torch; print(torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

The workflow runs on CPU, but a CUDA-capable PyTorch installation is recommended for training and batched online optimization. If your platform needs a different CUDA build, install the appropriate PyTorch package for that system after creating the environment.

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

## Tests

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
