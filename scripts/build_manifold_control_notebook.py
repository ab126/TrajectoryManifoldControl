"""Build the lightweight report notebook from deterministic cell sources."""

from __future__ import annotations

import json
from pathlib import Path


def markdown(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": text.splitlines(keepends=True)}


cells = [
    markdown(r"""# Data-driven trajectory-manifold control

This notebook is the primary debug and results view for the end-to-end experiment in
`scripts/run_inverted_pendulum_mc.py`. Expensive collection, training, and closed-loop
optimization run in that script; this notebook loads the resulting artifacts.

Following *Trajectory Manifolds for Nonlinear Data-Enabled Predictive Control, Part I*,
the terminal-state-augmented behavior is

\[
\mathcal B_N^+=\{w=\operatorname{col}(\mathbf u,\mathbf x):x_{k+1}=f(x_k,u_k)\},
\quad q=\operatorname{col}(x_0,\mathbf u),
\quad w=\Phi_N(q).
\]

The learned decoder implements this canonical chart directly. It passes the measured
initial state and proposed inputs through exactly, and predicts future states using a
transition learned only from trajectory samples. The online controller does not call the
simulated plant model. A local terminal policy is also identified from trajectory data."""),
    markdown("## Load artifacts and reproduce configuration\n"),
    code("""from pathlib import Path
import json
import math
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

project_root = Path.cwd()
if not (project_root / "src").exists():
    project_root = project_root.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

summary_path = project_root / "saves/simulation_results/canonical_manifold_dev.json"
results_path = project_root / "saves/simulation_results/canonical_manifold_dev.npz"
data_path = project_root / "saves/datasets/canonical_trajectories_dev.npz"
checkpoint_path = project_root / "saves/saved_models/canonical_decoder_dev.pt"

summary = json.loads(summary_path.read_text(encoding="utf-8"))
results = np.load(results_path)
dataset = np.load(data_path)
summary["config"]"""),
    markdown("## Data provenance, episode split, and coverage\n"),
    code("""X_data, U_data = dataset["X"], dataset["U"]
print("episode split:", summary["split_sizes"])
print("state coverage min:", np.array(summary["coverage_min"]))
print("state coverage max:", np.array(summary["coverage_max"]))
print("input coverage:", float(U_data.min()), float(U_data.max()))

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].hist(X_data[:, :, 2].ravel(), bins=80)
axes[0].axvline(np.pi / 3, color="tab:red", linestyle="--")
axes[0].axvline(-np.pi / 3, color="tab:red", linestyle="--")
axes[0].set(xlabel="theta [rad]", ylabel="samples", title="Collected state coverage")
axes[1].hist(U_data.ravel(), bins=80)
axes[1].set(xlabel="normalized input", ylabel="samples", title="Collected input coverage")
plt.tight_layout()"""),
    markdown("## Training history and held-out rollout error\n"),
    code("""history = summary["training_history"]
rollout_rmse = summary["prediction_metrics"]["rmse_by_step"]
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].semilogy(history["train"], label="train")
axes[0].semilogy(history["validation"], label="validation")
axes[0].set(xlabel="epoch", ylabel="standardized one-step MSE", title="Transition training")
axes[0].legend()
axes[1].plot(np.arange(len(rollout_rmse)) * 0.02, rollout_rmse)
axes[1].set(xlabel="prediction time [s]", ylabel="aggregate state RMSE",
            title="Untouched test episodes")
plt.tight_layout()
summary["prediction_metrics"]"""),
    markdown("## Decoder structural checks\n"),
    code("""from scripts.run_inverted_pendulum_mc import load_transition
from src.manifold_control import CanonicalBehaviorDecoder

transition, payload = load_transition(checkpoint_path, summary["config"]["device"])
decoder = CanonicalBehaviorDecoder(transition, summary["config"]["horizon"]).eval()
device = next(decoder.parameters()).device
x0 = torch.tensor([0.1, -0.2, 0.3, -0.4], device=device)
u = torch.linspace(-1, 1, decoder.horizon, device=device).reshape(-1, 1)
q = decoder.coordinates(x0, u)
x_decoded, u_decoded = decoder.unpack_behavior(decoder(q))
assert torch.equal(x_decoded[0], x0)
assert torch.equal(u_decoded, u)
print("q dimension:", decoder.alpha_dim)
print("w dimension:", decoder.w_dim)
print("exact x0/input passthrough: passed")"""),
    markdown("## Closed-loop results and checkpoints\n"),
    code("""metrics = summary["closed_loop_metrics"]
for case, values in metrics.items():
    print(case, values)

fig, axes = plt.subplots(4, 1, figsize=(11, 12), sharex=True)
for case in metrics:
    t = results[f"t_{case}"]
    X = results[f"X_{case}"]
    U = results[f"U_{case}"]
    axes[0].plot(t, X[:, 2], label=case)
    axes[1].plot(t, X[:, 3] * math.sqrt(9.81), label=case)
    axes[2].plot(t, X[:, 1] * math.sqrt(9.81), label=case)
    axes[3].step(t[:-1], U[:, 0], where="post", label=case)
axes[0].axhline(0.02, color="black", linestyle=":")
axes[0].axhline(-0.02, color="black", linestyle=":")
for axis, label in zip(axes, ("theta [rad]", "theta_dot [rad/s]", "x_dot [m/s]", "u=F/(mg)")):
    axis.set_ylabel(label)
axes[3].set_xlabel("time [s]")
axes[0].legend(ncol=2)
plt.tight_layout()"""),
    markdown("## Solver timing and terminal-policy usage\n"),
    code("""diagnostics = summary["solver_diagnostics"]
for case, values in diagnostics.items():
    solve_time = np.asarray(values["solve_seconds"])
    terminal = np.asarray(values["terminal_policy_used"], dtype=bool)
    print(f"{case}: manifold solves={len(solve_time)}, median={np.median(solve_time):.4f}s, "
          f"terminal-policy fraction={terminal.mean():.3f}")"""),
    markdown("## Existing pendulum animation\n"),
    code("""from IPython.display import HTML
from src.plotting import animate_point_mass

case = "theta0_+60deg"
t = results[f"t_{case}"]
X = results[f"X_{case}"]
U = results[f"U_{case}"][:, 0]
animation = animate_point_mass(t, X[:, 0], X[:, 2], 1.0, U, case_name=case, frame_step=5)
HTML(animation.to_jshtml())"""),
    markdown("""## Interpretation and limitations

- The reported controller is fully data-driven: its rollout decoder, local terminal model,
  and feedback gain are fitted from measured trajectories. The plant equations are used only
  to create the simulated experiment and score the resulting controller.
- All six staged initial angles from -60 to +60 degrees satisfy the angular residence test in
  the saved 14-second experiment. Cart rest is reported separately.
- Input saturation is active for the large-angle cases. These outcomes establish empirical
  performance on the tested simulation and seed; they are not a closed-loop stability proof.
- The 30-step test-rollout error grows with horizon. Optimizer-selected trajectory validation,
  additional seeds, disturbances, and transfer to a different plant remain useful next checks."""),
]

notebook = {"cells": cells, "metadata": {"kernelspec": {"display_name": "manifoldControl",
             "language": "python", "name": "python3"},
             "language_info": {"name": "python", "version": "3.11"}},
            "nbformat": 4, "nbformat_minor": 5}

target = Path(__file__).resolve().parents[1] / "manifold_control.ipynb"
target.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
print(target)
