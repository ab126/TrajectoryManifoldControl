"""End-to-end, data-driven trajectory-manifold control experiment.

The simulated plant is used only behind ``collect_trajectory_data`` and
``simulate_plant``. Training and control consume measured state/input arrays
and contain no inverted-pendulum equations or known model parameters.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.linalg import solve_discrete_are
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inverted_pendulum import rk4_step
from src.manifold_control import CanonicalBehaviorDecoder, DataDrivenCEMSolver, DataDrivenTransition


@dataclass
class ExperimentConfig:
    horizon: int = 40
    dt: float = 0.02 / math.sqrt(1.0 / 9.81)
    episodes: int = 5000
    epochs: int = 160
    batch_size: int = 2048
    hidden_dims: tuple[int, ...] = (256, 256, 256)
    umax: float = 10.0
    seed: int = 234
    simulation_seconds: float = 8.0
    controller_iterations: int = 80
    controller_lr: float = 0.06
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


PROFILES = {
    "smoke": dict(horizon=10, episodes=120, epochs=3, batch_size=256,
                  hidden_dims=(64, 64), simulation_seconds=0.4, controller_iterations=8),
    "dev": dict(horizon=30, episodes=3500, epochs=100, batch_size=2048,
                hidden_dims=(192, 192, 192), simulation_seconds=14.0, controller_iterations=60),
    "full": {},
}


def make_config(profile: str, **overrides) -> ExperimentConfig:
    values = asdict(ExperimentConfig())
    values.update(PROFILES[profile])
    values.update({key: value for key, value in overrides.items() if value is not None})
    values["hidden_dims"] = tuple(values["hidden_dims"])
    return ExperimentConfig(**values)


def _plant_step(x: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
    """Opaque simulated plant interface used only for collection/evaluation."""
    return rk4_step(x, float(np.asarray(u).reshape(-1)[0]), dt=dt, M=2.0)


def collect_trajectory_data(config: ExperimentConfig) -> tuple[np.ndarray, np.ndarray]:
    """Collect broad, bounded-input trajectories without a model-based policy."""
    rng = np.random.default_rng(config.seed)
    X = np.empty((config.episodes, config.horizon + 1, 4), dtype=np.float32)
    U = np.empty((config.episodes, config.horizon, 1), dtype=np.float32)
    state_low = np.array([-1.5, -2.5, -1.30, -3.5])
    state_high = np.array([1.5, 2.5, 1.30, 3.5])
    for episode in tqdm(range(config.episodes), desc="Collecting trajectories"):
        x = rng.uniform(state_low, state_high)
        X[episode, 0] = x
        u = rng.uniform(-config.umax, config.umax)
        hold = int(rng.integers(1, 6))
        for k in range(config.horizon):
            if k % hold == 0:
                if episode % 2:
                    u = rng.uniform(-config.umax, config.umax)
                else:
                    u = np.clip(0.75 * u + rng.normal(scale=3.0), -config.umax, config.umax)
                hold = int(rng.integers(1, 6))
            U[episode, k, 0] = u
            x = _plant_step(x, np.array([u]), config.dt)
            X[episode, k + 1] = x
    if not np.isfinite(X).all():
        raise RuntimeError("nonfinite data collected; reduce the collection domain")
    return X, U


def split_episodes(X: np.ndarray, U: np.ndarray, seed: int) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Split complete episodes before transition/window extraction."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(X))
    n_test = max(1, round(0.15 * len(X)))
    n_val = max(1, round(0.15 * len(X)))
    indices = {"test": order[:n_test], "validation": order[n_test:n_test + n_val],
               "train": order[n_test + n_val:]}
    return {name: (X[idx], U[idx]) for name, idx in indices.items()}


def train_transition(train, validation, config: ExperimentConfig, checkpoint: Path):
    """Train a system-agnostic transition from measured trajectory triples."""
    device = torch.device(config.device)
    X_train, U_train = (torch.from_numpy(array).to(device) for array in train)
    X_val, U_val = (torch.from_numpy(array).to(device) for array in validation)
    x, u, xn = X_train[:, :-1].reshape(-1, 4), U_train.reshape(-1, 1), X_train[:, 1:].reshape(-1, 4)
    xv, uv, xnv = X_val[:, :-1].reshape(-1, 4), U_val.reshape(-1, 1), X_val[:, 1:].reshape(-1, 4)
    model = DataDrivenTransition(4, 1, config.hidden_dims).to(device)
    model.fit_scaling(x, u, xn)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, config.epochs))
    generator = torch.Generator(device="cpu").manual_seed(config.seed)
    history = {"train": [], "validation": []}
    best, best_state = math.inf, None
    for epoch in range(config.epochs):
        model.train()
        order = torch.randperm(len(x), generator=generator)
        total = 0.0
        for start in range(0, len(x), config.batch_size):
            idx = order[start:start + config.batch_size].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = torch.mean(((model(x[idx], u[idx]) - xn[idx]) / model.dx_std) ** 2)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(idx)
        scheduler.step()
        model.eval()
        with torch.no_grad():
            val = torch.mean(((model(xv, uv) - xnv) / model.dx_std) ** 2).item()
        history["train"].append(total / len(x))
        history["validation"].append(val)
        if val < best:
            best = val
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if epoch % max(1, config.epochs // 10) == 0 or epoch + 1 == config.epochs:
            print(f"epoch={epoch:04d} train={history['train'][-1]:.6g} val={val:.6g}")
    if best_state is None:
        raise RuntimeError("training produced no finite checkpoint")
    model.load_state_dict(best_state)
    model.to(device).eval()
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"format": "canonical_behavior_decoder_v1", "transition_state_dict": best_state,
                "x_dim": 4, "u_dim": 1, "horizon": config.horizon,
                "hidden_dims": config.hidden_dims, "config": asdict(config), "history": history}, checkpoint)
    return model, history


def load_transition(checkpoint: Path, device: str):
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    if payload.get("format") != "canonical_behavior_decoder_v1":
        raise ValueError("checkpoint is not a canonical behavior decoder")
    model = DataDrivenTransition(payload["x_dim"], payload["u_dim"], tuple(payload["hidden_dims"])).to(device)
    model.load_state_dict(payload["transition_state_dict"])
    return model.eval(), payload


def prediction_metrics(decoder: CanonicalBehaviorDecoder, data) -> dict[str, object]:
    X_np, U_np = data
    device = next(decoder.parameters()).device
    errors = []
    with torch.no_grad():
        for start in range(0, len(X_np), 512):
            X = torch.from_numpy(X_np[start:start + 512]).to(device)
            U = torch.from_numpy(U_np[start:start + 512]).to(device)
            errors.append((decoder.rollout(X[:, 0], U) - X).cpu().numpy())
    error = np.concatenate(errors)
    absolute = np.abs(error)
    return {"mae_by_state": absolute.mean(axis=(0, 1)).tolist(),
            "p95_by_state": np.percentile(absolute, 95, axis=(0, 1)).tolist(),
            "final_p95_by_state": np.percentile(absolute[:, -1], 95, axis=0).tolist(),
            "rmse_by_step": np.sqrt(np.mean(error**2, axis=(0, 2))).tolist()}


def fit_local_feedback(data, Q: np.ndarray, R: np.ndarray, samples: int = 6000):
    """Identify a local linear model and LQR gain from trajectory data only."""
    X, U = data
    x = X[:, :-1].reshape(-1, X.shape[-1]).astype(float)
    u = U.reshape(-1, U.shape[-1]).astype(float)
    xn = X[:, 1:].reshape(-1, X.shape[-1]).astype(float)
    scale = np.std(x, axis=0).clip(1e-6)
    nearest = np.argsort(np.linalg.norm(x / scale, axis=1))[:min(samples, len(x))]
    regressors = np.concatenate((x[nearest], u[nearest]), axis=1)
    gram = regressors.T @ regressors + 1e-6 * np.eye(regressors.shape[1])
    coefficients = np.linalg.solve(gram, regressors.T @ xn[nearest])
    A, B = coefficients[:x.shape[1]].T, coefficients[x.shape[1]:].T
    P = solve_discrete_are(A, B, Q, R)
    gain = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
    residual = xn[nearest] - (x[nearest] @ A.T + u[nearest] @ B.T)
    return gain, {"A": A, "B": B, "gain": gain,
                  "fit_rmse": np.sqrt(np.mean(residual**2, axis=0))}


def make_controller(decoder: CanonicalBehaviorDecoder, config: ExperimentConfig, local_gain=None):
    device = torch.device(config.device)
    Q = torch.diag(torch.tensor([0.02, 30.0, 100.0, 10.0], device=device))
    R = torch.tensor([[0.03]], device=device)
    solver = DataDrivenCEMSolver(
        decoder, Q, R, Q_terminal=12.0 * Q,
        u_bounds=(-config.umax, config.umax),
        population=256, elite=32, iterations=max(3, config.controller_iterations // 20),
        input_rate_weight=0.002, seed=config.seed,
    )
    previous = torch.zeros(config.horizon, 1, device=device)
    replan_interval = 5
    pending = torch.empty(0, 1, device=device)
    diagnostics = {"loss": [], "iterations": [], "solve_seconds": [], "predictions": [],
                   "terminal_policy_used": []}

    def control(_k: int, state: np.ndarray) -> np.ndarray:
        nonlocal previous, pending
        tracking_cost = float(np.asarray(state) @ Q.cpu().numpy() @ np.asarray(state))
        if local_gain is not None and tracking_cost < 0.75:
            pending = torch.empty(0, 1, device=device)
            diagnostics["terminal_policy_used"].append(True)
            value = -np.asarray(local_gain) @ np.asarray(state)
            return np.clip(value, -config.umax, config.umax).reshape(1)
        diagnostics["terminal_policy_used"].append(False)
        if len(pending):
            value = pending[0].clone()
            pending = pending[1:]
            return value.cpu().numpy()
        current = torch.as_tensor(state, dtype=torch.float32, device=device)
        shifted = torch.cat((previous[replan_interval:], previous[-replan_interval:]))
        start = time.perf_counter()
        solution = solver.solve(current, shifted)
        diagnostics["solve_seconds"].append(time.perf_counter() - start)
        diagnostics["loss"].append(solution.loss)
        diagnostics["iterations"].append(solution.iterations)
        diagnostics["predictions"].append(solution.x.detach().cpu().numpy())
        if not solution.finite:
            previous = torch.zeros_like(previous)
            return np.zeros(1)
        previous = solution.u.detach()
        pending = previous[:replan_interval].clone()
        value = pending[0].clone()
        pending = pending[1:]
        return value.cpu().numpy()

    control.diagnostics = diagnostics
    return control


def simulate_plant(controller, y0: np.ndarray, config: ExperimentConfig, seconds: float | None = None):
    """Evaluate a controller through the opaque plant interface."""
    steps = round((config.simulation_seconds if seconds is None else seconds) / 0.02)
    X, U = np.empty((steps + 1, 4)), np.empty((steps, 1))
    X[0] = y0
    for k in tqdm(range(steps), desc=f"Closed loop theta0={y0[2]:.3f}"):
        U[k] = np.clip(controller(k, X[k]), -config.umax, config.umax)
        X[k + 1] = _plant_step(X[k], U[k], config.dt)
    return np.arange(steps + 1) * 0.02, X, U


def simulate_discrete_inverted_pendulum(u_caller, M_ratio, y0, dt, num_steps, umax=np.inf):
    """Backward-compatible fixed-step simulator used by unit tests/notebooks."""
    state = np.asarray(y0, dtype=float).reshape(4)
    X = np.empty((4, num_steps + 1), dtype=float)
    U = np.empty((1, num_steps), dtype=float)
    X[:, 0] = state
    for k in range(num_steps):
        value = float(np.asarray(u_caller(k, state)).reshape(-1)[0])
        value = float(np.clip(value, -umax, umax))
        U[0, k] = value
        state = rk4_step(state, value, dt=dt, M=M_ratio)
        X[:, k + 1] = state
    return np.arange(num_steps + 1, dtype=float) * dt, X, U


def success_metrics(t: np.ndarray, X: np.ndarray, U: np.ndarray) -> dict[str, float | bool]:
    final = t >= max(0.0, t[-1] - min(3.0, 0.4 * t[-1]))
    angular_velocity = X[:, 3] * math.sqrt(9.81)
    cart_velocity = X[:, 1] * math.sqrt(9.81)
    return {"balanced": bool(np.all(np.abs(X[final, 2]) <= 0.02) and
                              np.all(np.abs(angular_velocity[final]) <= 0.05)),
            "cart_rest": bool(np.all(np.abs(cart_velocity[final]) <= 0.05)),
            "final_theta": float(X[-1, 2]),
            "final_theta_dot_physical": float(angular_velocity[-1]),
            "final_x_dot_physical": float(cart_velocity[-1]),
            "max_abs_u": float(np.max(np.abs(U))) if len(U) else 0.0,
            "max_abs_theta": float(np.max(np.abs(X[:, 2])))}


def save_figure(results, path: Path) -> None:
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(4, 1, figsize=(10, 11), sharex=True)
    for label, (t, X, U) in results.items():
        axes[0].plot(t, X[:, 2], label=label)
        axes[1].plot(t, X[:, 3] * math.sqrt(9.81), label=label)
        axes[2].plot(t, X[:, 1] * math.sqrt(9.81), label=label)
        axes[3].step(t[:-1], U[:, 0], where="post", label=label)
    for axis, label in zip(axes, ("theta [rad]", "theta_dot [rad/s]", "x_dot [m/s]", "u=F/(mg)")):
        axis.set_ylabel(label)
    axes[3].set_xlabel("time [s]")
    axes[0].legend(ncol=2)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def run(args: argparse.Namespace) -> dict[str, object]:
    config = make_config(args.profile, device=args.device, episodes=args.episodes,
                         epochs=args.epochs, seed=args.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    root = Path(__file__).resolve().parents[1]
    data_path = root / "saves" / "datasets" / f"canonical_trajectories_{args.profile}.npz"
    checkpoint = root / "saves" / "saved_models" / f"canonical_decoder_{args.profile}.pt"
    results_path = root / "saves" / "simulation_results" / f"canonical_manifold_{args.profile}.npz"
    summary_path = root / "saves" / "simulation_results" / f"canonical_manifold_{args.profile}.json"
    figure_path = root / "saves" / "figures" / f"canonical_manifold_{args.profile}.png"
    if args.reuse_data and data_path.exists():
        stored = np.load(data_path)
        X, U = stored["X"], stored["U"]
    else:
        X, U = collect_trajectory_data(config)
        data_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(data_path, X=X, U=U, config=json.dumps(asdict(config)))
    splits = split_episodes(X, U, config.seed)
    print("episode splits:", {key: len(value[0]) for key, value in splits.items()})
    print("coverage min:", X.min(axis=(0, 1)), "max:", X.max(axis=(0, 1)))
    if args.reuse_model and checkpoint.exists():
        transition, payload = load_transition(checkpoint, config.device)
        history = payload["history"]
        if payload["horizon"] != config.horizon:
            raise ValueError("checkpoint horizon does not match configuration")
    else:
        transition, history = train_transition(splits["train"], splits["validation"], config, checkpoint)
    decoder = CanonicalBehaviorDecoder(transition, config.horizon).to(config.device).eval()
    metrics = prediction_metrics(decoder, splits["test"])
    print("test prediction metrics:", json.dumps(metrics, indent=2))
    Q_np = np.diag([0.02, 30.0, 100.0, 10.0])
    R_np = np.array([[0.03]])
    local_gain, local_model = fit_local_feedback(splits["train"], Q_np, R_np)
    print("data-driven local fit RMSE:", local_model["fit_rmse"])
    results, closed_loop_metrics, diagnostics = {}, {}, {}
    for degrees in (-60, -30, -15, 15, 30, 60):
        controller = make_controller(decoder, config, local_gain=local_gain)
        label = f"theta0_{degrees:+d}deg"
        result = simulate_plant(controller, np.array([0.0, 0.0, math.radians(degrees), 0.0]), config)
        results[label] = result
        closed_loop_metrics[label] = success_metrics(*result)
        diagnostics[label] = {key: controller.diagnostics[key]
                              for key in ("loss", "iterations", "solve_seconds", "terminal_policy_used")}
        print(label, closed_loop_metrics[label])
    save_figure(results, figure_path)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {"config_json": np.array(json.dumps(asdict(config)))}
    for label, (t, states, inputs) in results.items():
        arrays.update({f"t_{label}": t, f"X_{label}": states, f"U_{label}": inputs})
    np.savez_compressed(results_path, **arrays)
    summary = {"config": asdict(config),
               "paper_notation": {"q": "col(x0,u0,...,uN-1)", "w": "col(u0,...,uN-1,x0,...,xN)"},
               "split_sizes": {key: len(value[0]) for key, value in splits.items()},
               "coverage_min": X.min(axis=(0, 1)).tolist(), "coverage_max": X.max(axis=(0, 1)).tolist(),
               "prediction_metrics": metrics, "training_history": history,
               "local_model": {key: value.tolist() for key, value in local_model.items()},
               "closed_loop_metrics": closed_loop_metrics, "solver_diagnostics": diagnostics,
               "artifacts": {"data": str(data_path), "checkpoint": str(checkpoint),
                             "results": str(results_path), "figure": str(figure_path)}}
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("summary:", summary_path)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=PROFILES, default="dev")
    parser.add_argument("--device", default=None)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--reuse-data", action="store_true")
    parser.add_argument("--reuse-model", action="store_true")
    return parser


def build_arg_parser() -> argparse.ArgumentParser:
    """Compatibility parser for the earlier public script interface."""
    parser = build_parser()
    parser.add_argument("--H", type=int, default=25)
    parser.add_argument("--alpha-dim", type=int, default=29)
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
