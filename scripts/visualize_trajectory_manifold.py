"""Small visualization experiment; reuses learned solvers and existing plant.

CLI: python -m scripts.visualize_trajectory_manifold --checkpoint PATH
No model training or validation-score selection is performed here.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from src.trajectory_visualization import DecoderSampler, TrajectoryManifoldVisualization


def notebook_visualization(namespace, *, checkpoint=None, seconds=.6, samples=256,
                           candidates=24, tolerance=1e-4, method='pca', seed=234,
                           frame_stride=1):
    """Reuse notebook model/config when present, otherwise explicit checkpoint.

    Returns visualization and metadata. The live latent path uses the notebook's
    existing solver, not its optional LQR fallback; feasibility failures remain
    visible. Both paths run a NEW short visualization-only experiment, leaving
    existing notebook experiment arrays and diagnostics alone.
    """
    from scripts import run_inverted_pendulum_mc as runner
    from src.manifold_control import CanonicalBehaviorDecoder
    from src.manifold_experiments import NativeConfig, native_controller

    decoder = namespace.get('decoder')
    latent = decoder is not None and not hasattr(decoder, 'coordinates')
    if decoder is None:
        if checkpoint is None:
            raise ValueError('run the notebook model cells or explicitly supply a canonical checkpoint')
        transition, payload = runner.load_transition(Path(checkpoint), 'cpu')
        cfg = runner.make_config('dev', **payload['config'])
        decoder = CanonicalBehaviorDecoder(transition, cfg.horizon).eval()
        model_source = str(Path(checkpoint).resolve())
    else:
        cfg = namespace.get('cfg', namespace.get('config'))
        if cfg is None and 'summary' in namespace:
            cfg = namespace['summary'].get('config')
        if isinstance(cfg, dict):
            cfg = runner.make_config('dev', **cfg)
        model_source = 'live notebook decoder (not retrained)'
    chosen = {}
    if latent:
        original = namespace.get('single_step_solver')
        if original is None:
            raise ValueError('live latent decoder requires notebook single_step_solver')
        # Shallow copy isolates solver bookkeeping; model is read-only and shared.
        solver = copy.copy(original)
        if solver.A is not None or solver.B is not None:
            raise ValueError('visualization example does not support linear dynamics penalties')
        horizon = solver.horizon
        scale = np.asarray(namespace['state_scale'])
        force_scale, dt = float(namespace['mg']), float(namespace['control_dt'])
        dt_norm, mass_ratio = float(namespace['dt_norm']), float(namespace['M_ratio'])
        initial = np.asarray(namespace['y0_normalized'])
        bounds = solver.u_bounds
        if bounds is None:
            raise ValueError('example requires explicit input bounds')
        training = namespace.get('W_train')
        encoder = namespace.get('encoder')
        if training is None or encoder is None:
            raise ValueError('latent background requires W_train and the trained encoder')
        with torch.no_grad():
            pool = encoder(training[:min(len(training), 512)]).detach().cpu().numpy()
        sampler = DecoderSampler(decoder, horizon, bounds=bounds, latent_solver=solver,
                                 latent_samples=pool, seed=seed)
        previous = None
        def control(k, state):
            nonlocal previous
            solution = solver.solve(sampler.tensor(state), alpha_init=previous)
            if not solution.finite:
                raise RuntimeError('nonfinite latent plan; no hidden controller fallback')
            previous = solution.alpha.detach().clone()
            chosen[k] = solution
            return solution.u[0].detach().cpu().numpy()
        def objective(x, u, step):
            ex, eu = x[:, :-1]-solver.x_ref, u-solver.u_ref
            terminal = x[:, -1]-solver.x_ref
            return (torch.einsum('bki,ij,bkj->b', ex, solver.Q, ex) +
                    torch.einsum('bki,ij,bkj->b', eu, solver.R, eu) +
                    torch.einsum('bi,ij,bj->b', terminal, solver.Q_terminal, terminal))
        t_norm, states, inputs = runner.simulate_discrete_inverted_pendulum(
            control, mass_ratio, initial, dt_norm, round(seconds/dt), umax=max(abs(v) for v in bounds))
        states, inputs = states.T, inputs.T
        t = np.arange(len(states))*dt
        target = solver.x_ref.detach().cpu().numpy()
        config_record = {'solver': 'existing latent solve', 'objective_colors': 'existing tracking + terminal, excluding augmented feasibility/latent regularization'}
    else:
        if cfg is None:
            raise ValueError('live canonical decoder requires notebook cfg or config')
        # NativeSolver freezes parameter flags; isolate this visualization run
        # from any live notebook model that may later be used for training.
        decoder = copy.deepcopy(decoder)
        bounds = (-cfg.umax, cfg.umax)
        sampler = DecoderSampler(decoder, decoder.horizon, bounds=bounds, seed=seed)
        native_cfg = NativeConfig(solver='adam', iterations=20, replan_interval=1, seed=seed)
        policy = native_controller(decoder, native_cfg)
        solve = policy.solver.solve
        current_step = 0
        def record(*args, **kwargs):
            solution = solve(*args, **kwargs)
            if solution.finite:
                chosen[current_step] = solution
            return solution
        def control(k, state):
            nonlocal current_step
            current_step = k
            return policy(k, state)
        policy.solver.solve = record
        initial = np.asarray(namespace.get('y0_normalized', [0., 0., .15, 0.]))
        try:
            t, states, inputs = runner.simulate_plant(control, initial, cfg, seconds=seconds)
        finally:
            policy.solver.solve = solve
        from src.inverted_pendulum import _physical_state_scale
        _, scale, force_scale = _physical_state_scale(1., 9.81, 1.)
        target = np.zeros(4)
        def objective(x, u, step):
            previous = sampler.tensor(inputs[step-1]) if step else sampler.tensor([0.])
            return policy.solver.objective(x, u, previous)
        config_record = {'solver': asdict(native_cfg), 'simulation': asdict(cfg),
                         'fallback_events': policy.diagnostics['fallback_events']}
    state_bounds = np.array([[-.5, -.8, -.6, -1.], [.5, .8, .6, 1.]])
    visualization = TrajectoryManifoldVisualization(
        sampler, t, states, inputs, chosen, objective, state_bounds=state_bounds,
        target=target, background_samples=samples, candidates_per_frame=candidates,
        feasibility_tolerance=tolerance, method=method, frame_stride=frame_stride,
        state_scale=scale, force_scale=force_scale, seed=seed)
    metadata = {'seed': seed, 'model_source': model_source, 'config': config_record,
                'seconds': seconds, 'background_samples': samples, 'candidates': candidates,
                'feasibility_tolerance': tolerance, 'frame_stride': frame_stride,
                'projection': visualization.projection.method,
                'projection_diagnostics': visualization.projection.diagnostics,
                'background_rejected': visualization.background_rejected,
                'frames': len(visualization.frames),
                'state_scale_to_SI': np.asarray(scale).tolist(), 'force_scale_to_N': force_scale,
                'warning': 'Visualization experiment only; not benchmark evidence. Decoder feasibility is not true-plant feasibility.'}
    if model_source != 'live notebook decoder (not retrained)':
        metadata['checkpoint_sha256'] = hashlib.sha256(Path(model_source).read_bytes()).hexdigest()
    return visualization, metadata


def export(visualization, metadata, output, *, fps=15, movie_format='gif'):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    visualization.save_static(output/'static.png', step=visualization.frames[len(visualization.frames)//2].step)
    animation = visualization.save_animation(output/f'animation.{movie_format}', fps=fps)
    np.savez_compressed(output/'closed_loop.npz', t=visualization.t, X=visualization.states, U=visualization.inputs)
    # Full-horizon cache preserves what each point and selected plan represented.
    np.savez_compressed(output/'trajectories.npz',
        background_w=visualization.background.w,
        background_xyz=visualization.background_xyz,
        steps=[f.step for f in visualization.frames],
        candidates_w=np.stack([f.candidates.w for f in visualization.frames]),
        candidates_xyz=np.stack(visualization.candidate_xyz),
        costs=np.stack([f.costs for f in visualization.frames]),
        feasible=np.stack([f.feasible for f in visualization.frames]),
        initial_residual=np.stack([f.initial_residual for f in visualization.frames]),
        bound_violation=np.stack([f.bound_violation for f in visualization.frames]),
        selected_w=np.stack([f.selected.w[0] if f.selected is not None else
                             np.full(visualization.background.w.shape[1], np.nan)
                             for f in visualization.frames]),
        mean=visualization.projection.mean, scale=visualization.projection.scale,
        pca_components=visualization.projection.components)
    metadata = {**metadata, 'fps': fps, 'movie_format': movie_format}
    (output/'metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    return animation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('saves/visualizations/trajectory_manifold'))
    parser.add_argument('--seconds', type=float, default=.6)
    parser.add_argument('--samples', type=int, default=256)
    parser.add_argument('--candidates', type=int, default=24)
    parser.add_argument('--projection', choices=['pca', 'umap', 'compare'], default='pca')
    parser.add_argument('--fps', type=int, default=15)
    parser.add_argument('--format', choices=['gif', 'mp4'], default='gif')
    args = parser.parse_args()
    torch.set_num_threads(1)
    visualization, metadata = notebook_visualization({}, checkpoint=args.checkpoint, seconds=args.seconds,
        samples=args.samples, candidates=args.candidates, method=args.projection)
    export(visualization, metadata, args.output, fps=args.fps, movie_format=args.format)
    print(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()
