"""Observational trajectory-manifold visualization; no plant or policy fitting.

All embeddings use STATE-FIRST full behaviors, regardless of decoder packing.
A point is an entire horizon, never a state. Feasibility means decoder membership,
initial-state agreement and input bounds, NOT verified true-plant feasibility.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import warnings

import numpy as np
import torch

from .manifold_control import unpack_w


@dataclass
class Trajectories:
    x: np.ndarray  # (batch, H+1, n), in model state units
    u: np.ndarray  # (batch, H, m), in model input units

    def __post_init__(self):
        self.x = np.asarray(self.x, dtype=float)
        self.u = np.asarray(self.u, dtype=float)
        if (self.x.ndim != 3 or self.u.ndim != 3 or
                self.x.shape[0] != self.u.shape[0] or
                self.x.shape[1] != self.u.shape[1] + 1):
            raise ValueError('expected x=(B,H+1,n), u=(B,H,m)')

    @property
    def w(self):
        return np.concatenate((self.x.reshape(len(self.x), -1),
                               self.u.reshape(len(self.u), -1)), axis=1)

    def feasibility(self, state, bounds, tolerance):
        """Max absolute x0 residual in model units; tolerance is not a plant test."""
        residual = np.max(np.abs(self.x[:, 0] - state), axis=1)
        violation = np.maximum(bounds[0] - self.u, self.u - bounds[1]).clip(0).max(axis=(1, 2))
        finite = np.isfinite(self.w).all(axis=1)
        return finite & (residual <= tolerance) & (violation <= tolerance), residual, violation


class DecoderSampler:
    """Canonical exact conditioning or latent initial-state feasibility search.

    For legacy decoders pass the existing latent solver (for denormalization)
    and latent samples from its encoder. No simulator calls are made. Search
    changes only temporary latent coordinates, not decoder weights or gradients.
    """
    def __init__(self, decoder, horizon, *, x_dim=4, u_dim=1, bounds=(-10., 10.),
                 latent_solver=None, latent_samples=None, seed=234):
        if horizon < 1 or bounds[0] >= bounds[1]:
            raise ValueError('invalid horizon or bounds')
        self.decoder, self.horizon = decoder, horizon
        self.x_dim, self.u_dim, self.bounds = x_dim, u_dim, bounds
        self.latent_solver = latent_solver
        self.latent_samples = latent_samples
        self.rng = np.random.default_rng(seed)
        parameter = next(decoder.parameters())
        self.device, self.dtype = parameter.device, parameter.dtype
        self.canonical = hasattr(decoder, 'coordinates') and hasattr(decoder, 'rollout')
        if not self.canonical and (latent_solver is None or latent_samples is None):
            raise ValueError('latent decoder requires its solver and encoded latent_samples')

    def tensor(self, value):
        return torch.as_tensor(value, device=self.device, dtype=self.dtype)

    def _latent_decode(self, z):
        w = self.latent_solver.denormalize_w(self.decoder(z))
        return unpack_w(w, x_dim=self.x_dim, u_dim=self.u_dim, horizon=self.horizon)

    def sample(self, count, *, state=None, state_bounds=None, search_steps=80):
        if count < 1:
            raise ValueError('count must be positive')
        if self.canonical:
            if state is None:
                if state_bounds is None:
                    raise ValueError('background requires explicit model-unit state_bounds')
                x0 = self.rng.uniform(*np.asarray(state_bounds), size=(count, self.x_dim))
            else:
                x0 = np.broadcast_to(state, (count, self.x_dim)).copy()
            # Bounded canonical free-input coordinates: exact current-state slice.
            u = self.tensor(self.rng.uniform(*self.bounds, (count, self.horizon, self.u_dim)))
            with torch.no_grad():
                x = self.decoder.rollout(self.tensor(x0), u)
        else:
            pool = np.asarray(self.latent_samples)
            z = self.tensor(pool[self.rng.integers(len(pool), size=count)]).clone()
            if state is not None:
                z.requires_grad_(True)
                optimizer = torch.optim.Adam([z], lr=.03)
                for _ in range(search_steps):
                    optimizer.zero_grad(set_to_none=True)
                    x, u = self._latent_decode(z)
                    loss = ((x[:, 0] - self.tensor(state)) ** 2).sum()
                    loss = loss + (torch.relu(self.bounds[0]-u)**2 +
                                   torch.relu(u-self.bounds[1])**2).sum()
                    gradient, = torch.autograd.grad(loss, z)
                    if not torch.isfinite(gradient).all():
                        break
                    z.grad = gradient
                    optimizer.step()
            with torch.no_grad():
                x, u = self._latent_decode(z)
        return Trajectories(x.detach().cpu().numpy(), u.detach().cpu().numpy())


def _neighborhood_score(high, low, k=8):
    """Mean k-neighbor overlap, [0,1]; not a curvature/isometry measure."""
    k = min(k, len(high)-1)
    if k < 1:
        return 1.
    def neighbors(a):
        distance = ((a[:, None] - a[None])**2).sum(-1)
        np.fill_diagonal(distance, np.inf)
        return np.argsort(distance, axis=1)[:, :k]
    a, b = neighbors(high), neighbors(low)
    return float(np.mean([len(set(x) & set(y))/k for x, y in zip(a, b)]))


class FixedProjection:
    """Fit once on a fixed background; out-of-sample transform for every frame.

    Per-feature standardization and PCA use NumPy, without a sklearn dependency.
    UMAP is optional and rejected if its neighborhood or continuity diagnostics
    deteriorate relative to PCA. Diagnostics are empirical, not geometric proofs.
    """
    def __init__(self, background, *, method='pca', seed=234, continuity=None):
        w = np.asarray(background, dtype=float)
        if w.ndim != 2 or min(w.shape) < 3 or not np.isfinite(w).all():
            raise ValueError('projection needs at least 3 finite full trajectories')
        if method not in ('pca', 'umap', 'compare'):
            raise ValueError('method must be pca, umap or compare')
        self.mean = w.mean(0)
        self.scale = w.std(0)
        self.scale[self.scale < 1e-8] = 1.
        z = (w-self.mean)/self.scale
        _, singular, vt = np.linalg.svd(z, full_matrices=False)
        self.components = vt[:3].T
        self.method, self.umap = 'pca', None
        self.diagnostics = {'pca': {'variance_fraction': float((singular[:3]**2).sum()/max((singular**2).sum(), 1e-12))}}
        # Bound diagnostics cost independently of cloud size.
        indices = np.random.default_rng(seed).choice(len(z), min(len(z), 256), replace=False)
        diagnostic_z = z[indices]
        pca = z @ self.components
        def metrics(embedding, path):
            result = {'neighbor_overlap': _neighborhood_score(diagnostic_z, embedding[indices])}
            if continuity is not None and len(continuity) > 1:
                high = np.diff((continuity-self.mean)/self.scale, axis=0)
                low = np.diff(path, axis=0)
                # Relative to background RMS radius: global scale invariant.
                hr = np.sqrt(np.mean(np.sum(z*z, axis=1))) or 1.
                lr = np.sqrt(np.mean(np.sum((embedding-embedding.mean(0))**2, axis=1))) or 1.
                hstep = np.linalg.norm(high, axis=1)/hr
                lstep = np.linalg.norm(low, axis=1)/lr
                result['temporal_step_distortion'] = float(np.median(np.abs(lstep-hstep)/np.maximum(hstep, 1e-8)))
                result['temporal_max_step'] = float(lstep.max())
            return result
        pca_path = None if continuity is None else ((continuity-self.mean)/self.scale) @ self.components
        self.diagnostics['pca'].update(metrics(pca, pca_path))
        if method != 'pca':
            try:
                import umap
                reducer = umap.UMAP(n_components=3, random_state=seed, transform_seed=seed,
                                    n_neighbors=min(15, len(z)-1))
                embedding = reducer.fit_transform(z)
                path = None if continuity is None else reducer.transform((continuity-self.mean)/self.scale)
                if not np.isfinite(embedding).all() or (path is not None and not np.isfinite(path).all()):
                    raise ValueError('nonfinite UMAP embedding')
                score = metrics(embedding, path)
                self.diagnostics['umap'] = score
                reference = self.diagnostics['pca']
                reliable = score['neighbor_overlap'] >= reference['neighbor_overlap']
                if 'temporal_step_distortion' in reference:
                    reliable &= score['temporal_step_distortion'] <= reference['temporal_step_distortion'] + .1
                if reliable:
                    self.method, self.umap = 'umap', reducer
                else:
                    self.diagnostics['fallback'] = 'UMAP worse than PCA on empirical diagnostics'
            except (ImportError, ValueError, RuntimeError) as error:
                self.diagnostics['fallback'] = f'UMAP unavailable/unreliable: {error}'
            if 'fallback' in self.diagnostics:
                warnings.warn(self.diagnostics['fallback'], RuntimeWarning)

    def transform(self, w):
        z = (np.asarray(w)-self.mean)/self.scale
        return z @ self.components if self.umap is None else self.umap.transform(z)


@dataclass
class VisualizationFrame:
    step: int
    candidates: Trajectories
    selected: Trajectories | None
    costs: np.ndarray
    feasible: np.ndarray
    initial_residual: np.ndarray
    bound_violation: np.ndarray
    selected_feasible: bool = False
    selected_residual: float = float('nan')


class TrajectoryManifoldVisualization:
    """Prepare once, then render cached frames with a fixed projection and camera.

    objective(x_batch,u_batch,step) must call the *existing* MPC objective;
    selected maps measured sample indices to actual controller solution objects.
    Missing solutions (e.g. queued actions) are left absent, never fabricated.
    """
    def __init__(self, sampler, t, states, inputs, selected, objective, *,
                 state_bounds=None, target=None, background_samples=400,
                 candidates_per_frame=48, feasibility_tolerance=1e-4,
                 method='pca', frame_stride=1, state_scale=None, force_scale=1.,
                 seed=234, selected_path=True):
        self.sampler = sampler
        self.t, self.states, self.inputs = map(np.asarray, (t, states, inputs))
        if (self.states.shape != (len(t), 4) or self.inputs.shape != (len(t)-1, 1)
                or not np.isfinite(self.states).all() or not np.isfinite(self.inputs).all()
                or not np.isfinite(self.t).all() or np.any(np.diff(self.t) <= 0)):
            raise ValueError('expected finite increasing t, states=(T+1,4), inputs=(T,1)')
        if frame_stride < 1 or feasibility_tolerance < 0 or len(self.inputs) < 1:
            raise ValueError('invalid stride, tolerance or empty run')
        self.scale = np.ones(4) if state_scale is None else np.asarray(state_scale)
        if self.scale.shape != (4,) or np.any(self.scale <= 0):
            raise ValueError('state_scale must contain four positive SI conversion factors')
        self.force_scale = force_scale
        self.target = np.zeros(4) if target is None else np.asarray(target)
        self.selected_path = selected_path
        background = sampler.sample(background_samples, state_bounds=state_bounds)
        finite = np.isfinite(background.w).all(1)
        bounded = ((background.u >= sampler.bounds[0]) & (background.u <= sampler.bounds[1])).all((1, 2))
        self.background = Trajectories(background.x[finite & bounded], background.u[finite & bounded])
        self.background_rejected = int((~(finite & bounded)).sum())
        self.frames = []
        steps = sorted(set(range(0, len(states), frame_stride)) | {len(states)-1})
        for step in steps:
            candidates = sampler.sample(candidates_per_frame, state=self.states[step])
            feasible, residual, violation = candidates.feasibility(self.states[step], sampler.bounds, feasibility_tolerance)
            with torch.no_grad():
                costs = objective(sampler.tensor(candidates.x), sampler.tensor(candidates.u), step)
            if torch.is_tensor(costs):
                costs = costs.detach().cpu().numpy()
            costs = np.asarray(costs)
            if costs.shape != (candidates_per_frame,):
                raise ValueError('objective must return one MPC cost per candidate')
            plan = selected.get(step)
            selected_batch, accepted, selected_residual = None, False, float('nan')
            if plan is not None:
                def array(a):
                    return a.detach().cpu().numpy() if torch.is_tensor(a) else np.asarray(a)
                selected_batch = Trajectories(array(plan.x)[None], array(plan.u)[None])
                if selected_batch.x.shape[1:] != (sampler.horizon+1, 4):
                    raise ValueError('selected horizon differs from decoder horizon')
                mask, r, _ = selected_batch.feasibility(self.states[step], sampler.bounds, feasibility_tolerance)
                accepted, selected_residual = bool(mask[0]), float(r[0])
            self.frames.append(VisualizationFrame(step, candidates, selected_batch, costs,
                                                  feasible, residual, violation, accepted, selected_residual))
        path = [f.selected.w[0] for f in self.frames if f.selected is not None and np.isfinite(f.selected.w).all()]
        continuity = np.asarray(path) if path else None
        self.projection = FixedProjection(self.background.w, method=method, seed=seed, continuity=continuity)
        self.background_xyz = self.projection.transform(self.background.w)
        self.candidate_xyz, self.selected_xyz = [], []
        for frame in self.frames:
            def project_finite(w):
                result = np.full((len(w), 3), np.nan)
                valid = np.isfinite(w).all(1)
                if valid.any():
                    result[valid] = self.projection.transform(w[valid])
                return result
            self.candidate_xyz.append(project_finite(frame.candidates.w))
            self.selected_xyz.append(None if frame.selected is None else project_finite(frame.selected.w)[0])
        all_xyz = np.concatenate([self.background_xyz, *self.candidate_xyz,
                                  np.asarray([p for p in self.selected_xyz if p is not None]).reshape(-1, 3)])
        self.projected_limits = self._limits(all_xyz)
        physical = [self.states*self.scale, (self.target*self.scale)[None]]
        physical += [f.selected.x[0]*self.scale for f in self.frames if f.selected is not None]
        self.physical_limits = self._limits(np.concatenate(physical)[:, [0, 2, 3]])
        valid_costs = np.concatenate([f.costs[f.feasible & np.isfinite(f.costs)] for f in self.frames])
        self.cost_limits = (float(valid_costs.min()), float(valid_costs.max())) if len(valid_costs) else (0., 1.)

    @staticmethod
    def _limits(points):
        finite = points[np.isfinite(points).all(1)]
        low, high = finite.min(0), finite.max(0)
        pad = np.maximum(.08*(high-low), .05)
        return np.stack((low-pad, high+pad), axis=1)

    def figure(self):
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(13, 6))
        axes = (fig.add_subplot(121, projection='3d'), fig.add_subplot(122, projection='3d'))
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize
        mappable = ScalarMappable(norm=Normalize(*self.cost_limits), cmap='viridis')
        fig.colorbar(mappable, ax=axes[1], shrink=.5, pad=.1, label='MPC objective (feasible candidates)')
        fig.subplots_adjust(left=.02, right=.9, bottom=.15, top=.88, wspace=.25)
        return fig, axes

    def draw_frame(self, index, axes):
        """Pure cached rendering: no decoder, optimizer, fit or transform calls."""
        from matplotlib.colors import Normalize
        frame = self.frames[index]
        physical, manifold = axes
        for axis in axes:
            axis.clear()
            axis.view_init(elev=23, azim=-58)
        k = frame.step
        x = self.states*self.scale
        physical.plot(*x[:k+1, [0, 2, 3]].T, color='tab:blue', label='realized')
        physical.scatter(*x[k, [0, 2, 3]], color='tab:blue', s=55, label='measured current')
        physical.scatter(*(self.target*self.scale)[[0, 2, 3]], marker='*', s=110, color='green', label='target')
        if frame.selected is not None:
            plan = frame.selected.x[0]*self.scale
            physical.plot(*plan[:, [0, 2, 3]].T, color='tab:red', linestyle='--', label='controller plan')
        physical.set(xlabel='cart [m]', ylabel='unwrapped angle [rad]', zlabel='angular speed [rad/s]', title='Physical state space (3 of 4 coordinates)')
        physical.legend(fontsize=7, loc='upper left')
        manifold.scatter(*self.background_xyz.T, s=5, c='gray', alpha=.25, label='fixed decoded background')
        points = self.candidate_xyz[index]
        valid = frame.feasible & np.isfinite(frame.costs) & np.isfinite(points).all(1)
        if valid.any():
            manifold.scatter(*points[valid].T, c=frame.costs[valid], cmap='viridis',
                             norm=Normalize(*self.cost_limits), s=22, label='current-state feasible slice')
        bad = ~frame.feasible & np.isfinite(points).all(1)
        if bad.any():
            manifold.scatter(*points[bad].T, c='orange', marker='x', s=18, label='approximate/infeasible')
        point = self.selected_xyz[index]
        if point is not None and np.isfinite(point).all():
            manifold.scatter(*point, marker='*' if frame.selected_feasible else 'X',
                             s=170, c='red', label='selected controller horizon')
        if self.selected_path:
            path = np.asarray([p if p is not None else [np.nan]*3 for p in self.selected_xyz[:index+1]])
            manifold.plot(*path.T, color='tab:red', alpha=.6, linewidth=1)
        manifold.set(xlabel='projection 1', ylabel='projection 2', zlabel='projection 3',
                     title=f'Full-trajectory {self.projection.method.upper()} (not exact geometry)')
        manifold.legend(fontsize=7, loc='upper left')
        for axis, limits in zip(axes, (self.physical_limits, self.projected_limits)):
            axis.set_xlim(*limits[0]); axis.set_ylim(*limits[1]); axis.set_zlim(*limits[2])
        selected_u = 'unavailable' if frame.selected is None else f'{frame.selected.u[0, 0, 0]*self.force_scale:.3f} N'
        status = 'feasible' if frame.selected_feasible else 'approximate/infeasible or absent'
        applied = f'{self.inputs[k,0]*self.force_scale:.3f} N' if k < len(self.inputs) else 'none (run complete)'
        axes[0].figure.suptitle(f't={self.t[k]:.2f} s | selected u*={selected_u} | applied F={applied}')
        manifold.text2D(0., -.15, f'{valid.sum()}/{len(valid)} feasible; selected {status}\nx0 residual={frame.selected_residual:.2g}; cart speed={x[k,1]:.3f} m/s', transform=manifold.transAxes, fontsize=8)
        return axes

    def save_static(self, path, *, step=0):
        """Selected closed-loop step must have been included in frame preparation."""
        indices = [i for i, frame in enumerate(self.frames) if frame.step == step]
        if not indices:
            raise ValueError('step not cached; prepare with frame_stride=1 or choose a cached step')
        fig, axes = self.figure()
        self.draw_frame(indices[0], axes)
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=130)
        return fig

    def animation(self, *, fps=15):
        from matplotlib.animation import FuncAnimation
        if fps <= 0:
            raise ValueError('fps must be positive')
        fig, axes = self.figure()
        animation = FuncAnimation(fig, lambda i: self.draw_frame(i, axes),
                                  frames=len(self.frames), interval=1000/fps, blit=False,
                                  repeat=False, cache_frame_data=False)
        return animation

    def save_animation(self, path, *, fps=15):
        from matplotlib.animation import PillowWriter, FFMpegWriter, writers
        path = Path(path)
        if path.suffix.lower() not in ('.gif', '.mp4'):
            raise ValueError('animation output must be .gif or .mp4')
        if path.suffix.lower() == '.mp4' and not writers.is_available('ffmpeg'):
            raise RuntimeError('MP4 requires ffmpeg; use GIF with existing Pillow instead')
        path.parent.mkdir(parents=True, exist_ok=True)
        animation = self.animation(fps=fps)
        writer = PillowWriter(fps=fps) if path.suffix.lower() == '.gif' else FFMpegWriter(fps=fps)
        animation.save(path, writer=writer, dpi=90)
        return animation
