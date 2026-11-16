"""Target-aligned trajectory display; preserves the original PCA renderer/API."""
from __future__ import annotations

import copy
import numpy as np
from .trajectory_visualization import FixedProjection, Trajectories, TrajectoryManifoldVisualization


class TargetAlignedProjection:
    """Two training-fitted behavior PCs plus exact displayed-state error height.

    The height is log(1 + ||(x0-target)[angle, angle_dot, cart_dot]/tol||).
    This is a deliberately target-aligned summary, not a three-PC embedding.
    Its zero plane is the image of a state constraint, not a feasible surface.
    """
    indices = np.array([2, 3, 1])

    def __init__(self, training_w, target, state_scale, *, tolerances=(.02, .05, .05)):
        self.behavior = FixedProjection(training_w)
        self.target = np.asarray(target)
        self.state_scale = np.asarray(state_scale)
        self.tolerances = np.asarray(tolerances)
        if self.tolerances.shape != (3,) or np.any(self.tolerances <= 0):
            raise ValueError('three positive SI tolerances required')
        self.method = 'target-aligned'
        self.mean, self.scale, self.components = self.behavior.mean, self.behavior.scale, self.behavior.components
        self.diagnostics = self.behavior.diagnostics
        self.umap = None

    def state_error(self, states):
        error = ((np.asarray(states)-self.target)*self.state_scale)[..., self.indices]
        return np.linalg.norm(error/self.tolerances, axis=-1)

    def transform(self, w):
        w = np.atleast_2d(w)
        return np.column_stack((self.behavior.transform(w)[:, :2],
                                np.log1p(self.state_error(w[:, :4]))))


def target_plane_view(base, training, *, training_indices=None, max_training_points=256, seed=234):
    """Reuse a cached experiment; add actual measured training rows, no decoding.

    Training must already be restricted to the original episode training split.
    All training rows fit the fixed horizontal projection; a reproducible subset
    is drawn. Existing base object and all old source functions remain intact.
    """
    if not np.isfinite(training.w).all():
        raise ValueError('training trajectories must be finite')
    view = copy.copy(base)
    view.__class__ = TargetPlaneVisualization
    view.manifold_style = 'legacy'  # parent left renderer; right panel is replaced
    view.comparisons = {}
    view.projection = TargetAlignedProjection(training.w, base.target, base.scale)
    chosen = np.random.default_rng(seed).choice(len(training.x), min(max_training_points, len(training.x)), replace=False)
    view.training = Trajectories(training.x[chosen], training.u[chosen])
    view.training_indices = chosen if training_indices is None else np.asarray(training_indices)[chosen]
    view.training_xyz = view.projection.transform(view.training.w)
    view._cache_projection()
    view.projected_limits = view._limits(np.concatenate([
        view.training_xyz, view.background_xyz, *view.candidate_xyz,
        np.array([p for p in view.selected_xyz if p is not None]).reshape(-1, 3)]))
    view.projected_limits[2, 0] = -.08
    # Drawing must use cached arrays, never transform or fit.
    view.state_errors = view.projection.state_error(view.states)
    return view


class TargetPlaneVisualization(TrajectoryManifoldVisualization):
    def draw_frame(self, index, axes):
        super().draw_frame(index, axes)
        from matplotlib.colors import Normalize
        from matplotlib.patches import Patch
        axis = axes[1]
        axis.clear()
        axis.view_init(elev=23, azim=-58)
        limits = self.projected_limits
        xx, yy = np.meshgrid(limits[0], limits[1])
        axis.plot_surface(xx, yy, np.zeros_like(xx), color='red', alpha=.14, shade=False)
        axis.scatter(*self.training_xyz.T, s=9, c='dimgray', marker='s', alpha=.42,
                     label='actual training episodes (fixed subset)')
        axis.scatter(*self.background_xyz.T, s=12, facecolors='none', edgecolors='gray',
                     alpha=.25, label='decoded background (synthetic)')
        frame = self.frames[index]
        points = self.candidate_xyz[index]
        finite = np.isfinite(points).all(1)
        valid = frame.feasible & np.isfinite(frame.costs) & finite
        axis.scatter(*points[valid].T, c=frame.costs[valid], cmap='viridis',
                     norm=Normalize(*self.cost_limits), s=22, label='decoded current-state candidates')
        bad = ~frame.feasible & finite
        if bad.any():
            axis.scatter(*points[bad].T, c='orange', marker='x', s=18, label='infeasible decoded candidate')
        point = self.selected_xyz[index]
        if point is not None and np.isfinite(point).all():
            axis.scatter(*point, marker='D' if frame.selected_feasible else 'X', s=110,
                         color='dodgerblue', edgecolors='black', label='selected controller horizon')
            axis.plot([point[0]]*2, [point[1]]*2, [0, point[2]], color='dodgerblue', linestyle=':')
        path = np.asarray([p if p is not None else [np.nan]*3 for p in self.selected_xyz[:index+1]])
        axis.plot(*path.T, color='dodgerblue', alpha=.65, linewidth=1)
        axis.set(xlabel='training behavior PC 1', ylabel='training behavior PC 2',
                 zlabel='log(1 + displayed-state error)', title='Target-aligned trajectory view (not 3-PC PCA)')
        axis.set_xlim(*limits[0]); axis.set_ylim(*limits[1]); axis.set_zlim(*limits[2])
        handles, labels = axis.get_legend_handles_labels()
        axis.legend([Patch(color='red', alpha=.2), *handles],
                    ['target-state constraint (inputs free)', *labels], fontsize=6.3, loc='upper left')
        axis.text2D(0., -.15,
                    f'target matches left: angle, angular speed, cart speed; cart position free\n'
                    f'normalized state error={self.state_errors[frame.step]:.3g}; '
                    f'{valid.sum()}/{len(valid)} decoded candidates feasible',
                    transform=axis.transAxes, fontsize=8)
        return axes
