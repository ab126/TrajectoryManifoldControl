from types import SimpleNamespace

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch

from src.manifold_control import CanonicalBehaviorDecoder, DataDrivenTransition
from src.trajectory_visualization import (Trajectories, DecoderSampler,
    FixedProjection, TrajectoryManifoldVisualization)


class SimpleTransition(DataDrivenTransition):
    def __init__(self):
        super().__init__(4, 1, ())

    def forward(self, x, u):
        return x + .01*u


def sampler(seed=1):
    return DecoderSampler(CanonicalBehaviorDecoder(SimpleTransition(), 3), 3, seed=seed)


def test_dimensions_and_canonical_packing():
    source = sampler()
    batch = source.sample(8, state=np.arange(4))
    assert batch.x.shape == (8, 4, 4)
    assert batch.u.shape == (8, 3, 1)
    assert batch.w.shape == (8, 19)
    np.testing.assert_allclose(batch.w[:, :4], batch.x[:, 0])
    with pytest.raises(ValueError):
        Trajectories(np.zeros((2, 4, 4)), np.zeros((2, 4, 1)))


def test_conditioning_and_determinism():
    state = np.array([.1, .2, .3, .4])
    a, b = sampler().sample(8, state=state), sampler().sample(8, state=state)
    np.testing.assert_array_equal(a.w, b.w)
    feasible, residual, violation = a.feasibility(state, (-10, 10), 1e-6)
    assert feasible.all() and residual.max() < 1e-6 and violation.max() == 0
    assert not a.feasibility(state+1, (-10, 10), 1e-6)[0].any()
    a.u[0, 0] = 11
    assert not a.feasibility(state, (-10, 10), 1e-6)[0][0]


def test_latent_feasibility_search_does_not_change_model():
    # A state-first latent decoder with a known, nontrivial x0 slice.
    decoder = torch.nn.Linear(4, 19, bias=False)
    with torch.no_grad():
        decoder.weight.zero_()
        for i in range(4):
            decoder.weight[i*4:(i+1)*4] = torch.eye(4)
    adapter = SimpleNamespace(denormalize_w=lambda w: w)
    source = DecoderSampler(decoder, 3, latent_solver=adapter,
                            latent_samples=np.zeros((8, 4)), seed=0)
    before = decoder.weight.detach().clone()
    batch = source.sample(8, state=np.ones(4)*.2, search_steps=120)
    assert batch.feasibility(np.ones(4)*.2, (-10, 10), .002)[0].all()
    torch.testing.assert_close(before, decoder.weight)
    assert decoder.weight.grad is None


def test_fixed_projection_is_consistent_and_does_not_refit():
    w = np.random.default_rng(0).normal(size=(30, 19))
    projection = FixedProjection(w, continuity=w[:6])
    mean, scale, basis = projection.mean.copy(), projection.scale.copy(), projection.components.copy()
    np.testing.assert_allclose(projection.transform(w[:3]), projection.transform(w)[:3])
    np.testing.assert_array_equal(mean, projection.mean)
    np.testing.assert_array_equal(scale, projection.scale)
    np.testing.assert_array_equal(basis, projection.components)
    assert 0 <= projection.diagnostics['pca']['neighbor_overlap'] <= 1
    assert np.isfinite(projection.diagnostics['pca']['temporal_step_distortion'])


def test_continuity_diagnostic_is_dimension_independent_on_exact_embedding():
    # Rank-three behaviors have an exact 3D PCA embedding, even in 19D.
    rng = np.random.default_rng(0)
    w = rng.normal(size=(30, 3)) @ rng.normal(size=(3, 19))
    projection = FixedProjection(w, continuity=w[:6])
    assert projection.diagnostics['pca']['temporal_step_distortion'] < 1e-12


def test_umap_unavailable_falls_back(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, 'umap', None)
    with pytest.warns(RuntimeWarning, match='UMAP unavailable'):
        projection = FixedProjection(np.random.default_rng(0).normal(size=(20, 19)), method='compare')
    assert projection.method == 'pca'


def make_visualization():
    source = sampler()
    states = np.zeros((4, 4)); inputs = np.zeros((3, 1))
    batch = source.sample(1, state=states[0])
    selected = {0: SimpleNamespace(x=batch.x[0], u=batch.u[0])}
    return TrajectoryManifoldVisualization(source, np.arange(4)*.02, states, inputs,
        selected, lambda x, u, k: x.square().sum((1, 2))+u.square().sum((1, 2)),
        state_bounds=[[-1]*4, [1]*4], background_samples=20, candidates_per_frame=5)


def test_render_is_cached_fixed_camera_and_exports_gif(tmp_path, monkeypatch):
    viz = make_visualization()
    def forbidden(*args, **kwargs):
        raise AssertionError('rendering must not decode, sample or project')
    monkeypatch.setattr(viz.sampler, 'sample', forbidden)
    monkeypatch.setattr(viz.sampler.decoder, 'rollout', forbidden)
    monkeypatch.setattr(viz.projection, 'transform', forbidden)
    fig, axes = viz.figure()
    limits = []
    for index in (0, 1, 2, 3, 0):
        viz.draw_frame(index, axes)
        fig.canvas.draw()
        limits.append((axes[1].get_xlim(), axes[1].get_ylim(), axes[1].azim))
    assert limits.count(limits[0]) == len(limits)
    assert len(viz.frames) == 4
    assert viz.frames[-1].step == len(viz.states)-1
    assert viz.frames[0].selected_feasible
    assert viz.frames[1].selected is None
    viz.save_static(tmp_path/'frame.png')
    viz.save_animation(tmp_path/'run.gif', fps=5)
    from PIL import Image
    with Image.open(tmp_path/'run.gif') as gif:
        assert gif.n_frames == 4
    assert (tmp_path/'frame.png').stat().st_size > 0
    plt.close('all')


def test_bad_trace_and_uncached_step_rejected():
    viz = make_visualization()
    with pytest.raises(ValueError, match='not cached'):
        viz.save_static('/tmp/not-created.png', step=999)
    with pytest.raises(ValueError, match='fps'):
        viz.animation(fps=0)


def test_notebook_live_model_is_not_mutated():
    from dataclasses import asdict
    from scripts.visualize_trajectory_manifold import notebook_visualization
    from scripts.run_inverted_pendulum_mc import make_config
    torch.set_num_threads(1)
    decoder = CanonicalBehaviorDecoder(SimpleTransition(), 3)
    before = {key: value.clone() for key, value in decoder.state_dict().items()}
    flags = [p.requires_grad for p in decoder.parameters()]
    cfg = make_config('smoke', horizon=3, device='cpu')
    viz, metadata = notebook_visualization({'decoder': decoder, 'summary': {'config': asdict(cfg)}},
        seconds=.04, samples=12, candidates=3)
    assert len(viz.frames) == 3  # initial, second action, final measured state
    assert metadata['model_source'].startswith('live notebook')
    assert flags == [p.requires_grad for p in decoder.parameters()]
    for key, value in decoder.state_dict().items():
        torch.testing.assert_close(before[key], value)


def test_mp4_dependency_error_is_clear(monkeypatch, tmp_path):
    from matplotlib.animation import writers
    monkeypatch.setattr(writers, 'is_available', lambda writer: False)
    with pytest.raises(RuntimeError, match='ffmpeg'):
        make_visualization().save_animation(tmp_path/'run.mp4')


@pytest.mark.parametrize('notebook', ['manifold_control.ipynb', 'inverted_pendulum_mc_debug.ipynb'])
def test_notebook_cells_refresh_stale_imports_and_export(notebook, tmp_path):
    """An existing kernel may hold the class from before comparisons existed."""
    import os
    from pathlib import Path
    import subprocess
    import sys
    import textwrap

    script = textwrap.dedent('''
        import json
        from pathlib import Path
        import torch
        import src.trajectory_visualization as module
        import scripts.visualize_trajectory_manifold as driver
        from src.manifold_control import CanonicalBehaviorDecoder, DataDrivenTransition
        from scripts.run_inverted_pendulum_mc import make_config

        class StaleVisualization(module.TrajectoryManifoldVisualization):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                del self.comparisons

        module.TrajectoryManifoldVisualization = StaleVisualization
        driver.TrajectoryManifoldVisualization = StaleVisualization
        cells = [c for c in json.loads(Path(NOTEBOOK).read_text())['cells']
                 if c['cell_type'] == 'code' and c.get('id', '').startswith('tmc-viz-')]
        assert len(cells) == 3
        namespace = {'decoder': CanonicalBehaviorDecoder(DataDrivenTransition(4, 1, ()), 3),
                     'cfg': make_config('smoke', horizon=3, device='cpu')}
        exec(''.join(cells[0]['source']), namespace)
        namespace.update(VIZ_SECONDS=.04, VIZ_SAMPLES=12, VIZ_CANDIDATES=3,
                         VIZ_PROJECTION='pca',
                         VIZ_OUTPUT=Path(OUTPUT), display=lambda *args: None)
        for cell in cells[1:]:
            exec(''.join(cell['source']), namespace)
        assert namespace['viz'].comparisons == {}
        for filename in ('static.png', 'animation.gif', 'metadata.json',
                         'closed_loop.npz', 'trajectories.npz'):
            assert (Path(OUTPUT)/filename).stat().st_size > 0
    ''')
    script = 'NOTEBOOK = {!r}\nOUTPUT = {!r}\n'.format(notebook, str(tmp_path)) + script
    result = subprocess.run([sys.executable, '-c', script],
                            cwd=Path(__file__).resolve().parents[1],
                            env={**os.environ, 'MPLBACKEND': 'Agg'},
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize('method', ['umap', 'compare'])
def test_umap_quality_warning_does_not_hide_explicit_view(monkeypatch, tmp_path, method):
    import sys
    from scripts.visualize_trajectory_manifold import notebook_visualization, export
    from scripts.run_inverted_pendulum_mc import make_config

    class DistortedUMAP:
        def __init__(self, **kwargs):
            pass

        def fit_transform(self, z):
            return self.transform(z)

        def transform(self, z):
            return np.column_stack((np.sin(z[:, 0]*100), np.cos(z[:, 1]*100), z[:, 2]))

    monkeypatch.setitem(sys.modules, 'umap', SimpleNamespace(UMAP=DistortedUMAP))
    monkeypatch.setattr('src.trajectory_visualization._neighborhood_score',
                        lambda high, low: 1. if np.max(np.abs(low[:, 0])) > 1 else 0.)
    torch.set_num_threads(1)
    with pytest.warns(RuntimeWarning, match='UMAP worse'):
        viz, metadata = notebook_visualization(
            {'decoder': CanonicalBehaviorDecoder(SimpleTransition(), 3),
             'cfg': make_config('smoke', horizon=3, device='cpu')},
            method=method, seconds=.04, samples=20, candidates=3)
    assert metadata['projection_requested'] == method
    assert viz.projection.method == ('umap' if method == 'umap' else 'pca')
    views = viz.comparisons if method == 'compare' else {'umap': viz}
    assert set(views) == ({'pca', 'umap'} if method == 'compare' else {'umap'})
    for name, view in views.items():
        assert view.frames is viz.frames
        fig, axes = view.figure()
        view.draw_frame(0, axes)
        assert name.upper() in axes[1].get_title()
        plt.close(fig)
    export(viz, metadata, tmp_path, fps=5)
    if method == 'compare':
        import json
        for name in views:
            assert (tmp_path/name/'animation.gif').is_file()
            assert json.loads((tmp_path/name/'metadata.json').read_text())['projection'] == name
    plt.close('all')


@pytest.mark.parametrize('plant_state,seconds,reached,steps', [
    ([0., 0., 0., 0.], 1.4, True, 50),
    ([.2, 0., 0., 0.], 1.4, False, 70),
    ([0., 0., 0., 0.], .04, False, 2),
])
def test_convergence_export_requires_sustained_full_state_target(
        monkeypatch, plant_state, seconds, reached, steps):
    from scripts.visualize_trajectory_manifold import notebook_visualization
    from scripts.run_inverted_pendulum_mc import make_config

    monkeypatch.setattr('scripts.run_inverted_pendulum_mc._plant_step',
                        lambda state, action, dt: np.asarray(plant_state))
    torch.set_num_threads(1)
    viz, metadata = notebook_visualization(
        {'decoder': CanonicalBehaviorDecoder(SimpleTransition(), 3),
         'cfg': make_config('smoke', horizon=3, device='cpu')},
        seconds=seconds, until_converged=True, frame_stride=10,
        samples=12, candidates=3)
    assert metadata['convergence']['reached'] is reached
    assert len(viz.inputs) == steps
    assert viz.frames[-1].step == steps
    assert metadata['seconds'] == pytest.approx(steps*.02)
    fig, axes = viz.figure()
    viz.draw_frame(len(viz.frames)-1, axes)
    assert axes[0].get_xlabel() == 'unwrapped angle [rad]'
    assert axes[0].get_ylabel() == 'angular speed [rad/s]'
    assert axes[0].get_zlabel() == 'cart speed [m/s]'
    plt.close(fig)


def test_target_reference_and_cache_replay_preserve_behavior(tmp_path, monkeypatch):
    from scripts.visualize_trajectory_manifold import export
    viz = make_visualization()
    metadata = {'projection': 'pca', 'projection_diagnostics': viz.projection.diagnostics,
                'state_scale_to_SI': viz.scale.tolist(), 'force_scale_to_N': viz.force_scale,
                'feasibility_tolerance': 1e-4}
    export(viz, metadata, tmp_path, fps=5)
    def forbidden(*args, **kwargs):
        raise AssertionError('replay must not fit, sample, or run the controller')
    monkeypatch.setattr(FixedProjection, '__init__', forbidden)
    monkeypatch.setattr(DecoderSampler, 'sample', forbidden)
    replay, _ = TrajectoryManifoldVisualization.from_export(tmp_path)
    np.testing.assert_array_equal(replay.states, viz.states)
    np.testing.assert_array_equal(replay.inputs, viz.inputs)
    np.testing.assert_allclose(replay.background_xyz, viz.background_xyz)
    np.testing.assert_allclose(replay.reference_xyz, viz.projection.transform(viz.reference.w)[0])
    np.testing.assert_array_equal(replay.reference.x, np.zeros((1, 4, 4)))
    np.testing.assert_array_equal(replay.reference.u, np.zeros((1, 3, 1)))
    for original, restored in zip(viz.frames, replay.frames):
        np.testing.assert_array_equal(original.candidates.w, restored.candidates.w)
        assert original.selected_feasible == restored.selected_feasible
        assert (original.selected is None) == (restored.selected is None)
    fig, axes = replay.figure()
    replay.draw_frame(0, axes)
    labels = axes[1].get_legend_handles_labels()[1]
    assert 'selected controller horizon' in labels
    assert 'target reference (constant state, zero input)' in labels
    legacy, _ = TrajectoryManifoldVisualization.from_export(tmp_path, manifold_style='legacy')
    legacy.draw_frame(0, axes)
    assert 'target reference (constant state, zero input)' not in axes[1].get_legend_handles_labels()[1]
    plt.close('all')


def test_target_plane_contains_left_target_with_free_position_and_inputs():
    from src.trajectory_target_visualization import TargetAlignedProjection
    rng = np.random.default_rng(5)
    training = rng.normal(size=(30, 19))
    target = np.array([.7, .2, -.1, .3])
    scale = np.array([1., 3., 1., 3.])
    projection = TargetAlignedProjection(training, target, scale)
    behaviors = rng.normal(size=(20, 19))
    behaviors[:, :4] = target
    behaviors[:, 0] = rng.normal(size=20)*10  # cart position is omitted on left
    # Future states and all future inputs remain arbitrary.
    np.testing.assert_array_equal(projection.transform(behaviors)[:, 2], np.zeros(20))
    behaviors[:, 2] += .02
    np.testing.assert_allclose(projection.transform(behaviors)[:, 2], np.log(2))
    behaviors[:, 2] -= .02
    behaviors[:, 1] += .05/3
    np.testing.assert_allclose(projection.transform(behaviors)[:, 2], np.log(2))


def test_target_plane_reuses_cached_data_and_draws_training_separately(monkeypatch):
    from src.trajectory_target_visualization import target_plane_view
    base = make_visualization()
    original_projection = base.projection
    training = base.background
    view = target_plane_view(base, training, training_indices=np.arange(len(training.x)))
    assert base.projection is original_projection
    assert view.frames is base.frames
    np.testing.assert_array_equal(view.states, base.states)
    np.testing.assert_array_equal(view.training.w, training.w[view.training_indices])
    assert np.all(view.state_errors == 0)
    def forbidden(*args, **kwargs):
        raise AssertionError('drawing must use cached data only')
    monkeypatch.setattr(view.projection, 'transform', forbidden)
    monkeypatch.setattr(view.sampler, 'sample', forbidden)
    fig, axes = view.figure()
    view.draw_frame(0, axes)
    labels = axes[1].get_legend_handles_labels()[1]
    assert 'actual training episodes (fixed subset)' in labels
    assert 'decoded background (synthetic)' in labels
    assert 'decoded current-state candidates' in labels
    assert axes[1].get_zlabel() == 'log(1 + displayed-state error)'
    plt.close(fig)
