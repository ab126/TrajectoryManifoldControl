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
