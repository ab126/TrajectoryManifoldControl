import numpy as np
import torch

from scripts.run_inverted_pendulum_mc import split_episodes
from src.manifold_control import (
    CanonicalBehaviorDecoder,
    DataDrivenMPCSolver,
    DataDrivenTransition,
)


class AdditiveTransition(DataDrivenTransition):
    def __init__(self):
        super().__init__(x_dim=1, u_dim=1, hidden_dims=())

    def forward(self, x, u):
        return x + u


def test_canonical_decoder_uses_paper_order_and_exact_coordinates():
    decoder = CanonicalBehaviorDecoder(AdditiveTransition(), horizon=3)
    x0 = torch.tensor([2.0])
    u = torch.tensor([[1.0], [-2.0], [0.5]])
    q = decoder.coordinates(x0, u)
    w = decoder(q)
    x_out, u_out = decoder.unpack_behavior(w)
    assert torch.equal(u_out, u)
    assert torch.equal(x_out, torch.tensor([[2.0], [3.0], [1.0], [1.5]]))
    assert torch.equal(x_out[0], x0)


def test_decoder_is_causal():
    decoder = CanonicalBehaviorDecoder(AdditiveTransition(), horizon=3)
    x0 = torch.zeros(1)
    first = torch.zeros(3, 1)
    second = first.clone()
    second[2] = 4.0
    assert torch.equal(decoder.rollout(x0, first)[:3], decoder.rollout(x0, second)[:3])


def test_data_driven_mpc_enforces_bounds_by_construction():
    decoder = CanonicalBehaviorDecoder(AdditiveTransition(), horizon=4)
    solver = DataDrivenMPCSolver(
        decoder, torch.ones(1, 1), torch.zeros(1, 1),
        u_bounds=(-0.2, 0.2), max_iter=20, lr=0.1,
    )
    solution = solver.solve(torch.ones(1))
    assert torch.all(solution.u <= 0.2)
    assert torch.all(solution.u >= -0.2)
    assert torch.equal(solution.x[0], torch.ones(1))


def test_episode_split_has_no_overlap():
    X = np.repeat(np.arange(20)[:, None, None], 3, axis=1).astype(np.float32)
    U = np.zeros((20, 2, 1), dtype=np.float32)
    splits = split_episodes(X, U, seed=7)
    identities = {name: set(values[0][:, 0, 0].tolist()) for name, values in splits.items()}
    assert identities["train"].isdisjoint(identities["validation"])
    assert identities["train"].isdisjoint(identities["test"])
    assert identities["validation"].isdisjoint(identities["test"])
    assert set.union(*identities.values()) == set(range(20))
