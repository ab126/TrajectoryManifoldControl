import json

import numpy as np

from src.validation_sweep import primary_success, terminal_recovery, plot_sweep


def test_legacy_endpoint_distinguishes_recovery_from_no_fall():
    t = np.arange(701)*.02
    x = np.zeros((701, 4)); u = np.zeros((700, 1))
    assert primary_success(t, x, u)
    x[0, 2] = np.pi/2
    assert not primary_success(t, x, u)  # recovered later but crossed horizontal
    assert terminal_recovery(t, x, u)
    x[0, 2] = np.pi/2-1e-9
    assert primary_success(t, x, u)
    x[-1, 3] = .050001/np.sqrt(9.81)
    assert not primary_success(t, x, u)
    x[-1, 3] = 0
    x[:, 0] = 1000  # original endpoint has no cart restriction
    assert primary_success(t, x, u)
    assert not primary_success(t[:2], x[:2], u[:1])
    u[0] = np.nan
    assert not primary_success(t, x, u)


def test_grid_counts_require_all_seeds_and_preserve_orientation(tmp_path):
    states = [[0,0,float(a),float(w)] for a in [-1,1] for w in [-2,2]]
    (tmp_path/'manifest.json').write_text(json.dumps(dict(states=states,seeds=[0,1,2],arms=['H','L'])))
    (tmp_path/'cases').mkdir()
    for seed in [0,1,2]:
        r = dict(seed=seed,arm='H',ic_index=1,primary_success=seed!=1)
        (tmp_path/'cases'/f'H{seed}.json').write_text(json.dumps(r))
    (tmp_path/'cases'/'L0.json').write_text(json.dumps(dict(seed=0,arm='L',ic_index=0,primary_success=False)))
    result = plot_sweep(tmp_path)
    assert not result['complete']
    assert result['full_seed_cells']==dict(H=1,L=0)
    with np.load(tmp_path/'grid_counts.npz') as grid:
        assert grid['successes'][0,1,0]==2
        assert grid['tested'][1,0,0]==1
    assert (tmp_path/'initial_state_regimes.pdf').is_file()
