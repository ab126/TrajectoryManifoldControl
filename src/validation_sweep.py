"""Original fixed-cart angle/rate grid, legacy endpoint and seed-count heatmap."""
from pathlib import Path
import json
import numpy as np

UPSTREAM = '053b14890a107063db9b58859884f655173528b8'


def terminal_recovery(t, x, u):
    """Authored final-window balance (screenshot); earlier crossings allowed."""
    t, x, u = np.asarray(t), np.asarray(x), np.asarray(u)
    tail = t >= t[-1] - 3.0
    return bool(t[-1] >= 3 and np.isfinite(x).all() and np.isfinite(u).all()
                and (np.abs(x[tail, 2]) <= .02).all()
                and (np.abs(x[tail, 3])*np.sqrt(9.81) <= .05).all())


def primary_success(t, x, u):
    """Campaign primary_success: terminal recovery plus no horizontal crossing."""
    return bool(terminal_recovery(t, x, u) and (np.abs(np.asarray(x)[:, 2]) < np.pi/2).all())


def plot_sweep(output, endpoint=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm
    output = Path(output)
    manifest = json.loads((output/'manifest.json').read_text())
    selected_endpoint = endpoint or manifest.get('plot_endpoint', 'primary_success')
    suffix = '' if endpoint is None else '_no_crossing'
    states = np.asarray(manifest['states'])
    seeds = manifest['seeds']
    arms = manifest['arms']
    xs = np.unique(np.rad2deg(states[:, 2]))
    ys = np.unique(states[:, 3]*np.sqrt(9.81))
    if len(xs)<2 or len(ys)<2:
        raise ValueError('heatmap requires at least two points on each axis')
    records = {}
    for p in (output/'cases').glob('*.json'):
        r = json.loads(p.read_text())
        key = (r['seed'], r['arm'], r['ic_index'])
        if key in records:
            raise ValueError(f'duplicate case {key}')
        records[key] = r
    counts = np.zeros((len(arms), len(ys), len(xs)), dtype=int)
    tested = np.zeros_like(counts)
    for a, arm in enumerate(arms):
        for i, state in enumerate(states):
            iy = np.searchsorted(ys, state[3]*np.sqrt(9.81))
            ix = np.searchsorted(xs, np.rad2deg(state[2]))
            for seed in seeds:
                r = records.get((seed, arm, i))
                if r is not None:
                    tested[a, iy, ix] += 1
                    counts[a, iy, ix] += r[selected_endpoint]
    n = len(seeds)
    cmap = ListedColormap(['#f0f0ef', '#b8d6e3', '#5b9cb9', '#135373'] if n==3
                          else plt.get_cmap('Blues')(np.linspace(.08, .9, n+1)))
    cmap = cmap.with_extremes(bad='#bbbbbb')
    norm = BoundaryNorm(np.arange(-.5, n+1.5), n+1)
    fig, axes = plt.subplots(1, len(arms), figsize=(8.5, 4.1), sharex=True, sharey=True,
                             squeeze=False, layout='constrained')
    labels = {'H': 'TMC\n(hybrid)', 'TMC': 'TMC\n(standalone)', 'L': 'Data-driven LQR\n(linear)'}
    for a, (ax, arm) in enumerate(zip(axes[0], arms)):
        grid = np.ma.masked_where(tested[a] != n, counts[a])
        im = ax.imshow(grid, origin='lower', interpolation='nearest', aspect='auto',
                       cmap=cmap, norm=norm,
                       extent=[xs[0]-(xs[1]-xs[0])/2, xs[-1]+(xs[1]-xs[0])/2,
                               ys[0]-(ys[1]-ys[0])/2, ys[-1]+(ys[1]-ys[0])/2])
        ax.set_title(labels[arm], fontsize=19, pad=18)
        ax.set_xlabel(r'Initial angle $\theta_0$ (degrees)')
        ax.set_xticks([-60, -30, 0, 30, 60])
    axes[0, 0].set_ylabel('Initial angular speed (rad/s)')
    cb = fig.colorbar(im, ax=axes[0].tolist(), ticks=range(n+1), shrink=.85)
    cb.ax.set_yticklabels([f'{i}/{n}' for i in range(n+1)])
    cb.set_label('Successful fitting seeds at each initial state')
    expected = len(states)*len(arms)*n
    complete = len(records)==expected
    if not complete:
        fig.suptitle(f'PARTIAL: {len(records)}/{expected} cases; gray = incomplete seed coverage', fontsize=10)
    for ext in ('png', 'pdf'):
        fig.savefig(output/f'initial_state_regimes{suffix}.{ext}', dpi=180, bbox_inches='tight')
    plt.close(fig)
    np.savez_compressed(output/f'grid_counts{suffix}.npz', theta_degrees=xs, angular_speed_physical=ys,
                        successes=counts, tested=tested, arms=arms, seeds=seeds)
    summary = dict(endpoint=selected_endpoint, complete=complete, saved=len(records), expected=expected,
                   successes={arm:int(counts[a].sum()) for a, arm in enumerate(arms)},
                   full_seed_cells={arm:int((tested[a]==n).sum()) for a, arm in enumerate(arms)})
    (output/f'summary{suffix}.json').write_text(json.dumps(summary, indent=2)+'\n')
    return summary
