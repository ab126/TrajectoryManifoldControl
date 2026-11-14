"""Render saved TMC validation cases without rerunning controllers or changing evidence."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


def generate_report(directory: Path, output: Path, reference_image: Path | None = None) -> Path:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    directory, output = Path(directory), Path(output)
    manifest_path = directory / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    rows = [json.loads(p.read_text()) for p in sorted((directory/'cases').glob('*.json'))]
    if not rows:
        raise ValueError('no saved cases found')
    arms = manifest['arms']
    indexed = {(r['arm'], r['ic_index']): r for r in rows}
    output.mkdir(parents=True, exist_ok=True)
    expected = len(manifest['states']) * len(arms)
    fig, axes = plt.subplots(1, len(arms), figsize=(5*len(arms), 4), squeeze=False,
                             sharex=True, sharey=True, layout='constrained')
    states = np.asarray(manifest['states'])
    for ax, arm in zip(axes[0], arms):
        ax.scatter(np.rad2deg(states[:, 2]), states[:, 3]*np.sqrt(9.81),
                   c='#dddddd', s=16, label='not yet saved')
        group = [r for r in rows if r['arm'] == arm]
        ids = [r['ic_index'] for r in group]
        ax.scatter(np.rad2deg(states[ids, 2]), states[ids, 3]*np.sqrt(9.81),
                   c=[int(r['terminal_recovery']) for r in group],
                   cmap=ListedColormap(['#d95f02', '#167a9a']), vmin=0, vmax=1,
                   s=65, edgecolors='black', linewidths=.5)
        for r in group:
            i = r['ic_index']
            ax.annotate(str(i), (np.rad2deg(states[i, 2]), states[i, 3]*np.sqrt(9.81)),
                        xytext=(5, 4), textcoords='offset points', fontsize=8)
        ax.set(title=f'{arm}: {sum(r["terminal_recovery"] for r in group)}/{len(group)} recoveries',
               xlabel='Initial angle (degrees)', ylabel='Initial angular speed (rad/s)')
        ax.grid(alpha=.2)
    fig.suptitle(f'{manifest["state_set"]}: {len(rows)}/{expected} saved cases; blue=recovery, orange=no recovery, gray=missing\n'
                 'Projection of four-state starts; no interpolation or inferred recovery region', fontsize=10)
    fig.savefig(output/'recovery_map.png', dpi=160)
    plt.close(fig)

    shared = sorted(set.intersection(*[{r['ic_index'] for r in rows if r['arm']==a} for a in arms]))
    pages = []
    # Paginate every shared start, rather than selecting successful examples.
    for offset in range(0, len(shared), 4):
        ids = shared[offset:offset+4]
        fig, axes = plt.subplots(5, len(ids), figsize=(4.3*len(ids), 11), squeeze=False,
                                 sharex=True, layout='constrained')
        for col, i in enumerate(ids):
            for arm in arms:
                r = indexed[arm, i]
                with np.load(directory/'raw'/f'{arm}_{i:04d}.npz') as raw:
                    t, x, u = raw['t'], raw['X'], raw['U']
                np.testing.assert_allclose(x[0], states[i], rtol=0, atol=1e-12)
                error = x - np.asarray(manifest['x_ref'])
                values = [error[:, 2], error[:, 3]*np.sqrt(9.81), error[:, 0],
                          error[:, 1]*np.sqrt(9.81)]
                label = f'{arm}: '+('recovered' if r['terminal_recovery'] else 'no recovery')
                if not r['completed']:
                    label += ' (failed)'
                for j, values_j in enumerate(values):
                    axes[j, col].plot(t, values_j, label=label)
                axes[4, col].step(t[:-1], u[:, 0]-manifest['u_ref'][0], where='post', label=label)
            axes[0, col].set_title(f'IC {i}: '+np.array2string(states[i], precision=2))
            axes[0, col].legend(fontsize=8)
            for j, label in enumerate(['Angle error (rad)', 'Angular speed error (rad/s)',
                                        'Cart position error (m)', 'Cart speed error (m/s)', 'Input error F/(mg)']):
                ax = axes[j, col]
                ax.set_ylabel(label)
                ax.grid(alpha=.25)
                ax.axhline(0, color='gray', linewidth=.5)
                if manifest['seconds'] >= 3:
                    ax.axvspan(manifest['seconds']-3, manifest['seconds'], color='green', alpha=.06)
            for j, tolerance in [(0, .02), (1, .05)]:
                for sign in [-1, 1]:
                    axes[j, col].axhline(sign*tolerance, color='green', linestyle=':', linewidth=.8)
            axes[4, col].set_xlabel('Physical time (s)')
        fig.suptitle('Saved paired convergence traces; green band = final recovery window')
        name = f'convergence_{offset//4+1:03d}.png'
        fig.savefig(output/name, dpi=150)
        plt.close(fig)
        pages.append(name)

    lines = ['# TMC versus linear validation report', '',
             f'Run: `{directory.resolve()}`. State set **{manifest["state_set"]}**, '
             f'{manifest["seconds"]:g} s, training seed {manifest["plant_config"]["seed"]}.', '',
             f'**Partial run: {len(rows)}/{expected} cases saved.**' if len(rows)<expected else f'Complete: {len(rows)}/{expected} cases saved.', '',
             f'Gain source: `{manifest["gain_source"]["kind"]}`. Solver: `{manifest["config"]}`.', '',
             '| Arm | Saved | Recovered | Controller failures | Median case median latency (ms) |',
             '| --- | ---: | ---: | ---: | ---: |']
    for arm in arms:
        group = [r for r in rows if r['arm']==arm]
        lat = [r['controller_latency']['median'] for r in group if r['controller_latency']['median'] is not None]
        ms = f'{1000*np.median(lat):.3f}' if lat else 'n/a'
        lines.append(f'| {arm} | {len(group)} | {sum(r["terminal_recovery"] for r in group)} | '
                     f'{sum(not r["completed"] for r in group)} | {ms} |')
    if 'TMC' in arms and 'L' in arms:
        pairs = sorted({i for a,i in indexed if a=='TMC'} & {i for a,i in indexed if a=='L'})
        wins = sum(indexed['TMC',i]['terminal_recovery'] and not indexed['L',i]['terminal_recovery'] for i in pairs)
        losses = sum(indexed['L',i]['terminal_recovery'] and not indexed['TMC',i]['terminal_recovery'] for i in pairs)
        lines += ['', f'Paired starts: {len(pairs)}; TMC-only recoveries: {wins}; L-only recoveries: {losses}.', '']
    lines += ['', '![Recovery map](recovery_map.png)', '',
              'Points show saved outcomes only. The angle/rate projection does not fix cart position or velocity. '
              'Missing cases are not failures. Unequal arm counts must not be compared as a paired success rate.', '',
              '## Convergence across shared initial conditions', '']
    for page in pages:
        lines += [f'![Paired convergence]({page})', '']
    lines += ['Recovery requires continuous angle error ≤0.02 rad and physical angular speed error ≤0.05 rad/s '
              'throughout the final three seconds. It does not certify cart regulation. '
              'Solver latency is measured but not injected as feedback delay; the deadline is 20 ms. '
              'These results do not establish superiority across the full initial-condition set.', '',
              '## Provenance', '', f'Checkpoint SHA256: `{manifest["checkpoint_sha256"]}`.', '',
              f'Manifest SHA256: `{hashlib.sha256(manifest_path.read_bytes()).hexdigest()}`.', '',
              'Generated from saved case JSON and raw NPZ traces, without controller reruns.']
    if reference_image is not None:
        import shutil
        shutil.copyfile(reference_image, output/'historical_recovery_map.png')
        lines += ['', '## Historical reference supplied by the user', '',
                  '![Historical hybrid versus linear map](historical_recovery_map.png)', '',
                  'The screenshot labels TMC as hybrid and shows successes out of three seeds. '
                  'It is historical context, not a figure generated by this standalone TMC run. '
                  'Its underlying data and exact plotting provenance have not been verified. '
                  'Legacy phase-map plotting remains in `validation/experiments/figures.py`; '
                  'it expects legacy campaign records, not the current TMC schema. '
                  'The current C set samples four state coordinates rather than a fixed-cart angle/rate grid.']
    evidence = {'manifest': manifest,
                'manifest_sha256': hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                'cases': rows,
                'raw_sha256': {f'{r["arm"]}_{r["ic_index"]:04d}.npz':
                    hashlib.sha256((directory/'raw'/f'{r["arm"]}_{r["ic_index"]:04d}.npz').read_bytes()).hexdigest()
                    for r in rows}}
    (output/'evidence.json').write_text(json.dumps(evidence, indent=2, allow_nan=False)+'\n')
    lines += ['', 'Full manifest, case metrics and raw-trace hashes: [evidence.json](evidence.json).']
    report = output/'REPORT.md' 
    report.write_text('\n'.join(lines)+'\n')
    return report
