"""Readable per-IC-set E4 success figure from completed counts."""
import csv
import json
from pathlib import Path
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'validation/results_v1/E4'


def main():
    start = time.perf_counter()
    manifest = json.loads((OUT/'manifest.json').read_text())
    assert manifest['status'] == 'DONE'
    with (OUT/'rates.csv').open() as handle:
        rows = list(csv.DictReader(handle))
    arms = manifest['config']['arms']
    labels = []
    for arm in arms:
        if arm.startswith('DeePC'):
            _, kind, penalty = arm.split('_')
            labels.append(('State' if kind == 'state' else 'IO')+'\nλ='+penalty)
        else:
            labels.append(arm)
    fig, axes = plt.subplots(1, 2, figsize=(14, 4), sharey=True)
    for ax, ic in zip(axes, ('A', 'C50')):
        for a, arm in enumerate(arms):
            for seed in range(3):
                r = next(r for r in rows if r['arm'] == arm and r['seed'] == str(seed)
                         and r['ic_set'] == ic and float(r['seconds']) == 14)
                ax.scatter(a+.06*(seed-1), int(r['successes'])/int(r['n']), color=f'C{seed}',
                           label=f'seed {seed}' if a == 0 else None)
        ax.set(xticks=range(len(arms)), xticklabels=labels, ylim=(-.03, 1.03),
               title='IC-'+ic+('; n=6 per seed' if ic == 'A' else '; n=50 per seed'))
        ax.tick_params(axis='x', labelsize=8)
    axes[0].set_ylabel('Angle stabilization and no fall at 14 s')
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc='upper center', ncol=3, bbox_to_anchor=(.5, .93))
    fig.suptitle('E4: shared 32-step collection; all prespecified DeePC penalties shown')
    fig.tight_layout(rect=(0, 0, 1, .87))
    fig.savefig(OUT/'matched_success_rates.png', dpi=150)
    plt.close(fig)
    elapsed = time.perf_counter()-start
    manifest['readable_figure_wall_seconds'] = elapsed
    manifest['wall_seconds'] += elapsed
    (OUT/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')


if __name__ == '__main__':
    main()
