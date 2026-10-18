"""Summarize local adapter runs, compare archived cases, and plot raw traces."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

ROOT=Path(__file__).resolve().parents[1]
if __package__:
    from .validation_benchmark import load_reference_cases
else:
    from validation_benchmark import load_reference_cases
ENDPOINTS=('legacy_uninterrupted','terminal_recovery','periodic_terminal_recovery')


def summarize(directories):
    records=[]; comparisons=[]
    for directory in directories:
        manifest=json.loads((directory/'manifest.json').read_text())
        for path in sorted((directory/'cases').glob('*.json')):
            row=json.loads(path.read_text()); row['directory']=str(directory)
            row['artifact_kind']=manifest['artifact_kind']; records.append(row)
            experiment='E0' if row['ic_set']=='A' else 'E1'
            name=f"seed{row['seed']}_{row['arm']}_{row['ic_set']}_{row['ic_index']:04d}_{manifest['seconds']:g}s.json"
            prior=load_reference_cases().get(f'{experiment}/cases/{name}')
            if prior is not None:
                comparison=dict(case=name,artifact_kind=row['artifact_kind'],
                    legacy_agrees=row['legacy_uninterrupted']==prior['primary_success'],
                    recovery_agrees=row['terminal_recovery']==prior['balanced'])
                for key in ('final_theta','final_p','max_abs_p','local_fraction'):
                    old,new=prior.get(key),row['upstream'].get(key)
                    comparison[key+'_difference']=new-old if old is not None and new is not None else None
                comparisons.append(comparison)
    rates=[]; pairs=[]
    groups={}
    for row in records:
        key=(row['seed'],row['ic_set'],row['artifact_kind'],row['arm'])
        groups.setdefault(key,{})
        if row['ic_index'] in groups[key]:
            raise ValueError(f'duplicate case across directories: {key}, {row["ic_index"]}')
        groups[key][row['ic_index']]=row
    for (seed,ic_set,kind,arm),items in groups.items():
        rates.append(dict(seed=seed,ic_set=ic_set,artifact_kind=kind,arm=arm,n=len(items),
                     **{key:sum(r[key] for r in items.values()) for key in ENDPOINTS}))
        if arm=='L':continue
        baseline=groups.get((seed,ic_set,kind,'L'),{})
        # Archived-gain L is a different artifact from a reconstructed-data L.
        shared=sorted(items.keys() & baseline.keys())
        for endpoint in ENDPOINTS:
            wins=sum(items[i][endpoint] and not baseline[i][endpoint] for i in shared)
            losses=sum(baseline[i][endpoint] and not items[i][endpoint] for i in shared)
            if shared:
                pairs.append(dict(seed=seed,ic_set=ic_set,artifact_kind=kind,arm=arm,endpoint=endpoint,
                    n=len(shared),wins=wins,losses=losses,
                    exact_mcnemar_p=float(binomtest(wins,wins+losses,.5).pvalue) if wins+losses else 1.))
    return dict(rates=rates,paired=pairs,archived_comparisons=comparisons)


def plot_examples(directory, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    chosen={}
    for case in sorted((directory/'cases').glob('*.json')):
        row=json.loads(case.read_text())
        category='success' if row['terminal_recovery'] else 'failure'
        chosen.setdefault((row['arm'],category),case)
    for (arm,category),case in chosen.items():
        with np.load(directory/'raw'/f'{case.stem}.npz') as raw:
            t,x,u=raw['t'],raw['X'],raw['U']
        fig,axes=plt.subplots(5,1,figsize=(9,10),sharex=True)
        axes[0].plot(t,x[:,2],label='continuous angle')
        axes[0].plot(t,np.arctan2(np.sin(x[:,2]),np.cos(x[:,2])),ls='--',label='periodic error')
        axes[1].plot(t,x[:,3]*np.sqrt(9.81))
        axes[2].plot(t,x[:,0]);axes[3].plot(t,x[:,1]*np.sqrt(9.81))
        axes[4].step(t[:-1],u[:,0],where='post')
        for ax,tolerance in zip(axes[:2],[.02,.05]):
            ax.axhline(tolerance,color='green',ls=':');ax.axhline(-tolerance,color='green',ls=':')
        for ax,label in zip(axes,['angle (rad)','angular speed (rad/s)','cart position (m)','cart speed (m/s)','force / (m_p g)']):
            ax.set_ylabel(label);ax.axvspan(t[-1]-3,t[-1],alpha=.1,color='green');ax.grid(alpha=.25)
        axes[0].legend();axes[-1].set_xlabel('physical time (s)')
        fig.suptitle(f'{directory.name}: {case.stem}; unwrapped recovery {category}')
        fig.tight_layout();fig.savefig(output/f'{directory.name}_{arm}_{category}.png',dpi=140);plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directories',nargs='+',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--figures',action='store_true')
    parser.add_argument('--figures-only',action='store_true',help='Plot frozen-evaluator or ablation cases without baseline-table assumptions')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    if not args.figures_only:
        result=summarize(args.directories)
        (args.output/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        print(json.dumps(result['rates'],indent=2))
    if args.figures or args.figures_only:
        for directory in args.directories:plot_examples(directory,args.output)
