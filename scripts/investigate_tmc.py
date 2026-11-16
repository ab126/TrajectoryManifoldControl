"""Audit a completed grid and run bounded independent-tuning model/optimizer probes."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
from validation_benchmark import ROOT, load_upstream, sha256, write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,default=Path('saves/validation/grid-hybrid-3seed-v1'))
    p.add_argument('--models-root',type=Path,default=Path('saves/validation/models'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--probe-starts',type=int,default=8)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    c,_,_=load_upstream(args.output);np,torch=c.np,c.torch
    from src.manifold_experiments import NativeSolver,NativeConfig
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    manifest=json.loads((args.run/'manifest.json').read_text())
    rows=[json.loads(f.read_text()) for f in sorted((args.run/'cases').glob('*.json'))]
    indexed={(r['seed'],r['arm'],r['ic_index']):r for r in rows}
    if len(indexed)!=10086 or len(rows)!=10086 or manifest['seeds']!=[0,1,2] or len(manifest['states'])!=1681:
        raise ValueError('investigation requires the completed three-seed 41x41 H/L grid')
    result=dict(script_sha256=sha256(__file__),run=str(args.run),manifest_sha256=sha256(args.run/'manifest.json'),
        cases=len(rows),expected=len(manifest['states'])*len(manifest['seeds'])*2,
        per_seed=[],models=[],probes=[],replays=[],case_hashes={f.name:sha256(f) for f in sorted((args.run/'cases').glob('*.json'))})
    wins=[];losses=[]
    for seed in manifest['seeds']:
        ids=sorted(i for s,a,i in indexed if s==seed and a=='H' and (seed,'L',i) in indexed)
        h=[indexed[seed,'H',i] for i in ids];l=[indexed[seed,'L',i] for i in ids]
        hw=[i for i in ids if indexed[seed,'H',i]['terminal_recovery'] and not indexed[seed,'L',i]['terminal_recovery']]
        lw=[i for i in ids if indexed[seed,'L',i]['terminal_recovery'] and not indexed[seed,'H',i]['terminal_recovery']]
        wins+= [(seed,i) for i in hw];losses+= [(seed,i) for i in lw]
        row=dict(seed=seed,n=len(ids),H=sum(r['terminal_recovery'] for r in h),L=sum(r['terminal_recovery'] for r in l),
            H_primary=sum(r['primary_success'] for r in h),L_primary=sum(r['primary_success'] for r in l),H_only=len(hw),L_only=len(lw),
            H_failures=sum(r['failure'] is not None for r in h),L_failures=sum(r['failure'] is not None for r in l),
            H_median_linear_fraction=float(np.median([r['metrics']['local_fraction'] for r in h])),
            H_nonfinite_cem=sum(r['metrics']['nonfinite_cem'] for r in h))
        result['per_seed'].append(row)
    # Two matched witnesses, deterministic middle index in each disagreement category.
    chosen=[]
    for label,pairs in [('L_only',losses),('H_only',wins)]:
        candidates=[(s,i) for s,i in pairs if s==0]
        if candidates:
            s,i=candidates[len(candidates)//2];chosen.append((label,s,i))
    fig,axes=plt.subplots(3,len(chosen),figsize=(10,7),squeeze=False,layout='constrained')
    for col,(label,seed,i) in enumerate(chosen):
        for arm in ('H','L'):
            raw_path=args.run/'raw'/f'seed{seed}_{arm}_{i:04d}.npz'
            if sha256(raw_path)!=indexed[seed,arm,i]['raw_sha256']:raise ValueError('witness raw trace mismatch')
            with np.load(raw_path) as raw:
                t,x=raw['t'],raw['X']
            axes[0,col].plot(t,x[:,2],label=arm)
            axes[1,col].plot(t,x[:,3]*np.sqrt(9.81))
            axes[2,col].plot(t,x[:,0])
        for ax in axes[:,col]:ax.grid(alpha=.2)
        axes[0,col].set_title(f'{label}: seed {seed}, IC {i}')
        axes[0,col].legend();axes[2,col].set_xlabel('Time (s)')
    for j,label in enumerate(['Continuous angle (rad)','Angular speed (rad/s)','Cart position (m)']):axes[j,0].set_ylabel(label)
    fig.savefig(args.output/'witnesses.png',dpi=150);plt.close(fig)
    tuning=np.random.default_rng(20261017).uniform([-.5,-.8,-.6,-1.],[.5,.8,.6,1.],(8,4))[:args.probe_starts]
    for seed in manifest['seeds']:
        model=args.models_root/f'reconstructed-seed{seed}'
        # Never attribute a different model's diagnostics to this grid.
        for name in ('transition.pt','data.npz'):
            if sha256(model/name)!=manifest['artifacts'][str(seed)][name]:raise ValueError('artifact mismatch')
        transition,payload=c.runner.load_transition(model/'transition.pt','cpu')
        cfg=c.runner.ExperimentConfig(**{**payload['config'],'device':'cpu'})
        decoder=c.CanonicalBehaviorDecoder(transition,30).eval()
        with np.load(model/'data.npz') as data:split=c.runner.split_episodes(data['X'],data['U'],seed)
        gain,local=c.runner.fit_local_feedback(split['train'],c.Q,c.R)
        with torch.no_grad():bias=transition(torch.zeros(4),torch.zeros(1)).numpy()
        # True-plant Jacobian is diagnostic only; never supplied to a controller.
        eps=1e-5
        closed=np.column_stack([(c.runner._plant_step(eps*np.eye(4)[j],-gain@(eps*np.eye(4)[j]),cfg.dt)-
                                  c.runner._plant_step(-eps*np.eye(4)[j],gain@(eps*np.eye(4)[j]),cfg.dt))/(2*eps) for j in range(4)])
        d=json.loads((model/'diagnostics.json').read_text())
        result['models'].append(dict(seed=seed,config=payload['config'],bias=bias.tolist(),
            true_local_spectral_radius=float(max(abs(np.linalg.eigvals(closed)))),
            prediction=d['prediction'],history=payload['history']))
        # Replay fixed measured inputs from evenly spaced L-only cases at several times.
        ids=[i for s,i in losses if s==seed]
        for i in ids[::max(1,len(ids)//12)][:12]:
            raw_path=args.run/'raw'/f'seed{seed}_H_{i:04d}.npz'
            if sha256(raw_path)!=indexed[seed,'H',i]['raw_sha256']:raise ValueError('replay raw trace mismatch')
            with np.load(raw_path) as raw:x,u=raw['X'],raw['U']
            for k in (0,5,50,400,650):
                if k+30>len(u):continue
                with torch.no_grad():pred=decoder.rollout(torch.tensor(x[k],dtype=torch.float32),torch.tensor(u[k:k+30],dtype=torch.float32)).numpy()
                delta=pred-x[k:k+31]
                result['replays'].append(dict(seed=seed,ic=i,step=k,theta_rmse=float(np.sqrt(np.mean(delta[1:,2]**2))),
                    omega_rmse=float(np.sqrt(np.mean(delta[1:,3]**2))*np.sqrt(9.81)),
                    actual_theta=float(x[k,2]),terminal_theta_error=float(delta[-1,2])))
        Q=torch.tensor(c.Q,dtype=torch.float32);R=torch.tensor(c.R,dtype=torch.float32)
        for i,state in enumerate(tuning):
            for name,conf in [('CEM3',NativeConfig(solver='original_cem',seed=seed,cem_iterations=3)),
                              ('CEM10',NativeConfig(solver='original_cem',seed=seed,cem_iterations=10)),
                              ('Adam40',NativeConfig(solver='adam',seed=seed,iterations=40,learning_rate=.1,starts=5))]:
                solver=NativeSolver(decoder,Q,R,conf);tick=time.perf_counter()
                sol=solver.solve(torch.tensor(state,dtype=torch.float32));elapsed=time.perf_counter()-tick
                proposed=sol.u.detach().numpy();actual=[state]
                for action in proposed:actual.append(c.runner._plant_step(actual[-1],action,cfg.dt))
                actual=np.asarray(actual)
                with torch.no_grad():true_cost=float(solver.objective(torch.tensor(actual,dtype=torch.float32)[None],sol.u[None])[0])
                result['probes'].append(dict(seed=seed,start=i,solver=name,predicted_cost=float(sol.loss),
                    replay_cost=true_cost,seconds=elapsed,finite=bool(sol.finite),
                    angle_rmse=float(np.sqrt(np.mean((sol.x.detach().numpy()[:,2]-actual[:,2])**2)))))
        print(f'Finished model and tuning probes for seed {seed}',flush=True)
    write_json(args.output/'investigation.json',result)
    # Coverage + actual paired counts, not unrelated percentages.
    fig,axes=plt.subplots(1,3,figsize=(12,3.6),layout='constrained')
    seeds=np.arange(3);w=.35
    for offset,arm in [(-w/2,'H'),(w/2,'L')]:
        axes[0].bar(seeds+offset,[100*r[arm]/r['n'] for r in result['per_seed']],w,label=arm)
    axes[0].set(xticks=seeds,xlabel='Independent fitting seed',ylabel='Recovery (%)',title='Seed matters: hybrid wins seed 2');axes[0].legend()
    totals=[sum(r[k] for r in result['per_seed']) for k in ['H_only','L_only']]
    axes[1].bar(['H only','L only'],totals,color=['#197c96','#df8535']);axes[1].set(title='Same-start disagreements',ylabel='Seed × initial-state cases')
    h=sum(r['H'] for r in result['per_seed']);hp=sum(r['H_primary'] for r in result['per_seed']);l=sum(r['L'] for r in result['per_seed'])
    axes[2].bar(['H recovery','H no crossing','L (both)'],[h,hp,l],color=['#197c96','#86bac8','#df8535']);axes[2].set(title='Recovery ≠ no fall',ylabel='Successes / 5,043')
    fig.savefig(args.output/'coverage.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(12,3.5),layout='constrained')
    for model in result['models']:
        err=np.asarray(model['prediction']['rmse_by_step_and_state'])
        axes[0].plot(np.arange(len(err))*.02,err[:,2],label=f'seed {model["seed"]}')
    axes[0].axhline(.02,color='gray',ls=':');axes[0].set(title='Held-out rollout error grows',xlabel='Open-loop horizon (s)',ylabel='Angle RMSE (rad)');axes[0].legend()
    replay=result['replays']
    axes[1].boxplot([[r['theta_rmse'] for r in replay if r['step']==k] for k in (0,5,50,400,650)],tick_labels=['0','.1','1','8','13'],showfliers=False)
    axes[1].set(yscale='log',title='Replay of H inputs on L-only cases',xlabel='Rollout start time (s)',ylabel='0.6s angle forecast RMSE (rad)')
    probes=result['probes'];names=['CEM3','CEM10','Adam40']
    # Matched cost ratios to original CEM at the identical seed/start.
    base={(r['seed'],r['start']):r for r in probes if r['solver']=='CEM3'}
    for name in names:
        rr=[r for r in probes if r['solver']==name]
        axes[2].scatter([r['predicted_cost']/base[r['seed'],r['start']]['predicted_cost'] for r in rr],
                        [r['replay_cost']/base[r['seed'],r['start']]['replay_cost'] for r in rr],label=name,alpha=.6,s=20)
    axes[2].axhline(1,color='gray',ls=':');axes[2].axvline(1,color='gray',ls=':');axes[2].set(title='Better plan ≠ wider recovery region',xlabel='Learned cost / CEM3',ylabel='True replay cost / CEM3');axes[2].legend(fontsize=8)
    fig.savefig(args.output/'model_and_solver.png',dpi=160);plt.close(fig)
    from matplotlib.colors import ListedColormap,BoundaryNorm
    fig,axes=plt.subplots(1,3,figsize=(12,3.8),sharex=True,sharey=True,layout='constrained')
    cmap=ListedColormap(['#dedede','#e39b53','#4a97b2','#224b5e'])
    for seed,ax in enumerate(axes):
        z=np.array([2*int(indexed[seed,'H',i]['terminal_recovery'])+int(indexed[seed,'L',i]['terminal_recovery']) for i in range(1681)]).reshape(41,41).T
        image=ax.imshow(z,origin='lower',aspect='auto',interpolation='nearest',cmap=cmap,
            norm=BoundaryNorm([-.5,.5,1.5,2.5,3.5],4),
            extent=[-87.125,87.125,-3.5875*np.sqrt(9.81),3.5875*np.sqrt(9.81)])
        ax.set(title=f'Seed {seed}',xlabel='Initial angle (degrees)')
    axes[0].set_ylabel('Initial angular speed (rad/s)')
    cb=fig.colorbar(image,ax=axes,ticks=range(4),shrink=.8)
    cb.ax.set_yticklabels(['Neither','L only','Hybrid only','Both'])
    fig.suptitle('Same grid, same endpoints: the successful regions differ by fit')
    fig.savefig(args.output/'paired_regions.png',dpi=150);plt.close(fig)
    print(args.output/'investigation.json')


if __name__=='__main__':main()
