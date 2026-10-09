"""Summaries and figures derived only from saved per-case scientific outputs."""
from campaign import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def figures(experiment):
    out=OUT/experiment
    rows,rates=summarize(out)
    if not rows:return
    distributions=[]
    for arm in sorted({r['arm'] for r in rows}):
        for name in sorted({r['ic_set'] for r in rows}):
            for seconds in sorted({r['seconds'] for r in rows}):
                items=[r for r in rows if r['arm']==arm and r['ic_set']==name and r['seconds']==seconds]
                if not items:continue
                for field in ('max_abs_p','final_p','angle_settle_seconds','full_settle_seconds','max_abs_u','input_limit_seconds','local_fraction','switch_toggles'):
                    values=[r[field] for r in items if r[field] is not None]
                    distributions.append(dict(arm=arm,ic_set=name,seconds=seconds,field=field,n=len(items),n_observed=len(values),
                        **{f'q{int(q*100):02d}':float(np.quantile(values,q)) if values else None for q in (0,.25,.5,.75,.95,1)}))
    csv_write(out/'distributions.csv',distributions)
    chosen=[r for r in rates if r['seconds']==14.]
    arms=sorted({r['arm'] for r in chosen})
    fig,ax=plt.subplots(figsize=(8,4))
    for i,arm in enumerate(arms):
        subset=[r for r in chosen if r['arm']==arm]
        ax.scatter([i+(j-len(subset)/2)*.04 for j in range(len(subset))],[r['rate'] for r in subset],s=24,label=arm)
    ax.set_xticks(range(len(arms)),arms); ax.set_ylim(-.03,1.03)
    ax.set_ylabel('Angle stabilization and no fall, rate per seed/set')
    ax.set_title(f'{experiment}: 14 s; repeated seeds shown separately')
    fig.tight_layout(); fig.savefig(out/'success_rates.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(12,4))
    for ax,field,label in zip(axes,('max_abs_p','angle_settle_seconds','local_fraction'),('Max |p| [m]','Angle settling [s]','Fraction local feedback')):
        series=[[r[field] for r in rows if r['arm']==arm and r['seconds']==14. and r[field] is not None] for arm in arms]
        ax.boxplot(series,labels=arms,showfliers=False);ax.set_ylabel(label)
        if field=='max_abs_p':ax.set_yscale('symlog',linthresh=1)
    fig.suptitle(f'{experiment}: descriptive distributions; unsettled times omitted')
    fig.tight_layout();fig.savefig(out/'distributions.png',dpi=150);plt.close(fig)
    if experiment=='E0':
        diagnostics=[]
        for path in sorted((OUT/'models').glob('episodes3500_seed*/diagnostics.json')):
            item=json.loads(path.read_text()); local=item['local'];pred=item['prediction']
            row=dict(seed=item['seed'],**{k:v for k,v in local.items() if k not in ('A','B','gain','fit_rmse')})
            for name in ('one_step_rmse_by_state','N_step_rmse_by_state'):
                row.update({f'{name}_{state}':value for state,value in zip(('p','v','theta','omega'),pred[name])})
            row.update({f'local_rmse_{state}':value for state,value in zip(('p','v','theta','omega'),local['fit_rmse'])})
            diagnostics.append(row)
        csv_write(out/'model_diagnostics.csv',diagnostics)
        fig,ax=plt.subplots(figsize=(8,4))
        for path in sorted((OUT/'models').glob('episodes3500_seed*/diagnostics.json')):
            item=json.loads(path.read_text());errors=np.asarray(item['prediction']['rmse_by_step_and_state'])
            ax.plot(np.arange(len(errors))*.02,np.sqrt(np.mean(errors**2,axis=1)),label=f"seed {item['seed']}")
        ax.set(xlabel='Prediction horizon [physical s]',ylabel='RMSE, normalized states',title='Held-out free-running predictions')
        ax.legend(ncol=2,fontsize=8);fig.tight_layout();fig.savefig(out/'prediction_rmse.png',dpi=150);plt.close(fig)
    if experiment=='E1':
        categories=[]
        for rate in rates:
            items=[r for r in rows if all(r[k]==rate[k] for k in ('seed','arm','ic_set','seconds'))]
            toggles=[r['switch_toggles'] for r in items]
            categories.append(dict(seed=rate['seed'],arm=rate['arm'],ic_set=rate['ic_set'],seconds=rate['seconds'],n=len(items),
                primary_failures=sum(not r['primary_success'] for r in items),
                **{field:sum(bool(r[field]) for r in items) for field in ('fell','never_entered_switch','balanced_with_cart_drift','nonfinite_cem','fell_within_1s_of_switch_exit')},
                cases_with_switch_toggles=sum(v>0 for v in toggles),switch_toggles_median=float(np.median(toggles)),
                switch_toggles_p95=float(np.quantile(toggles,.95)),switch_toggles_max=max(toggles)))
        csv_write(out/'failure_categories.csv',categories)
        for seed in sorted({r['seed'] for r in rows}):
            for name in ('B','D'):
                items=[r for r in rows if r['ic_set']==name and r['seed']==seed and r['seconds']==14.]
                if not items:continue
                fig,axes=plt.subplots(1,len(arms),figsize=(3*len(arms),3.2),squeeze=False)
                for ax,arm in zip(axes[0],arms):
                    arm_rows=[r for r in items if r['arm']==arm]
                    if name=='B':
                        ax.scatter([np.rad2deg(r['initial_state'][2]) for r in arm_rows],[r['primary_success'] for r in arm_rows],s=9)
                        ax.set(xlabel='θ0 [degrees]',ylabel='Primary success',ylim=(-.1,1.1))
                    else:
                        ax.scatter([np.rad2deg(r['initial_state'][2]) for r in arm_rows],
                                   [r['initial_state'][3]*np.sqrt(9.81) for r in arm_rows],
                                   c=[r['primary_success'] for r in arm_rows],cmap='RdYlGn',vmin=0,vmax=1,s=10)
                        ax.set(xlabel='θ0 [degrees]',ylabel='θdot0 [physical rad/s]')
                    ax.set_title(arm)
                fig.suptitle(f'IC-{name}, seed {seed}; green=success, red=failure')
                fig.tight_layout();fig.savefig(out/f'map_{name}_seed{seed}.png',dpi=150);plt.close(fig)
        # Descriptive cart-excursion maps use 14 s cases only, with no rail cutoff.
        for seed in sorted({r['seed'] for r in rows}):
            fig,axes=plt.subplots(1,len(arms),figsize=(3*len(arms),3.2),squeeze=False)
            from matplotlib.colors import SymLogNorm
            values=[r['max_abs_p'] for r in rows if r['ic_set']=='D' and r['seed']==seed and r['seconds']==14. and r['max_abs_p'] is not None]
            if not values:plt.close(fig);continue
            norm=SymLogNorm(linthresh=1.,vmin=0.,vmax=max(1.,max(values)))
            for ax,arm in zip(axes[0],arms):
                items=[r for r in rows if r['ic_set']=='D' and r['seed']==seed and r['seconds']==14. and r['arm']==arm and r['max_abs_p'] is not None]
                dots=ax.scatter([np.rad2deg(r['initial_state'][2]) for r in items],
                    [r['initial_state'][3]*np.sqrt(9.81) for r in items],c=[r['max_abs_p'] for r in items],norm=norm,cmap='viridis',s=10)
                ax.set(title=arm,xlabel='θ0 [degrees]',ylabel='θdot0 [physical rad/s]')
            fig.colorbar(dots,ax=list(axes[0]),label='Max |p| [m]',shrink=.8)
            fig.suptitle(f'IC-D cart excursion, seed {seed}; descriptive, no rail limit')
            fig.subplots_adjust(top=.8,bottom=.2,wspace=.4,right=.85)
            fig.savefig(out/f'cart_excursion_D_seed{seed}.png',dpi=150);plt.close(fig)
        # Five failures spread across arms. Replay inputs in the true plant to compare
        # the predicted horizon against the same candidate input sequence.
        failures=[r for r in rows if not r['primary_success'] and r['arm']!='L']
        chosen=[]
        for arm in ('H','M','M1','Hstar'):
            candidates=[r for r in failures if r['arm']==arm]
            if candidates:chosen.append(candidates[0])
        chosen.extend(r for r in failures if r not in chosen and len(chosen)<5)
        for i,row in enumerate(chosen[:5]):
            data=np.load(ROOT/row['raw_path']);t,X,U=data['t'],data['X'],data['U']
            steps=data['solve_steps'];pred=data['predictions']
            fig,axes=plt.subplots(4,1,figsize=(8,9),sharex=False)
            axes[0].plot(t,X[:,2],label='Actual closed loop')
            for j in np.unique(np.linspace(0,max(0,len(steps)-1),min(5,len(steps))).astype(int)) if len(steps) else []:
                k=int(steps[j]);prediction=pred[j];end=min(len(X),k+len(prediction))
                axes[0].plot(t[k:end],prediction[:end-k,2],alpha=.6,linestyle='--',label='Predicted continuation' if j==0 else None)
            axes[0].axhline(np.pi/2,color='gray',ls=':');axes[0].axhline(-np.pi/2,color='gray',ls=':')
            axes[0].set_ylabel('θ [rad]');axes[0].legend(fontsize=8)
            matched=[]
            if 'plan_inputs' in data:
                for j in np.unique(np.linspace(0,max(0,len(steps)-1),min(5,len(steps))).astype(int)) if len(steps) else []:
                    actual=[X[int(steps[j])].copy()]
                    for value in data['plan_inputs'][j]:
                        actual.append(runner._plant_step(actual[-1],value,runner.make_config('dev').dt))
                    actual=np.asarray(actual);prediction=pred[j]
                    error=prediction-actual
                    axes[1].plot(np.arange(len(error))*.02,error[:,2],label=f'At t={steps[j]*.02:g}s')
                    matched.append(dict(planning_step=int(steps[j]),rmse_by_state=[float(v) if np.isfinite(v) else None for v in np.sqrt(np.mean(error**2,axis=0))],
                                        final_error_by_state=[float(v) if np.isfinite(v) else None for v in error[-1]]))
            dump(out/f'failure_{i+1}_prediction_errors.json',dict(case=row['case'],matched_input_replays=matched))
            axes[1].set(xlabel='Matched plan horizon [s]',ylabel='Predicted minus true θ [rad]');axes[1].legend(fontsize=8)
            axes[2].plot(t,X[:,0]);axes[2].set_ylabel('p [m]')
            axes[3].step(t[:-1],U[:,0]);axes[3].set_ylabel('u');axes[3].set_xlabel('Physical time [s]')
            fig.suptitle(row['case']+'; matched-input replay separates model error')
            fig.tight_layout();fig.savefig(out/f'failure_{i+1}.png',dpi=150);plt.close(fig)
        dump(out/'representative_failures.json',chosen[:5])
    if experiment=='E2':
        fig,ax=plt.subplots(figsize=(8,4))
        for arm in arms:
            for name in ('A','C50'):
                subset=[r for r in rates if r['arm']==arm and r['ic_set']==name]
                budgets=sorted({r['episodes'] for r in subset})
                med=[np.mean([r['rate'] for r in subset if r['episodes']==b]) for b in budgets]
                fit_counts=[json.loads((OUT/'models'/f'episodes{b}_seed0/diagnostics.json').read_text())['local']['training_transitions'] for b in budgets]
                ax.plot(fit_counts,med,'o-',label=f'{arm}, IC-{name}')
        ax.set(xscale='log',xlabel='Fitted training transitions (whole-episode split)',ylabel='Mean primary success rate across training seeds')
        ax.legend(fontsize=8);fig.tight_layout();fig.savefig(out/'data_efficiency.png',dpi=150);plt.close(fig)

if __name__=='__main__':
    import sys
    figures(sys.argv[1])
