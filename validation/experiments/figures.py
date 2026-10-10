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
        from failure_diagnostics import from_rest_spans
        intervals=[]
        for seed in sorted({r['seed'] for r in rows}):
            for arm in arms:
                items=[r for r in rows if r['seed']==seed and r['arm']==arm and r['ic_set']=='B' and r['seconds']==14.]
                spans=from_rest_spans(items)
                for span in spans:intervals.append(dict(seed=seed,arm=arm,**span))
                if not spans:intervals.append(dict(seed=seed,arm=arm,start_degrees=None,end_degrees=None,tested_points=0))
        csv_write(out/'from_rest_intervals.csv',intervals)
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
        from failure_diagnostics import witness,select_representative_failures
        chosen=select_representative_failures(rows)
        for i,row in enumerate(chosen[:5]):
            data=np.load(ROOT/row['raw_path']);t,X,U=data['t'],data['X'],data['U']
            steps=data['solve_steps'];pred=data['predictions']
            fig,axes=plt.subplots(4,1,figsize=(8,9),sharex=False)
            axes[0].plot(t,X[:,2],label='Actual closed loop')
            for j in np.unique(np.linspace(0,max(0,len(steps)-1),min(5,len(steps))).astype(int)) if len(steps) else []:
                k=int(steps[j]);prediction=pred[j];end=min(len(X),k+len(prediction))
                axes[0].plot(t[k:end],prediction[:end-k,2],alpha=.6,linestyle='--',label='Predicted continuation' if j==0 else None)
            axes[0].axhline(np.pi/2,color='gray',ls=':');axes[0].axhline(-np.pi/2,color='gray',ls=':')
            exits=np.flatnonzero(data['local'][:-1]&~data['local'][1:])+1
            for j,k in enumerate(exits):
                axes[0].axvline(t[k],color='tab:orange',ls=':',alpha=.45,label='Exit local feedback' if j==0 else None)
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
                    matched.append(dict(planning_step=int(steps[j]),optimizer_and_model_witness=witness(row['seed'],row['arm'],X[int(steps[j])],data['plan_inputs'][j],prediction,data['loss'][j]),rmse_by_state=[float(v) if np.isfinite(v) else None for v in np.sqrt(np.mean(error**2,axis=0))],
                                        final_error_by_state=[float(v) if np.isfinite(v) else None for v in error[-1]]))
            dump(out/f'failure_{i+1}_prediction_errors.json',dict(case=row['case'],matched_input_replays=matched,
                switch_exit_times_seconds=t[exits].tolist(),
                fell_within_1s_of_switch_exit=row['fell_within_1s_of_switch_exit']))
            axes[1].set(xlabel='Matched plan horizon [s]',ylabel='Predicted minus true θ [rad]');axes[1].legend(fontsize=8)
            axes[2].plot(t,X[:,0]);axes[2].set_ylabel('p [m]')
            axes[3].step(t[:-1],U[:,0]);axes[3].set_ylabel('u');axes[3].set_xlabel('Physical time [s]')
            fig.suptitle(row['case']+'\n'+row['diagnostic_selection']+'; matched-input replay')
            fig.tight_layout();fig.savefig(out/f'failure_{i+1}.png',dpi=150);plt.close(fig)
            row['diagnostic_figure_path']=str((out/f'failure_{i+1}.png').relative_to(OUT))
            row['matched_input_replays_path']=str((out/f'failure_{i+1}_prediction_errors.json').relative_to(OUT))
        dump(out/'representative_failures.json',chosen[:5])
        if json.loads((out/'manifest.json').read_text())['status']=='DONE':
            update_e1_answer(rates)
    if experiment=='E3':
        fig,axes=plt.subplots(1,3,figsize=(15,4),sharey=True)
        modes=[('measurement',['measurement_0.01','measurement_0.03','measurement_0.1']),
               ('process',['process_0.01','process_0.03','process_0.1']),
               ('mismatch/delay',['mass_minus25','mass_plus25','length_minus25','length_plus25','input_delay'])]
        for ax,(mode,conditions) in zip(axes,modes):
            for a,arm in enumerate(('H','L')):
                for j,name in enumerate(('A','C50')):
                    for seed in (0,1,2):
                        subset=[r for r in rates if r['arm']==arm and r['seed']==seed and r['ic_set'] in [condition+'_'+name for condition in conditions]]
                        subset=sorted(subset,key=lambda r:conditions.index(r['ic_set'].rsplit('_',1)[0]))
                        positions=[conditions.index(r['ic_set'].rsplit('_',1)[0])+.04*(seed-1)+.15*(a-.5) for r in subset]
                        ax.plot(positions,[r['rate'] for r in subset],marker='o' if j==0 else 'x',color=f'C{a}',alpha=.55,
                                label=arm+', '+name if seed==0 else None)
            ax.set(xticks=range(len(conditions)),xticklabels=[c.split('_',1)[-1] for c in conditions],title=mode,ylim=(-.03,1.03))
            ax.tick_params(axis='x',rotation=30)
        axes[0].set_ylabel('Primary success rate; seeds shown separately');axes[-1].legend(fontsize=8)
        fig.tight_layout();fig.savefig(out/'robustness_conditions.png',dpi=150);plt.close(fig)
    if experiment=='E2':
        diagnostics=[]
        budgets=sorted({r['episodes'] for r in rates})
        for budget in budgets:
            for seed in sorted({r['seed'] for r in rates if r['episodes']==budget}):
                item=json.loads((OUT/'models'/f'episodes{budget}_seed{seed}/diagnostics.json').read_text())
                local=item['local'];pred=item['prediction']
                row=dict(episodes=budget,seed=seed,**{k:v for k,v in local.items() if k not in ('A','B','gain','fit_rmse')})
                for name in ('one_step_rmse_by_state','N_step_rmse_by_state'):
                    row.update({f'{name}_{state}':value for state,value in zip(('p','v','theta','omega'),pred[name])})
                row.update({f'local_rmse_{state}':value for state,value in zip(('p','v','theta','omega'),local['fit_rmse'])})
                diagnostics.append(row)
        csv_write(out/'model_diagnostics.csv',diagnostics)
        fig,axes=plt.subplots(1,3,figsize=(13,4))
        for ax,prefix,title in zip(axes[:2],('N_step_rmse_by_state','local_rmse'),('Held-out 30-step RMSE','Local linear fit RMSE')):
            for i,state in enumerate(('p','v','theta','omega')):
                for seed in sorted({r['seed'] for r in diagnostics}):
                    items=sorted([r for r in diagnostics if r['seed']==seed],key=lambda r:r['training_transitions'])
                    ax.plot([r['training_transitions'] for r in items],[r[prefix+'_'+state] for r in items],'.-',alpha=.5,color=f'C{i}',label=state if seed==0 else None)
            ax.set(xscale='log',yscale='log',title=title,xlabel='Fitted training transitions',ylabel='RMSE in stored state units');ax.legend(fontsize=8)
        for seed in sorted({r['seed'] for r in diagnostics}):
            items=sorted([r for r in diagnostics if r['seed']==seed],key=lambda r:r['training_transitions'])
            axes[2].plot([r['training_transitions'] for r in items],[np.rad2deg(r['max_abs_theta']) for r in items],'.-',label=f'seed {seed}')
        axes[2].set(xscale='log',title='6000-nearest neighborhood',xlabel='Fitted training transitions',ylabel='Max |θ| in selected fit [degrees]');axes[2].legend(fontsize=8)
        fig.tight_layout();fig.savefig(out/'model_quality_vs_data.png',dpi=150);plt.close(fig)
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

def update_e1_answer(rates):
    report=OUT/'RESULTS.md';text=report.read_text()
    old='**E1 is running; data efficiency and robustness remain unresolved pending E2–E3.**'
    if old not in text:return
    answer='**E1 is complete; operating regions are measured for three seeds. Data efficiency and robustness remain unresolved pending E2–E3.**\n\n'
    answer+='The table gives the range of primary success counts across seeds 0–2, with each seed kept separate in the [rates and Wilson intervals](E1/rates.csv). These are tested ICs, not continuous-domain guarantees.\n\n'
    answer+='| IC set | H | L | M | M1 | H* |\n|---|---|---|---|---|---|\n'
    for name,label in [('B','B: from rest'),('C','C: random box'),('D','D: angle/rate grid')]:
        cells=[]
        for arm in ('H','L','M','M1','Hstar'):
            items=[r for r in rates if r['ic_set']==name and r['arm']==arm and r['seconds']==14.]
            counts=[r['successes'] for r in items];denominators={r['n'] for r in items}
            if len(items)!=3 or len(denominators)!=1:raise ValueError('Completed E1 requires matched full sets for all three seeds')
            lo,hi=min(counts),max(counts);n=next(iter(denominators))
            cells.append(f'{lo}/{n}' if lo==hi else f'{lo}–{hi}/{n}')
        answer+='| '+label+' | '+' | '.join(cells)+' |\n'
    answer+='\n[Exact tested success intervals from rest](E1/from_rest_intervals.csv) preserve holes in the 1° grid. The [D map for seed 0](E1/map_D_seed0.png) and [cart-excursion map](E1/cart_excursion_D_seed0.png) have counterparts for seeds 1–2. [Paired comparisons](E1/paired_comparisons.csv), [overlapping failure categories](E1/failure_categories.csv) and [five selected failure diagnoses and evidence paths](E1/representative_failures.json) ([example plot](E1/failure_1.png)) separate the endpoints, switch associations and model/optimizer witnesses. Cart excursions remain descriptive; no rail criterion is used.\n\n'
    report.write_text(text.replace(old,answer))


if __name__=='__main__':
    import sys
    figures(sys.argv[1])
