"""Standalone checkpoint figures: common axes, actual tests, separate cohorts."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import *

PASS='#238b45';FAIL='#cb181d';MISSING='#c7c7c7'


def row_index(row):
    if 'index' in row:return row['index']
    case=row.get('identity',{}).get('case',{})
    return case.get('index',int(row['case'].rsplit('_',2)[1]))


def row_state(row):
    if 'initial_state' in row:return row['initial_state']
    return row['identity']['case']['state']


def main():
    isolation('dfine');budget=Budget('preparation',2);budget.checkpoint();out=OUT/'figures';out.mkdir(exist_ok=True);original=json.loads((ROOT/'validation/results_v1/initial_conditions.json').read_text())['sets']
    # Introductory map is H/L only. The source cases are read, never rerun.
    fig,axes=plt.subplots(1,2,figsize=(10,4),sharex=True,sharey=True)
    D=np.asarray(original['D']);positions=D[:,[2,3]]*np.array([180/np.pi,np.sqrt(9.81)])
    for ax,arm in zip(axes,['H','L']):
        ax.scatter(*positions.T,c=MISSING,s=8,marker='s');tested=[];colors=[]
        for p in (ROOT/'validation/results_v1/E1/cases').glob(f'seed0_{arm}_D_*_14s.json'):
            row=json.loads(p.read_text());i=row_index(row);assert row_state(row)==original['D'][i];tested.append(positions[i]);colors.append(PASS if row['primary_success'] else FAIL)
        if tested:ax.scatter(*np.asarray(tested).T,c=colors,s=8,marker='s')
        ax.set_title(f'{arm}, original seed 0, {len(tested)}/{len(D)} tested');ax.set_xlabel('Initial angle (degrees)')
    axes[0].set_ylabel('Initial angular speed (rad/s)');fig.suptitle('Authored zero-position/zero-cart-speed map: H and L');fig.tight_layout();fig.savefig(out/'intro_H_L.png',dpi=180);fig.savefig(out/'intro_H_L.svg');plt.close(fig);budget.checkpoint()
    panels=[('edmd','QP','original'),('edmd','QP+L','original'),('sindy','CEM','original'),('sindy','CEM+L','original'),('sindy','NMPC_exact','original'),('sindy','NMPC_exact+L','original'),('smc_C200','SMC','original'),('pets','PETS','original'),('pets/deterministic_reference','deterministic_ensemble','original'),('deepkoopman_control','MPC','original'),('deepkoopman_control','MPC+L','original'),('neural_first_action','NN_MPC','original'),('neural_first_action','NN_MPC+L','original'),('dfine_control','MPC','original'),('dfine_control','MPC+L','original'),('dfine_control','filtered_L','original'),('rokdeepc','RoKDeePC','E4'),('rokdeepc','linear_kernel','E4')]
    panels += [('dfine_control/output_only',arm,'output-only') for arm in ['DFINE_MPC','DFINE_MPC+L','DFINE_filtered_L','NN_history_MPC','NN_history_MPC+L','history_L']]
    states=np.asarray(original['C']);xy=states[:,[2,3]]*np.array([180/np.pi,np.sqrt(9.81)])
    fig,axes=plt.subplots(8,3,figsize=(13,26),sharex=True,sharey=True)
    inventories=[]
    for ax,(method,arm,cohort) in zip(axes.flat,panels):
        ax.scatter(*xy.T,c=MISSING,s=13);rows=[]
        for p in (OUT/method/'cases').glob('seed0_*.json'):
            row=json.loads(p.read_text())
            if row['arm']==arm and row.get('set','C')=='C' and row.get('seconds',14.)==14.:rows.append(row)
        indices=[row_index(r) for r in rows];assert len(set(indices))==len(indices)
        for r,i in zip(rows,indices):assert row_state(r)==original['C'][i]
        if indices:ax.scatter(*xy[indices].T,c=[PASS if r['primary_success'] else FAIL for r in rows],s=13)
        successes=sum(r['primary_success'] for r in rows);ax.set_title(f'{cohort}: {method} {arm}\n{successes}/{len(rows)} tested, requested 200',fontsize=9);inventories.append(dict(method=method,arm=arm,cohort=cohort,seed=0,tested_indices=indices))
    for ax in axes[-1]:ax.set_xlabel('Initial angle (degrees)')
    for ax in axes[:,0]:ax.set_ylabel('Initial angular speed (rad/s)')
    fig.suptitle('Random-state projections; cart position/speed vary\nGreen: success; red: unsuccessful; gray: untested. No interpolation.',fontsize=12);fig.tight_layout(rect=[0,0,1,.965]);fig.savefig(out/'controller_random_projections.png',dpi=160);fig.savefig(out/'controller_random_projections.svg');plt.close(fig);budget.checkpoint()
    # Fixed operating regions are distinct from random four-state projections.
    targets=[i*41+j for i in range(0,41,2) for j in range(0,41,2)]
    assert len(targets)==441 and np.all(D[:,:2]==0.)
    fig,axes=plt.subplots(3,2,figsize=(10,12),sharex=True,sharey=True);regions=[]
    for seed in range(3):
        records=[json.loads(p.read_text()) for p in (OUT/'edmd/cases').glob(f'seed{seed}_*_D_*_14s.json')]
        for column,arm in enumerate(['QP','QP+L']):
            ax=axes[seed,column];rows=[r for r in records if r['arm']==arm];indices=[row_index(r) for r in rows]
            assert len(indices)==len(set(indices)) and set(indices)<=set(targets)
            for r,i in zip(rows,indices):assert row_state(r)==original['D'][i]
            ax.scatter(*positions[targets].T,c=MISSING,s=15,marker='s')
            if indices:ax.scatter(*positions[indices].T,c=[PASS if r['primary_success'] else FAIL for r in rows],s=15,marker='s')
            ax.set_title(f'EDMD {arm}, seed {seed}: {sum(r["primary_success"] for r in rows)}/{len(rows)}, targeted 441')
            ax.set_xlabel('Initial angle (degrees)');ax.set_ylabel('Initial angular speed (rad/s)')
            regions.append(dict(method='edmd',arm=arm,seed=seed,set='D',targeted_indices=targets,tested_indices=indices))
        budget.checkpoint()
    fig.suptitle('Fixed zero-position/zero-cart-speed operating regions\nGray points are untested; no interpolation or region certification')
    fig.tight_layout(rect=[0,0,1,.95]);fig.savefig(out/'edmd_fixed_regions.png',dpi=180);fig.savefig(out/'edmd_fixed_regions.svg');plt.close(fig)
    rows=json.loads((OUT/'prediction_tables.json').read_text());fig,ax=plt.subplots(figsize=(8,4))
    for method in ['passive_linear','passive_edmd','passive_neural','deepkoopman_passive','dfine_passive']:
        group=[r for r in rows if r['method']==method];values=np.array([[r['standardized_mse_by_horizon'][str(h)] if r['standardized_mse_by_horizon'][str(h)] is not None else np.nan for h in [1,10,30]] for r in group]);valid=np.isfinite(values).all(0)
        if np.any(valid):ax.plot(np.array([1,10,30])[valid],np.mean(values[:,valid],axis=0),marker='o',label=method)
    ax.set_yscale('log');ax.set_xlabel('Forecast horizon (samples, 20 ms each)');ax.set_ylabel('Train-standardized squared error');ax.set_title('Matched passive held-out windows; mean across three fitting seeds');ax.legend(fontsize=8);fig.tight_layout();fig.savefig(out/'passive_forecasts.png',dpi=180);fig.savefig(out/'passive_forecasts.svg');plt.close(fig)
    fig,ax=plt.subplots(figsize=(11,5));labels=[];values=[]
    for method,arm,cohort in panels:
        samples=[]
        for p in (OUT/method/'cases').glob('*.json'):
            row=json.loads(p.read_text());quantiles=row.get('solve_latency_quantiles',[])
            if row['arm']==arm and quantiles and quantiles[0]>0:samples.append(1000*quantiles[0])
        if samples:labels.append(f'{cohort}\n{method}\n{arm}');values.append(samples)
    if values:ax.boxplot(values,tick_labels=labels,showfliers=False)
    ax.axhline(20.,color='black',linestyle='--',label='Physical sample period, 20 ms');ax.set_yscale('log');ax.set_ylabel('Case-median solver time (ms)');ax.set_title('Between-case solver latency; filtering/housekeeping excluded\nOffline fixed-step simulation does not establish a real-time guarantee');ax.tick_params(axis='x',labelsize=6);ax.legend(fontsize=8);fig.tight_layout();fig.savefig(out/'solver_latency.png',dpi=180);fig.savefig(out/'solver_latency.svg');plt.close(fig);budget.checkpoint()
    fig,ax=plt.subplots(figsize=(11,4));labels=['Original\nacquired','Original\ntraining pool','E4\nacquired','E4\ntraining pool','E4 kernel\nfit targets','Passive\nacquired','Passive\ntraining pool','Passive\nfit targets'];counts=[105000,73500,104992,73504,7936,99000,69300,63000];ax.bar(labels,counts,color=['#2171b5','#6baed6','#2171b5','#6baed6','#9ecae1','#238b45','#74c476','#c7e9c0']);ax.set_ylabel('State-transition intervals per collection');ax.set_title('Acquisition, available training pool and direct fit targets\nKernel: 512 overlapping windows / 256 episodes; passive latent: 90 intervals per training episode');fig.tight_layout();fig.savefig(out/'data_cohorts.png',dpi=180);fig.savefig(out/'data_cohorts.svg');plt.close(fig)
    dump(out/'manifest.json',dict(source_sha256=digest(__file__),initial_conditions_sha256=digest(ROOT/'validation/results_v1/initial_conditions.json'),panels=inventories,fixed_regions=regions,interpolation=False,colors=dict(success=PASS,unsuccessful=FAIL,untested=MISSING),checkpoint=True,cohorts_separate=True));dump(OUT/'figures/completion.json',dict(state='complete',accounting=budget.checkpoint()))


if __name__=='__main__':main()
