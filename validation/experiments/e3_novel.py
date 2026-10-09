"""Novel LTI and uniform-rod acrobot experiments after E2 and robustness."""
from novel_plants import *
import concurrent.futures
import multiprocessing
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CACHE={}


def init():
    torch.set_num_threads(1);runner.tqdm=lambda it,**kw:it


def initial_conditions(out,plant):
    path=out/'initial_conditions.json'
    if path.exists():return json.loads(path.read_text())['sets']
    rng=np.random.default_rng(20261009)
    if plant.name=='linear':A=[(np.ones(4)*v/2).tolist() for v in (-1.,-.5,-.25,.25,.5,1.)]
    else:A=[[float(np.deg2rad(v)),0.,0.,0.] for v in (-60,-30,-15,15,30,60)]
    C=[]
    while len(C)<50:
        x=rng.uniform(plant.low,plant.high)
        if plant.name=='acrobot' and abs(x[0]+x[1])>1.3:continue
        C.append(x.tolist())
    sets=dict(A=A,C50=C)
    dump(path,dict(seed=20261009,sets=sets,notes='Acrobot random ICs reject absolute second-link angles outside ±1.3 rad; collection does not.'))
    return sets


def fit_model(out,plant,seed):
    config=runner.make_config('dev',seed=seed,device='cuda',umax=plant.umax)
    np.random.seed(seed);torch.manual_seed(seed)
    path=out/'models'/f'seed{seed}';path.mkdir(parents=True,exist_ok=True)
    if (path/'data.npz').exists():
        data=np.load(path/'data.npz');X,U=data['X'],data['U']
    else:
        X,U=collect(plant,config);np.savez_compressed(path/'data.npz',X=X,U=U)
    data=runner.split_episodes(X,U,seed)
    if (path/'transition.pt').exists():model,_=runner.load_transition(path/'transition.pt','cuda')
    else:model,_=train(plant,config,data,path/'transition.pt')
    gain,local=runner.fit_local_feedback(data['train'],plant.Q,plant.R)
    x=data['train'][0][:,:-1].reshape(-1,plant.n).astype(float)
    norms=np.linalg.norm(x/np.std(x,axis=0).clip(1e-6),axis=1)
    nearest=np.argsort(norms)[:min(6000,len(x))]
    decoder=CanonicalBehaviorDecoder(model,30).to('cuda').eval()
    dump(path/'diagnostics.json',dict(config=asdict(config),plant=plant.name,n=plant.n,m=plant.m,
        prediction=prediction_rmse(decoder,data['test']),
        local=dict(gain=gain.tolist(),A=local['A'].tolist(),B=local['B'].tolist(),fit_rmse=local['fit_rmse'].tolist(),
                   training_transitions=len(x),selected_samples=len(nearest),uses_all_data=len(x)<=6000,
                   max_normalized_norm=float(norms[nearest].max()),max_abs_by_state=np.max(np.abs(x[nearest]),axis=0).tolist())))


def load_model(out,plant,seed):
    path=out/'models'/f'seed{seed}'
    model,payload=runner.load_transition(path/'transition.pt','cuda')
    decoder=CanonicalBehaviorDecoder(model,30).to('cuda').eval()
    gain=np.asarray(json.loads((path/'diagnostics.json').read_text())['local']['gain'])
    config=runner.make_config('dev',seed=seed,device='cuda',umax=plant.umax)
    return decoder,gain,config


def exact_linear_reference(plant):
    A=np.diag([1.06,1.02,.96,.90]);B=np.array([.2,.2,.1,.1])[:,None]
    from scipy.linalg import solve_discrete_are
    P=solve_discrete_are(A,B,plant.Q,plant.R)
    gain=np.linalg.solve(plant.R+B.T@P@B,B.T@P@A)
    return gain,P


def evaluate(task):
    name,seed,arm,ic_set,index,state,seconds=task
    out=OUT/('E3_'+name);key=f'seed{seed}_{arm}_{ic_set}_{index:04d}_{seconds:g}s'
    path=out/'cases'/(key+'.json')
    if path.exists():return json.loads(path.read_text())
    if (name,seed) not in CACHE:
        plant=make_plant(name);CACHE[name,seed]=(plant,*load_model(out,plant,seed))
    plant,decoder,gain,config=CACHE[name,seed]
    P=None
    if arm=='LQR':
        gain,P=exact_linear_reference(plant);ctl=controller(decoder,config,gain,'L')
    else:ctl=make_control(plant,decoder,config,gain,arm)
    start=time.perf_counter()
    with np.errstate(over='ignore',invalid='ignore'):
        t,X,U=simulate(plant,ctl,np.asarray(state),config,seconds)
    row=dict(case=key,plant=name,seed=seed,arm=arm,ic_set=ic_set,ic_index=index,seconds=seconds,
        initial_state=state,wall_seconds=time.perf_counter()-start,
        **endpoint(plant,t,X,U,ctl.diagnostics['terminal_policy_used']))
    cost=float(np.einsum('bi,ij,bj->',X[:-1],plant.Q,X[:-1])+np.einsum('bi,ij,bj->',U,plant.R,U))
    row['accumulated_stage_cost']=cost if np.isfinite(cost) else None
    if name=='linear':
        _,P=exact_linear_reference(plant)
        reference=float(np.asarray(state)@P@np.asarray(state))
        tail=float(X[-1]@P@X[-1])
        row.update(unconstrained_optimal_infinite_cost=reference,
                   cost_plus_optimal_tail=cost+tail if np.isfinite(cost+tail) else None)
    row['nonfinite_cem']=int(np.sum(~np.isfinite(ctl.diagnostics['loss'])))
    row['cem_solves']=len(ctl.diagnostics['loss'])
    raw=out/'raw'/(key+'.npz');raw.parent.mkdir(exist_ok=True)
    np.savez_compressed(raw,t=t,X=X,U=U,local=ctl.diagnostics['terminal_policy_used'],
                        predictions=np.asarray(ctl.diagnostics['predictions']))
    row['raw_path']=str(raw.relative_to(ROOT));dump(path,row)
    print(json.dumps({k:row[k] for k in ('case','primary_success','settle_seconds','local_fraction','wall_seconds')}),flush=True)
    return row


def summarize_novel(out,rows):
    rows=sorted(rows,key=lambda r:r['case']);csv_write(out/'summary.csv',rows)
    rates=[];paired=[]
    from scipy.stats import binomtest
    for seed in range(3):
        for name in ('A','C50'):
            for seconds in (14.,60.):
                conditions=[r for r in rows if r['seed']==seed and r['ic_set']==name and r['seconds']==seconds]
                H={r['ic_index']:r['primary_success'] for r in conditions if r['arm']=='H'}
                for arm in sorted({r['arm'] for r in conditions}):
                    subset=[r for r in conditions if r['arm']==arm];n=len(subset);s=sum(r['primary_success'] for r in subset)
                    lo,hi=wilson(s,n)
                    rates.append(dict(seed=seed,ic_set=name,seconds=seconds,arm=arm,n=n,successes=s,wilson95_low=lo,wilson95_high=hi))
                    if arm=='H':continue
                    A={r['ic_index']:r['primary_success'] for r in subset};shared=H.keys()&A.keys()
                    if not shared:continue
                    b=sum(H[i] and not A[i] for i in shared);c=sum(A[i] and not H[i] for i in shared)
                    paired.append(dict(seed=seed,ic_set=name,seconds=seconds,arm=arm,n_paired=len(shared),H_only=b,arm_only=c,
                        exact_mcnemar_p=float(binomtest(b,b+c,.5).pvalue) if b+c else 1.))
    csv_write(out/'rates.csv',rates);csv_write(out/'paired_comparisons.csv',paired)
    return rates


def main(name):
    plant=make_plant(name);out=OUT/('E3_'+name);out.mkdir(exist_ok=True)
    arms=['H','L','M1']+(['LQR'] if name=='linear' else [])
    choices=dict(n=plant.n,m=plant.m,box_low=plant.low.tolist(),box_high=plant.high.tolist(),Q=plant.Q.tolist(),R=plant.R.tolist(),
        umax=plant.umax,arms=arms,workers=8,profile='dev',episodes=3500,horizon=30,epochs=100,simulation_seconds=[14,60],
        cost_rule='LTI identity Q; acrobot authored 100-per-angle/10-per-rate; R=.03, no per-arm tuning',
        endpoint='LTI all |state components|≤.02 in last 3 s; acrobot both absolute angles≤.02 and physical angular rates≤.05 in last 3 s, no link falls')
    if name=='linear':choices.update(A=np.diag([1.06,1.02,.96,.90]).tolist(),B=[[.2],[.2],[.1],[.1]],sampling='A/B are discrete per 0.02 physical seconds',reference='Infinite-horizon unconstrained LQR, clipped at ±10 for simulation; tail-cost lower bound remains unconstrained')
    else:choices.update(masses=[1,1],lengths=[1,1],COM=[.5,.5],COM_inertias=[1/12,1/12],gravity=9.81,
        coordinates=['q1 from upright','q2 relative','q1_dot/sqrt(g)','q2_dot/sqrt(g)'],input='torque/(m2*g*l2), limit ±1',
        integration='Eight RK4 substeps per 0.02 physical sample; no angle wrapping or velocity clipping',
        source='Uniform-rod Lagrange mass matrix; same equation structure as https://raw.githubusercontent.com/Farama-Foundation/Gymnasium/main/gymnasium/envs/classic_control/acrobot.py, with explicit inertia and coordinate changes')
    manifest=start_manifest(out,'E3_'+name,[0,1,2],choices);start=time.perf_counter()
    try:
        sets=initial_conditions(out,plant);tasks=[]
        for seed in range(3):
            fit_model(out,plant,seed)
            for ic_set,states in sets.items():
                for i,state in enumerate(states):
                    for arm in arms:tasks.append((name,seed,arm,ic_set,i,state,14.))
            for i in (0,2,5):
                for arm in arms:tasks.append((name,seed,arm,'A',i,sets['A'][i],60.))
        saved={r['case']:r for r in [json.loads(p.read_text()) for p in (out/'cases').glob('*.json')]};last=time.perf_counter()
        with concurrent.futures.ProcessPoolExecutor(8,mp_context=multiprocessing.get_context('spawn'),initializer=init) as pool:
            futures=[pool.submit(evaluate,task) for task in tasks]
            for future in concurrent.futures.as_completed(futures):
                row=future.result();saved[row['case']]=row
                if time.perf_counter()-last>60:
                    summarize_novel(out,list(saved.values()));last=time.perf_counter()
                    manifest['wall_seconds']=time.perf_counter()-start;dump(out/'manifest.json',manifest)
        rates=summarize_novel(out,list(saved.values()))
        fig,axes=plt.subplots(1,2,figsize=(9,4))
        for ax,ic_set in zip(axes,('A','C50')):
            for i,arm in enumerate(arms):
                vals=[r['successes']/r['n'] for r in rates if r['arm']==arm and r['seconds']==14. and r['ic_set']==ic_set]
                ax.scatter([i]*len(vals),vals)
            ax.set(xticks=range(len(arms)),xticklabels=arms,ylim=(-.03,1.03),title=ic_set,ylabel='Primary success per seed')
        fig.suptitle(name+'; plant-specific endpoints');fig.tight_layout();fig.savefig(out/'success_rates.png',dpi=150);plt.close(fig)
        manifest['status']='DONE'
    except BaseException as error:manifest.update(status='ERROR',error=repr(error));raise
    finally:manifest['wall_seconds']=time.perf_counter()-start;dump(out/'manifest.json',manifest)

if __name__=='__main__':
    import sys
    init();main(sys.argv[1])
