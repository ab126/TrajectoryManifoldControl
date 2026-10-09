"""Resumable matched-arm validation; submitted training and H remain unchanged."""
from bootstrap import *
import argparse
import csv
import json
import math
import subprocess
import time
import types
from dataclasses import asdict

OUT = ROOT / 'validation/results_v1'
Q = np.diag([.02, 30., 100., 10.])
R = np.array([[.03]])
IC_SEED = 20261009
ARMS = ('H', 'L', 'M', 'M1', 'Hstar')


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def csv_write(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, keys)
        writer.writeheader()
        writer.writerows(rows)


def ic_sets():
    """Saved once; velocities are in the runner's stored normalized coordinates."""
    path = OUT / 'initial_conditions.json'
    if path.exists():
        return json.loads(path.read_text())['sets']
    def states(angles):
        return [[0., 0., float(np.deg2rad(a)), 0.] for a in angles]
    rng = np.random.default_rng(IC_SEED)
    sets = dict(A=states([-60,-30,-15,15,30,60]), B=states(range(-85,86)),
                C=rng.uniform([-1.5,-2.5,-1.3,-3.5],[1.5,2.5,1.3,3.5],(200,4)).tolist(),
                D=[[0.,0.,float(theta),float(omega)] for theta in np.linspace(-np.deg2rad(85),np.deg2rad(85),41)
                   for omega in np.linspace(-3.5,3.5,41)])
    dump(path, dict(seed=IC_SEED, sets=sets, C_subset=list(range(50)),
                    D_axes=dict(theta_degrees=[-85,85,41], theta_dot_normalized=[-3.5,3.5,41])))
    return sets


def settle_time(t, predicate):
    good = np.logical_and.accumulate(np.asarray(predicate, bool)[::-1])[::-1]
    eligible = np.flatnonzero(good & (t <= t[-1]-3.+1e-9))
    return float(t[eligible[0]]) if len(eligible) else None


def metrics(t, X, U, local, solve_steps=(), losses=()):
    """Reported angular endpoint delegates to the authored success_metrics."""
    authored = runner.success_metrics(t,X,U)
    finite = bool(np.isfinite(X).all() and np.isfinite(U).all())
    no_fall = finite and bool(np.all(np.abs(X[:,2]) < np.pi/2))
    angular_good = (np.abs(X[:,2]) <= .02) & (np.abs(X[:,3])*np.sqrt(9.81) <= .05)
    physical_speed = X[:,1]*np.sqrt(9.81)
    span = max(1, round(1./(t[1]-t[0])))
    rate = np.full(len(X), np.inf)
    rate[span:] = np.abs(X[span:,0]-X[:-span,0])/(t[span:]-t[:-span])
    full_good = angular_good & (np.abs(physical_speed)<=.05) & (rate<=.01)
    mask = np.asarray(local, bool)
    cost = np.einsum('bi,ij,bj->b', X[:-1], Q, X[:-1])
    exits = np.flatnonzero(mask[:-1] & ~mask[1:])+1 if len(mask)>1 else np.array([],int)
    falls = np.flatnonzero(~np.isfinite(X).all(axis=1) | (np.abs(X[:,2])>=np.pi/2))
    final = t>=t[-1]-3.-1e-9
    def number(v):
        return float(v) if np.isfinite(v) else None
    return dict(balanced=authored['balanced'],cart_rest=authored['cart_rest'], no_fall=no_fall,
        primary_success=bool(authored['balanced'] and no_fall), nonfinite_state=not finite,
        fell=bool(len(falls)), first_fall_seconds=float(t[falls[0]]) if len(falls) else None,
        max_abs_p=number(np.max(np.abs(X[:,0]))), final_p=number(X[-1,0]),
        angle_settle_seconds=settle_time(t, angular_good), full_settle_seconds=settle_time(t, full_good),
        final_theta=number(X[-1,2]), final_cart_speed_physical=number(physical_speed[-1]),
        tail_position_rate=number((X[-1,0]-X[np.flatnonzero(final)[0],0])/(t[-1]-t[np.flatnonzero(final)[0]])),
        max_abs_u=number(np.max(np.abs(U))), input_limit_seconds=float(np.sum(np.abs(U[:,0])>=10.-1e-6)*.02),
        local_fraction=float(mask.mean()) if len(mask) else 0., switch_toggles=int(np.sum(mask[1:]!=mask[:-1])),
        switch_exits=int(len(exits)), never_entered_switch=bool(not np.any(cost<.75)),
        balanced_with_cart_drift=bool(authored['balanced'] and (not authored['cart_rest'] or abs(X[-1,0]-X[np.flatnonzero(final)[0],0])/3.>.01)),
        nonfinite_cem=int(np.sum(~np.isfinite(losses))), cem_solves=len(losses),
        fell_within_1s_of_switch_exit=bool(len(falls) and np.any((falls[0]-exits>=0)&(falls[0]-exits<=50))))


def controller(decoder, config, gain, arm):
    if arm=='L':
        def control(k, state):
            control.diagnostics['terminal_policy_used'].append(True)
            return np.clip(-gain@state,-config.umax,config.umax).reshape(1)
        control.diagnostics=dict(terminal_policy_used=[], loss=[], predictions=[], solve_seconds=[])
        return control
    if arm=='M1':
        # Parameterize only the literal replan_interval=5 in the imported function.
        code = runner.make_controller.__code__
        indices = [i for i,c in enumerate(code.co_consts) if type(c) is int and c==5]
        if len(indices)!=1:
            raise RuntimeError('Submitted replan constant changed; inspect before parameterizing')
        values = list(code.co_consts)
        values[indices[0]]=1
        factory = types.FunctionType(code.replace(co_consts=tuple(values)), runner.make_controller.__globals__)
        return factory(decoder, config, None)
    return runner.make_controller(decoder, config, gain if arm in ('H','Hstar') else None)


def fit_diagnostics(data, local):
    X,U = data
    x=X[:,:-1].reshape(-1,4).astype(float)
    u=U.reshape(-1,1)
    radii=np.linalg.norm(x/np.std(x,axis=0).clip(1e-6),axis=1)
    nearest=np.argsort(radii)[:min(6000,len(x))]
    return dict(training_transitions=len(x),selected_samples=len(nearest),uses_all_data=len(x)<=6000,
                max_abs_theta=float(np.max(np.abs(x[nearest,2]))),
                max_normalized_norm=float(np.max(radii[nearest])),max_raw_norm=float(np.max(np.linalg.norm(x[nearest],axis=1))),
                input_range=[float(u[nearest].min()),float(u[nearest].max())],
                fit_rmse=local['fit_rmse'].tolist(),gain=local['gain'].tolist(), A=local['A'].tolist(), B=local['B'].tolist())


def prediction_rmse(decoder, data):
    X,U=data
    errors=[]; one=[]
    with torch.no_grad():
        for start in range(0,len(X),512):
            x=torch.from_numpy(X[start:start+512]).to(next(decoder.parameters()).device)
            u=torch.from_numpy(U[start:start+512]).to(x.device)
            errors.append((decoder.rollout(x[:,0],u)-x).cpu().numpy())
            one.append((decoder.transition(x[:,:-1].reshape(-1,4),u.reshape(-1,1))-x[:,1:].reshape(-1,4)).cpu().numpy())
    e=np.concatenate(errors); a=np.concatenate(one)
    return dict(one_step_rmse_by_state=np.sqrt(np.mean(a*a,axis=0)).tolist(),
        N_step_rmse_by_state=np.sqrt(np.mean(e[:,-1]**2,axis=0)).tolist(),
        rollout_rmse_by_state=np.sqrt(np.mean(e[:,1:]**2,axis=(0,1))).tolist(),
        rmse_by_step_and_state=np.sqrt(np.mean(e**2,axis=0)).tolist())


def train_model(seed, episodes=3500, data_root=None, collect=None):
    path=(OUT/'models' if data_root is None else Path(data_root))/f'episodes{episodes}_seed{seed}'
    path.mkdir(parents=True,exist_ok=True)
    config=runner.make_config('dev',seed=seed,episodes=episodes,device='cuda')
    np.random.seed(seed); torch.manual_seed(seed)
    data_path=path/'data.npz'
    if data_path.exists():
        data=np.load(data_path); X,U=data['X'],data['U']
    else:
        X,U=(runner.collect_trajectory_data if collect is None else collect)(config)
        np.savez_compressed(data_path,X=X,U=U)
    splits=runner.split_episodes(X,U,seed)
    checkpoint=path/'transition.pt'
    if checkpoint.exists():
        transition,_=runner.load_transition(checkpoint,'cuda')
    else:
        transition,_=runner.train_transition(splits['train'],splits['validation'],config,checkpoint)
    decoder=CanonicalBehaviorDecoder(transition,config.horizon).to('cuda').eval()
    gain,local=runner.fit_local_feedback(splits['train'],Q,R)
    if not (path/'diagnostics.json').exists():
        dump(path/'diagnostics.json',dict(seed=seed,config=asdict(config),
             prediction=prediction_rmse(decoder,splits['test']), local=fit_diagnostics(splits['train'],local)))
    return decoder,gain,config


def run_case(out, decoder, gain, config, arm, set_name, index, state, seconds=14., observe=None):
    key=f'seed{config.seed}_{arm}_{set_name}_{index:04d}_{seconds:g}s'
    if out.name=='E2':key=f'episodes{config.episodes}_'+key
    path=out/'cases'/f'{key}.json'
    if path.exists():
        return json.loads(path.read_text())
    ctl=controller(decoder,config,gain,arm)
    solve_steps=[]
    plan_inputs=[]
    cells=dict(zip(ctl.__code__.co_freevars,ctl.__closure__ or ()))
    if 'solver' in cells:
        solver=cells['solver'].cell_contents
        original_solve=solver.solve
        def record_solve(*args,**kwargs):
            solution=original_solve(*args,**kwargs)
            plan_inputs.append(solution.u.detach().cpu().numpy().copy())
            return solution
        solver.solve=record_solve
    def measured(k,state):
        before=len(ctl.diagnostics['loss'])
        u=ctl(k,state if observe is None else observe(k,state))
        if len(ctl.diagnostics['loss'])>before:
            solve_steps.append(k)
        return u
    start=time.perf_counter()
    # The original plant evaluator, including dt and physical time conversion.
    with np.errstate(over='ignore',invalid='ignore'):
        t,X,U=runner.simulate_plant(measured,np.asarray(state),config,seconds=seconds)
    elapsed=time.perf_counter()-start
    row=dict(case=key,seed=config.seed,episodes=config.episodes,arm=arm,ic_set=set_name,ic_index=index,
             seconds=seconds,initial_state=state,wall_seconds=elapsed,
             **metrics(t,X,U,ctl.diagnostics['terminal_policy_used'],solve_steps,ctl.diagnostics['loss']))
    raw=out/'raw'/f'{key}.npz'; raw.parent.mkdir(exist_ok=True)
    # Save trajectories and prediction locations for failure diagnosis.
    np.savez_compressed(raw,t=t,X=X,U=U,local=np.asarray(ctl.diagnostics['terminal_policy_used']),
        solve_steps=solve_steps,predictions=np.asarray(ctl.diagnostics['predictions']),plan_inputs=np.asarray(plan_inputs),
        loss=np.asarray(ctl.diagnostics['loss']),solve_seconds=ctl.diagnostics['solve_seconds'])
    row['raw_path']=str(raw.relative_to(ROOT))
    dump(path,row)
    print(json.dumps({k:row[k] for k in ('case','primary_success','balanced','cart_rest','max_abs_p','local_fraction','wall_seconds')}),flush=True)
    return row


def wilson(successes,n):
    if not n:return (None,None)
    z=1.959963984540054; p=successes/n; den=1+z*z/n
    mid=(p+z*z/(2*n))/den; spread=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return (float(mid-spread),float(mid+spread))


def summarize(out,rows=None):
    if rows is None:rows=[json.loads(p.read_text()) for p in sorted((out/'cases').glob('*.json'))]
    rows=sorted(rows,key=lambda row:row['case'])
    csv_write(out/'summary.csv',rows)
    groups={}
    for row in rows:
        key=(row['episodes'],row['seed'],row['ic_set'],row['seconds'],row['arm'])
        groups.setdefault(key,[]).append(row)
    rates=[]
    for key,items in groups.items():
        n=len(items); s=sum(r['primary_success'] for r in items); lo,hi=wilson(s,n)
        rates.append(dict(zip(('episodes','seed','ic_set','seconds','arm'),key),n=n,successes=s,
                          rate=s/n,wilson95_low=lo,wilson95_high=hi,
                          balanced=sum(r['balanced'] for r in items),cart_rest=sum(r['cart_rest'] for r in items),
                          fell=sum(r['fell'] for r in items),local_fraction=np.mean([r['local_fraction'] for r in items]),
                          median_excursion=np.median([r['max_abs_p'] for r in items if r['max_abs_p'] is not None])))
    csv_write(out/'rates.csv',rates)
    from scipy.stats import binomtest
    comparisons=[]
    for key in sorted({k[:-1] for k in groups}):
        h={r['ic_index']:r for r in groups.get((*key,'H'),[])}
        for arm in ('L','M','M1','Hstar'):
            a={r['ic_index']:r for r in groups.get((*key,arm),[])}
            shared=sorted(h.keys()&a.keys())
            if not shared:continue
            b=sum(h[i]['primary_success'] and not a[i]['primary_success'] for i in shared)
            c=sum(a[i]['primary_success'] and not h[i]['primary_success'] for i in shared)
            differences=np.array([int(h[i]['primary_success'])-int(a[i]['primary_success']) for i in shared])
            comparisons.append(dict(zip(('episodes','seed','ic_set','seconds'),key),reference='H',arm=arm,
                n_paired=len(shared),H_only=b,arm_only=c,success_rate_difference=float(differences.mean()),
                exact_mcnemar_p=float(binomtest(b,b+c,.5).pvalue) if b+c else 1.))
    csv_write(out/'paired_comparisons.csv',comparisons)
    return rows,rates


def start_manifest(out,experiment,seeds,config):
    path=out/'manifest.json'
    if path.exists():
        value=json.loads(path.read_text())
    else:
        value=dict(experiment=experiment,git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                   config=config,seeds=seeds,device='cuda',wall_seconds=0.,status='RUNNING',
                   cvxpy_unused_import_workaround=CVXPY_WORKAROUND,initial_conditions='validation/results_v1/initial_conditions.json',
                   oracle_exploration_scaling='learned training scaling; plant replaced only',
                   full_stabilization_position_rate_m_s=.01)
    value['status']='RUNNING'; dump(path,value)
    return value


def run_experiment(experiment,sets=None):
    out=OUT/experiment; out.mkdir(parents=True,exist_ok=True)
    seeds=list(range(10)) if experiment=='E0' else [0,1,2]
    episodes_list=[3500] if experiment!='E2' else [35,100,350,1000,3500,10000]
    if experiment=='E2':seeds=list(range(5))
    sets=sets or (['A'] if experiment=='E0' else ['B','C','D'] if experiment=='E1' else ['A','C50'])
    manifest=start_manifest(out,experiment,seeds,dict(profile='dev',episodes=episodes_list,arms=ARMS if experiment!='E2' else ['H','L','M1'],ic_sets=sets))
    start=time.perf_counter()
    try:
        all_ics=ic_sets()
        for episodes in episodes_list:
            for seed in seeds:
                decoder,gain,config=train_model(seed,episodes)
                from models import OracleTransition,check_oracle
                if experiment!='E2':
                    error=check_oracle('cpu'); manifest['oracle_equivalence_max_abs_error']=error
                    transition=OracleTransition('cuda')
                    transition.xu_std=decoder.transition.xu_std.detach().clone()
                    oracle=CanonicalBehaviorDecoder(transition,config.horizon).to('cuda').eval()
                for name in sets:
                    states=all_ics['C'][:50] if name=='C50' else all_ics[name]
                    for index,state in enumerate(states):
                        for arm in (ARMS if experiment!='E2' else ('H','L','M1')):
                            run_case(out,oracle if arm=='Hstar' else decoder,gain,config,arm,name,index,state)
                    summarize(out)
                if experiment=='E0':
                    for index,state in enumerate(all_ics['A']):
                        for arm in ('H','L','Hstar'):
                            run_case(out,oracle if arm=='Hstar' else decoder,gain,config,arm,'A',index,state,60.)
                    summarize(out)
                manifest['wall_seconds']+=time.perf_counter()-start; start=time.perf_counter()
                dump(out/'manifest.json',manifest)
        summarize(out); manifest['status']='DONE'
    except BaseException as error:
        manifest.update(status='ERROR',error=repr(error)); raise
    finally:
        manifest['wall_seconds']+=time.perf_counter()-start
        dump(out/'manifest.json',manifest)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('experiment',choices=['E0','E1','E2'])
    parser.add_argument('--sets',nargs='+')
    args=parser.parse_args()
    # Limit CPU threading, but retain the runner's default CUDA numerical settings.
    torch.set_num_threads(1)
    run_experiment(args.experiment,args.sets)
