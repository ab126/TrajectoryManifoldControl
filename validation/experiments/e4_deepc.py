"""Matched-data H/L/M1 and regularized DeePC comparison, after E0–E3.

32-step collection permits two observed past samples plus a 30-step prediction;
3281 episodes give 104992 transitions, versus authored 105000. Every arm in this
experiment uses that same new collection and episode split. No episodes joined.
"""
from campaign import *
from deepc import BehaviorBasis,windows,compress,DeePC
import concurrent.futures
import multiprocessing
from figures import figures
import osqp
from ensemble_baseline import train_members,load_members,make_ensemble_controller

CACHE={}
REGULARIZERS=(.001,.01,.1)
ARMS_E4=['H','L','M1','E5']+[f'DeePC_{kind}_{value:g}' for kind in ('state','output') for value in REGULARIZERS]


def initialize(include_ensemble=True):
    torch.set_num_threads(1);runner.tqdm=lambda it,**kw:it
    if not include_ensemble and 'E5' in ARMS_E4:ARMS_E4.remove('E5')


def save_basis(path,basis):
    np.savez_compressed(path.with_suffix('.npz'),B=basis.B,singular_values=basis.singular_values,scale=basis.scale)
    dump(path.with_suffix('.json'),dict(columns=basis.columns,rank=basis.B.shape[1],rows=basis.B.shape[0],
        horizon=basis.horizon,inputs=basis.inputs,tini=basis.tini,layout={key:[value.start,value.stop] for key,value in basis.layout.items()},
        relative_svd_tolerance=1e-8,coefficient_transform='Within the retained SVD subspace: Zg=B alpha; min ||g||²=||alpha||²/columns',
        retained_singular_values=basis.singular_values.tolist()))


def load_basis(path):
    data=np.load(path.with_suffix('.npz'));meta=json.loads(path.with_suffix('.json').read_text())
    return BehaviorBasis(data['B'],data['singular_values'],data['scale'],meta['columns'],
        {key:slice(*value) for key,value in meta['layout'].items()},meta['horizon'],meta['inputs'],meta['tini'])


def fit(seed):
    path=OUT/'E4/models'/f'seed{seed}';path.mkdir(parents=True,exist_ok=True)
    np.random.seed(seed);torch.manual_seed(seed)
    collection=runner.make_config('dev',seed=seed,episodes=3281,horizon=32,device='cuda')
    if (path/'data.npz').exists():
        data=np.load(path/'data.npz');X,U=data['X'],data['U']
    else:
        X,U=runner.collect_trajectory_data(collection);np.savez_compressed(path/'data.npz',X=X,U=U)
    splits=runner.split_episodes(X,U,seed)
    if (path/'transition.pt').exists():model,_=runner.load_transition(path/'transition.pt','cuda')
    else:model,_=runner.train_transition(splits['train'],splits['validation'],collection,path/'transition.pt')
    if 'E5' in ARMS_E4:train_members(path,splits['train'],splits['validation'],collection)
    decoder=CanonicalBehaviorDecoder(model,30).to('cuda').eval()
    gain,local=runner.fit_local_feedback(splits['train'],Q,R)
    for kind,tini,indices in [('state',0,None),('output',2,[0,2])]:
        if not path.joinpath(kind+'_basis.npz').exists():
            Z,layout=windows(*splits['train'],30,tini=tini,output_indices=indices)
            basis=compress(Z,layout,30,1,tini=tini);save_basis(path/(kind+'_basis'),basis)
    if not (path/'diagnostics.json').exists():
        test=splits['test']
        dump(path/'diagnostics.json',dict(config=asdict(collection),evaluation_horizon=30,
            collected_transitions=len(U)*U.shape[1],training_transitions=len(splits['train'][1])*32,
            prediction=prediction_rmse(decoder,(test[0][:,:31],test[1][:,:30])),local=fit_diagnostics(splits['train'],local)))


def load(seed):
    path=OUT/'E4/models'/f'seed{seed}'
    model,payload=runner.load_transition(path/'transition.pt','cuda')
    config=runner.ExperimentConfig(**{**payload['config'],'horizon':30,'hidden_dims':tuple(payload['config']['hidden_dims'])})
    decoder=CanonicalBehaviorDecoder(model,30).to('cuda').eval()
    gain=np.asarray(json.loads((path/'diagnostics.json').read_text())['local']['gain'])
    members=load_members(path,'cuda') if 'E5' in ARMS_E4 else None
    return decoder,gain,config,{kind:load_basis(path/(kind+'_basis')) for kind in ('state','output')},members


def evaluate(task):
    seed,arm,name,index,state,seconds=task;out=OUT/'E4'
    key=f'seed{seed}_{arm}_{name}_{index:04d}_{seconds:g}s';path=out/'cases'/(key+'.json')
    if path.exists():return json.loads(path.read_text())
    if seed not in CACHE:CACHE[seed]=load(seed)
    decoder,gain,config,bases,members=CACHE[seed]
    if arm in ('H','L','M1'):return run_case(out,decoder,gain,config,arm,name,index,state,seconds)
    if arm=='E5':return run_case(out,decoder,gain,config,arm,name,index,state,seconds,control=make_ensemble_controller(members,config))
    _,kind,regularization=arm.split('_')
    ctl=DeePC(bases[kind],Q,R,lambda_per_column=float(regularization),dt=config.dt)
    # The IO controller sees only p and theta; simulation still records all states
    # for the common primary endpoint. No supplied velocity enters its decisions.
    measured=lambda k,x:ctl(k,x[[0,2]] if kind=='output' else x)
    start=time.perf_counter()
    with np.errstate(over='ignore',invalid='ignore'):
        t,X,U=runner.simulate_plant(measured,np.asarray(state),config,seconds=seconds)
    diag=ctl.diagnostics
    row=dict(case=key,seed=seed,episodes=config.episodes,arm=arm,ic_set=name,ic_index=index,
        initial_state=state,seconds=seconds,wall_seconds=time.perf_counter()-start,
        **metrics(t,X,U,diag['terminal_policy_used'],diag['solve_steps'],()))
    failed=sum(not np.isfinite(loss) for loss in diag['loss'])
    def maximum(values):
        finite=[float(v) for v in values if v is not None and np.isfinite(v)]
        return max(finite) if finite else None
    row.update(qp_solves=len(diag['qp_status']),qp_failures=failed,
        qp_max_initial_residual=maximum(diag['initial_residual']),qp_max_primal_residual=maximum(diag['qp_primal_residual']),
        qp_max_dual_residual=maximum(diag['qp_dual_residual']),qp_max_history_residual=maximum(diag['history_residual']),
        qp_iteration_median=float(np.median([v for v in diag['qp_iterations'] if np.isfinite(v)])) if any(np.isfinite(v) for v in diag['qp_iterations']) else None,
        qp_solve_seconds=float(np.sum(diag['solve_seconds'])) if np.isfinite(diag['solve_seconds']).all() else None,
        lambda_per_column=float(regularization),lambda_g=bases[kind].columns*float(regularization),
        output_only=kind=='output',startup_zero_input_steps=2 if kind=='output' else 0)
    raw=out/'raw'/(key+'.npz');raw.parent.mkdir(exist_ok=True)
    p=2 if kind=='output' else 4
    predictions=np.full((len(diag['predictions']),31,p),np.nan)
    for i,prediction in enumerate(diag['predictions']):
        if prediction.shape==(31,p):predictions[i]=prediction
    np.savez_compressed(raw,t=t,X=X,U=U,local=diag['terminal_policy_used'],solve_steps=diag['solve_steps'],
        predictions=predictions,plan_inputs=diag['plan_inputs'],qp_status=diag['qp_status'],loss=diag['loss'],
        qp_iterations=diag['qp_iterations'],qp_primal_residual=diag['qp_primal_residual'],qp_dual_residual=diag['qp_dual_residual'])
    row['raw_path']=str(raw.relative_to(ROOT));dump(path,row)
    print(json.dumps({k:row[k] for k in ('case','primary_success','qp_failures','wall_seconds')}),flush=True)
    return row


def main():
    out=OUT/'E4';out.mkdir(exist_ok=True)
    config=dict(arms=ARMS_E4,ic_sets=['A','C50'],seeds=[0,1,2],workers=8,profile='dev',
        collection_episodes=3281,collection_horizon=32,collected_transitions=104992,control_horizon=30,
        lambda_per_column=list(REGULARIZERS),lambda_g_rule='number of windows × lambda_per_column; all choices reported',
        lambda_history=1000.,history_penalty='squared residual in training-row-standardized past outputs',
        full_state_initialization='exact current measured state',output_initialization='two actual zero-input samples, past inputs exact, past outputs soft, current outputs exact',
        output_channels=['p','theta'],output_cost='position/angle as authored; velocities from forward normalized-time differences; terminal velocity from last difference',
        common_cost=dict(Q=Q.tolist(),R=R.tolist(),terminal_factor=12.,input_rate_weight=.002),input_bounds=[-10,10],
        solver='OSQP '+osqp.__version__,sources=['https://arxiv.org/pdf/1811.05890','https://osqp.org/docs/interfaces/python.html'],
        neighbor_code_audit='pe.hankel_matrix flattens multichannel data; dataset.py has different units/plant/collection, so neither reused',
        ensemble='Optional E5: five episode-bootstrap deterministic models, mean member cost, first-action feedback; PETS-lite, not full PETS',
        limitations='Mosaic trajectory span is exact only for adequately excited LTI data; regularization does not make it exact for nonlinear cart-pole. The relative SVD cutoff discards numerically small nonzero directions as well as exact null directions.')
    manifest=start_manifest(out,'E4',[0,1,2],config);prior_wall=manifest.get('wall_seconds',0.);start=time.perf_counter()
    try:
        tasks=[];sets=ic_sets()
        for seed in range(3):
            fit(seed)
            for name,states in [('A',sets['A']),('C50',sets['C'][:50])]:
                for index,state in enumerate(states):
                    for arm in ARMS_E4:tasks.append((seed,arm,name,index,state,14.))
            for index in (0,2,5):
                for arm in ARMS_E4:tasks.append((seed,arm,'A',index,sets['A'][index],60.))
        saved={r['case']:r for r in [json.loads(p.read_text()) for p in (out/'cases').glob('*.json')]};last=time.perf_counter()
        with concurrent.futures.ProcessPoolExecutor(8,mp_context=multiprocessing.get_context('spawn'),initializer=initialize,initargs=('E5' in ARMS_E4,)) as pool:
            futures=[pool.submit(evaluate,task) for task in tasks]
            for future in concurrent.futures.as_completed(futures):
                row=future.result();saved[row['case']]=row
                if time.perf_counter()-last>=60:
                    summarize(out,list(saved.values()));last=time.perf_counter()
                    manifest.update(completed_cases=len(saved),total_cases=len(tasks),wall_seconds=prior_wall+time.perf_counter()-start)
                    dump(out/'manifest.json',manifest)
        summarize(out,list(saved.values()));figures('E4');manifest.update(status='DONE',completed_cases=len(saved),total_cases=len(tasks))
    except BaseException as error:manifest.update(status='ERROR',error=repr(error));raise
    finally:manifest['wall_seconds']=prior_wall+time.perf_counter()-start;dump(out/'manifest.json',manifest)

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--skip-ensemble',action='store_true');args=parser.parse_args()
    if args.skip_ensemble:ARMS_E4.remove('E5')
    initialize();main()
    # The active continuation already loaded its code before this extension was
    # requested. This future entry point therefore also schedules the post-core
    # comparison, without restarting any running scientific job or supervisor.
    from queued_smc import launch_after_core
    print('Post-core SMC:',launch_after_core(),flush=True)
