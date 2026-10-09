"""Noise, physical mismatch and input-delay tests of nominal H and local L."""
from distribution_ablation import *
import concurrent.futures
import multiprocessing
from figures import figures

CACHE={}
NOMINAL_STEP=runner._plant_step
CONDITIONS={**{f'measurement_{v:g}':dict(mode='measurement',level=v) for v in (.01,.03,.1)},
            **{f'process_{v:g}':dict(mode='process',level=v) for v in (.01,.03,.1)},
            'mass_minus25':dict(mode='mismatch',mass=1.5,length=1.),
            'mass_plus25':dict(mode='mismatch',mass=2.5,length=1.),
            'length_minus25':dict(mode='mismatch',mass=2.,length=.75),
            'length_plus25':dict(mode='mismatch',mass=2.,length=1.25),
            'input_delay':dict(mode='delay',samples=1)}


def nominal_scale(seed):
    data=np.load(OUT/'models'/f'episodes3500_seed{seed}'/'data.npz')
    splits=runner.split_episodes(data['X'],data['U'],seed)
    return np.std(splits['train'][0][:,:-1].reshape(-1,4),axis=0).clip(1e-6)


def physical_mismatch_step(x,u,dt,mass=2.,length=1.):
    # Map the fixed nominal measurement coordinates to actual-length normalized
    # coordinates, use the submitted RK4 map, then map back. Input F/(mg) stays fixed.
    root=np.sqrt(length);scale=np.array([1/length,1/root,1.,root])
    return runner.rk4_step(np.asarray(x)*scale,float(np.asarray(u).reshape(-1)[0]),dt/root,mass)/scale


def normal_sequence(seed,condition,index,steps,scale):
    # The arm is deliberately absent from this key: matched disturbances.
    code=list(CONDITIONS).index(condition)
    return np.random.default_rng(np.random.SeedSequence([20261009,seed,code,index])).standard_normal((steps,4))*scale


def applied_inputs(commands,delay=False):
    commands=np.asarray(commands)
    return np.concatenate((np.zeros_like(commands[:1]),commands[:-1])) if delay else commands.copy()


def init():
    torch.set_num_threads(1);runner.tqdm=lambda it,**kw:it


def evaluate(task):
    condition,seed,arm,name,index,state=task
    key=(condition if CONDITIONS[condition]['mode']=='process' else 'nominal',seed)
    if key not in CACHE:
        path=(OUT/'E3/models'/condition if key[0]!='nominal' else OUT/'models')/f'episodes3500_seed{seed}'
        model,payload=runner.load_transition(path/'transition.pt','cuda')
        decoder=CanonicalBehaviorDecoder(model,30).to('cuda').eval()
        gain=np.asarray(json.loads((path/'diagnostics.json').read_text())['local']['gain'])
        CACHE[key]=(decoder,gain,runner.make_config('dev',seed=seed,device='cuda'),nominal_scale(seed))
    decoder,gain,config,std=CACHE[key]
    spec=CONDITIONS[condition]
    noise=normal_sequence(seed,condition,index+(0 if name=='A' else 1000),700,std*spec.get('level',0))
    observe=None;sample=0;last=np.zeros(1)
    def plant(x,u,dt):
        nonlocal sample,last
        value=np.asarray(u).copy()
        if spec['mode']=='delay':value,last=last,value
        xn=physical_mismatch_step(x,value,dt,spec['mass'],spec['length']) if spec['mode']=='mismatch' else NOMINAL_STEP(x,value,dt)
        if spec['mode']=='process':xn=xn+noise[sample]
        sample+=1
        return xn
    if spec['mode']=='measurement':observe=lambda k,state:state+noise[k]
    runner._plant_step=plant
    try:row=run_case(OUT/'E3',decoder,gain,config,arm,condition+'_'+name,index,state,observe=observe)
    finally:runner._plant_step=NOMINAL_STEP
    # Derive applied inputs from saved commands even after a pause between the
    # generic case save and these robustness-specific diagnostics.
    raw=ROOT/row['raw_path'];data=np.load(raw);arrays={k:data[k] for k in data.files};data.close()
    arrays['U_applied']=applied_inputs(arrays['U'],spec['mode']=='delay')
    if spec['mode']=='measurement':arrays['observation_noise']=noise
    if spec['mode']=='process':arrays['process_noise']=noise
    temporary=raw.with_suffix('.tmp.npz');np.savez_compressed(temporary,**arrays);temporary.replace(raw)
    observed=arrays['X'][:-1]+noise if spec['mode']=='measurement' else arrays['X'][:-1]
    observed_cost=np.einsum('bi,ij,bj->b',observed,Q,observed)
    def maximum(values):return float(np.max(np.abs(values))) if np.isfinite(values).all() else None
    row.update(max_command_u=maximum(arrays['U']),command_input_limit_seconds=float(np.sum(np.abs(arrays['U'][:,0])>=10-1e-6)*.02),
               max_abs_u=maximum(arrays['U_applied']),input_limit_seconds=float(np.sum(np.abs(arrays['U_applied'][:,0])>=10-1e-6)*.02),
               controller_never_used_local=bool(not np.any(arrays['local'])),never_entered_switch=bool(not np.any(observed_cost<.75)),
               noise_std_by_state=(std*spec['level']).tolist() if 'level' in spec else None)
    dump(OUT/'E3/cases'/f"{row['case']}.json",row)
    return row


def main():
    out=OUT/'E3';out.mkdir(exist_ok=True)
    manifest=start_manifest(out,'E3',[0,1,2],dict(profile='dev',episodes=3500,conditions=CONDITIONS,
        arms=['H','L'],ICs=['A','C50'],workers=8,
        noise_scale='Per-state population std of nominal training transitions; same noise arrays per IC/seed across arms',
        process_training='Add independent per-step noise during data collection; nominal initial collection states',
        physical_mismatch='Mass ratio changed; length changed while nominal measurement/time/force coordinates remain fixed',
        delay='One input sample, 0.02 physical seconds, initial applied input zero'))
    prior_wall=manifest.get('wall_seconds',0.);start=time.perf_counter()
    try:
        sets=ic_sets();tasks=[]
        for seed in range(3):
            train_model(seed)
            std=nominal_scale(seed)
            manifest.setdefault('nominal_state_std_by_seed',{})[str(seed)]=std.tolist()
            for condition,spec in CONDITIONS.items():
                if spec['mode']=='process':
                    rng=np.random.default_rng(np.random.SeedSequence([20261009,seed,list(CONDITIONS).index(condition),987654]))
                    fn=collector()
                    def noisy_step(x,u,dt,rng=rng,level=spec['level'],std=std):
                        return NOMINAL_STEP(x,u,dt)+rng.standard_normal(4)*level*std
                    fn.__globals__['_plant_step']=noisy_step
                    train_model(seed,data_root=out/'models'/condition,collect=fn)
                for name,states in [('A',sets['A']),('C50',sets['C'][:50])]:
                    for i,state in enumerate(states):
                        for arm in ('H','L'):tasks.append((condition,seed,arm,name,i,state))
        saved={r['case']:r for r in [json.loads(p.read_text()) for p in (out/'cases').glob('*.json')]};last=time.perf_counter()
        with concurrent.futures.ProcessPoolExecutor(8,mp_context=multiprocessing.get_context('spawn'),initializer=init) as pool:
            futures=[pool.submit(evaluate,task) for task in tasks]
            for future in concurrent.futures.as_completed(futures):
                row=future.result();saved[row['case']]=row
                if time.perf_counter()-last>60:
                    summarize(out,list(saved.values()));last=time.perf_counter()
                    manifest['wall_seconds']=prior_wall+time.perf_counter()-start;dump(out/'manifest.json',manifest)
        figures('E3');manifest['status']='DONE'
    except BaseException as error:manifest.update(status='ERROR',error=repr(error));raise
    finally:manifest['wall_seconds']=prior_wall+time.perf_counter()-start;dump(out/'manifest.json',manifest)

if __name__=='__main__':init();main()
