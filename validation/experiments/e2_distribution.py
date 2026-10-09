"""Matched collection-distribution ablations after the main E2 budget sweep."""
from distribution_ablation import *
import concurrent.futures
import multiprocessing
from figures import figures

CACHE={}
VARIANTS={'half_box':(.5,False),'uniform_only':(1.,True)}

def init():
    torch.set_num_threads(1);runner.tqdm=lambda it,**kw:it

def evaluate(task):
    variant,seed,arm,name,index,state=task
    key=(variant,seed)
    if key not in CACHE:
        path=OUT/'E2_distribution/models'/variant/f'episodes3500_seed{seed}'
        transition,payload=runner.load_transition(path/'transition.pt','cuda')
        config=runner.make_config('dev',device='cuda',seed=seed)
        decoder=CanonicalBehaviorDecoder(transition,30).to('cuda').eval()
        gain=np.asarray(json.loads((path/'diagnostics.json').read_text())['local']['gain'])
        CACHE[key]=(decoder,gain,config)
    decoder,gain,config=CACHE[key]
    row=run_case(OUT/'E2_distribution',decoder,gain,config,arm,variant+'_'+name,index,state)
    return row


def main():
    out=OUT/'E2_distribution';out.mkdir(exist_ok=True)
    manifest=start_manifest(out,'E2_distribution',list(range(5)),dict(profile='dev',episodes=3500,
        variants={k:dict(box_scale=v[0],uniform_only=v[1]) for k,v in VARIANTS.items()},arms=['H','L','M1'],workers=8))
    prior_wall=manifest.get('wall_seconds',0.);start=time.perf_counter()
    try:
        sets=ic_sets();tasks=[]
        for variant,(box,uniform) in VARIANTS.items():
            for seed in range(5):
                train_model(seed,data_root=out/'models'/variant,collect=collector(box,uniform))
                for name,states in [('A',sets['A']),('C50',sets['C'][:50])]:
                    for i,state in enumerate(states):
                        for arm in ('H','L','M1'):tasks.append((variant,seed,arm,name,i,state))
        saved={r['case']:r for r in [json.loads(p.read_text()) for p in (out/'cases').glob('*.json')]}
        last=time.perf_counter()
        with concurrent.futures.ProcessPoolExecutor(8,mp_context=multiprocessing.get_context('spawn'),initializer=init) as pool:
            pending=[pool.submit(evaluate,task) for task in tasks]
            for future in concurrent.futures.as_completed(pending):
                row=future.result();saved[row['case']]=row
                if time.perf_counter()-last>60:
                    summarize(out,list(saved.values()));last=time.perf_counter()
                    manifest['wall_seconds']=prior_wall+time.perf_counter()-start;dump(out/'manifest.json',manifest)
        figures('E2_distribution');manifest['status']='DONE'
    except BaseException as error:manifest.update(status='ERROR',error=repr(error));raise
    finally:manifest['wall_seconds']=prior_wall+time.perf_counter()-start;dump(out/'manifest.json',manifest)

if __name__=='__main__':init();main()
