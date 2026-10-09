"""Run independent saved cases in a small process pool; no shared RNG or leases."""
from campaign import *
import concurrent.futures
import multiprocessing
from figures import figures

CACHE={}

def evaluate(task):
    experiment,episodes,seed,arm,name,index,state,seconds=task
    key=(episodes,seed)
    if key not in CACHE:
        path=OUT/'models'/f'episodes{episodes}_seed{seed}'
        model,payload=runner.load_transition(path/'transition.pt','cuda')
        config=runner.ExperimentConfig(**{**payload['config'],'hidden_dims':tuple(payload['config']['hidden_dims'])})
        decoder=CanonicalBehaviorDecoder(model,config.horizon).to('cuda').eval()
        gain=np.asarray(json.loads((path/'diagnostics.json').read_text())['local']['gain'])
        from models import OracleTransition
        transition=OracleTransition('cuda')
        transition.xu_std=decoder.transition.xu_std.detach().clone()
        oracle=CanonicalBehaviorDecoder(transition,config.horizon).to('cuda').eval()
        CACHE[key]=(decoder,oracle,gain,config)
    decoder,oracle,gain,config=CACHE[key]
    return run_case(OUT/experiment,oracle if arm=='Hstar' else decoder,gain,config,arm,name,index,state,seconds)


def initialize():
    torch.set_num_threads(1)
    # Suppress display-only progress bars. Algorithmic runner functions stay intact.
    runner.tqdm=lambda iterable,**kwargs:iterable


def main(experiment,workers,sets):
    out=OUT/experiment;out.mkdir(parents=True,exist_ok=True)
    seed_list=list(range(10)) if experiment=='E0' else [0,1,2] if experiment=='E1' else list(range(5))
    budgets=[35,100,350,1000,3500,10000] if experiment=='E2' else [3500]
    names=sets or (['A'] if experiment=='E0' else ['B','C','D'] if experiment=='E1' else ['A','C50'])
    arms=ARMS if experiment!='E2' else ('H','L','M1')
    manifest=start_manifest(out,experiment,seed_list,dict(profile='dev',episodes=budgets,arms=list(arms),ic_sets=names,workers=workers))
    manifest['config'].update(workers=workers,ic_sets=names)
    start=time.perf_counter();last_update=start
    from models import check_oracle
    manifest['oracle_equivalence_max_abs_error']=check_oracle('cpu')
    tasks=[]; all_ics=ic_sets()
    try:
        for episodes in budgets:
            for seed in seed_list:
                train_model(seed,episodes)
                for name in names:
                    states=all_ics['C'][:50] if name=='C50' else all_ics[name]
                    for index,state in enumerate(states):
                        for arm in arms:
                            tasks.append((experiment,episodes,seed,arm,name,index,state,14.))
                if experiment=='E1':
                    long_indices={'B':[25,85,145],'C':[0,1,2],'D':[20,840,1660]}
                    for name in names:
                        for index in long_indices.get(name,[]):
                            for arm in arms:
                                tasks.append((experiment,episodes,seed,arm,name,index,all_ics[name][index],60.))
                if experiment=='E0':
                    # All six authored ICs are a fixed small subset; additional 60 s
                    # runs assess eventual stabilization without imposing recentering.
                    for index,state in enumerate(all_ics['A']):
                        for arm in ('H','L','Hstar'):
                            tasks.append((experiment,episodes,seed,arm,'A',index,state,60.))
        # Case identifiers include budget for E2, whose seeds repeat across budgets.
        completed=0
        with concurrent.futures.ProcessPoolExecutor(workers,mp_context=multiprocessing.get_context('spawn'),initializer=initialize) as pool:
            futures=[pool.submit(evaluate,task) for task in tasks]
            for future in concurrent.futures.as_completed(futures):
                future.result();completed+=1
                now=time.perf_counter()
                if now-last_update>=60 or completed==len(tasks):
                    summarize(out)
                    manifest['wall_seconds']+=now-start;start=now;last_update=now
                    manifest.update(completed_cases=completed,total_cases=len(tasks))
                    dump(out/'manifest.json',manifest)
        figures(experiment)
        manifest['status']='DONE'
    except BaseException as error:
        manifest.update(status='ERROR',error=repr(error));raise
    finally:
        manifest['wall_seconds']+=time.perf_counter()-start
        dump(out/'manifest.json',manifest)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('experiment',choices=['E0','E1','E2'])
    parser.add_argument('--workers',type=int,default=8)
    parser.add_argument('--sets',nargs='+')
    args=parser.parse_args();initialize();main(args.experiment,args.workers,args.sets)
