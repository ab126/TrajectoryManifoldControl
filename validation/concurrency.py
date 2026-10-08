"""Bounded process-worker scaling; one BLAS/PyTorch thread per worker.

CUDA uses the same logical GPU for all workers; no MPS or CPU fallback.
Worker occupancy is charged in finally, including failures and termination.
"""
from common import *
import argparse
import signal
import multiprocessing as mp
from benchmark import models
from src.manifold_control import DataDrivenCEMSolver
from models import EnsembleCosts

def worker(index,members,device,barrier,queue,repeats):
    configure(indexed_seed('concurrency',index))
    cores=sorted(os.sched_getaffinity(0)); os.sched_setaffinity(0,{cores[index % len(cores)]})
    if device=='cuda':
        if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable; no CPU fallback')
        torch.cuda.set_device(0); torch.cuda.reset_peak_memory_stats()
    decoders,_=models(device,members)
    Q=torch.diag(torch.tensor([.02,30.,100.,10.],device=device)); R=torch.tensor([[.03]],device=device)
    kw=dict(Q_terminal=12*Q,u_bounds=(-10,10),population=256,elite=32,iterations=3,input_rate_weight=.002,seed=SEED+index)
    solver=(DataDrivenCEMSolver(decoders[0],Q,R,**kw) if members==1 else
            EnsembleCosts(decoders[0],Q,R,members=decoders,**kw))
    x=torch.tensor([0.,0.,.6,0.],device=device); previous=torch.zeros(30,1,device=device)
    for _ in range(3): solver.solve(x,previous)
    sync(device); barrier.wait(timeout=120)
    start=time.perf_counter(); latencies=[]
    for _ in range(repeats):
        sync(device); t=time.perf_counter(); result=solver.solve(x,previous); sync(device)
        latencies.append(time.perf_counter()-t)
        assert result.finite
        previous=torch.cat((result.u[5:],result.u[-5:]))
    end=time.perf_counter()
    queue.put({'index':index,'steady_start':start,'steady_end':end,'completed':repeats,
               'latency':summary(latencies),'peak_rss_MiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
               'peak_allocated_VRAM_MiB':torch.cuda.max_memory_allocated(device)/2**20 if device=='cuda' else None,
               'peak_reserved_VRAM_MiB':torch.cuda.max_memory_reserved(device)/2**20 if device=='cuda' else None})

def run(device='cpu',worker_counts=(1,2,4,8)):
    assert json.loads((OUT/'plant_checks.json').read_text())['passed']
    assert json.loads((OUT/('equivalence.json' if device=='cpu' else 'equivalence_cuda.json')).read_text())['passed']
    ctx=mp.get_context('spawn'); rows=[]
    destination='concurrency.json' if device=='cpu' else 'concurrency_cuda.json'
    for members in (1,5):
        for n in worker_counts:
            used=sum(json.loads(s)['worker_seconds'] for s in (OUT/'usage.jsonl').read_text().splitlines())
            reservation=(240+6*n)*(n+1) # bounded child occupancy plus supervisor
            if used+reservation>3600:
                rows.append({'device':device,'members':members,'workers':n,'skipped':'remaining worst-case reservation insufficient'})
                save(destination,rows); continue
            if device=='cuda':
                free,total=torch.cuda.mem_get_info()
                if free<2*2**30:
                    raise RuntimeError('Less than 2 GiB currently free; defer shared-GPU scaling')
            barrier=ctx.Barrier(n+1); queue=ctx.Queue(); processes=[]; launched={}; finished={}
            measurements=[]; passed=False
            try:
                for i in range(n):
                    process=ctx.Process(target=worker,args=(i,members,device,barrier,queue,24 if members==1 else 12))
                    launched[i]=time.perf_counter(); process.start(); processes.append(process)
                barrier.wait(timeout=120)
                deadline=min(launched.values())+240
                measurements=[queue.get(timeout=max(.1,deadline-time.perf_counter())) for _ in range(n)]
                for i,process in enumerate(processes):
                    process.join(timeout=1)
                    if not process.is_alive(): finished[i]=time.perf_counter()
                passed=all(p.exitcode==0 for p in processes)
                if not passed: raise RuntimeError('Concurrency worker failed')
            finally:
                for i,process in enumerate(processes):
                    if process.is_alive(): process.terminate(); process.join(timeout=5)
                    if process.is_alive(): process.kill(); process.join()
                    finished.setdefault(i,time.perf_counter())
                occupancy=sum(finished[i]-launched[i] for i in range(len(processes)))
                with (OUT/'usage.jsonl').open('a') as file:
                    file.write(json.dumps({'label':f'concurrency {device} members={members} workers={n}',
                                           'worker_seconds':occupancy,'passed':passed})+'\n')
            earliest=min(r['steady_start'] for r in measurements); latest=max(r['steady_end'] for r in measurements)
            row={'device':device,'members':members,'workers':n,'aggregate_solves_per_s':sum(r['completed'] for r in measurements)/(latest-earliest),
                 'steady_wall_s':latest-earliest,'worker_seconds_including_startup':occupancy,
                 'sum_peak_RSS_MiB':sum(r['peak_rss_MiB'] for r in measurements),'workers_detail':measurements,
                 'scope':'solver-only scaling; validate end-to-end deadlines separately under chosen concurrency'}
            rows.append(row)
            save(destination,rows); print(json.dumps({k:v for k,v in row.items() if k!='workers_detail'}),flush=True)
    return rows

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    parser.add_argument('--workers',type=int,nargs='+',default=None)
    args=parser.parse_args()
    counts=args.workers or ([1,2,4] if args.device=='cuda' else [1,2,4,8])
    if any(n not in (1,2,4,8) for n in counts): parser.error('workers must be 1,2,4,8')
    def terminate(_signum,_frame): raise RuntimeError('Concurrency cancelled; charge worker cleanup')
    signal.signal(signal.SIGTERM,terminate)
    configure(); run(args.device,counts)
