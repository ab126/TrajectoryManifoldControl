"""Performance only: original dev architecture/control and five-member mean-cost CEM."""
from common import *
import argparse
from unittest.mock import patch
from scripts import run_inverted_pendulum_mc as dev
from src.manifold_control import DataDrivenTransition, CanonicalBehaviorDecoder, DataDrivenCEMSolver
from models import EnsembleCosts, OracleTransition, check_oracle

def models(device='cpu',members=1):
    checkpoint=ROOT/'validation/smoke_pipeline/saves/saved_models/canonical_decoder_dev.pt'
    base,payload=dev.load_transition(checkpoint,device)
    all_models=[base]
    for i in range(1,members):
        # Independently initialized members with fitted smoke scaling. NOT a fitted
        # ensemble baseline: architecture/latency only, no outcome comparison.
        torch.manual_seed(indexed_seed('performance_member',i))
        model=DataDrivenTransition(4,1,(192,192,192)).to(device)
        for name in ('xu_mean','xu_std','dx_mean','dx_std'):
            getattr(model,name).copy_(getattr(base,name))
        all_models.append(model.eval())
    return [CanonicalBehaviorDecoder(m,30).to(device).eval() for m in all_models],payload

def control_benchmark(device='cpu',members=1,repeats=120,warmup=10):
    decoders,payload=models(device,members)
    cfg=dev.make_config('dev',device=device)
    smoke_summary=json.loads((ROOT/'validation/smoke_pipeline/saves/simulation_results/canonical_manifold_dev.json').read_text())
    gain=np.asarray(smoke_summary['local_model']['gain'])
    factory=DataDrivenCEMSolver if members==1 else lambda *a,**kw: EnsembleCosts(*a,members=decoders,**kw)
    with patch.object(dev,'DataDrivenCEMSolver',side_effect=factory):
        control=dev.make_controller(decoders[0],cfg,local_gain=gain)
    # Force a planning control call every repetition by clearing the queue through
    # the actual local branch. Previous plan is deliberately preserved by runner.
    planning=[]; local=[]; queued=[]
    rng=np.random.default_rng(SEED)
    if str(device).startswith('cuda'): torch.cuda.reset_peak_memory_stats(device)
    for i in range(warmup+repeats):
        sync(device); start=time.perf_counter(); control(i,np.zeros(4)); sync(device)
        elapsed=time.perf_counter()-start
        if i>=warmup: local.append(elapsed)
        x=rng.uniform(-1,1,4)*[1.5,2.5,1.3,3.5]
        x[2]=np.sign(x[2])*(.2+abs(x[2]))
        sync(device); start=time.perf_counter(); control(i,x); sync(device)
        elapsed=time.perf_counter()-start
        if i>=warmup: planning.append(elapsed)
        sync(device); start=time.perf_counter(); control(i,x); sync(device)
        elapsed=time.perf_counter()-start
        if i>=warmup: queued.append(elapsed)
    return {'device':device,'members':members,'architecture':[192,192,192], 'horizon':30,
            'population':256,'elites':32,'iterations':3,'warmup':warmup,'planning':summary(planning),
            'local':summary(local),'queue':summary(queued), 'raw_seconds':{'planning':planning,'local':local,'queue':queued},
            'planning_p99_meets_0.1s':bool(np.quantile(planning,.99)<=.1),
            'local_p99_meets_0.02s':bool(np.quantile(local,.99)<=.02),
            'deadline_miss_fraction':float(np.mean(np.array(planning)>.1)),
            'peak_rss_MiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
            'peak_allocated_VRAM_MiB':torch.cuda.max_memory_allocated(device)/2**20 if str(device).startswith('cuda') else None,
            'peak_reserved_VRAM_MiB':torch.cuda.max_memory_reserved(device)/2**20 if str(device).startswith('cuda') else None,
            'ensemble_limit':'five-member dev architecture; four members untrained; original local branch measured for end-to-end parity (E scientific recipe has no switch)'}

def oracle_benchmark(device='cpu',repeats=12,warmup=3):
    Q=torch.diag(torch.tensor([.02,30,100,10],dtype=torch.float64,device=device)); R=torch.tensor([[.03]],dtype=torch.float64,device=device)
    f=OracleTransition(device); decoder=CanonicalBehaviorDecoder(f,60)
    solver=DataDrivenCEMSolver(decoder,Q,R,Q_terminal=12*Q,u_bounds=(-10,10),population=256,elite=32,iterations=3,input_rate_weight=.002,seed=SEED)
    x=torch.tensor([0.,0.,.6,0.],dtype=torch.float64,device=device); previous=torch.zeros(60,1,dtype=torch.float64,device=device)
    from src.inverted_pendulum import rk4_step
    from scipy.linalg import solve_discrete_are
    eps=1e-7; eye=np.eye(4)
    A=np.column_stack([(rk4_step(eps*v,0.,f.dt,2.)-rk4_step(-eps*v,0.,f.dt,2.))/(2*eps) for v in eye])
    B=((rk4_step(np.zeros(4),eps,f.dt,2.)-rk4_step(np.zeros(4),-eps,f.dt,2.))/(2*eps))[:,None]
    Qnp=Q.cpu().numpy(); Rnp=R.cpu().numpy(); P=solve_discrete_are(A,B,Qnp,Rnp)
    K=np.linalg.solve(Rnp+B.T@P@B,B.T@P@A)
    if str(device).startswith('cuda'): torch.cuda.reset_peak_memory_stats(device)
    vals=[]; local=[]; queue=torch.empty(0,1,dtype=torch.float64,device=device)
    for i in range(warmup+repeats):
        # Time the actual switch, clipping, and queue clearing in local mode.
        state=np.zeros(4)
        sync(device); t=time.perf_counter()
        if float(state@Q.cpu().numpy()@state)<.75:
            queue=torch.empty(0,1,dtype=torch.float64,device=device)
            action=np.clip(-K@state,-10,10).reshape(1)
        sync(device)
        if i>=warmup: local.append(time.perf_counter()-t)
        # End-to-end planning includes switch check, stale warm start, predicted
        # trajectory CPU transfer, queue creation, and first returned action.
        state=x.cpu().numpy()
        sync(device); t=time.perf_counter()
        assert float(state@Q.cpu().numpy()@state)>=.75 and not len(queue)
        current=torch.as_tensor(state,dtype=torch.float64,device=device)
        shifted=torch.cat((previous[5:],previous[-5:]))
        result=solver.solve(current,shifted)
        prediction=result.x.detach().cpu().numpy()
        previous=result.u.detach(); queue=previous[:5].clone()
        action=queue[0].clone().cpu().numpy(); queue=queue[1:]
        sync(device)
        if i>=warmup: vals.append(time.perf_counter()-t)
        assert result.finite
    return {'device':device,'N':60,'dtype':'float64','warmup':warmup,'population':256,'elites':32,'iterations':3,
            'single_start':summary(vals),'local':summary(local),'raw_seconds':vals,'raw_local_seconds':local,
            'planning_p99_meets_0.1s':bool(np.quantile(vals,.99)<=.1),
            'local_p99_meets_0.02s':bool(np.quantile(local,.99)<=.02),
            'five_start_mean_s':5*float(np.mean(vals)),
            'scope':'few performance solves; no oracle attainability calibration',
            'peak_rss_MiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
            'peak_allocated_VRAM_MiB':torch.cuda.max_memory_allocated(device)/2**20 if str(device).startswith('cuda') else None,
            'peak_reserved_VRAM_MiB':torch.cuda.max_memory_reserved(device)/2**20 if str(device).startswith('cuda') else None}

def training_benchmark(device='cpu',members=1):
    # Replay measured smoke training triples to exactly 84000 rows: only a
    # throughput workload, not a scientific acquisition or trained checkpoint.
    stored=np.load(ROOT/'validation/smoke_pipeline/saves/datasets/canonical_trajectories_dev.npz')
    split=dev.split_episodes(stored['X'],stored['U'],234)
    X,U=split['train']; train=(np.tile(X,(34,1,1))[:2800].copy(),np.tile(U,(34,1,1))[:2800].copy())
    Xv,Uv=split['validation']
    validation=(np.tile(Xv,(40,1,1))[:700].copy(),np.tile(Uv,(40,1,1))[:700].copy())
    assert train[1].shape[0]*train[1].shape[1]==84000
    cfg=dev.make_config('dev',device=device,epochs=3)
    # Warm optimizer/kernel and validation path with a complete one-epoch call.
    warm=dev.make_config('dev',device=device,epochs=1)
    dev.train_transition((train[0][:70],train[1][:70]),split['validation'],warm,OUT/f'train_warm_{device}.pt')
    sync(device)
    if str(device).startswith('cuda'): torch.cuda.reset_peak_memory_stats(device)
    t=time.perf_counter()
    for member in range(members):
        if members>1:
            torch.manual_seed(indexed_seed('performance_training_member',member))
            rng=np.random.default_rng(indexed_seed('performance_bootstrap',member))
            ids=rng.integers(0,len(train[0]),len(train[0]))
            member_train=(train[0][ids],train[1][ids])
        else: member_train=train
        model,hist=dev.train_transition(member_train,validation,cfg,OUT/f'train_performance_{device}_{members}_{member}.pt')
    sync(device); elapsed=time.perf_counter()-t
    return {'device':device,'members':members,'T_train':84000,'epochs_per_member':3,'batch_size':2048,'architecture':[192,192,192],
            'wall_s':elapsed,'epochs_per_s':3*members/elapsed,'epoch_seconds':elapsed/(3*members),
            'ensemble_epochs_per_s':3/elapsed,'training_schedule':'serial members; original train_transition for each member',
            'T_validation':21000,
            'timing_scope':'original train_transition end-to-end; includes full 21000-row validation/checkpoint',
            'dataset':'repeat smoke training episodes, performance only',
            'peak_rss_MiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
            'peak_allocated_VRAM_MiB':torch.cuda.max_memory_allocated(device)/2**20 if str(device).startswith('cuda') else None,
            'peak_reserved_VRAM_MiB':torch.cuda.max_memory_reserved(device)/2**20 if str(device).startswith('cuda') else None}

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--device',default='cpu'); p.add_argument('--members',type=int,default=1)
    p.add_argument('--kind',choices=['control','training','oracle'],default='control'); p.add_argument('--repeats',type=int,default=120)
    args=p.parse_args(); configure()
    assert json.loads((OUT/'plant_checks.json').read_text())['passed']
    assert json.loads((OUT/('equivalence.json' if args.device=='cpu' else 'equivalence_cuda.json')).read_text())['passed']
    with Meter(f'performance {args.kind} {args.device} members={args.members}',reservation=600):
        if args.kind=='control': result=control_benchmark(args.device,args.members,args.repeats)
        elif args.kind=='training': result=training_benchmark(args.device,args.members)
        else:
            error=check_oracle(args.device); result=oracle_benchmark(args.device); result['RK4_vectorization_max_abs_error']=error
        save(f'benchmark_{args.kind}_{args.device}_{args.members}.json',result)
        print(json.dumps({k:v for k,v in result.items() if k!='raw_seconds'},indent=2))
