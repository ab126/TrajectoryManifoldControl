"""Next-stage oracle attainability calibration. NOT run during readiness.

Default is preflight only. --execute requires a feasible worst-case reservation,
then enforces the separate future worker-hour cap. Saves every true plant sample.
This is a development calibration, not confirmatory evidence or a certificate.
"""
from common import *
import argparse
import signal
from scipy.linalg import solve_discrete_are
from src.inverted_pendulum import rk4_step,linearize_upright_dynamics
from src.manifold_control import CanonicalBehaviorDecoder,DataDrivenCEMSolver
from models import OracleTransition,check_oracle
from unittest.mock import patch

def strict_physical(X):
    scale=np.array([1.,np.sqrt(9.81),1.,np.sqrt(9.81)])
    Y=X*scale
    if not np.isfinite(Y).all(): return False
    tail=Y[-151:]
    return bool(np.all(abs(Y[:,0])<=2.4) and np.all(abs(Y[:,2])<=np.pi/2) and
                np.all(abs(tail[:,2])<=.02) and np.all(abs(tail[:,3])<=.05) and
                np.all(abs(tail[:,1])<=.05) and np.all(abs(tail[:,0])<=.1))

STOP_REQUESTED=False

def atomic_json(path,value):
    temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)

def atomic_trace(path,**arrays):
    temporary=path.with_name(path.name+'.tmp')
    with temporary.open('wb') as file:
        np.savez_compressed(file,**arrays)
        file.flush();os.fsync(file.fileno())
    temporary.replace(path)

def main(args):
    plant=json.loads((OUT/'plant_checks.json').read_text()); eq=json.loads((OUT/'equivalence.json').read_text())
    assert plant['passed'] and eq['passed']
    device=args.device
    if device!='cpu':
        assert json.loads((OUT/'equivalence_cuda.json').read_text())['passed']
        assert json.loads((OUT/'gpu_probe.json').read_text())['passed']
    timing_path=OUT/f'benchmark_oracle_{device}_1.json'
    if not timing_path.exists(): raise RuntimeError(f'Measure oracle timing on {device} first: {timing_path}')
    measured=json.loads(timing_path.read_text())['single_start']['mean_s']
    reserve=args.cases*1000*args.starts*measured/3600
    preflight={'device':device,'timing_source':str(timing_path),'cases':args.cases,'starts':args.starts,'N':60,'physical_seconds':20,
               'worst_case_planning_worker_hours':reserve,'cap_worker_hours':args.max_worker_hours,
               'reservation_with_20pct_overhead_h':reserve*1.2,
               'reservation_fits':bool(reserve*1.2<=args.max_worker_hours),
               'note':'planning floor only; reserve extra plant/scoring/logging overhead',
               'requested_box_scale':args.box_scale,'execute':args.execute}
    print(json.dumps(preflight,indent=2))
    if not args.execute: return
    if reserve*1.2>args.max_worker_hours:
        raise RuntimeError('Worst-case planning plus 20% overhead exceeds cap; revise allocation before Gate A')
    if device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable in this execution context')
    error=check_oracle(device)
    destination=Path(args.output_dir).resolve() if args.output_dir else ROOT/'validation'/('gate_a_artifacts' if device=='cpu' else 'gate_a_artifacts_cuda')
    if not destination.is_relative_to((ROOT/'validation').resolve()):raise RuntimeError('Gate A output must be under validation/')
    if destination.exists() and any(destination.iterdir()):raise RuntimeError('Refusing to overwrite existing calibration artifacts; choose a fresh directory')
    destination.mkdir(parents=True,exist_ok=True)
    f=OracleTransition(device); h=f.dt
    Ac,Bc=linearize_upright_dynamics(2.)
    # Exact linearization of the submitted held-input RK4 transition.
    A=np.eye(4)+h*Ac+h*h/2*(Ac@Ac)+h**3/6*np.linalg.matrix_power(Ac,3)+h**4/24*np.linalg.matrix_power(Ac,4)
    B=(h*np.eye(4)+h*h/2*Ac+h**3/6*(Ac@Ac)+h**4/24*np.linalg.matrix_power(Ac,3))@Bc
    Q=np.diag([.02,30.,100.,10.]); R=np.array([[.03]])
    P=solve_discrete_are(A,B,Q,R); K=np.linalg.solve(R+B.T@P@B,B.T@P@A)
    Qt=torch.tensor(Q,dtype=torch.float64,device=device); Rt=torch.tensor(R,dtype=torch.float64,device=device)
    solver=DataDrivenCEMSolver(CanonicalBehaviorDecoder(f,60),Qt,Rt,Q_terminal=12*Qt,
                             u_bounds=(-10,10),population=256,elite=32,iterations=3,input_rate_weight=.002,seed=SEED)
    rng=np.random.default_rng(indexed_seed('gate_a_cases',args.box_scale))
    initial=rng.uniform(-1,1,(args.cases,4))*np.array([1.5,2.5,1.3,3.5])*args.box_scale
    np.save(destination/'initial_states.npy',initial)
    rows=[]; start=time.perf_counter(); starts_executed=0
    def checkpoint(reason,case,k,X,U,branches):
        atomic_trace(destination/f'incomplete_case_{case}.npz',X=X[:k+1],U=U[:k],branches=branches)
        partial={'device':device,'passed':False,'incomplete':True,'reason':reason,'cases':len(rows),'case_in_progress':case,
                 'sample_in_progress':k,'worker_seconds':time.perf_counter()-start,'starts_executed':starts_executed}
        atomic_json(destination/'summary.json',partial)
        atomic_json(destination/'usage.json',partial)
    atomic_json(destination/'cases.json',[])
    atomic_json(destination/'config.json',{'preflight':preflight,'root_seed':SEED,'box_half_widths_normalized':[1.5,2.5,1.3,3.5],
        'case_seed':indexed_seed('gate_a_cases',args.box_scale),'dtype':'float64','dt_physical':.02,'dt_normalized':h,
        'cost_Q':Q.tolist(),'cost_R':R.tolist(),'LQR_gain':K.tolist(),'timing_exempt':True})
    for case,x0 in enumerate(initial):
        X=np.empty((1001,4)); U=np.empty((1000,1)); X[0]=x0
        previous=torch.zeros(60,1,dtype=torch.float64,device=device); queue=[]; branches=[]; solve_times=[]
        for k in range(1000):
            elapsed=time.perf_counter()-start
            if STOP_REQUESTED or elapsed+args.starts*measured*1.2>args.max_worker_hours*3600:
                reason='termination requested' if STOP_REQUESTED else 'Gate A worker cap reached'
                checkpoint(reason,case,k,X,U,branches)
                raise RuntimeError(reason+'; checkpoint saved, incomplete, do not infer attainability')
            if float(X[k]@Q@X[k])<.75:
                u=np.clip(-K@X[k],-10,10); queue=[]; branch='local'
            elif queue:
                u=queue.pop(0); branch='queue'
            else:
                state=torch.tensor(X[k],dtype=torch.float64,device=device); best=None
                sync(device); solve_start=time.perf_counter()
                for attempt in range(args.starts):
                    warm=torch.cat((previous[5:],previous[-5:]))
                    if attempt:
                        gen=torch.Generator(device='cpu').manual_seed(indexed_seed('oracle_start',case,k,attempt))
                        warm=(warm+5*torch.randn(warm.shape,generator=gen,dtype=torch.float64).to(device)).clamp(-10,10)
                    draws=normals(['oracle',case,attempt],k,horizon=60,device=device).double(); iterator=iter(draws)
                    with patch('torch.randn',side_effect=lambda shape,**kw:next(iterator)):
                        candidate=solver.solve(state,warm)
                    starts_executed+=1
                    if candidate.finite and (best is None or candidate.loss<best.loss): best=candidate
                sync(device); solve_times.append(time.perf_counter()-solve_start)
                if best is None: u=np.zeros(1); previous.zero_(); branch='nonfinite'
                else:
                    previous=best.u; queue=[v.cpu().numpy().copy() for v in previous[:5]]
                    u=queue.pop(0); branch='plan'
            U[k]=u; X[k+1]=rk4_step(X[k],float(u[0]),h,2.); branches.append(branch)
            if not np.isfinite(X[k+1]).all():
                X[k+2:]=np.nan; U[k+1:]=np.nan; break
        success=strict_physical(X)
        atomic_trace(destination/f'case_{case:03d}.npz',X=X,U=U,branches=branches,solve_seconds=solve_times)
        rows.append({'case':case,'positive_angle':bool(x0[2]>=0),'strict_physical_success':success,
                     'planning_calls':len(solve_times),'executed_samples':len(branches),'timing_exempt':True})
        atomic_json(destination/'cases.json',rows)
        atomic_json(destination/'progress.json',{'completed_cases':len(rows),'total_cases':args.cases,'last_case':case,
            'successes_so_far':sum(r['strict_physical_success'] for r in rows),'elapsed_s':time.perf_counter()-start})
        print(f'case {case+1}/{args.cases}: success={success}, replans={len(solve_times)}, elapsed={time.perf_counter()-start:.1f}s',flush=True)
    success=np.mean([r['strict_physical_success'] for r in rows])
    halves=[np.mean([r['strict_physical_success'] for r in rows if r['positive_angle']==sign]) for sign in (False,True)]
    result={'device':device,'incomplete':False,'development_only':True,'physical_success_rate':float(success),'angle_half_rates':list(map(float,halves)),
            'passed':bool(success>=.98 and all(v>=.95 for v in halves)), 'cases':len(rows),
            'box_scale':args.box_scale,'starts_executed':starts_executed,'worker_seconds':time.perf_counter()-start,
            'RK4_vectorization_error':error,'determinism':determinism_manifest(),
            'warning':'no automatic box reduction; diagnose and amend protocol on failure'}
    atomic_json(destination/'summary.json',result); print(json.dumps(result,indent=2))
    return 0 if result['passed'] else 3

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--execute',action='store_true'); p.add_argument('--cases',type=int,default=200)
    p.add_argument('--starts',type=int,default=5); p.add_argument('--box-scale',type=float,default=1.)
    p.add_argument('--max-worker-hours',type=float,default=8.)
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--output-dir',default=None)
    args=p.parse_args()
    if not 1<=args.starts<=5 or args.cases!=200 or args.box_scale not in (1.,.5,.25): p.error('Gate A requires 200 cases, 1..5 starts, box scale 1/.5/.25')
    def request_stop(_signum,_frame):
        global STOP_REQUESTED
        STOP_REQUESTED=True
    signal.signal(signal.SIGTERM,request_stop);signal.signal(signal.SIGINT,request_stop)
    configure();raise SystemExit(main(args))
