"""Evaluate configurable candidate models on independent tuning starts or the saved D grid."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
from validation_benchmark import ROOT,load_upstream,sha256,write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-dir',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--set',choices=['tuning','D'],default='tuning')
    p.add_argument('--limit',type=int)
    p.add_argument('--solver',choices=['original_cem','safe_cem','adam'],default='original_cem')
    p.add_argument('--cem-iterations',type=int,default=10)
    p.add_argument('--population',type=int,default=256)
    p.add_argument('--iterations',type=int,default=40)
    p.add_argument('--starts',type=int,default=5)
    p.add_argument('--learning-rate',type=float,default=.1)
    p.add_argument('--replan-interval',type=int,default=1)
    p.add_argument('--seconds',type=float,default=14.)
    p.add_argument('--max-wall-seconds',type=float,default=900.)
    args=p.parse_args()
    if args.output.resolve().is_relative_to(ROOT/'validation'):p.error('output outside frozen validation required')
    if args.seconds<=0 or args.max_wall_seconds<=0 or (args.limit is not None and args.limit<1):p.error('positive durations/limit required')
    args.output.mkdir(parents=True,exist_ok=True)
    c,sets,_=load_upstream(args.output);np,torch=c.np,c.torch
    from src.manifold_experiments import NativeConfig
    from src.validation_controllers import validation_lqr,trajectory_controller
    from src.validation_sweep import terminal_recovery,primary_success
    model,payload=c.runner.load_transition(args.model_dir/'transition.pt','cpu')
    cfg=c.runner.ExperimentConfig(**{**payload['config'],'device':'cpu'})
    if payload['horizon']!=30 or not np.isclose(cfg.dt,.02*np.sqrt(9.81)):p.error('requires benchmark N30 and sampling time')
    if not 1<=args.replan_interval<=30 or args.population<32 or min(args.iterations,args.starts,args.cem_iterations)<1 or args.learning_rate<=0:p.error('invalid solver configuration')
    decoder=c.CanonicalBehaviorDecoder(model,30).eval()
    with np.load(args.model_dir/'data.npz') as data:split=c.runner.split_episodes(data['X'],data['U'],cfg.seed)
    gain,_=c.runner.fit_local_feedback(split['train'],c.Q,c.R)
    config=NativeConfig(solver=args.solver,seed=cfg.seed,iterations=args.iterations,learning_rate=args.learning_rate,
                        starts=args.starts,replan_interval=args.replan_interval,population=args.population,cem_iterations=args.cem_iterations)
    from dataclasses import asdict
    states=sets['D'] if args.set=='D' else np.random.default_rng(20261017).uniform([-.5,-.8,-.6,-1.],[.5,.8,.6,1.],(8,4)).tolist()
    spec=dict(checkpoint=sha256(args.model_dir/'transition.pt'),data=sha256(args.model_dir/'data.npz'),
              source={s:sha256(ROOT/s) for s in ['scripts/screen_tmc_candidate.py','src/validation_controllers.py','src/manifold_experiments.py','src/validation_sweep.py']},
              config=asdict(config),states=states,seconds=args.seconds,gain=gain.tolist(),seed=cfg.seed,
              torch=str(torch.__version__),numpy=str(np.__version__),controller='current hybrid; same fitted L for switch/failure and baseline')
    path=args.output/'manifest.json'
    if path.exists() and json.loads(path.read_text())!=json.loads(json.dumps(spec)):p.error('protocol differs; new output required')
    write_json(path,spec)
    torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    tick=time.perf_counter();stop=False
    for i,state in enumerate(states[:args.limit] if args.limit else states):
        for arm in ['hybrid','L']:
            key=f'{arm}_{i:04d}';case=args.output/'cases'/f'{key}.json'
            if case.exists():continue
            if time.perf_counter()-tick>=args.max_wall_seconds:stop=True;break
            linear=validation_lqr(gain,x_ref=[0.]*4,u_ref=[0.],umax=cfg.umax)
            ctl=linear if arm=='L' else trajectory_controller(decoder,config,x_ref=[0.]*4,u_ref=[0.],umax=cfg.umax,linear=linear)
            x=[np.asarray(state)];u=[];lat=[];failure=None
            for k in range(round(args.seconds/.02)):
                before=time.perf_counter();action=np.asarray(ctl(k,x[-1])).reshape(1);lat.append(time.perf_counter()-before)
                if not np.isfinite(action).all():failure='nonfinite action';break
                with np.errstate(over='ignore',invalid='ignore'):next_x=c.runner._plant_step(x[-1],action,cfg.dt)
                x.append(next_x);u.append(action)
                if not np.isfinite(next_x).all():failure='nonfinite plant';break
            t=np.arange(len(x))*.02;x=np.asarray(x);u=np.asarray(u).reshape(-1,1)
            raw=args.output/'raw'/f'{key}.npz';raw.parent.mkdir(exist_ok=True)
            np.savez_compressed(raw,t=t,X=x,U=u,controller_seconds=lat)
            row=dict(arm=arm,ic_index=i,seed=cfg.seed,failure=failure,
                     terminal_recovery=failure is None and terminal_recovery(t,x,u),
                     primary_success=failure is None and primary_success(t,x,u),median_seconds=float(np.median(lat)))
            write_json(case,row);print(json.dumps(row),flush=True)
        if stop:break
    rows=[json.loads(f.read_text()) for f in (args.output/'cases').glob('*.json')]
    write_json(args.output/'summary.json',dict(complete=len(rows)==2*len(states),saved=len(rows),expected=2*len(states),
        rates={arm:dict(n=sum(r['arm']==arm for r in rows),successes=sum(r['terminal_recovery'] for r in rows if r['arm']==arm)) for arm in ['hybrid','L']}))


if __name__=='__main__':main()
