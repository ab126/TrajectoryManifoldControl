"""Train a configurable candidate from measured data without changing frozen recipes."""
from __future__ import annotations
import argparse
from dataclasses import asdict
from pathlib import Path
import subprocess
import sys
from validation_benchmark import ROOT,load_upstream,write_json,sha256


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--episodes',type=int,default=14000)
    p.add_argument('--epochs',type=int,default=200)
    p.add_argument('--hidden-dims',type=int,nargs='+',default=[256,256,256])
    p.add_argument('--multistep-epochs',type=int,default=20)
    p.add_argument('--window',type=int,default=5)
    p.add_argument('--angle-limit-deg',type=float,default=85.)
    p.add_argument('--local-fraction',type=float,default=.25)
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    args=p.parse_args();args.output=args.output.resolve()
    if args.output.is_relative_to(ROOT/'validation'):p.error('output must be outside frozen validation')
    if min(args.episodes,args.epochs,*args.hidden_dims)<1 or args.multistep_epochs<0 or not 1<=args.window<=30:p.error('invalid training sizes')
    if not 0<=args.local_fraction<=1 or not 0<args.angle_limit_deg<180:p.error('invalid collection domain')
    args.output.mkdir(parents=True,exist_ok=True)
    c,_,_=load_upstream(args.output);np,torch=c.np,c.torch
    cfg=c.runner.make_config('dev',seed=args.seed,episodes=args.episodes,epochs=args.epochs,hidden_dims=tuple(args.hidden_dims),device=args.device)
    spec=dict(config=asdict(cfg),angle_limit_deg=args.angle_limit_deg,local_fraction=args.local_fraction,multistep_epochs=args.multistep_epochs,window=args.window,
              script_sha256=sha256(__file__),recipe_sha256=sha256(ROOT/'scripts/run_inverted_pendulum_mc.py'))
    import json
    path=args.output/'training_manifest.json'
    if path.exists() and json.loads(path.read_text())!=json.loads(json.dumps(spec)):p.error('training protocol differs; use a new directory')
    write_json(path,spec)
    np.random.seed(args.seed);torch.manual_seed(args.seed);torch.use_deterministic_algorithms(True)
    data_path=args.output/'data.npz'
    if data_path.exists():
        with np.load(data_path) as data:X,U=data['X'],data['U']
    else:
        rng=np.random.default_rng(args.seed)
        X=np.empty((cfg.episodes,31,4),np.float32);U=np.empty((cfg.episodes,30,1),np.float32)
        for episode in range(cfg.episodes):
            local=rng.random()<args.local_fraction
            bound=np.array([.5,.3,.15,.3]) if local else np.array([1.5,2.5,np.deg2rad(args.angle_limit_deg),3.5])
            state=rng.uniform(-bound,bound);X[episode,0]=state
            umax=.5 if local else cfg.umax
            action=rng.uniform(-umax,umax);hold=int(rng.integers(1,6))
            for k in range(30):
                if k%hold==0:
                    action=rng.uniform(-umax,umax) if episode%2 else np.clip(.75*action+rng.normal(scale=.3*umax),-umax,umax)
                    hold=int(rng.integers(1,6))
                U[episode,k,0]=action
                state=c.runner._plant_step(state,np.array([action]),cfg.dt);X[episode,k+1]=state
        if not np.isfinite(X).all():raise RuntimeError('nonfinite collected data')
        np.savez_compressed(data_path,X=X,U=U)
    split=c.runner.split_episodes(X,U,args.seed)
    checkpoint=args.output/'transition.pt'
    if not checkpoint.exists():c.runner.train_transition(split['train'],split['validation'],cfg,checkpoint)
    transition,payload=c.runner.load_transition(checkpoint,args.device)
    decoder=c.CanonicalBehaviorDecoder(transition,30).to(args.device).eval()
    gain,local=c.runner.fit_local_feedback(split['train'],c.Q,c.R)
    write_json(args.output/'diagnostics.json',dict(config=asdict(cfg),prediction=c.prediction_rmse(decoder,split['test']),local=c.fit_diagnostics(split['train'],local)))
    if args.multistep_epochs:
        target=args.output/'finetuned'
        if not (target/'training_summary.json').exists():
            subprocess.run([sys.executable,str(ROOT/'scripts/train_multistep_ablation.py'),
                '--model-dir',str(args.output),'--output',str(target),'--epochs',str(args.multistep_epochs),
                '--window',str(args.window),'--device',args.device],check=True)
    print('Candidate checkpoint:',args.output/('finetuned/multistep/transition.pt' if args.multistep_epochs else 'transition.pt'))


if __name__=='__main__':main()
