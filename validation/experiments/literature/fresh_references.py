"""Unmodified H/L on fresh holdout; core reserve, dedicated project interpreter."""
import ast,argparse,json,math,time,typing
from dataclasses import dataclass
from types import SimpleNamespace
import numpy as np
import torch
from common import *
from neural_common import GPUBudget
from reference_transition import saved_decoder


def factory(decoder,config,gain):
    namespace=dict(typing.__dict__);namespace.update(torch=torch,np=np,math=math,time=time,dataclass=dataclass,TensorLike=torch.Tensor,CanonicalBehaviorDecoder=type(decoder),ExperimentConfig=SimpleNamespace)
    source=ROOT/'src/manifold_control.py';tree=ast.parse(source.read_text());nodes=[n for n in tree.body if isinstance(n,ast.ClassDef) and n.name in ['DataDrivenMPCSolution','DataDrivenCEMSolver']];exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),namespace)
    source=ROOT/'scripts/run_inverted_pendulum_mc.py';tree=ast.parse(source.read_text());nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='make_controller'];exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),namespace)
    return namespace['make_controller'](decoder,config,gain)


def evaluate(seed,arm,index,state,decoder,config,gain,identity,budget,name='F'):
    root=OUT/'fresh_references';key=f'seed{seed}_{arm}_{name}_{index:04d}_14s';p=root/'cases'/f'{key}.json';expected=dict(model=identity,seed=seed,arm=arm,index=index,set=name,initial_state=state,source_sha256=digest(__file__))
    if p.exists():r=json.loads(p.read_text());assert r['identity']==expected;return r
    law=factory(decoder,config,gain) if arm=='H' else None;step=plant_interface();X=np.empty((701,4));U=np.empty((700,1));X[0]=state;start=time.monotonic();cpu=cpu_seconds()
    for k in range(700):
        if k%25==0:budget.checkpoint()
        U[k]=law(k,X[k]) if law is not None else np.clip(-gain@X[k],-10,10)
        with np.errstate(over='ignore',invalid='ignore'):X[k+1]=step(X[k],U[k,0])
        if not np.isfinite(X[k+1]).all():X=X[:k+2];U=U[:k+1];break
    t=np.arange(len(X))*.02;tail=t>=11.;finite=bool(np.isfinite(X).all());complete=len(U)==700;no_fall=bool(finite and np.all(np.abs(X[:,2])<np.pi/2));balanced=bool(complete and finite and np.all(np.abs(X[tail,2])<=.02) and np.all(np.abs(X[tail,3])*np.sqrt(9.81)<=.05));raw=root/'raw'/f'{key}.npz';raw.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(raw,X=X,U=U,t=t)
    r=dict(case=key,identity=expected,seed=seed,arm=arm,index=index,set=name,seconds=14.,initial_state=state,primary_success=balanced and no_fall,balanced=balanced,no_fall=no_fall,complete=complete,raw_path=str(raw.relative_to(ROOT)),cpu_seconds=cpu_seconds()-cpu,wall_seconds=time.monotonic()-start);dump(p,r);budget.checkpoint();return r


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['pilot','matrix'],required=True);args=parser.parse_args();isolation('deepkoopman');budget=GPUBudget('core_reserve',4.,2);budget.checkpoint();root=OUT/'fresh_references';states=json.loads((OUT/'inventory/fresh_200_states.json').read_text())['states']
    if args.phase=='matrix':
        projection=json.loads((root/'pilot_projection.json').read_text());assert projection['projected_gpu_hours']<=2-budget.checkpoint()['gpu_seconds']/3600,'Fresh reference matrix exceeds core reserve GPU cap'
    for seed in ([0] if args.phase=='pilot' else range(3)):
        decoder=saved_decoder(seed);path=ROOT/f'validation/results_v1/models/episodes3500_seed{seed}/diagnostics.json';saved=json.loads(path.read_text());config=SimpleNamespace(**{**saved['config'],'device':'cuda'});gain=np.asarray(saved['local']['gain']);identity=dict(data_sha256=digest(path.with_name('data.npz')),checkpoint_sha256=digest(path.with_name('transition.pt')),gain_sha256=digest(path),controller_sha256=digest(ROOT/'scripts/run_inverted_pendulum_mc.py'),class_sha256=digest(ROOT/'src/manifold_control.py'),simulator_sha256=digest(ROOT/'src/inverted_pendulum.py'),fresh_states_sha256=digest(OUT/'inventory/fresh_200_states.json'),H='unaltered source five-step pending queue and common L switch; AST extraction excludes batch/plant code',L='same saved fitted gain; saturated +/-10',allocation='core reserve CPU4 h + GPU2 h; separate SMC<=1 CPU h and pending PID<=1 CPU h keep reserve <=6 CPU h')
        for arm in ['H','L']:
            if args.phase=='pilot':
                authored=json.loads((ROOT/'validation/results_v1/initial_conditions.json').read_text())['sets']['A']
                for index,state in enumerate(authored):evaluate(seed,arm,index,state,decoder,config,gain,identity,budget,name='A')
            for index,state in enumerate(states[:10] if args.phase=='pilot' else states):evaluate(seed,arm,index,state,decoder,config,gain,identity,budget)
    if args.phase=='pilot':
        rows=[json.loads(p.read_text()) for p in (root/'cases').glob('*.json')];means={arm:float(np.mean([r['wall_seconds'] for r in rows if r['arm']==arm])) for arm in ['H','L']};dump(root/'pilot_projection.json',dict(projected_gpu_hours=sum(means.values())*600/3600,mean_case_seconds=means,remaining_gpu_hours=2-budget.checkpoint()['gpu_seconds']/3600))
    dump(root/f'{args.phase}_completion.json',dict(state='complete',accounting=budget.checkpoint()))


if __name__=='__main__':main()
