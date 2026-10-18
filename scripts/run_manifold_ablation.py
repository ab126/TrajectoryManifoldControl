"""Resumable small native-controller ablations on independent tuning starts.

No test-set success counts select settings. Defaults are a bounded screening
experiment, not final independent validation or a claim of improvement.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import sys
import time

from validation_benchmark import load_upstream, write_json, sha256, ROOT, PIN


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-dir',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--variants',nargs='+',default=['M','H','L','cem5','cem1','adam1','periodic1','window1'],
                   choices=['M','M1','H','L','cem5','cem1','adam1','periodic1','window1'])
    p.add_argument('--limit',type=int,default=2)
    p.add_argument('--seconds',type=float,default=14.)
    p.add_argument('--adam-iterations',type=int,default=20)
    p.add_argument('--max-wall-seconds',type=float,default=300.)
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    args=p.parse_args();args.output=args.output.resolve()
    if args.output.is_relative_to(ROOT/'validation'):p.error('output must be outside validation snapshot')
    if not 1<=args.limit<=8:p.error('limit must be between1 and8')
    args.output.mkdir(parents=True,exist_ok=True)
    c,benchmark_sets,scoring=load_upstream(args.output)
    np,torch=c.np,c.torch
    path=ROOT/'src/manifold_experiments.py'
    spec=importlib.util.spec_from_file_location('src.manifold_experiments',path)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    transition,payload=c.runner.load_transition(args.model_dir/'transition.pt',args.device)
    cfg=c.runner.make_config('dev',seed=payload['config']['seed'],device=args.device)
    decoder=c.CanonicalBehaviorDecoder(transition,30).to(args.device).eval()
    with np.load(args.model_dir/'data.npz') as data:
        split=c.runner.split_episodes(data['X'],data['U'],cfg.seed)
    gain,_=c.runner.fit_local_feedback(split['train'],c.Q,c.R)
    # Fixed, separately seeded local recovery screen; these are never final C200.
    tuning=np.random.default_rng(20261017).uniform([-.5,-.8,-.6,-1.],[.5,.8,.6,1.],(8,4))
    assert not any(np.array_equal(x,y) for x in tuning for y in benchmark_sets['C'])
    common=dict(seed=cfg.seed,iterations=args.adam_iterations)
    configurations={
        'cem5':module.NativeConfig(solver='safe_cem',replan_interval=5,**common),
        'cem1':module.NativeConfig(solver='safe_cem',replan_interval=1,**common),
        'adam1':module.NativeConfig(solver='adam',replan_interval=1,**common),
        'periodic1':module.NativeConfig(solver='adam',replan_interval=1,periodic=True,**common),
        'window1':module.NativeConfig(solver='adam',replan_interval=1,periodic=True,terminal_window=5,**common),
    }
    protocol=dict(upstream=PIN,seed=cfg.seed,checkpoint_sha256=sha256(args.model_dir/'transition.pt'),
        data_sha256=sha256(args.model_dir/'data.npz'),code_sha256=sha256(path),runner_sha256=sha256(__file__),
        seconds=args.seconds,tuning_seed=20261017,tuning_states=tuning[:args.limit].tolist(),
        variants={key:asdict(configurations[key]) if key in configurations else 'unchanged_upstream'
                  for key in args.variants},device=args.device,torch=str(torch.__version__),
        limitations='Local recovery screen only; no configuration selection or final claims.')
    manifest=args.output/'manifest.json'
    if manifest.exists() and json.loads(manifest.read_text())!=protocol:
        raise ValueError('different protocol already exists; use a new output')
    write_json(manifest,protocol)
    start=time.perf_counter();rows=[];finished=True
    for index,state in enumerate(tuning[:args.limit]):
        for variant in args.variants:
            case=args.output/'cases'/f'{variant}_{index:04d}.json'
            if case.exists():rows.append(json.loads(case.read_text()));continue
            if time.perf_counter()-start>args.max_wall_seconds:
                finished=False;break
            if variant in configurations:
                control=module.native_controller(decoder,configurations[variant])
            else:control=c.controller(decoder,cfg,gain,variant)
            then=time.perf_counter()
            with np.errstate(over='ignore',invalid='ignore'):
                t,x,u=c.runner.simulate_plant(control,state,cfg,seconds=args.seconds)
            d=control.diagnostics
            upstream=c.metrics(t,x,u,d['terminal_policy_used'],losses=d['loss'])
            fallback=len(d.get('fallback_events',[])) if variant in configurations else int(np.sum(~np.isfinite(d['loss'])))
            scores=scoring.score_trace(t,x,u,upstream=upstream,fallback_events=fallback)
            times=np.asarray(d['solve_seconds'])
            row=dict(arm=variant,seed=cfg.seed,ic_set='tuning20261017',ic_index=index,initial_state=state.tolist(),
                wall_seconds=time.perf_counter()-then,upstream=upstream,**scores,
                solver_latency=dict(n=len(times),median=float(np.median(times)) if len(times) else None,
                    p95=float(np.quantile(times,.95)) if len(times) else None,
                    deadline_miss_fraction=float(np.mean(times>.02)) if len(times) else None),
                solver_evaluations=sum(v.get('evaluations',0) for v in d.get('solver',[])),
                nonfinite_candidates=sum(v.get('nonfinite_candidates',0) for v in d.get('solver',[])),
                warm_start_age=d.get('warm_start_age',[]))
            raw=args.output/'raw'/f'{variant}_{index:04d}.npz';raw.parent.mkdir(exist_ok=True)
            np.savez_compressed(raw,t=t,X=x,U=u,loss=d['loss'],solve_seconds=times,
                solve_steps=d.get('solve_steps',[]),local=d['terminal_policy_used'])
            write_json(case,row);rows.append(row)
            print(json.dumps({k:row[k] for k in ('arm','ic_index','terminal_recovery','periodic_terminal_recovery','terminal_max_angle_unwrapped','wall_seconds')}),flush=True)
        if not finished:break
    rates=[];paired=[]
    for variant in args.variants:
        group=[r for r in rows if r['arm']==variant]
        rates.append(dict(variant=variant,n=len(group),
            **{key:sum(r[key] for r in group) for key in ('legacy_uninterrupted','terminal_recovery','periodic_terminal_recovery')}))
        baseline={r['ic_index']:r for r in rows if r['arm']=='L'}
        shared=[r for r in group if r['ic_index'] in baseline]
        paired.append(dict(variant=variant,n=len(shared),
            wins=sum(r['terminal_recovery'] and not baseline[r['ic_index']]['terminal_recovery'] for r in shared),
            losses=sum(baseline[r['ic_index']]['terminal_recovery'] and not r['terminal_recovery'] for r in shared)))
    write_json(args.output/'summary.json',dict(complete=finished,rates=rates,paired_recovery=paired,
        invocation_wall_seconds=time.perf_counter()-start))
    print(json.dumps(dict(complete=finished,rates=rates),indent=2))


if __name__=='__main__':main()
