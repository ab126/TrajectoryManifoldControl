"""Freeze a validation-selected configuration, then resume independent evaluation.

Freezing and evaluating are separate commands. Evaluation never alters weights,
costs or solver settings. Source/model/data hashes must match the frozen bundle.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import sys
import time

from validation_benchmark import ROOT,VALIDATION,PIN,SOURCE_SHA256,load_upstream,sha256,write_json,verify_snapshot


def freeze(args):
    if args.bundle.resolve().is_relative_to(VALIDATION):
        raise ValueError('freeze bundle must be outside the validation snapshot')
    evidence=args.screen.resolve()
    summary=json.loads((evidence/'summary.json').read_text())
    manifest=json.loads((evidence/'manifest.json').read_text())
    if not summary['complete']:
        raise ValueError('complete the declared tuning screen before freezing')
    if sha256(ROOT/'src/manifold_experiments.py')!=manifest['code_sha256']:
        raise ValueError('native source changed after the tuning screen')
    if args.variant not in manifest['variants'] or not isinstance(manifest['variants'][args.variant],dict):
        raise ValueError('variant must identify a native configuration in the completed tuning screen')
    if args.bundle.exists():raise ValueError('freeze bundle already exists')
    verify_snapshot()
    baselines=args.baseline_models or args.models
    if len(baselines)!=len(args.models):
        raise ValueError('one original baseline model is required per candidate model')
    models=[]
    for directory,baseline in zip(args.models,baselines):
        directory=directory.resolve()
        baseline=baseline.resolve()
        models.append(dict(directory=str(directory),checkpoint_sha256=sha256(directory/'transition.pt'),
                           data_sha256=sha256(directory/'data.npz'),
                           baseline_directory=str(baseline),
                           baseline_checkpoint_sha256=sha256(baseline/'transition.pt'),
                           baseline_data_sha256=sha256(baseline/'data.npz')))
    if not any(row['checkpoint_sha256']==manifest['checkpoint_sha256'] for row in models):
        raise ValueError('selection model must be included in frozen bundle')
    write_json(args.bundle,dict(upstream=PIN,validation_manifest_sha256=SOURCE_SHA256,
        configuration=manifest['variants'][args.variant],
        source_sha256=sha256(ROOT/'src/manifold_experiments.py'),
        scoring_sha256=sha256(ROOT/'src/validation_metrics.py'),
        evaluator_sha256=sha256(__file__),models=models,
        adapter_sha256=sha256(ROOT/'scripts/validation_benchmark.py'),
        selection_summary_sha256=sha256(evidence/'summary.json'),
        selection_manifest_sha256=sha256(evidence/'manifest.json'),
        selection_variant=args.variant,
        seed_policy='Solver RNG seed equals each independently fitted model training seed.',
        scope='Selected using tuning starts only. Final results must not change this bundle.'))


def evaluate(args):
    bundle=json.loads(args.bundle.read_text());args.output=args.output.resolve()
    if args.output.is_relative_to(VALIDATION):raise ValueError('output must be outside validation snapshot')
    if bundle['upstream']!=PIN:raise ValueError('upstream pin mismatch')
    for filename,key in [('src/manifold_experiments.py','source_sha256'),
                         ('src/validation_metrics.py','scoring_sha256'),
                         ('scripts/validation_benchmark.py','adapter_sha256'),
                         ('scripts/frozen_manifold_validation.py','evaluator_sha256')]:
        if sha256(ROOT/filename)!=bundle[key]:raise ValueError(f'changed frozen source: {filename}')
    args.output.mkdir(parents=True,exist_ok=True)
    c,sets,scoring=load_upstream(args.output);np,torch=c.np,c.torch
    spec=importlib.util.spec_from_file_location('src.manifold_experiments',ROOT/'src/manifold_experiments.py')
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    model_record=bundle['models'][args.model_index]
    directory=Path(model_record['directory'])
    for filename,key in [('transition.pt','checkpoint_sha256'),('data.npz','data_sha256')]:
        if sha256(directory/filename)!=model_record[key]:raise ValueError(f'changed frozen artifact: {filename}')
    transition,payload=c.runner.load_transition(directory/'transition.pt',args.device)
    seed=payload['config']['seed'];cfg=c.runner.make_config('dev',seed=seed,device=args.device)
    decoder=c.CanonicalBehaviorDecoder(transition,30).to(args.device).eval()
    baseline=Path(model_record['baseline_directory'])
    for filename,key in [('transition.pt','baseline_checkpoint_sha256'),('data.npz','baseline_data_sha256')]:
        if sha256(baseline/filename)!=model_record[key]:raise ValueError(f'changed baseline artifact: {filename}')
    base_model,base_payload=c.runner.load_transition(baseline/'transition.pt',args.device)
    if 'fine_tuning' in base_payload or base_payload['config']['seed']!=seed:
        raise ValueError('baseline must be the original training recipe with the matched seed')
    base_configuration=base_payload['config'].copy()
    base_configuration['device']=cfg.device
    if base_configuration!=asdict(cfg):raise ValueError('baseline is not the original dev recipe')
    baseline_decoder=c.CanonicalBehaviorDecoder(base_model,30).to(args.device).eval()
    with np.load(baseline/'data.npz') as data:split=c.runner.split_episodes(data['X'],data['U'],seed)
    gain,_=c.runner.fit_local_feedback(split['train'],c.Q,c.R)
    configuration=bundle['configuration'].copy();configuration['seed']=seed
    native_config=module.NativeConfig(**configuration)
    fresh_path=VALIDATION/'results_literature/inventory/fresh_200_states.json'
    identities=json.loads(fresh_path.read_text())['states'] if args.set=='fresh' else sets[args.set]
    indices=list(range(len(identities))) if args.indices is None else args.indices
    if len(set(indices))!=len(indices) or any(i<0 or i>=len(identities) for i in indices):
        raise ValueError('indices must identify distinct saved initial states')
    protocol=dict(bundle_sha256=sha256(args.bundle),model_index=args.model_index,seed=seed,ic_set=args.set,
        initial_states=identities,seconds=args.seconds,arms=args.arms,device=args.device,
        indices=indices,condition=args.condition,
        torch=str(torch.__version__),numpy=str(np.__version__),ideal_simulation_feedback=True,
        timing='Controller computation recorded separately; wall latency is not injected as plant delay.')
    manifest=args.output/'manifest.json'
    if manifest.exists() and json.loads(manifest.read_text())!=protocol:raise ValueError('different existing evaluation')
    write_json(manifest,protocol)
    if args.verify_only:
        print('Frozen sources/artifacts/configuration verified; no evaluation executed.');return
    start=time.perf_counter();completed=True
    for index in indices:
        state=identities[index]
        for arm in args.arms:
            key=f'seed{seed}_{arm}_{args.set}_{index:04d}'
            path=args.output/'cases'/f'{key}.json'
            if path.exists():continue
            if time.perf_counter()-start>args.max_wall_seconds:
                completed=False;break
            ctl=module.native_controller(decoder,native_config) if arm=='native' else c.controller(baseline_decoder,cfg,gain,arm)
            call_times=[]
            def timed(k,x):
                before=time.perf_counter();action=ctl(k,x)
                if args.device=='cuda':torch.cuda.synchronize()
                call_times.append(time.perf_counter()-before);return action
            original_step=c.runner._plant_step
            if args.condition!='nominal':
                # This is evaluation-only plant intervention. The controller's
                # measurements remain in the fixed nominal coordinates and it
                # receives no mismatch parameter or predictive plant access.
                import e3_robustness as robustness
                condition=robustness.CONDITIONS[args.condition]
                previous_applied=np.zeros(1)
                def changed_plant(x,command,dt):
                    nonlocal previous_applied
                    applied=np.asarray(command).copy()
                    if condition['mode']=='delay':applied,previous_applied=previous_applied,applied
                    if condition['mode']=='mismatch':
                        return robustness.physical_mismatch_step(x,applied,dt,condition['mass'],condition['length'])
                    return original_step(x,applied,dt)
                c.runner._plant_step=changed_plant
            try:
                with np.errstate(over='ignore',invalid='ignore'):
                    t,x,u=c.runner.simulate_plant(timed,np.asarray(state),cfg,seconds=args.seconds)
            finally:c.runner._plant_step=original_step
            if args.condition!='nominal':u_applied=robustness.applied_inputs(u,args.condition=='input_delay')
            else:u_applied=u
            d=ctl.diagnostics
            upstream=c.metrics(t,x,u_applied,d['terminal_policy_used'],losses=d['loss'])
            fallback=len(d.get('fallback_events',[])) if arm=='native' else int(np.sum(~np.isfinite(d['loss'])))
            score=scoring.score_trace(t,x,u_applied,upstream=upstream,fallback_events=fallback)
            times=np.asarray(call_times)
            result=dict(seed=seed,arm=arm,ic_set=args.set,ic_index=index,initial_state=state,
                upstream=upstream,**score,
                controller_latency=dict(median=float(np.median(times)),p95=float(np.quantile(times,.95)),
                    p99=float(np.quantile(times,.99)),deadline_miss_fraction=float(np.mean(times>.02))))
            raw=args.output/'raw'/f'{key}.npz';raw.parent.mkdir(exist_ok=True)
            np.savez_compressed(raw,t=t,X=x,U=u_applied,U_commands=u,loss=d['loss'],solve_seconds=d['solve_seconds'],controller_seconds=times,
                                local=d['terminal_policy_used'])
            write_json(path,result)
            print(key,score['legacy_uninterrupted'],score['terminal_recovery'],score['periodic_terminal_recovery'],flush=True)
        if not completed:break
    # Within-seed paired comparisons, separately for all three endpoints.
    rows=[json.loads(path.read_text()) for path in (args.output/'cases').glob('*.json')]
    endpoints=('legacy_uninterrupted','terminal_recovery','periodic_terminal_recovery')
    rates=[];paired=[]
    from scipy.stats import binomtest
    baseline={r['ic_index']:r for r in rows if r['arm']=='L'}
    for arm in args.arms:
        group=[r for r in rows if r['arm']==arm]
        rates.append(dict(arm=arm,n=len(group),**{key:sum(r[key] for r in group) for key in endpoints}))
        for key in endpoints:
            shared=[r for r in group if r['ic_index'] in baseline]
            wins=sum(r[key] and not baseline[r['ic_index']][key] for r in shared)
            losses=sum(baseline[r['ic_index']][key] and not r[key] for r in shared)
            paired.append(dict(arm=arm,endpoint=key,n=len(shared),wins=wins,losses=losses,
                exact_mcnemar_p=float(binomtest(wins,wins+losses,.5).pvalue) if wins+losses else 1.))
    write_json(args.output/'summary.json',dict(complete=completed,rates=rates,paired=paired,
        invocation_seconds=time.perf_counter()-start,seed=seed))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    f=sub.add_parser('freeze');f.add_argument('--screen',type=Path,required=True)
    f.add_argument('--variant',required=True);f.add_argument('--models',nargs='+',type=Path,required=True)
    f.add_argument('--baseline-models',nargs='+',type=Path,
                   help='Matched original checkpoints for H/L when candidate models were fine-tuned')
    f.add_argument('--bundle',type=Path,required=True)
    e=sub.add_parser('evaluate');e.add_argument('--bundle',type=Path,required=True)
    e.add_argument('--model-index',type=int,default=0);e.add_argument('--output',type=Path,required=True)
    e.add_argument('--set',choices=['A','C','fresh'],default='C');e.add_argument('--seconds',type=float,default=14.)
    e.add_argument('--arms',nargs='+',choices=['H','L','M','M1','native'],default=['H','native','L'])
    e.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    e.add_argument('--indices',nargs='+',type=int,help='Predeclared subset for limited/long-run diagnostics; default all')
    e.add_argument('--condition',choices=['nominal','input_delay','mass_minus25','mass_plus25','length_minus25','length_plus25'],default='nominal')
    e.add_argument('--max-wall-seconds',type=float,default=300.);e.add_argument('--verify-only',action='store_true')
    args=p.parse_args()
    if args.command=='freeze':freeze(args)
    else:evaluate(args)
