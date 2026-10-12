"""Resumable frozen-grid selection and matched full-state controller evaluation."""
import argparse
import json
import pickle
import time
from pathlib import Path
import numpy as np
from common import *
from edmd import EDMD, QP


def model_grid(method):
    if method=='edmd':return [dict(trig=t,ridge=r) for t in [False,True] for r in [1e-6,1e-4,1e-2]]
    return [dict(degree=d,trig=t,threshold=v) for d in [2,3] for t in [False,True] for v in [.001,.01,.1]]


def code_identity():
    return {p.name:digest(p) for p in Path(__file__).parent.glob('*.py') if p.name in ['common.py','edmd.py','sindy.py','run_classical.py']}


def select(method,seed,budget):
    stage=OUT/method;dest=stage/f'seed{seed}';dest.mkdir(parents=True,exist_ok=True)
    data,identity=load_data(seed)
    identity.update(adapter_hashes=code_identity(),candidate_grid=model_grid(method),selection='minimum mean training-standardized rollout MSE across steps 1–30 and four states; candidate-order tie break')
    path=dest/'identity.json'
    if path.exists():assert json.loads(path.read_text())==identity,'Resume identity mismatch'
    else:dump(path,identity)
    dump(stage/'grid.json',dict(candidates=model_grid(method),frozen_before_final_tests=True,
         sindy_scaling='RMS feature / increment std, no centering; ridge 1e-6; max STLSQ iterations 20',
         solvers=dict(qp_max_iterations=4000,nmpc_max_iterations=50,nmpc_wall_seconds=2.,cem_population=256,cem_elite=32,cem_iterations=3),
         fallback='explicit zero input on rejected/nonfinite solve; common L only in declared +L arms'))
    scores=[]
    X,U=data['train'];scale=X[:,:-1].reshape(-1,4).std(0).clip(1e-8)
    for index,config in enumerate(model_grid(method)):
        scorepath=dest/f'candidate{index}.json';modelpath=dest/f'candidate{index}.pkl'
        if scorepath.exists() and modelpath.exists():scores.append(json.loads(scorepath.read_text()));continue
        budget.checkpoint();start=time.monotonic();cpu=cpu_seconds()
        if method=='edmd':model=EDMD(**config).fit(X,U)
        else:
            from sindy import SINDYc
            model=SINDYc(**config).fit(X,U)
        score=forecast_score(model,data['validation'],scale)
        row=dict(index=index,config=config,validation_30step_standardized_mse=score,
                 fit_cpu_seconds=cpu_seconds()-cpu,fit_wall_seconds=time.monotonic()-start,
                 condition=getattr(model,'condition',None),nonzero_coefficients=int(np.count_nonzero(model.coef)) if method=='sindy' else None)
        modelpath.write_bytes(pickle.dumps(model));dump(scorepath,row);scores.append(row);budget.checkpoint()
        print(json.dumps(dict(method=method,seed=seed,selection=row)),flush=True)
    finite=[r for r in scores if r['validation_30step_standardized_mse'] is not None]
    if not finite:raise RuntimeError('All candidate rollouts nonfinite: identification failure, not controller score')
    chosen=min(finite,key=lambda r:(r['validation_30step_standardized_mse'],r['index']))
    model=pickle.loads((dest/f"candidate{chosen['index']}.pkl").read_bytes())
    # Save choice before looking at test forecasts/cases.
    selection=dest/'selection.json'
    if selection.exists():assert json.loads(selection.read_text())==chosen
    else:dump(selection,chosen)
    if not (dest/'test_forecast.json').exists():
        score=forecast_score(model,data['test'],scale)
        dump(dest/'test_forecast.json',dict(standardized_30step_mse=score,selection_unchanged=True))
    if method=='sindy':dump(dest/'coefficients.json',dict(terms=model.names,raw_coordinate_increment_coefficients=model.coef.tolist()))
    return model,identity,chosen


def cases():
    original=json.loads((ROOT/'validation/results_v1/initial_conditions.json').read_text())
    rows=[dict(set='A',index=i,state=x,seconds=14.) for i,x in enumerate(original['sets']['A'])]
    rows += [dict(set='C',index=i,state=original['sets']['C'][i],seconds=14.) for i in original['C_subset']]
    rows += [dict(set='A',index=i,state=original['sets']['A'][i],seconds=60.) for i in (0,2,5)]
    assert len(rows)==59
    return rows


def evaluate(method,seed,arm,model,identity,chosen,case,budget):
    stage=OUT/method;key=f"seed{seed}_{arm}_{case['set']}_{case['index']:04d}_{case['seconds']:g}s"
    path=stage/'cases'/f'{key}.json'
    expected=dict(data_sha256=identity['sha256'],adapter_hashes=identity['adapter_hashes'],selected=chosen['config'],case=case,arm=arm,seed=seed)
    if path.exists():
        row=json.loads(path.read_text());assert row['identity']==expected;return row
    if method=='edmd':solver=QP(model)
    else:
        from sindy import CEM,NMPC
        solver=NMPC(model) if arm.startswith('NMPC') else CEM(model,seed)
    gain=np.array(json.loads((ROOT/f'validation/results_v1/models/episodes3500_seed{seed}/diagnostics.json').read_text())['local']['gain'])
    step=plant_interface();steps=round(case['seconds']/.02)
    X=np.empty((steps+1,4));U=np.empty((steps,1));X[0]=case['state']
    local=[];diagnostics=[];plans=[];locations=[];latencies=[]
    start=time.monotonic();start_cpu=cpu_seconds()
    for k in range(steps):
        if k%25==0:budget.checkpoint()
        catch=bool(arm.endswith('+L') and X[k]@Q@X[k]<.75)
        local.append(catch)
        if catch:u=np.clip(-gain@X[k],-10,10)
        else:
            before=time.monotonic()
            proposal,info=solver.solve(X[k]);latencies.append(time.monotonic()-before)
            diagnostics.append(info);plans.append(proposal);locations.append(k);u=proposal[0]
        U[k]=np.clip(u,-10,10)
        with np.errstate(over='ignore',invalid='ignore'):X[k+1]=step(X[k],U[k,0])
        if not np.isfinite(X[k+1]).all():
            # Save partial raw trajectory; distinguish state nonfiniteness explicitly.
            X=X[:k+2];U=U[:k+1];break
    t=np.arange(len(X))*.02;tail=t>=case['seconds']-3.
    complete=len(U)==steps;finite=bool(np.isfinite(X).all() and np.isfinite(U).all())
    balanced=bool(complete and finite and np.all(np.abs(X[tail,2])<=.02) and np.all(np.abs(X[tail,3])*np.sqrt(9.81)<=.05))
    no_fall=bool(finite and np.all(np.abs(X[:,2])<np.pi/2))
    def number(v):return float(v) if np.isfinite(v) else None
    row=dict(identity=expected,case=key,seed=seed,arm=arm,set=case['set'],seconds=case['seconds'],
             complete=complete,balanced=balanced,no_fall=no_fall,primary_success=balanced and no_fall,
             cart_rest=bool(complete and finite and np.all(np.abs(X[tail,1])*np.sqrt(9.81)<=.05)),
             nonfinite_state=not finite,max_abs_p=number(np.max(np.abs(X[:,0]))),final_theta=number(X[-1,2]),
             max_abs_u=number(np.max(np.abs(U))),effort=number(np.sum(U*U)*.02),
             input_limit_seconds=float(np.sum(np.abs(U)>=10-1e-6)*.02),
             local_fraction=float(np.mean(local)),switch_toggles=int(np.sum(np.diff(local)!=0)),
             solver_failures=sum(not d['accepted'] for d in diagnostics),solve_count=len(diagnostics),
             solve_latency_quantiles=np.quantile(latencies,[.5,.95,1.]).tolist() if latencies else [],
             wall_seconds=time.monotonic()-start,cpu_seconds=cpu_seconds()-start_cpu)
    raw=stage/'raw'/f'{key}.npz';raw.parent.mkdir(exist_ok=True)
    np.savez_compressed(raw,t=t,X=X,U=U,local=local,plan_inputs=plans,solve_steps=locations,solve_seconds=latencies)
    row['raw_path']=str(raw.relative_to(ROOT));dump(stage/'solver_logs'/f'{key}.json',diagnostics);dump(path,row)
    budget.checkpoint();print(json.dumps({k:row[k] for k in ['case','primary_success','solver_failures','wall_seconds']}),flush=True)
    return row


def main():
    parser=argparse.ArgumentParser();parser.add_argument('method',choices=['edmd','sindy']);parser.add_argument('--phase',choices=['fit','pilot','matrix'],default='fit');args=parser.parse_args()
    method=args.method;isolation(method);budget=Budget(method,8 if method=='edmd' else 12);budget.checkpoint()
    checks=json.loads((OUT/method/'checks.json').read_text());assert checks['passed']
    arms=['QP','QP+L'] if method=='edmd' else ['CEM','CEM+L','NMPC','NMPC+L']
    for seed in (range(3) if args.phase!='pilot' else [0]):
        model,identity,chosen=select(method,seed,budget)
        if args.phase=='fit':continue
        subset=cases()[:16] if args.phase=='pilot' else cases()
        if args.phase=='matrix':
            forecast=json.loads((OUT/method/'pilot_projection.json').read_text())
            if forecast['projected_core_hours']>forecast['remaining_stage_core_hours']:
                raise RuntimeError('Minimum matrix projection exceeds stage allocation; retain pilot partial counts')
        for arm in arms:
            for case in subset:evaluate(method,seed,arm,model,identity,chosen,case,budget)
    if args.phase=='pilot':
        rows=[json.loads(p.read_text()) for p in (OUT/method/'cases').glob('*.json')]
        timing={arm:float(np.mean([r['cpu_seconds'] for r in rows if r['arm']==arm])) for arm in arms}
        projected=sum(timing.values())*177/3600
        remaining=(8 if method=='edmd' else 12)-json.loads((OUT/method/'accounting.json').read_text())['cpu_seconds']/3600
        dump(OUT/method/'pilot_projection.json',dict(mean_case_cpu_seconds=timing,projected_core_hours=projected,remaining_stage_core_hours=remaining,minimum_runs_per_arm=177,arms=arms))
    budget.checkpoint()
    dump(OUT/method/f'{args.phase}_completion.json',dict(state='complete',phase=args.phase,adapter_hashes=code_identity()))


if __name__=='__main__':main()
