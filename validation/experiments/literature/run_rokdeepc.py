"""Source-faithful bounded RoKDeePC study on the separately matched E4 collection."""
import argparse
import json
import pickle
import time
import numpy as np
from scipy.optimize import minimize
from common import *
from rokdeepc import RoKDeePC


def data(seed):
    path=ROOT/f'validation/results_v1/E4/models/seed{seed}/data.npz';saved=np.load(path);X,U=saved['X'].astype(float),saved['U'].astype(float)
    assert X.shape==(3281,33,4) and U.shape==(3281,32,1)
    order=np.random.default_rng(seed).permutation(len(X));n=round(.15*len(X));indices=dict(test=order[:n],validation=order[n:2*n],train=order[2*n:])
    return {k:(X[v],U[v]) for k,v in indices.items()},dict(path=str(path.relative_to(ROOT)),sha256=digest(path),split_indices={k:v.tolist() for k,v in indices.items()},cohort='E4, 3281 × 32, separate from original H/L')


def score(model,data,scale):
    X,U=data;error=[]
    for e in range(len(X)):
        pred=model.predict_inputs(X[e,1],U[e,1:31],float(U[e,0,0]))
        error.append((pred[1:]-X[e,2:32])/scale)
    return float(np.mean(np.array(error)**2))


def checks(budget):
    rng=np.random.default_rng(15);X=rng.normal(size=(40,33,4));U=rng.uniform(-1,1,(40,32,1))
    model=RoKDeePC().fit(X,U);x=np.array([.1,.2,-.1,.05]);u=np.linspace(-.1,.1,30)
    value,gradient,g,k=model.value_gradient(x,u,0.)
    independent=np.linalg.solve(model.M,model.lambda_k*model.K.T@k)
    assert np.max(np.abs(g-independent))<1e-9
    literal=x@Q@x/1000.+u@model.R@u+np.sum(model.W*(model.Y@g)**2)+.005*np.sum(g*g)+1000*np.sum((model.K@g-k)**2)
    assert abs(value-literal)<1e-9
    finite=[]
    for j in range(30):
        plus=u.copy();minus=u.copy();plus[j]+=1e-6;minus[j]-=1e-6
        finite.append((model.value_gradient(x,plus,0.)[0]-model.value_gradient(x,minus,0.)[0])/2e-6)
    error=float(np.max(np.abs(np.array(finite)-gradient)));assert error<1e-5,error
    linear=RoKDeePC(linear=True).fit(X,U);context=np.concatenate(([0.],x,u));lk,ldk=linear.query(x,u,0.)
    assert np.allclose(lk,linear.Z@((context-linear.mean)/linear.scale))
    assert np.allclose(ldk,linear.Z[:,5:]/linear.scale[5:])
    split,identity=data(0);X,U=split['train'];step=plant_interface();replay=[]
    for e in range(3):
        for k in range(1,31):replay.append(np.max(np.abs(step(X[e,k],U[e,k,0])-X[e,k+1])))
    assert max(replay)<2e-6
    dump(OUT/'rokdeepc/checks.json',dict(passed=True,source_equations=[29,34,35],independent_g_error=float(np.max(np.abs(g-independent))),literal_objective_error=float(abs(value-literal)),gradient_max_error=error,linear_kernel_limit=True,saved_post_control_replay_max_error=float(max(replay)),data_sha256=identity['sha256'],source_scope='author quadratic reformulation; no certified uncertainty radii or plant stability claim'))
    budget.checkpoint()


def fit(seed,budget):
    split,identity=data(seed);dest=OUT/'rokdeepc'/f'seed{seed}';dest.mkdir(parents=True,exist_ok=True)
    grid=[dict(bandwidth_factor=b,gamma=r) for b in [.5,1.,2.] for r in [.02,.2]]
    identity.update(grid=grid,source_hashes={name:digest(Path(__file__).with_name(name)) for name in ['rokdeepc.py','run_rokdeepc.py']},selection='mean training-standardized 30-step rollout MSE, validation episodes only',retention='first 512 training-window columns, train split order; starts 1,2 per episode; no episode stitching')
    p=dest/'identity.json'
    if p.exists():assert json.loads(p.read_text())==identity
    else:dump(p,identity)
    dump(OUT/'rokdeepc/grid.json',dict(candidates=grid,lambda_k_prime=1000.,lambda_g=.005,cost_rescaling=1000.,solver='bounded L-BFGS-B, exact kernel gradient, maxiter=100, wall ceiling 2 s; paper section IV allows generic nonlinear optimizers',source_pgd='retained for parity: max_iter=100 (99 updates), step .02, old-gradient momentum',acceptance='finite bounded proposal at iteration limit, convergence separately recorded',input_bounds=[-10,10],state_bounds=None,frozen_before_test=True))
    X,U=split['train'];scale=X[:,:-1].reshape(-1,4).std(0).clip(1e-8);rows=[]
    for index,config in enumerate(grid):
        p=dest/f'candidate{index}.json';modelpath=dest/f'candidate{index}.pkl'
        if p.exists() and modelpath.exists():rows.append(json.loads(p.read_text()));continue
        budget.checkpoint();model=RoKDeePC(**config).fit(X,U);value=score(model,split['validation'],scale)
        modelpath.write_bytes(pickle.dumps(model));row=dict(index=index,config=config,validation_rollout_mse=value,retained_windows=model.retained,alpha=model.alpha);dump(p,row);rows.append(row)
    selected=min(rows,key=lambda r:(r['validation_rollout_mse'],r['index']));p=dest/'selection.json'
    if p.exists():assert json.loads(p.read_text())==selected
    else:dump(p,selected)
    model=pickle.loads((dest/f"candidate{selected['index']}.pkl").read_bytes());linear=RoKDeePC(**selected['config'],linear=True).fit(X,U)
    if not (dest/'test_forecasts.json').exists():dump(dest/'test_forecasts.json',dict(kernel=score(model,split['test'],scale),linear=score(linear,split['test'],scale),configuration_frozen=True))
    return model,linear,identity


def evaluate(seed,arm,model,identity,case,budget):
    key=f"seed{seed}_{arm}_{case['set']}_{case['index']:04d}_{case['seconds']:g}s";dest=OUT/'rokdeepc';p=dest/'cases'/f'{key}.json';expected=dict(fit=identity,case=case,arm=arm,seed=seed)
    if p.exists():row=json.loads(p.read_text());assert row['identity']==expected;return row
    model.reset();step=plant_interface();steps=round(case['seconds']/.02);X=np.empty((steps+1,4));U=np.empty((steps,1));X[0]=case['state'];logs=[];latencies=[];start=time.monotonic();cpu=cpu_seconds()
    for k in range(steps):
        if k%25==0:budget.checkpoint()
        before=time.monotonic();sequence,info=model.solve(X[k]);latencies.append(time.monotonic()-before);logs.append(info);U[k]=sequence[0]
        with np.errstate(over='ignore',invalid='ignore'):X[k+1]=step(X[k],U[k,0])
        if not np.isfinite(X[k+1]).all():X=X[:k+2];U=U[:k+1];break
    t=np.arange(len(X))*.02;tail=t>=case['seconds']-3.;complete=len(U)==steps;finite=bool(np.isfinite(X).all() and np.isfinite(U).all());no_fall=bool(finite and np.all(np.abs(X[:,2])<np.pi/2));balanced=bool(complete and finite and np.all(np.abs(X[tail,2])<=.02) and np.all(np.abs(X[tail,3])*np.sqrt(9.81)<=.05))
    def number(v):return float(v) if np.isfinite(v) else None
    row=dict(case=key,identity=expected,seed=seed,arm=arm,set=case['set'],seconds=case['seconds'],initial_state=case['state'],primary_success=balanced and no_fall,balanced=balanced,no_fall=no_fall,complete=complete,nonfinite_state=not finite,
             cart_rest=bool(complete and finite and np.all(np.abs(X[tail,1])*np.sqrt(9.81)<=.05)),max_abs_p=number(np.max(np.abs(X[:,0]))),final_theta=number(X[-1,2]),
             max_abs_u=number(np.max(np.abs(U))),effort=number(np.sum(U*U)*.02),input_limit_seconds=float(np.sum(np.abs(U)>=10-1e-6)*.02),
             tail_theta_std=number(np.std(X[tail,2])),tail_angular_speed_std_physical=number(np.std(X[tail,3])*np.sqrt(9.81)),
             solver_failures=sum(not i['accepted'] for i in logs),solve_latency_quantiles=np.quantile(latencies,[.5,.95,1.]).tolist(),wall_seconds=time.monotonic()-start,cpu_seconds=cpu_seconds()-cpu)
    raw=dest/'raw'/f'{key}.npz';raw.parent.mkdir(exist_ok=True);np.savez_compressed(raw,t=t,X=X,U=U);row['raw_path']=str(raw.relative_to(ROOT));dump(dest/'solver_logs'/f'{key}.json',logs);dump(p,row);budget.checkpoint();print(json.dumps({k:row[k] for k in ['case','primary_success','wall_seconds']}),flush=True);return row


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['checks','fit','pilot','matrix'],required=True);args=parser.parse_args();isolation('rokdeepc');budget=Budget('rokdeepc',12);budget.checkpoint()
    if args.phase=='checks':checks(budget);return
    assert json.loads((OUT/'rokdeepc/checks.json').read_text())['passed']
    from casebook import cases
    if args.phase=='matrix':
        projection=json.loads((OUT/'rokdeepc/pilot_projection.json').read_text())
        if projection['projected_core_hours']>projection['remaining_core_hours']:raise RuntimeError('RoKDeePC minimum projects beyond cap; retain partial')
    for seed in ([0] if args.phase=='pilot' else range(3)):
        kernel,linear,identity=fit(seed,budget)
        if args.phase=='fit':continue
        for arm,model in [('RoKDeePC',kernel),('linear_kernel',linear)]:
            for case in (cases()[:16] if args.phase=='pilot' else cases()):evaluate(seed,arm,model,identity,case,budget)
    if args.phase=='pilot':
        rows=[json.loads(p.read_text()) for p in (OUT/'rokdeepc/cases').glob('*.json')];means={arm:float(np.mean([r['cpu_seconds'] for r in rows if r['arm']==arm])) for arm in ['RoKDeePC','linear_kernel']};projection=sum(means.values())*(168+9*3000/700)/3600
        dump(OUT/'rokdeepc/pilot_projection.json',dict(mean_case_cpu_seconds=means,projected_core_hours=projection,remaining_core_hours=12-json.loads((OUT/'rokdeepc/accounting.json').read_text())['cpu_seconds']/3600))
    dump(OUT/'rokdeepc'/f'{args.phase}_completion.json',dict(state='complete',accounting=budget.checkpoint()))


if __name__=='__main__':main()
