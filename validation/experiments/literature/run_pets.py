"""Bounded fixed-data PETS fitting, calibration, source checks and timing pilot."""
import argparse
import json
import time
import numpy as np
import torch
from common import *
from neural_common import GPUBudget, atomic_save
from pets import PETS, TSInfinityCEM


def checks(budget):
    torch.manual_seed(892);model=PETS().cuda().double()
    xu=torch.linspace(-1,1,5*7*5,device='cuda',dtype=torch.float64).reshape(5,7,5)
    with torch.no_grad():
        mean,lv=model(xu)
        current=xu.cpu().numpy()
        for i,layer in enumerate(model.layers):
            current=np.matmul(current,layer.weight.cpu().numpy())+layer.bias.cpu().numpy()
            if i<len(model.layers)-1:current=current/(1+np.exp(-current))
        maxlv=model.max_logvar.cpu().numpy();minlv=model.min_logvar.cpu().numpy()
        independent=maxlv-np.logaddexp(0,maxlv-current[...,4:]);independent=minlv+np.logaddexp(0,independent-minlv)
        err=float(np.max(np.abs(lv.cpu().numpy()-independent)))
        assert err<1e-12 and np.max(np.abs(mean.cpu().numpy()-current[...,:4]))<1e-12
    # Instrument particle-member assignments across the complete input sequence.
    class Instrumented:
        input_mean=torch.zeros(5,device='cuda')
        input_std=torch.ones(5,device='cuda')
        def __init__(self):self.calls=[]
        def moments(self,x):
            self.calls.append(x[...,0].detach().cpu().numpy())
            m=torch.arange(1,6,device='cuda')[:,None,None].expand(5,x.shape[1],4)
            return m,torch.zeros_like(m)
    probe=Instrumented();solver=TSInfinityCEM(probe,0,particles=20)
    cost=solver.costs(torch.zeros(4,device='cuda'),torch.zeros(256,30,1,device='cuda'))
    assert torch.isfinite(cost).all()
    for k,seen in enumerate(probe.calls):assert np.all(seen==np.arange(1,6)[:,None]*k)
    result=dict(passed=True,bounded_logvariance_numpy_max_error=err,particle_member_preserved_30steps=True,
                particle_count=20,ensemble_members=5,source='handful-of-trials BNN.py:_compile_outputs/_compile_losses; MPC.py:TSinf',
                adaptation='PyTorch port; fixed episode-bootstrap data; normalized increment targets; shared state/input costs; source plant and online data loop excluded')
    dump(OUT/'pets/checks.json',result);budget.checkpoint();return result


def fit(seed,budget):
    dest=OUT/'pets'/f'seed{seed}';dest.mkdir(parents=True,exist_ok=True)
    data,identity=load_data(seed)
    config=dict(ensemble_members=5,hidden=[200]*4,activation='swish',bootstrap='whole training episodes per member',
                epochs=100,patience=10,batch_size=2048,lr=.001,target_normalization='train increment mean/std',
                source_variance_bounds=[.5,-10.],variance_bound_penalty=.01,seed=seed)
    identity.update(config=config,adapter_hashes={p.name:digest(p) for p in Path(__file__).parent.glob('*.py') if p.name in ['pets.py','run_pets.py','neural_common.py','common.py']})
    identitypath=dest/'identity.json'
    if identitypath.exists():assert json.loads(identitypath.read_text())==identity,'PETS resume identity mismatch'
    else:dump(identitypath,identity)
    torch.manual_seed(seed);model=PETS().cuda();optimizer=torch.optim.Adam(model.parameters(),lr=.001)
    X,U=data['train'];x=X[:,:-1].reshape(-1,4);y=X[:,1:].reshape(-1,4)-x;xu=np.column_stack((x,U.reshape(-1,1)))
    with torch.no_grad():
        model.input_mean.copy_(torch.tensor(xu.mean(0),device='cuda'));model.input_std.copy_(torch.tensor(xu.std(0).clip(1e-6),device='cuda'))
        model.target_mean.copy_(torch.tensor(y.mean(0),device='cuda'));model.target_std.copy_(torch.tensor(y.std(0).clip(1e-6),device='cuda'))
    bootstrap=[]
    for member in range(5):
        ids=np.random.default_rng(seed*100+member).integers(0,len(X),len(X));bootstrap.append((ids[:,None]*30+np.arange(30)).reshape(-1))
    boot=np.stack(bootstrap);dump(dest/'bootstrap_identity.json',dict(sha256=__import__('hashlib').sha256(boot.tobytes()).hexdigest(),shape=list(boot.shape)))
    inputs=torch.as_tensor(xu[boot],dtype=torch.float32,device='cuda');targets=torch.as_tensor(y[boot],dtype=torch.float32,device='cuda')
    VX,VU=data['validation'];vx=VX[:,:-1].reshape(-1,4);vy=VX[:,1:].reshape(-1,4)-vx
    vi=torch.tensor(np.column_stack((vx,VU.reshape(-1,1))),dtype=torch.float32,device='cuda').unsqueeze(0).expand(5,-1,-1)
    vt=torch.tensor(vy,dtype=torch.float32,device='cuda').unsqueeze(0).expand(5,-1,-1)
    statepath=dest/'training.pt';start_epoch=0;best=float('inf');stale=0;history=[]
    if statepath.exists():
        state=torch.load(statepath,weights_only=False);model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);start_epoch=state['epoch']+1;best=state['best'];stale=state['stale'];history=state['history']
    if not (dest/'fit_completion.json').exists():
        for epoch in range(start_epoch,100):
            budget.checkpoint();model.train()
            order=torch.as_tensor(np.random.default_rng(seed+10000*epoch).permutation(inputs.shape[1]),device='cuda')
            for start in range(0,len(order),2048):
                ids=order[start:start+2048];optimizer.zero_grad(set_to_none=True)
                loss=model.normalized_loss(inputs[:,ids],targets[:,ids])
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite probabilistic training loss')
                loss.backward();optimizer.step()
            model.eval()
            with torch.no_grad():
                mean,lv=model(vi);v=float((((mean-(vt-model.target_mean)/model.target_std)**2)*torch.exp(-lv)+lv).mean())
            if v<best:
                best=v;stale=0;atomic_save(dest/'best.pt',dict(model=model.state_dict(),epoch=epoch,validation_nll=v))
            else:stale+=1
            history.append(dict(epoch=epoch,validation_normalized_nll=v));atomic_save(statepath,dict(model=model.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch,best=best,stale=stale,history=history))
            dump(dest/'history.json',history);budget.checkpoint();print(json.dumps(dict(method='pets',seed=seed,epoch=epoch,validation_nll=v)),flush=True)
            if stale>=10:break
        dump(dest/'fit_completion.json',dict(state='complete',epochs=len(history),best_validation_nll=best))
    model.load_state_dict(torch.load(dest/'best.pt',weights_only=False)['model']);model.eval()
    if not (dest/'test_forecast.json').exists():
        score=forecast_score(model,data['test'],x.std(0).clip(1e-8))
        dump(dest/'test_forecast.json',dict(standardized_30step_rollout_mse=score))
        TX,TU=data['test'];tx=TX[:,:-1].reshape(-1,4);delta=TX[:,1:].reshape(-1,4)-tx
        with torch.no_grad():
            ti=torch.tensor(np.column_stack((tx,TU.reshape(-1,1))),dtype=torch.float32,device='cuda').unsqueeze(0).expand(5,-1,-1)
            means,vars=model.moments(ti);mean=means.mean(0);variance=vars.mean(0)+means.var(0,unbiased=False)
            err=torch.tensor(delta,dtype=torch.float32,device='cuda')-mean
            nll=(.5*(torch.log(2*torch.pi*variance)+err.square()/variance)).mean(0)
            coverage=(err.abs()<=1.96*variance.sqrt()).float().mean(0)
        dump(dest/'calibration.json',dict(moment_matched_gaussian_nll_by_state=nll.cpu().tolist(),coverage95_by_state=coverage.cpu().tolist(),scope='one-step held-out increments; Gaussian moment approximation to ensemble mixture'))
    return model


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['checks','fit','timing','pilot','matrix'],default='checks');args=parser.parse_args()
    isolation('pets');budget=GPUBudget('pets',8,8);budget.checkpoint()
    if args.phase=='checks':checks(budget);return
    assert json.loads((OUT/'pets/checks.json').read_text())['passed']
    if args.phase=='fit':
        for seed in range(3):fit(seed,budget)
    elif args.phase=='timing':
        model=fit(0,budget);times=[]
        for particles,population in [(20,256),(5,256),(5,128)]:
            solver=TSInfinityCEM(model,0,particles,population);samples=[]
            for _ in range(3):
                torch.cuda.synchronize();start=time.monotonic();u,info=solver.solve(np.array([0.,0.,.3,0.]));torch.cuda.synchronize();samples.append(time.monotonic()-start);assert info['accepted'];budget.checkpoint()
            times.append(dict(particles=particles,population=population,solve_seconds=samples))
        dump(OUT/'pets/timing.json',times)
    else:
        from control import evaluate
        from run_classical import cases
        timing=json.loads((OUT/'pets/timing.json').read_text())
        remaining=8-json.loads((OUT/'pets/accounting.json').read_text())['gpu_seconds']/3600
        choices=[]
        for config in timing:
            # 168 short cases plus 9 long cases; all controls applied every sample.
            projection=max(config['solve_seconds'][1:])*(168*700+9*3000)/3600
            if projection<=remaining:choices.append((config,projection))
        if not choices:raise RuntimeError('Even reduced PETS minimum matrix exceeds measured allocation; preserve partial')
        config,projection=choices[0]
        selected=OUT/'pets/planner_selection.json'
        selection=dict(particles=config['particles'],population=config['population'],iterations=3,projected_gpu_hours=projection,
                       rule='largest prespecified particle/population configuration whose timed solves fit minimum matrix')
        if selected.exists():selection=json.loads(selected.read_text())
        else:dump(selected,selection)
        if args.phase=='matrix':
            pilot=json.loads((OUT/'pets/pilot_projection.json').read_text())
            if pilot['projected_gpu_hours']>pilot['remaining_gpu_hours']:raise RuntimeError('PETS pilot projects beyond allocation; cap-limited partial')
        seeds=[0] if args.phase=='pilot' else range(3)
        for seed in seeds:
            model=fit(seed,budget)
            identity=dict(fit=json.loads((OUT/'pets'/f'seed{seed}/identity.json').read_text()),
                          model_sha256=digest(OUT/'pets'/f'seed{seed}/best.pt'),planner=selection,semantics='TS-infinity, conditional particle sampled costs, native first-action MPC')
            factory=lambda:TSInfinityCEM(model,seed,selection['particles'],selection['population'])
            for case in (cases()[:16] if args.phase=='pilot' else cases()):evaluate('pets',seed,'PETS',case,identity,factory,budget)
        if args.phase=='pilot':
            rows=[json.loads(p.read_text()) for p in (OUT/'pets/cases').glob('*.json')]
            mean=float(np.mean([r['wall_seconds'] for r in rows]));projection=mean*(168+9*3000/700)/3600
            dump(OUT/'pets/pilot_projection.json',dict(mean_short_case_seconds=mean,projected_gpu_hours=projection,remaining_gpu_hours=8-json.loads((OUT/'pets/accounting.json').read_text())['gpu_seconds']/3600,minimum_runs=177))
        dump(OUT/'pets'/f'{args.phase}_completion.json',dict(state='complete',phase=args.phase))
    budget.checkpoint()


if __name__=='__main__':main()
