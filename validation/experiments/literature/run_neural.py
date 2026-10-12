"""Frozen six-candidate passive/controlled latent experiments, resumable by identity."""
import argparse
import json
import time
import numpy as np
import torch
from common import *
from neural_common import GPUBudget,atomic_save
from passive_data import load_passive,windows


def grid(method):
    key='recon_lam' if method=='deepkoopman' else 'regularization'
    values=[.001,.01] if method=='deepkoopman' else [0.,1e-4]
    return [dict(latent=d,**{key:v}) for d in [4,8,16] for v in values]


def instantiate(method,config,seed,controlled,device):
    if method=='deepkoopman':
        from deepkoopman import DeepKoopman
        return DeepKoopman(config['latent'],controlled=controlled).to(device)
    from dfine import DFINEAdapter
    return DFINEAdapter(config['latent'],regularization=config['regularization'],seed=seed,controlled=controlled).to(device)


def normalize(model,X):
    flat=X.reshape(-1,4)
    with torch.no_grad():
        model.mean.copy_(flat.mean(0));model.scale.copy_(flat.std(0).clamp_min(1e-6))


def fit_candidate(method,mode,seed,index,config,data,identity,budget,device):
    stage=f'{method}_{mode}';dest=OUT/stage/f'seed{seed}'/f'candidate{index}';dest.mkdir(parents=True,exist_ok=True)
    X,U=data['train'];TX,TU=data['validation'];scale=X[:,:-1].reshape(-1,4).std(0).clip(1e-8)
    torch.manual_seed(seed);model=instantiate(method,config,seed,mode=='control',device)
    trainX=torch.tensor(X,dtype=torch.float32,device=device);trainU=torch.tensor(U,dtype=torch.float32,device=device);normalize(model,trainX)
    optimizer=torch.optim.Adam(model.parameters(),lr=.001)
    path=dest/'training.pt';epoch_start=0;best=float('inf');stale=0;history=[]
    if path.exists():
        state=torch.load(path,map_location=device,weights_only=False);assert state['identity']==identity
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);epoch_start=state['epoch']+1;best=state['best'];stale=state['stale'];history=state['history']
    else:
        if method=='deepkoopman' and mode=='control':
            # Forced-only initialization: bounded reconstruction fitting, then B residual fit.
            # This diagnostic does not reuse passive data or weights.
            for epoch in range(5):
                budget.checkpoint();optimizer.zero_grad(set_to_none=True)
                normalized=(trainX-model.mean)/model.scale;encoded=model.encoder(normalized)
                reconstruction=(model.decoder(encoded)-normalized).square().mean();reconstruction.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
            model.fit_fixed_encoder_input(trainX,trainU)
            dump(dest/'fixed_encoder_input_diagnostic.json',dict(B=model.B.detach().cpu().tolist(),forced_data_only=True,encoder_reconstruction_epochs=5,B_source='measured input least-squares latent residual; no plant equations'))
    if not (dest/'completion.json').exists():
        for epoch in range(epoch_start,100):
            budget.checkpoint();model.train();order=np.random.default_rng(seed+1000*epoch).permutation(len(X));batch_losses=[]
            for start in range(0,len(order),256):
                ids=order[start:start+256];optimizer.zero_grad(set_to_none=True)
                loss=(model.loss(trainX[ids],trainU[ids],recon_lam=config['recon_lam']) if method=='deepkoopman' else model.loss(trainX[ids],trainU[ids]))
                if not torch.isfinite(loss):
                    dump(dest/'failure.json',dict(state='identification_failure',reason='nonfinite training loss',epoch=epoch,config=config))
                    budget.checkpoint();return None
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();batch_losses.append(float(loss.detach()))
            model.eval();value=forecast_score(model,(TX,TU),scale)
            if value is not None and value<best:
                best=value;stale=0;atomic_save(dest/'best.pt',dict(model=model.state_dict(),epoch=epoch,validation_rollout_mse=value,identity=identity))
            else:stale+=1
            history.append(dict(epoch=epoch,train_loss=float(np.mean(batch_losses)),validation_rollout_mse=value))
            atomic_save(path,dict(model=model.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch,best=best,stale=stale,history=history,identity=identity))
            dump(dest/'history.json',history);budget.checkpoint()
            if epoch%10==0:print(json.dumps(dict(stage=stage,seed=seed,candidate=index,epoch=epoch,validation_rollout_mse=value)),flush=True)
            if stale>=10:break
        if not (dest/'best.pt').exists():
            dump(dest/'failure.json',dict(state='identification_failure',reason='no finite validation forecast',config=config));return None
        dump(dest/'completion.json',dict(state='complete',epochs=len(history),validation_rollout_mse=best,config=config))
    model.load_state_dict(torch.load(dest/'best.pt',map_location=device,weights_only=False)['model']);model.eval()
    return model


def fit_seed(method,mode,seed,budget,device):
    stage=f'{method}_{mode}';dest=OUT/stage/f'seed{seed}';dest.mkdir(parents=True,exist_ok=True)
    if mode=='passive':
        episodes,identity=load_passive(budget);data={k:windows(v) for k,v in episodes.items()}
    else:data,identity=load_data(seed)
    code={name:digest(Path(__file__).with_name(name)) for name in ['run_neural.py',f'{method}.py','neural_common.py','common.py','passive_data.py']}
    identity.update(method=method,mode=mode,seed=seed,candidates=grid(method),adapter_hashes=code,
                    epochs=100,patience=10,batch_size=256,optimizer='Adam lr .001, gradient norm clip 1',
                    selection='mean train-standardized rollout squared error across steps 1–30 and states',
                    passive_pretraining=False,training_inputs='zero' if mode=='passive' else 'saved time-aligned applied controls')
    p=dest/'identity.json'
    identity['training_device']=json.loads(p.read_text()).get('training_device',device) if p.exists() else device
    if p.exists():assert json.loads(p.read_text())==identity,'Neural resume identity mismatch'
    else:dump(p,identity)
    dump(OUT/stage/'grid.json',dict(candidates=grid(method),frozen_before_tests=True,loss_definition='source reconstruction/prediction/latent consistency + Linf/L2' if method=='deepkoopman' else 'source [1,2,3,4]-step causal prediction MSE + sum weight norms',state_mean_scale='training only'))
    rows=[]
    for index,config in enumerate(grid(method)):
        candidate_identity=dict(experiment=identity,candidate_index=index,config=config)
        completion=dest/f'candidate{index}/completion.json';failure=dest/f'candidate{index}/failure.json'
        if completion.exists():rows.append(dict(index=index,**json.loads(completion.read_text())));continue
        if failure.exists():rows.append(dict(index=index,**json.loads(failure.read_text())));continue
        model=fit_candidate(method,mode,seed,index,config,data,candidate_identity,budget,device)
        rows.append(dict(index=index,**json.loads((completion if model is not None else failure).read_text())))
    finite=[r for r in rows if r['state']=='complete' and r['validation_rollout_mse'] is not None]
    dump(dest/'candidate_scores.json',rows)
    if not finite:raise RuntimeError('No implemented latent model with finite validation forecasts; identification failure, no controller score')
    selected=min(finite,key=lambda r:(r['validation_rollout_mse'],r['index']))
    p=dest/'selection.json'
    if p.exists():assert json.loads(p.read_text())==selected
    else:dump(p,selected)
    config=selected['config'];model=instantiate(method,config,seed,mode=='control',device)
    bestpath=dest/f"candidate{selected['index']}/best.pt";model.load_state_dict(torch.load(bestpath,map_location=device,weights_only=False)['model']);model.eval()
    if not (dest/'test_forecast.json').exists():
        X,U=data['test'];pred=model.predict_inputs(X[:,0],U);scale=data['train'][0][:,:-1].reshape(-1,4).std(0).clip(1e-8)
        error=(pred-X)/scale
        def number(value):return float(value) if np.isfinite(value) else None
        result=dict(reconstruction_standardized_mse=number(np.mean(error[:,0]**2)),standardized_mse_by_horizon={str(h):number(np.mean(error[:,h]**2)) for h in [1,10,30]},rollout_mse=number(np.mean(error[:,1:]**2)),configuration_frozen=True)
        if any(v is None for v in result['standardized_mse_by_horizon'].values()):result['state']='nonfinite_test_prediction'
        dump(dest/'test_forecast.json',result)
    return model,dict(fit_identity=identity,config=config,model_sha256=digest(bestpath),evaluation_device=device)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('method',choices=['deepkoopman','dfine']);parser.add_argument('--mode',choices=['passive','control'],required=True);parser.add_argument('--phase',choices=['collect','fit','pilot','matrix'],default='fit');parser.add_argument('--device',choices=['cpu','cuda'],default='cuda');args=parser.parse_args()
    method=args.method;stage=f'{method}_{args.mode}';isolation(method)
    cpu_hours=12 if method=='dfine' and args.mode=='control' else 8
    gpu_hours={'deepkoopman_passive':3,'deepkoopman_control':4,'dfine_passive':5,'dfine_control':6}[stage]
    if args.device=='cuda':budget=GPUBudget(stage,cpu_hours,gpu_hours)
    else:torch.set_num_threads(1);torch.set_num_interop_threads(1);budget=Budget(stage,cpu_hours)
    budget.checkpoint();assert json.loads((OUT/method/'checks.json').read_text())['passed']
    if args.phase=='collect':load_passive(budget);budget.checkpoint();return
    if args.phase=='fit':
        for seed in range(3):fit_seed(method,args.mode,seed,budget,args.device)
        dump(OUT/stage/'fit_completion.json',dict(state='complete',accounting=budget.checkpoint()));return
    if args.mode!='control':raise ValueError('Passive stage has forecast/inference tables, no invented controller success scores')
    from control import TorchCEM,evaluate
    from run_classical import cases
    arms=['MPC','MPC+L']+(['filtered_L'] if method=='dfine' else ['NN_MPC','NN_MPC+L'])
    for seed in ([0] if args.phase=='pilot' else range(3)):
        model,identity=fit_seed(method,'control',seed,budget,args.device)
        def factory():
            if method=='deepkoopman':forecast=model.rollout
            else:
                def forecast(x,u):
                    z=model.current_mean.expand(len(u),-1);out=[model.decode_dynamic(z)]
                    for k in range(HORIZON):z=z@model.source.ldm.A.T+u[:,k]@model.B.T;out.append(model.decode_dynamic(z))
                    return torch.stack(out,dim=1)
            return TorchCEM(forecast,model.scale,seed,args.device)
        if args.phase=='matrix':
            projection=json.loads((OUT/stage/'pilot_projection.json').read_text())
            if projection['projected_hours']>projection['remaining_stage_hours']:raise RuntimeError('Latent-control minimum projects beyond stage cap; partial retained')
        for arm in arms:
            if arm.startswith('NN_'):
                from reference_transition import saved_decoder
                decoder=saved_decoder(seed,args.device)
                reference_identity=dict(method='submitted neural transition, every-step sampled MPC',checkpoint_sha256=digest(ROOT/f'validation/results_v1/models/episodes3500_seed{seed}/transition.pt'),data_sha256=identity['fit_identity']['sha256'],source_sha256=digest(ROOT/'src/manifold_control.py'),planner=dict(population=256,elite=32,iterations=3,first_action=True),allocation='controlled Deep Koopman stage')
                reference_factory=lambda:TorchCEM(decoder.rollout,decoder.transition.xu_std[:4],seed,args.device)
                for case in (cases()[:16] if args.phase=='pilot' else cases()):evaluate('neural_first_action',seed,arm,case,reference_identity,reference_factory,budget)
            else:
                for case in (cases()[:16] if args.phase=='pilot' else cases()):evaluate(stage,seed,arm,case,identity,factory,budget,model if method=='dfine' else None)
    if args.phase=='pilot':
        rows=[json.loads(p.read_text()) for p in (OUT/stage/'cases').glob('*.json')]
        if method=='deepkoopman':rows += [json.loads(p.read_text()) for p in (OUT/'neural_first_action/cases').glob('*.json')]
        times={arm:float(np.mean([r['wall_seconds'] for r in rows if r['arm']==arm])) for arm in arms}
        projection=sum(times.values())*(168+9*3000/700)/3600
        spent=json.loads((OUT/stage/'accounting.json').read_text())
        remaining=(gpu_hours-spent['gpu_seconds']/3600) if args.device=='cuda' else (cpu_hours-spent['cpu_seconds']/3600)
        dump(OUT/stage/'pilot_projection.json',dict(mean_case_seconds=times,projected_hours=projection,remaining_stage_hours=remaining,arms=arms))
    dump(OUT/stage/f'{args.phase}_completion.json',dict(state='complete',accounting=budget.checkpoint()))


if __name__=='__main__':main()
