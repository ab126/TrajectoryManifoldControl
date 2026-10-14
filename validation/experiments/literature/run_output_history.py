"""Separate, matched output-only DFINE track; no hidden simulated observations.

Same frozen latent settings selected in the corresponding full-state stage.
All adapters receive only causal position/angle and previous applied input.
Supervised velocity labels enter readout fitting only, identically for competitors.
"""
import argparse,json,time
import numpy as np
import torch
from common import *
from neural_common import atomic_save,GPUBudget
from dfine import DFINEAdapter
from passive_data import load_passive,windows
from transition_training import fit_transition
from reference_transition import classes
from output_history import histories,fit_observer,HistoryObserver,readout,apply_readout
from control import TorchCEM
from casebook import cases


def dataset(mode,budget,seed):
    if mode=='control':return load_data(seed)
    data,identity=load_passive(budget);return {key:windows(value) for key,value in data.items()},identity


def latent_forecast(model,z,U):
    out=[model.decode_dynamic(z)]
    for k in range(U.shape[-2]):
        z=z@model.source.ldm.A.T+(U[...,k,:]@model.B.T if model.controlled else 0.)
        out.append(model.decode_dynamic(z))
    return torch.stack(out,-2)


def fit(seed,mode,budget):
    stage='dfine_'+mode;root=OUT/stage/'output_only'/f'seed{seed}';root.mkdir(parents=True,exist_ok=True)
    data,data_id=dataset(mode,budget,seed);selected=json.loads((OUT/stage/f'seed{seed}/selection.json').read_text())['config']
    protocol=dict(data_identity=data_id,config=selected,observations=['position','angle'],history='all available causal samples; DFINE latent filter or fitted five-lag linear Kalman observer',hidden_labels='velocity/angular velocity used equally for all supervised linear readouts during training only',full_state_comparison=False,epochs=100,patience=10,lr=.001,batch_size=256,selection='minimum 30-step observed-channel validation rollout MSE; no new architecture search',adapter_sha256={p:digest(Path(__file__).with_name(p)) for p in ['run_output_history.py','output_history.py','dfine.py','common.py']})
    p=root/'protocol.json'
    if p.exists():assert json.loads(p.read_text())==protocol
    else:dump(p,protocol)
    X,U=data['train'];VX,VU=data['validation'];Y=X[:,:,[0,2]];VY=VX[:,:,[0,2]]
    torch.manual_seed(seed);model=DFINEAdapter(selected['latent'],selected['regularization'],seed,controlled=mode=='control',observations=2)
    model.mean.copy_(torch.tensor(Y[:,:-1].reshape(-1,2).mean(0),dtype=torch.float32));model.scale.copy_(torch.tensor(Y[:,:-1].reshape(-1,2).std(0).clip(1e-6),dtype=torch.float32))
    optimizer=torch.optim.Adam(model.parameters(),lr=.001);best=float('inf');stale=0;start=0;history=[];last=root/'training.pt'
    if last.exists():
        saved=torch.load(last,weights_only=False);assert saved['protocol']==protocol;model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer']);start=saved['epoch']+1;best=saved['best'];stale=saved['stale'];history=saved['history']
    TY=torch.tensor(Y,dtype=torch.float32);TU=torch.tensor(U,dtype=torch.float32)
    if not (root/'latent_completion.json').exists():
        for epoch in range(start,100):
            budget.checkpoint();model.train();order=np.random.default_rng(seed+1000*epoch).permutation(len(Y))
            for offset in range(0,len(order),256):
                ids=order[offset:offset+256];optimizer.zero_grad(set_to_none=True);loss=model.loss(TY[ids],TU[ids]);assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
            model.eval();score=forecast_score(model,(VY,VU),model.scale.detach().numpy())
            if score is not None and score<best:best=score;stale=0;atomic_save(root/'best.pt',dict(model=model.state_dict(),epoch=epoch,validation_rollout_mse=score,protocol=protocol))
            else:stale+=1
            history.append(dict(epoch=epoch,validation_rollout_mse=score));atomic_save(last,dict(model=model.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch,best=best,stale=stale,history=history,protocol=protocol));dump(root/'history.json',history);budget.checkpoint()
            if epoch%10==0:print(json.dumps(dict(stage=stage,track='output_only',seed=seed,epoch=epoch,validation_rollout_mse=score)),flush=True)
            if stale>=10:break
        if not (root/'best.pt').exists():raise RuntimeError('Output-only DFINE fitting has no finite validation checkpoint')
        dump(root/'latent_completion.json',dict(state='complete',best_validation_rollout_mse=best))
    model.load_state_dict(torch.load(root/'best.pt',weights_only=False)['model']);model.eval()
    observer_model=fit_observer(X,U);observer=HistoryObserver(observer_model)
    history_train=histories(Y);history_validation=histories(VY)
    decoder=fit_transition((history_train,U),(history_validation,VU),seed,root/'neural',budget)
    # Same hidden labels for both causal latent and history readouts. The history
    # readout uses filtered training histories, matching its inference adapter.
    readout_path=root/'readouts.npz'
    if not readout_path.exists():
        with torch.no_grad():mu,_,_=model.filter_sequence(TY,TU)
        filtered=[]
        for episode in range(len(Y)):
            if episode%50==0:budget.checkpoint()
            observer.reset();filtered.append([observer.observe(y,None if k==0 else U[episode,k-1]) for k,y in enumerate(Y[episode])])
        weights=readout(mu.numpy(),X[:,:,[1,3]]);other=readout(np.asarray(filtered),X[:,:,[1,3]])
        np.savez_compressed(readout_path,latent=weights,history=other,**observer_model)
    saved=np.load(readout_path);weights={k:saved[k] for k in ['latent','history']};observer_model={k:saved[k] for k in ['A','B','b','mean','scale','W','R']}
    dump(root/'completion.json',dict(state='complete',model_sha256=digest(root/'best.pt'),neural_sha256=digest(root/'neural/best.pt'),readout_sha256=digest(readout_path)))
    return model,decoder,observer_model,weights,protocol,data


def inference(budget):
    out=OUT/'dfine_passive/output_only';rows=[]
    arrays=np.load(OUT/'dfine_passive/inference/noise_masks.npz');noise=arrays['noise'][:,:,[0,2]];missing=arrays['missing']
    episodes,_=load_passive(budget);truth=episodes['test'][0][:,:,[0,2]]
    for seed in range(3):
        model,decoder,kf,weights,protocol,data=fit(seed,'passive',budget);scale=model.scale.numpy()
        for level in [0.,.01,.05]:
            for missing_fraction in [0.,.1]:
                key=f'seed{seed}_noise{level:g}_missing{missing_fraction:g}';p=out/f'{key}.json'
                if p.exists():rows+=json.loads(p.read_text());continue
                mask=np.ones_like(missing) if missing_fraction==0 else missing;measured=np.where(mask.astype(bool),truth+level*noise*scale,0.);filtered=[]
                observer=HistoryObserver(kf)
                for i in range(len(truth)):
                    observer.reset();filtered.append([observer.observe(y,None if k==0 else 0.,bool(mask[i,k,0])) for k,y in enumerate(measured[i])])
                filtered=np.asarray(filtered)
                with torch.no_grad():
                    Y=torch.tensor(measured,dtype=torch.float32);M=torch.tensor(mask,dtype=torch.float32);mu,cov,_=model.filter_sequence(Y,mask=M);changed=Y.clone();changed[:,50:]+=100.;test,_,_=model.filter_sequence(changed,mask=M);assert torch.equal(mu[:,:50],test[:,:50]);assert torch.linalg.eigvalsh((cov+cov.transpose(-1,-2))/2).min()>=-1e-5
                local=[];held=np.zeros((len(truth),2));persistence=[]
                for k in range(100):held=np.where(mask[:,k].astype(bool),measured[:,k],held);persistence.append(held.copy())
                persistence=np.stack(persistence,1)
                for horizon in [1,10,30]:
                    n=100-horizon;initial=filtered[:,:n];z=(initial-kf['mean'])/kf['scale']
                    for _ in range(horizon):z=z@kf['A'].T+kf['b']
                    linear=(z*kf['scale']+kf['mean'])[...,:2]
                    with torch.no_grad():
                        latent=mu[:,:n]
                        for _ in range(horizon):latent=latent@model.source.ldm.A.T
                        df=model.decode_dynamic(latent).numpy();neural=torch.tensor(initial.reshape(-1,10),dtype=torch.float32)
                        for _ in range(horizon):neural=decoder.transition(neural,torch.zeros(len(neural),1))
                        nn=neural.numpy().reshape(len(truth),n,10)[...,:2]
                    for name,pred in [('raw_persistence',persistence[:,:n]),('linear_history_Kalman',linear),('neural_history_Kalman',nn),('DFINE_output_causal',df)]:
                        score=float(np.mean(((pred-truth[:,horizon:])/scale)**2));local.append(dict(seed=seed,noise_level=level,missing_fraction=missing_fraction,method=name,horizon=horizon,standardized_observed_mse=score if np.isfinite(score) else None,n_forecasts=len(truth)*n,causal=True,scored_channels=['position','angle']))
                dump(p,local);rows+=local;budget.checkpoint()
    dump(out/'inference_summary.json',rows);dump(out/'inference_completion.json',dict(state='complete',conditions=18,reused_masks_sha256=digest(OUT/'dfine_passive/inference/noise_masks.npz'),no_hidden_observations=True))


def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['passive','control'],required=True);p.add_argument('--phase',choices=['fit','inference'],required=True);args=p.parse_args();isolation('dfine');torch.set_num_threads(1);torch.set_num_interop_threads(1);budget=Budget('dfine_'+args.mode,8 if args.mode=='passive' else 12);budget.checkpoint()
    if args.phase=='inference':inference(budget)
    else:
        for seed in range(3):fit(seed,args.mode,budget)
        dump(OUT/('dfine_'+args.mode)/'output_only/fit_completion.json',dict(state='complete',accounting=budget.checkpoint()))


if __name__=='__main__':main()
