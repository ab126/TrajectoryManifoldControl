"""Matched fine-tuning ablation from an unchanged original checkpoint.

Both arms see identical training episodes, normalization, minibatch orders and
prefix windows. A common validation criterion (one-step + multistep) selects
checkpoints; final test trajectories/starts are never loaded for selection.
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path

from validation_benchmark import load_upstream, write_json, sha256, ROOT, PIN


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-dir',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--epochs',type=int,default=10)
    p.add_argument('--window',type=int,default=5)
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    args=p.parse_args();args.output=args.output.resolve()
    if args.output.is_relative_to(ROOT/'external'):p.error('output must be outside external')
    if args.output.exists() and any(args.output.iterdir()):p.error('output must be new or empty')
    if args.epochs<1 or not 1<=args.window<=30:p.error('invalid epochs/window')
    args.output.mkdir(parents=True,exist_ok=True)
    c,_,_=load_upstream(args.output);np,torch=c.np,c.torch
    base,payload=c.runner.load_transition(args.model_dir/'transition.pt',args.device)
    seed=payload['config']['seed']
    with np.load(args.model_dir/'data.npz') as data:
        split=c.runner.split_episodes(data['X'],data['U'],seed)
    train=tuple(torch.from_numpy(a).to(args.device) for a in split['train'])
    val=tuple(torch.from_numpy(a).to(args.device) for a in split['validation'])
    X,U=train;V,W=val
    state_std=X[:,:-1].reshape(-1,4).std(0).clamp_min(1e-6)
    # Every overlapping validation window; shared criterion for both arms.
    vx=torch.cat([V[:,start:start+args.window+1] for start in range(31-args.window)])
    vu=torch.cat([W[:,start:start+args.window] for start in range(31-args.window)])
    results={}
    for arm,weight in [('one_step',0.),('multistep',1.)]:
        model=copy.deepcopy(base)
        for parameter in model.parameters():parameter.requires_grad_(True)
        decoder=c.CanonicalBehaviorDecoder(model,args.window)
        optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-6)
        rng=np.random.default_rng(20261017+seed)
        best=float('inf');best_state=None;history=[]
        for epoch in range(args.epochs):
            model.train();order=rng.permutation(len(X));starts=rng.integers(0,31-args.window,len(X))
            for offset in range(0,len(X),256):
                ids=order[offset:offset+256];start=starts[ids]
                xx=X[ids];uu=U[ids]
                optimizer.zero_grad(set_to_none=True)
                one=(((model(xx[:,:-1].reshape(-1,4),uu.reshape(-1,1))-xx[:,1:].reshape(-1,4))/model.dx_std)**2).mean()
                if weight:
                    idx=torch.as_tensor(start,device=X.device)[:,None]+torch.arange(args.window+1,device=X.device)[None,:]
                    window_x=xx.gather(1,idx[:,:,None].expand(-1,-1,4))
                    window_u=uu.gather(1,idx[:,:-1,None])
                    roll=(((decoder.rollout(window_x[:,0],window_u)[:,1:]-window_x[:,1:])/state_std)**2).mean()
                    loss=one+weight*roll
                else:loss=one
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),100.);optimizer.step()
            model.eval()
            with torch.no_grad():
                one=(((model(V[:,:-1].reshape(-1,4),W.reshape(-1,1))-V[:,1:].reshape(-1,4))/model.dx_std)**2).mean()
                total=0.
                for offset in range(0,len(vx),512):
                    xx=vx[offset:offset+512];uu=vu[offset:offset+512]
                    error=(decoder.rollout(xx[:,0],uu)[:,1:]-xx[:,1:])/state_std
                    total+=float(error.square().mean())*len(xx)
                roll=total/len(vx);criterion=float(one)+roll
            history.append(dict(epoch=epoch,one_step_validation=float(one),multistep_validation=roll,criterion=criterion))
            if criterion<best:
                best=criterion;best_state={key:value.detach().cpu().clone() for key,value in model.state_dict().items()}
        target=args.output/arm;target.mkdir()
        saved=payload.copy();saved['transition_state_dict']=best_state
        saved['fine_tuning']=dict(arm=arm,epochs=args.epochs,window=args.window,rollout_weight=weight,
            lr=1e-4,validation='one_step_dx_standardized + rollout_state_standardized',history=history,
            parent_checkpoint_sha256=sha256(args.model_dir/'transition.pt'))
        torch.save(saved,target/'transition.pt')
        # Keep the measured dataset available through a symlink; no duplication or edits.
        (target/'data.npz').symlink_to((args.model_dir/'data.npz').resolve())
        results[arm]=dict(checkpoint_sha256=sha256(target/'transition.pt'),best_validation_criterion=best,
                         history=history)
    write_json(args.output/'training_summary.json',dict(upstream=PIN,parent_checkpoint_sha256=sha256(args.model_dir/'transition.pt'),
        data_sha256=sha256(args.model_dir/'data.npz'),script_sha256=sha256(__file__),
        seed=seed,window=args.window,epochs=args.epochs,results=results,
        consistency='Shared recursive transition gives exact predicted-prefix consistency; multistep targets use measured overlapping windows. No direct decoder or geometric penalty selected.'))
    print({arm:row['best_validation_criterion'] for arm,row in results.items()})


if __name__=='__main__':main()
