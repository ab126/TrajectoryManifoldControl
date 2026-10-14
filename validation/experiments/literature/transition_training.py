"""Resumable submitted transition architecture/AdamW-cosine training recipe."""
import json
import math
from pathlib import Path
import hashlib
import numpy as np
import torch
from reference_transition import classes
from neural_common import atomic_save
from common import dump,digest,ROOT


def fit_transition(train,validation,seed,dest,budget,device='cpu'):
    dest=Path(dest);dest.mkdir(parents=True,exist_ok=True);X,U=train;VX,VU=validation
    identity=dict(seed=seed,shapes=[list(X.shape),list(U.shape)],training_hash=hashlib.sha256(X.tobytes()+U.tobytes()).hexdigest(),validation_hash=hashlib.sha256(VX.tobytes()+VU.tobytes()).hexdigest(),
                  architecture=[192,192,192],activation='SiLU',epochs=100,batch_size=2048,optimizer='AdamW lr .001, weight decay 1e-6',scheduler='CosineAnnealingLR T_max=100',selection='minimum held-out normalized one-step increment MSE, original recipe',source_sha256=digest(ROOT/'src/manifold_control.py'),recipe_sha256=digest(ROOT/'scripts/run_inverted_pendulum_mc.py'),adapter_sha256=digest(__file__),device=device)
    p=dest/'identity.json'
    if p.exists():assert json.loads(p.read_text())==identity
    else:dump(p,identity)
    Transition,Decoder=classes();torch.manual_seed(seed);model=Transition(X.shape[-1],1,[192]*3).to(device)
    x=torch.tensor(X[:,:-1].reshape(-1,X.shape[-1]),dtype=torch.float32,device=device);u=torch.tensor(U.reshape(-1,1),dtype=torch.float32,device=device);xn=torch.tensor(X[:,1:].reshape(-1,X.shape[-1]),dtype=torch.float32,device=device)
    xv=torch.tensor(VX[:,:-1].reshape(-1,X.shape[-1]),dtype=torch.float32,device=device);uv=torch.tensor(VU.reshape(-1,1),dtype=torch.float32,device=device);xnv=torch.tensor(VX[:,1:].reshape(-1,X.shape[-1]),dtype=torch.float32,device=device)
    model.fit_scaling(x,u,xn);optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=1e-6);scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=100)
    generator=torch.Generator(device='cpu').manual_seed(seed);history=[];best=float('inf');start=0;last=dest/'training.pt'
    if last.exists():
        saved=torch.load(last,map_location=device,weights_only=False);assert saved['identity']==identity;model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer']);scheduler.load_state_dict(saved['scheduler']);generator.set_state(saved['generator_state'].cpu());start=saved['epoch']+1;best=saved['best'];history=saved['history']
    if not (dest/'completion.json').exists():
        for epoch in range(start,100):
            budget.checkpoint();model.train();order=torch.randperm(len(x),generator=generator);total=0.
            for offset in range(0,len(order),2048):
                ids=order[offset:offset+2048].to(device);optimizer.zero_grad(set_to_none=True);loss=(((model(x[ids],u[ids])-xn[ids])/model.dx_std)**2).mean()
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite reference training loss')
                loss.backward();optimizer.step();total+=float(loss.detach())*len(ids)
            scheduler.step();model.eval()
            with torch.no_grad():value=float((((model(xv,uv)-xnv)/model.dx_std)**2).mean())
            if np.isfinite(value) and value<best:best=value;atomic_save(dest/'best.pt',dict(model=model.state_dict(),epoch=epoch,validation_loss=value,identity=identity))
            history.append(dict(epoch=epoch,train_loss=total/len(x),validation_loss=value))
            atomic_save(last,dict(model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),generator_state=generator.get_state(),epoch=epoch,best=best,history=history,identity=identity));dump(dest/'history.json',history);budget.checkpoint()
            if epoch%10==0:print(json.dumps(dict(reference=str(dest),seed=seed,epoch=epoch,validation_loss=value)),flush=True)
        if not (dest/'best.pt').exists():raise RuntimeError('No finite reference checkpoint')
        dump(dest/'completion.json',dict(state='complete',best_validation_loss=best,epochs=100))
    model.load_state_dict(torch.load(dest/'best.pt',map_location=device,weights_only=False)['model']);model.eval()
    return Decoder(model,30).to(device).eval()


class NeuralPredictor:
    def __init__(self,decoder):self.decoder=decoder
    def predict_inputs(self,x,U):
        with torch.no_grad():
            device=next(self.decoder.parameters()).device
            return self.decoder.rollout(torch.as_tensor(x,dtype=torch.float32,device=device),torch.as_tensor(U,dtype=torch.float32,device=device)).cpu().numpy()
