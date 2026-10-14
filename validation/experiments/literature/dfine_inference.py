"""Causal passive inference/forecasting with shared frozen noise and missing masks."""
import json
import numpy as np
import torch
from common import *
from passive_data import load_passive,windows
from dfine import DFINEAdapter
from reference_transition import classes


def linear_filter(Y,mask,train):
    X,_=train;x=X[:,:-1].reshape(-1,4);next_x=X[:,1:].reshape(-1,4);mean=x.mean(0);scale=x.std(0).clip(1e-8)
    z=(x-mean)/scale;target=(next_x-mean)/scale;v=np.column_stack((z,np.ones(len(z))))
    fitted=np.linalg.solve(v.T@v/len(v)+1e-6*np.eye(5),v.T@target/len(v));A=fitted[:4].T;b=fitted[4]
    residual=target-v@fitted;W=np.cov(residual,rowvar=False)+1e-6*np.eye(4);R=.01*np.eye(4)
    n,t,_=Y.shape;mu=np.zeros((n,4));P=np.repeat(np.eye(4)[None],n,axis=0);output=[]
    for k in range(t):
        if k:mu=mu@A.T+b;P=A@P@A.T+W
        K=P@np.linalg.inv(P+R);K=K*mask[:,k,None]
        mu=mu+np.einsum('bij,bj->bi',K,(Y[:,k]-mean)/scale-mu);P=(np.eye(4)-K)@P;output.append(mu.copy())
    return np.stack(output,axis=1),dict(A=A,b=b,mean=mean,scale=scale,W=W,R=R)


def main():
    isolation('dfine');torch.set_num_threads(1);torch.set_num_interop_threads(1);budget=Budget('dfine_passive',8);budget.checkpoint()
    assert json.loads((OUT/'dfine_passive/fit_completion.json').read_text())['state']=='complete'
    assert json.loads((OUT/'passive_baselines/completion.json').read_text())['state']=='complete'
    episodes,identity=load_passive(budget);train=windows(episodes['train']);truth=episodes['test'][0];scale=train[0][:,:-1].reshape(-1,4).std(0).clip(1e-8)
    out=OUT/'dfine_passive/inference';out.mkdir(exist_ok=True)
    rng=np.random.default_rng(20261013);noise=rng.normal(size=truth.shape);missing=(rng.uniform(size=truth.shape[:-1]+(1,))>=.1).astype(float);missing[:,0]=1.
    arrays=out/'noise_masks.npz'
    if arrays.exists():
        saved=np.load(arrays);assert np.array_equal(saved['noise'],noise) and np.array_equal(saved['missing'],missing)
    else:np.savez_compressed(arrays,noise=noise,missing=missing)
    protocol=dict(seed=20261013,noise_levels_training_std=[0.,.01,.05],missing_probability=[0.,.1],initial_observation_available=True,data_identity=identity,
                  shared_masks_sha256=digest(arrays),noise_std_normalized=scale.tolist(),physical_scale_multipliers=[1.,float(np.sqrt(9.81)),1.,float(np.sqrt(9.81))],
                  kalman='affine dynamics/residual covariance fit on same clean training windows; fixed normalized measurement covariance .01 I; no noise-specific tuning',
                  neural_history='same available causal observation history through the declared fitted linear Kalman adapter',
                  dfine='source causal latent filtering, fitted W/R, fixed identity prior covariance, decoded conditional means; no smoothing',score='common train-standardized squared forecast error; full observed channels; noisy data enter only through current/past observations')
    dump(out/'protocol.json',protocol)
    Transition,Decoder=classes();rows=[]
    for seed in range(3):
        root=OUT/'dfine_passive'/f'seed{seed}';selected=json.loads((root/'selection.json').read_text());config=selected['config']
        model=DFINEAdapter(config['latent'],config['regularization'],seed);model.load_state_dict(torch.load(root/f"candidate{selected['index']}/best.pt",map_location='cpu',weights_only=False)['model']);model.eval()
        neural=Transition(4,1,[192]*3);neural.load_state_dict(torch.load(OUT/'passive_baselines'/f'seed{seed}/neural/best.pt',map_location='cpu',weights_only=False)['model']);neural.eval();decoder=Decoder(neural,30)
        for level in [0.,.01,.05]:
            for missing_fraction in [0.,.1]:
                key=f'seed{seed}_noise{level:g}_missing{missing_fraction:g}';path=out/f'{key}.json'
                if path.exists():rows+=json.loads(path.read_text());continue
                budget.checkpoint();mask=np.ones_like(missing) if missing_fraction==0 else missing;observations=truth+level*noise*scale
                # Erase unavailable observations before any observation adapter sees them.
                measured=np.where(mask.astype(bool),observations,0.)
                filtered,kf=linear_filter(measured,mask,train)
                with torch.no_grad():
                    Y=torch.tensor(measured,dtype=torch.float32);M=torch.tensor(mask,dtype=torch.float32)
                    mu,cov,_=model.filter_sequence(Y,torch.zeros(len(Y),99,1),M)
                    reconstruction=model.decode_dynamic(mu).numpy();minimum=float(torch.linalg.eigvalsh((cov+cov.transpose(-1,-2))/2).min());assert minimum>=-1e-5
                    changed=Y.clone();changed[:,50:]+=100
                    causal,_,_=model.filter_sequence(changed,torch.zeros(len(Y),99,1),M);assert torch.equal(causal[:,:50],mu[:,:50])
                persistence=[];held=np.zeros((len(truth),4))
                for k in range(100):held=np.where(mask[:,k].astype(bool),observations[:,k],held);persistence.append(held.copy())
                persistence=np.stack(persistence,1);local=[]
                for horizon in [1,10,30]:
                    target=truth[:,horizon:];nsteps=100-horizon
                    raw=persistence[:,:nsteps]
                    z=filtered[:,:nsteps].copy()
                    for _ in range(horizon):z=z@kf['A'].T+kf['b']
                    linear=z*kf['scale']+kf['mean']
                    with torch.no_grad():
                        latent=mu[:,:nsteps]
                        for _ in range(horizon):latent=latent@model.source.ldm.A.T
                        predicted=model.decode_dynamic(latent).numpy()
                        initial=torch.tensor(filtered[:,:nsteps]*kf['scale']+kf['mean'],dtype=torch.float32).reshape(-1,4)
                        neural_state=initial
                        for _ in range(horizon):neural_state=neural(neural_state,torch.zeros(len(neural_state),1))
                        neural_pred=neural_state.numpy().reshape(len(truth),nsteps,4)
                    for method,prediction in [('raw_persistence',raw),('linear_Kalman',linear),('neural_Kalman_history',neural_pred),('DFINE_causal',predicted)]:
                        error=(prediction-target)/scale;value=np.mean(error**2)
                        local.append(dict(seed=seed,noise_level=level,missing_fraction=missing_fraction,method=method,horizon=horizon,n_episodes=len(truth),n_forecasts=int(len(truth)*nsteps),standardized_mse=float(value) if np.isfinite(value) else None,rmse_by_state=[float(v) if np.isfinite(v) else None for v in np.sqrt(np.mean((prediction-target)**2,axis=(0,1)))],causal=True,covariance_minimum_eigenvalue=minimum if method=='DFINE_causal' else None))
                dump(path,local);rows+=local;budget.checkpoint();print(json.dumps(dict(condition=key,state='complete')),flush=True)
    dump(out/'summary.json',rows);dump(out/'completion.json',dict(state='complete',conditions=18,causal_future_perturbation_checks_passed=True,accounting=budget.checkpoint()))


if __name__=='__main__':main()
