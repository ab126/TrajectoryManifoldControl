"""Matched passive prediction references, including retrained submitted architecture."""
import json
import pickle
import numpy as np
import torch
from common import *
from passive_data import load_passive,windows
from edmd import EDMD
from transition_training import fit_transition,NeuralPredictor


class LinearPrediction:
    def fit(self,X,U):
        x=X[:,:-1].reshape(-1,4);y=X[:,1:].reshape(-1,4);self.mean=x.mean(0);self.scale=x.std(0).clip(1e-8)
        z=(x-self.mean)/self.scale;v=np.column_stack((z,np.ones(len(z))));target=(y-self.mean)/self.scale
        self.coefficients=np.linalg.solve(v.T@v/len(v)+1e-6*np.eye(5),v.T@target/len(v));return self
    def predict_inputs(self,x,U):
        assert np.all(U==0),'Autonomous reference cannot model forced control'
        x=np.asarray(x);z=(x-self.mean)/self.scale;out=[x]
        for k in range(U.shape[-2]):
            z=np.concatenate((z,np.ones((*z.shape[:-1],1))),axis=-1)@self.coefficients;out.append(z*self.scale+self.mean)
        return np.stack(out,axis=-2)


def metrics(model,data,scale):
    X,U=data;prediction=model.predict_inputs(X[:,0],U);error=prediction-X
    def number(value):return float(value) if np.isfinite(value) else None
    return dict(reconstruction_standardized_mse=number(np.mean((error[:,0]/scale)**2)),standardized_mse_by_horizon={str(k):number(np.mean((error[:,k]/scale)**2)) for k in [1,10,30]},rmse_by_state_and_horizon={str(k):[number(v) for v in np.sqrt(np.mean(error[:,k]**2,axis=0))] for k in [1,10,30]},rollout_mse=number(np.mean((error[:,1:]/scale)**2)),scope='passive zero-input, common held-out windows')


def main():
    isolation('deepkoopman');torch.set_num_threads(1);torch.set_num_interop_threads(1);budget=Budget('deepkoopman_passive',8);budget.checkpoint()
    episodes,identity=load_passive(budget);data={k:windows(v) for k,v in episodes.items()};scale=data['train'][0][:,:-1].reshape(-1,4).std(0).clip(1e-8)
    out=OUT/'passive_baselines';dump(out/'manifest.json',dict(data_identity=identity,scope='same passive data and whole-episode splits as stages 5/6',fitting_allocation='Deep Koopman passive CPU allocation',neural_recipe='submitted architecture/AdamW-cosine 100 epochs; no forced weights reused',edmd_grid=[dict(trig=t,ridge=r) for t in [False,True] for r in [1e-6,1e-4,1e-2]]))
    for seed in range(3):
        dest=out/f'seed{seed}';dest.mkdir(parents=True,exist_ok=True)
        neural=NeuralPredictor(fit_transition(data['train'],data['validation'],seed,dest/'neural',budget))
        dump(dest/'neural_metrics.json',metrics(neural,data['test'],scale))
        configs=[dict(trig=t,ridge=r) for t in [False,True] for r in [1e-6,1e-4,1e-2]];scores=[]
        for index,config in enumerate(configs):
            budget.checkpoint();model=EDMD(**config).fit(*data['train']);value=forecast_score(model,data['validation'],scale);scores.append(dict(index=index,config=config,validation_rollout_mse=value));(dest/f'edmd_candidate{index}.pkl').write_bytes(pickle.dumps(model))
        selected=min([r for r in scores if r['validation_rollout_mse'] is not None],key=lambda r:(r['validation_rollout_mse'],r['index']));dump(dest/'edmd_selection.json',selected);dump(dest/'edmd_scores.json',scores)
        model=pickle.loads((dest/f"edmd_candidate{selected['index']}.pkl").read_bytes());dump(dest/'edmd_metrics.json',metrics(model,data['test'],scale))
        linear=LinearPrediction().fit(*data['train']);(dest/'linear.pkl').write_bytes(pickle.dumps(linear));dump(dest/'linear_metrics.json',metrics(linear,data['test'],scale))
    dump(out/'completion.json',dict(state='complete',accounting=budget.checkpoint()))


if __name__=='__main__':main()
