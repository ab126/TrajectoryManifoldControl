"""Exploratory frozen initial-plan predictions versus identical-input plant replay."""
import json,pickle
import numpy as np
from common import *
from rok_stress import PreservingBudget
from run_rokdeepc import data


def main():
    isolation('rokdeepc');budget=PreservingBudget('rokdeepc',12);budget.checkpoint()
    root=OUT/'rokdeepc/plan_replay';original=json.loads((ROOT/'validation/results_v1/initial_conditions.json').read_text())['sets'];step=plant_interface();rows=[]
    for seed in range(3):
        selected=json.loads((OUT/f'rokdeepc/seed{seed}/selection.json').read_text());i=selected['index'];path=OUT/f'rokdeepc/seed{seed}/candidate{i}.pkl';model=pickle.loads(path.read_bytes());split,_=data(seed);scale=split['train'][0][:,:-1].reshape(-1,4).std(0).clip(1e-8)
        for name,indices in [('A',[0,2,5]),('C',[0,10,25])]:
            for index in indices:
                budget.checkpoint();x=np.asarray(original[name][index]);model.reset();U,info=model.solve(x);value,gradient,g,k=model.value_gradient(x,U[:,0],0.)
                optimized=np.concatenate((x[None],(model.Y@g).reshape(30,4)));forecast=model.predict_inputs(x,U,0.);actual=np.empty((31,4));actual[0]=x
                for j in range(30):actual[j+1]=step(actual[j],float(U[j,0]))
                literal=float(objective(optimized,U))/1000.+model.lambda_g*float(g@g)+model.lambda_k*float(np.sum((model.K@g-k)**2))
                assert np.isclose(value,literal,rtol=1e-10,atol=1e-9),(seed,name,index,'joint objective')
                context=np.concatenate(([0.],x,U[:,0]));z=(context-model.mean)/model.scale
                distance=np.sqrt(np.sum((model.Z-z)**2,axis=1));casepath=OUT/f'rokdeepc/cases/seed{seed}_RoKDeePC_{name}_{index:04d}_14s.json';saved=json.loads(casepath.read_text());raw=np.load(ROOT/saved['raw_path']);command_error=float(np.max(np.abs(U[0]-raw['U'][0])));raw.close()
                key=f'seed{seed}_{name}_{index:04d}';artifact=root/'raw'/f'{key}.npz';artifact.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(artifact,U=U,optimized=optimized,forecast=forecast,plant_replay=actual,t=np.arange(31)*.02)
                rows.append(dict(seed=seed,set=name,index=index,model_sha256=digest(path),solver=info,first_action_difference_from_saved_case=command_error,optimized_forecast_mse=float(np.mean(((optimized[1:]-actual[1:])/scale)**2)),pure_kernel_forecast_mse=float(np.mean(((forecast[1:]-actual[1:])/scale)**2)),optimized_physical_cost=float(objective(optimized,U)),replayed_physical_cost=float(objective(actual,U)),joint_objective=value,joint_literal_error=abs(value-literal),max_kernel_similarity=float(np.max(k)),nearest_normalized_context_distance=float(np.min(distance)),g_stationarity_residual=float(np.max(np.abs(model.M@g-model.lambda_k*model.K.T@k))),raw_path=str(artifact.relative_to(ROOT))))
    dump(root/'summary.json',dict(rows=rows,source_sha256=digest(__file__),semantics='Fresh initial frozen-controller solve at predeclared A0/A2/A5 and C0/C10/C25, previous input zero. Replay holds all 30 proposed controls; this is not the later replanned closed-loop trajectory. First action is compared to saved case, with differences retained.',interpretation='Exploratory diagnostics separate feasible optimization, kernel support and physical prediction error; no model/controller repair or family-wide impossibility claim.',accounting=budget.checkpoint()))
    dump(root/'completion.json',dict(state='complete',cases=len(rows)))


if __name__=='__main__':main()
