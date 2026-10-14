"""Same prescribed, frozen SMC on the new 200-state holdout; reserve subcap."""
import importlib.util,json,time
import numpy as np
from common import *


def main():
    isolation('smc');budget=Budget('smc_C200',1);budget.checkpoint();states=json.loads((OUT/'inventory/fresh_200_states.json').read_text())['states'];source=ROOT/'validation/experiments/smc_baseline.py';spec=importlib.util.spec_from_file_location('reference_smc',source);smc=importlib.util.module_from_spec(spec);spec.loader.exec_module(smc);root=OUT/'smc_C200';saved=json.loads((root/'manifest.json').read_text());step=plant_interface()
    for seed in range(3):
        model=saved['models'][seed];assert model['seed']==seed;path=ROOT/f'validation/results_v1/E6_SMC/models/nominal_seed{seed}.json';original=json.loads(path.read_text());assert np.array_equal(original['C'],model['C']) and original['calibration']['phi']==model['phi'];law=smc.SlidingFeedback(np.asarray(model['A']),np.asarray(model['C']),model['phi'],correction=True,rho=.9,bound=10.);identity=dict(original_model=model,model_sha256=digest(path),fresh_states_sha256=digest(OUT/'inventory/fresh_200_states.json'),source_sha256=digest(source),adapter_sha256=digest(__file__),simulator_sha256=digest(ROOT/'src/inverted_pendulum.py'))
        for index,state in enumerate(states):
            p=root/'cases'/f'seed{seed}_SMC_F_{index:04d}_14s.json'
            if p.exists():r=json.loads(p.read_text());assert r['identity']==identity and r['initial_state']==state;continue
            budget.checkpoint();X=np.empty((701,4));U=np.empty((700,1));X[0]=state;start=time.monotonic();cpu=cpu_seconds()
            for k in range(700):
                U[k]=law.command(X[k])[0]
                with np.errstate(over='ignore',invalid='ignore'):X[k+1]=step(X[k],U[k,0])
                if not np.isfinite(X[k+1]).all():X=X[:k+2];U=U[:k+1];break
            t=np.arange(len(X))*.02;tail=t>=11.;finite=bool(np.isfinite(X).all());complete=len(U)==700;no_fall=bool(finite and np.all(np.abs(X[:,2])<np.pi/2));balanced=bool(complete and finite and np.all(np.abs(X[tail,2])<=.02) and np.all(np.abs(X[tail,3])*np.sqrt(9.81)<=.05));key=p.stem;raw=root/'raw'/f'{key}.npz';raw.parent.mkdir(exist_ok=True);np.savez_compressed(raw,X=X,U=U,t=t);r=dict(case=key,identity=identity,seed=seed,arm='SMC',set='F',index=index,seconds=14.,initial_state=state,primary_success=balanced and no_fall,balanced=balanced,no_fall=no_fall,complete=complete,raw_path=str(raw.relative_to(ROOT)),cpu_seconds=cpu_seconds()-cpu,wall_seconds=time.monotonic()-start);dump(p,r)
    dump(root/'fresh_completion.json',dict(state='complete',cases=600,accounting=budget.checkpoint()))


if __name__=='__main__':main()
