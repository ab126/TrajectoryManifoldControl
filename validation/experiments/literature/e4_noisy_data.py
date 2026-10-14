"""Resumable E4 process-noise recollection using saved initial states/forces.

This is new measured data, not the original E3 corpus or a noisy-label transform.
The opaque plant is used only to acquire transitions, never inside the controller.
"""
import json
import numpy as np
from common import ROOT,OUT,dump,digest,plant_interface
from stress_protocol import CONDITIONS


def collect(seed,budget):
    source=ROOT/f'validation/results_v1/E4/models/seed{seed}/data.npz'
    saved=np.load(source);initial=saved['X'][:,0].astype(float);U=saved['U'].copy()
    order=np.random.default_rng(seed).permutation(len(initial));n=round(.15*len(initial))
    scale=saved['X'][order[2*n:],:-1].reshape(-1,4).std(0).clip(1e-6)
    condition='process_0.01';root=OUT/'rokdeepc/stress/process_0.01'/f'seed{seed}/data'
    identity=dict(source_sha256=digest(source),source_shape=list(saved['X'].shape),seed=seed,
                  initial_states='exact saved E4 x[0]; all episodes retained',
                  inputs='exact saved E4 force sequences; no new control policy',
                  noise='Gaussian increments after opaque RK4 steps, level .01 times nominal E4 training population standard deviations',
                  noise_scale=scale.tolist(),rng=[20261009,seed,list(CONDITIONS).index(condition),987654],
                  adapter_sha256=digest(__file__),simulator_sha256=digest(ROOT/'src/inverted_pendulum.py'))
    p=root/'identity.json'
    if p.exists():assert json.loads(p.read_text())==identity
    else:dump(p,identity)
    path=root/'data.npz'
    if path.exists():return path,identity
    noise=np.random.default_rng(np.random.SeedSequence(identity['rng'])).standard_normal((len(initial),32,4))*(scale*.01)
    step=plant_interface();chunks=[]
    for start in range(0,len(initial),200):
        budget.checkpoint();end=min(start+200,len(initial));p=root/f'chunk{start:04d}.npz'
        if not p.exists():
            X=np.empty((end-start,33,4),dtype=np.float32)
            for e in range(start,end):
                x=initial[e].copy();X[e-start,0]=x
                for k in range(32):x=step(x,float(U[e,k,0]))+noise[e,k];X[e-start,k+1]=x
            assert np.isfinite(X).all(),'Nonfinite noisy E4 acquisition; retain all episodes and diagnose'
            temp=p.with_suffix('.tmp')
            with temp.open('wb') as f:np.savez_compressed(f,X=X)
            temp.replace(p)
        with np.load(p) as chunk:chunks.append(chunk['X'])
    X=np.concatenate(chunks);temp=path.with_suffix('.tmp')
    with temp.open('wb') as f:np.savez_compressed(f,X=X,U=U)
    temp.replace(path);dump(root/'completion.json',dict(identity=identity,data_sha256=digest(path),acquired_transitions=len(initial)*32))
    budget.checkpoint();return path,identity
