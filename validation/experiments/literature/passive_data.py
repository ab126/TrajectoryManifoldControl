"""Predeclared zero-force collection with fixed whole-episode splits, no exclusions."""
import json
import numpy as np
from common import *


def load_passive(budget):
    dest=OUT/'passive_data';path=dest/'data.npz'
    protocol=dict(seed=20261012,episodes=1000,state_samples_per_episode=100,control_intervals_per_episode=99,
                  split_seed=20261012,split_counts=dict(train=700,validation=150,test=150),
                  initial_ranges=[dict(label='near',low=[-.25,-.5,-.3,-.6],high=[.25,.5,.3,.6],episodes=500),
                                  dict(label='broad',low=[-1.5,-2.5,-1.3,-3.5],high=[1.5,2.5,1.3,3.5],episodes=500)],
                  force=0.,physical_dt=.02,normalized_dt=DT,plant_source_sha256=digest(ROOT/'src/inverted_pendulum.py'),
                  exclusions='none, including fallen trajectories',allocation='Deep Koopman passive stage',
                  training_windows='starts 0,30,60; 31 state samples each; extraction only after whole-episode split')
    manifest=dest/'protocol.json'
    if manifest.exists():assert json.loads(manifest.read_text())==protocol
    else:dump(manifest,protocol)
    if not path.exists():
        rng=np.random.default_rng(20261012);step=plant_interface();X=np.empty((1000,100,4));U=np.zeros((1000,99,1));regions=np.repeat([0,1],500)
        for e in range(1000):
            if e%25==0:budget.checkpoint()
            region=protocol['initial_ranges'][regions[e]];X[e,0]=rng.uniform(region['low'],region['high'])
            for k in range(99):X[e,k+1]=step(X[e,k],0.)
        if not np.isfinite(X).all():raise RuntimeError('Passive collection nonfinite; keep diagnostic rather than exclude trajectories')
        temp=path.with_suffix('.tmp.npz');np.savez_compressed(temp,X=X,U=U,regions=regions);temp.replace(path)
    arrays=np.load(path);X,U=arrays['X'],arrays['U'];order=np.random.default_rng(20261012).permutation(1000)
    indices=dict(test=order[:150],validation=order[150:300],train=order[300:])
    identity=dict(data_sha256=digest(path),protocol=protocol,split_indices={k:v.tolist() for k,v in indices.items()},total_acquired_control_intervals=99000)
    dump(dest/'identity.json',identity)
    return {k:(X[v],U[v]) for k,v in indices.items()},identity


def windows(data):
    X,U=data
    return (np.concatenate([X[:,k:k+31] for k in [0,30,60]],axis=0),
            np.concatenate([U[:,k:k+30] for k in [0,30,60]],axis=0))
