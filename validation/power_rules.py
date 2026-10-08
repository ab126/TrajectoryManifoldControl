"""Inference scaffold only; no sizing simulations launched in readiness.

Input shape: bundle x common case x method (H,L,E), binary paired outcomes.
Resample both crossed factors, preserving method pairing. Use both one-sided
95% COMPONENT lower bounds for the intersection-union primary. These are NOT
simultaneous coverage bounds. Retain Holm for the six gated secondary tests.
"""
import numpy as np
COMPONENT_ALPHA=.05

def crossed_primary_bounds(outcomes,replicates=1000,seed=20261006):
    a=np.asarray(outcomes,dtype=float)
    if a.ndim!=3 or a.shape[2]!=3 or not np.isin(a,[0,1]).all():
        raise ValueError('binary bundle x case x (H,L,E) required')
    rng=np.random.default_rng(seed); draws=[]
    for _ in range(replicates):
        s=rng.integers(0,a.shape[0],a.shape[0]); c=rng.integers(0,a.shape[1],a.shape[1])
        p=a[s[:,None],c[None,:],:].mean((0,1))
        draws.append([p[0]-p[1],p[0]-p[2],p[0]])
    lower=np.quantile(draws,COMPONENT_ALPHA,axis=0)
    estimate=a.mean((0,1))
    return {'component_lower_95':lower[:2].tolist(), 'primary_reject':bool(np.all(lower[:2]>.05)),
            'absolute_target':bool(estimate[0]>=.95 and lower[2]>=.90),
            'component_alpha':COMPONENT_ALPHA,'iut':'both component nulls must reject',
            'limitation':'bootstrap rule scaffold; boundary/mixed-null and crossed-effects power calibration still required'}

def holm(pvalues,alpha=.05):
    p=np.asarray(pvalues); order=np.argsort(p); rejected=np.zeros(len(p),dtype=bool)
    for rank,index in enumerate(order):
        if p[index]>alpha/(len(p)-rank): break
        rejected[index]=True
    return rejected
