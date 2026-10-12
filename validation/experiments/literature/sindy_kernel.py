"""Serial compiled evaluation of fitted PySINDy polynomial coefficients, no refit."""
import numpy as np
from numba import njit


@njit(cache=False,parallel=False,error_model='numpy')
def rollout(x,U,powers,coef,trig):
    batch,horizon,_=U.shape
    out=np.empty((batch,horizon+1,4))
    for b in range(batch):
        for j in range(4):out[b,0,j]=x[b,j]
        for k in range(horizon):
            value=np.empty(5)
            for j in range(4):value[j]=out[b,k,j]
            value[4]=U[b,k,0]
            increment=np.zeros(4)
            for term in range(len(powers)):
                v=1.
                for j in range(5):
                    for _ in range(powers[term,j]):v*=value[j]
                for j in range(4):increment[j]+=v*coef[term,j]
            if trig:
                for j in range(4):increment[j]+=np.sin(value[2])*coef[len(powers),j]+np.cos(value[2])*coef[len(powers)+1,j]
            for j in range(4):out[b,k+1,j]=out[b,k,j]+increment[j]
    return out
