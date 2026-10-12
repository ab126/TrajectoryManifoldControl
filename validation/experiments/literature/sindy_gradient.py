"""Exact serial sensitivities of the learned SINDYc map; no plant derivatives."""
import time
import numpy as np
from numba import njit
from scipy.optimize import minimize
from common import HORIZON


@njit(cache=False,parallel=False,error_model='numpy')
def cost_gradient(x,u,powers,coef,trig):
    n=len(u);state=x.copy();sensitivity=np.zeros((4,n));gradient=np.zeros(n);cost=0.
    q=np.array([.02,30.,100.,10.])
    for k in range(n):
        for j in range(4):
            cost+=q[j]*state[j]*state[j]
            for h in range(n):gradient[h]+=2*q[j]*state[j]*sensitivity[j,h]
        cost+=.03*u[k]*u[k];gradient[k]+=.06*u[k]
        if k>0:
            difference=u[k]-u[k-1];cost+=.002*difference*difference;gradient[k]+=.004*difference;gradient[k-1]-=.004*difference
        xu=np.empty(5)
        for j in range(4):xu[j]=state[j]
        xu[4]=u[k];increment=np.zeros(4);jacobian=np.zeros((4,5))
        for term in range(len(powers)):
            feature=1.
            for d in range(5):
                for _ in range(powers[term,d]):feature*=xu[d]
            for j in range(4):increment[j]+=feature*coef[term,j]
            for d in range(5):
                exponent=powers[term,d]
                if exponent==0:continue
                derivative=float(exponent)
                for a in range(5):
                    p=powers[term,a]-(1 if a==d else 0)
                    for _ in range(p):derivative*=xu[a]
                for j in range(4):jacobian[j,d]+=derivative*coef[term,j]
        if trig:
            sine,cosine=np.sin(state[2]),np.cos(state[2])
            for j in range(4):
                increment[j]+=sine*coef[len(powers),j]+cosine*coef[len(powers)+1,j]
                jacobian[j,2]+=cosine*coef[len(powers),j]-sine*coef[len(powers)+1,j]
        following=sensitivity+jacobian[:,:4]@sensitivity
        for j in range(4):following[j,k]+=jacobian[j,4];state[j]+=increment[j]
        sensitivity=following
    for j in range(4):
        cost+=12*q[j]*state[j]*state[j]
        for h in range(n):gradient[h]+=24*q[j]*state[j]*sensitivity[j,h]
    if not np.isfinite(cost) or not np.isfinite(gradient).all():return 1e100,np.zeros(n)
    return cost,gradient


class ExactGradientNMPC:
    def __init__(self,model):self.model=model;self.previous=np.zeros((HORIZON,1))

    def solve(self,x):
        start=time.monotonic();model=self.model
        def evaluate(u):
            if time.monotonic()-start>2.:raise TimeoutError('2 s per-solve ceiling')
            return cost_gradient(x,u,model.library.powers_,model.coef,model.trig)
        guess=np.concatenate((self.previous[1:],self.previous[-1:])).ravel()
        try:
            result=minimize(evaluate,guess,jac=True,method='L-BFGS-B',bounds=[(-10,10)]*HORIZON,
                            options=dict(maxiter=50,maxls=20,ftol=1e-8))
            accepted=bool(result.success and np.isfinite(result.fun) and result.fun<1e100 and np.max(np.abs(result.x))<=10+1e-8)
            value=result.x if accepted else np.zeros(HORIZON)
            info=dict(accepted=accepted,status=str(result.message),iterations=int(result.nit),constraint_residual=float(max(0,np.max(np.abs(value))-10)),gradient='analytic learned-polynomial sensitivities')
        except TimeoutError as error:
            value=np.zeros(HORIZON);info=dict(accepted=False,status=str(error),constraint_residual=0.,gradient='analytic learned-polynomial sensitivities')
        self.previous=value[:,None];return self.previous,info
