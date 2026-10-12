"""Discrete input-aware increments, using public PySINDy libraries and STLSQ."""
import numpy as np
import pysindy as ps
from common import HORIZON, objective
from scipy.optimize import minimize


class SINDYc:
    def __init__(self, degree=2,trig=False,threshold=.001):
        self.degree,self.trig,self.threshold=degree,trig,threshold
        self.library=ps.PolynomialLibrary(degree=degree,include_bias=True)
        self.library.fit(np.zeros((1,5)))

    def features(self,x,u):
        v=np.asarray(self.library.transform(np.column_stack((x,u))))
        if self.trig: v=np.column_stack((v,np.sin(x[:,2]),np.cos(x[:,2])))
        return v

    def fit(self,X,U):
        x=X[:,:-1].reshape(-1,4); y=X[:,1:].reshape(-1,4)-x
        f=self.features(x,U.reshape(-1,1))
        self.state_scale=x.std(0).clip(1e-8)
        self.feature_scale=np.sqrt(np.mean(f*f,axis=0)).clip(1e-8)
        self.target_scale=y.std(0).clip(1e-8)
        # RMS feature scaling keeps the constant feature and permits raw coefficients.
        opt=ps.STLSQ(threshold=self.threshold,alpha=1e-6,max_iter=20,normalize_columns=False,unbias=False)
        opt.fit(f/self.feature_scale,y/self.target_scale)
        self.coef=opt.coef_.T*self.target_scale/self.feature_scale[:,None]
        self.names=list(self.library.get_feature_names(['p','v','theta','omega','u']))
        if self.trig:self.names+=['sin(theta)','cos(theta)']
        return self

    def transition(self,x,u):
        shape=x.shape
        return x+(self.features(x.reshape(-1,4),u.reshape(-1,1))@self.coef).reshape(shape)

    def predict_inputs(self,x,U):
        out=[np.asarray(x)]
        for k in range(U.shape[-2]):out.append(self.transition(out[-1],U[...,k,:]))
        return np.stack(out,axis=-2)


class CEM:
    """NumPy port of submitted CEM semantics: same budget, costs and first action."""
    def __init__(self,model,seed):
        self.model=model; self.rng=np.random.default_rng(seed); self.previous=np.zeros((HORIZON,1))

    def solve(self,x):
        mean=np.concatenate((self.previous[1:],self.previous[-1:]))
        fraction=min(1.,2*np.sqrt(np.mean((x/self.model.state_scale)**2)))
        std=np.full_like(mean,(.05+.45*fraction)*20)
        best_cost=np.inf; best=np.zeros_like(mean)
        with np.errstate(over='ignore',invalid='ignore'):
            for _ in range(3):
                candidates=np.clip(mean+std*self.rng.normal(size=(256,HORIZON,1)),-10,10)
                candidates[0]=np.clip(mean,-10,10)
                prediction=self.model.predict_inputs(np.broadcast_to(x,(256,4)),candidates)
                costs=objective(prediction,candidates); costs=np.where(np.isfinite(costs),costs,np.inf)
                elite=candidates[np.argsort(costs)[:32]]
                mean=.15*mean+.85*elite.mean(0)
                std=.15*std+.85*elite.std(0,ddof=1).clip(.02)
                index=np.argmin(costs)
                if costs[index]<best_cost:best_cost=costs[index];best=candidates[index].copy()
        accepted=bool(np.isfinite(best_cost))
        self.previous=best if accepted else np.zeros_like(best)
        return self.previous,dict(accepted=accepted,status='finite' if accepted else 'nonfinite',objective=float(best_cost) if accepted else None)


class NMPC:
    def __init__(self,model):
        self.model=model;self.previous=np.zeros((HORIZON,1))

    def solve(self,x):
        guess=np.concatenate((self.previous[1:],self.previous[-1:])).ravel()
        def cost(u):
            with np.errstate(over='ignore',invalid='ignore'):
                value=objective(self.model.predict_inputs(x,u[:,None]),u[:,None])
            return float(value) if np.isfinite(value) else 1e100
        start=__import__('time').monotonic()
        def bounded(u):
            if __import__('time').monotonic()-start>2.:raise TimeoutError('2 s per-solve ceiling')
            return cost(u)
        try:
            result=minimize(bounded,guess,method='L-BFGS-B',bounds=[(-10,10)]*HORIZON,
                            options=dict(maxiter=50,maxls=20,ftol=1e-8))
            accepted=bool(result.success and np.isfinite(result.fun) and result.fun<1e100 and np.max(np.abs(result.x))<=10+1e-8)
            value=result.x if accepted else np.zeros(HORIZON)
            info=dict(accepted=accepted,status=str(result.message),iterations=int(result.nit),constraint_residual=float(max(0,np.max(np.abs(value))-10)))
        except TimeoutError as error:
            value=np.zeros(HORIZON);info=dict(accepted=False,status=str(error),constraint_residual=0.)
        self.previous=value[:,None]
        return self.previous,info
