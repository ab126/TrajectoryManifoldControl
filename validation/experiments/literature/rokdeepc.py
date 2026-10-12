"""RoKDeePC quadratic formulation (Huang et al., equations 29/34/35), source port.

Author code: Linbin-Huang/RoKDeePC GD_kernel.m, f_cal.m, gradient_cal.m.
Apache-2.0 notice/license preserved in results_literature/sources.
Task changes: four full-state outputs, training standardization, physical costs,
bounded ±10 input, 512 training-only windows, and post-control output timestamps.
This retains joint trajectory-weight/kernel-residual optimization, not kernel-MPC.
"""
import numpy as np
from scipy.linalg import cho_factor,cho_solve
from scipy.optimize import minimize
import time
from common import Q,HORIZON


class RoKDeePC:
    def __init__(self,bandwidth_factor=1.,gamma=.02,linear=False):
        self.bandwidth_factor,self.gamma,self.linear=bandwidth_factor,gamma,linear
        self.lambda_k=1000.;self.lambda_g=.005

    def fit(self,X,U):
        # Each column: applied u[k-1], current measured x[k], planned u[k:k+30].
        # Targets are post-control x[k+1:k+31], preserving source direct-I/O indexing.
        contexts=[];targets=[];retained=[]
        for e in range(len(X)):
            for k in [1,2]:
                contexts.append(np.concatenate((U[e,k-1],X[e,k],U[e,k:k+30,0])))
                targets.append(X[e,k+1:k+31].reshape(-1));retained.append([e,k])
                if len(contexts)==512:break
            if len(contexts)==512:break
        raw=np.array(contexts);self.Y=np.array(targets).T;self.retained=retained
        self.mean,self.scale=raw.mean(0),raw.std(0).clip(1e-6);self.Z=(raw-self.mean)/self.scale
        distances=np.sum((self.Z[:,None]-self.Z[None,:])**2,axis=-1)
        self.alpha=float(np.median(distances[np.triu_indices(len(raw),1)]))*self.bandwidth_factor**2
        self.K=(self.Z@self.Z.T if self.linear else np.exp(-distances/self.alpha))+self.gamma*np.eye(len(raw))
        weights=np.concatenate((np.tile(np.diag(Q),HORIZON-1),12*np.diag(Q)))/1000.
        self.W=weights
        M=(self.Y.T*weights)@self.Y+self.lambda_g*np.eye(len(raw))+self.lambda_k*(self.K.T@self.K)
        self.M=M;self.Mk=cho_solve(cho_factor(M),self.lambda_k*self.K.T)
        self.predictor=self.Y@cho_solve(cho_factor(self.K),np.eye(len(raw)))
        D=np.diff(np.eye(HORIZON),axis=0);self.R=(.03*np.eye(HORIZON)+.002*D.T@D)/1000.
        self.reset();return self

    def reset(self):self.previous_input=0.;self.previous=np.zeros(HORIZON)

    def query(self,x,u,previous_input):
        context=np.concatenate(([previous_input],x,u));z=(context-self.mean)/self.scale
        if self.linear:k=self.Z@z;dk=self.Z[:,5:]/self.scale[5:]
        else:
            difference=z-self.Z;k=np.exp(-np.sum(difference**2,axis=1)/self.alpha)
            dk=(-2/self.alpha)*k[:,None]*difference[:,5:]/self.scale[5:]
        return k,dk

    def predict_inputs(self,x,U,previous_input=0.):
        x=np.asarray(x);U=np.asarray(U)
        if x.ndim>1:
            previous=np.broadcast_to(previous_input,x.shape[:-1]).reshape(-1)
            return np.stack([self.predict_inputs(a,b,float(p)) for a,b,p in zip(x,U,previous)])
        k,_=self.query(x,U[:,0],previous_input)
        future=(self.predictor@k).reshape(HORIZON,4)
        return np.concatenate((x[None],future),axis=0)

    def joint_cost(self,x,u,k,g):
        future=self.Y@g;residual=self.K@g-k
        return float(x@Q@x/1000.+u@self.R@u+np.sum(self.W*future**2)+self.lambda_g*(g@g)+self.lambda_k*(residual@residual))

    def value_gradient(self,x,u,previous_input):
        k,dk=self.query(x,u,previous_input);g=self.Mk@k
        value=self.joint_cost(x,u,k,g)
        gradient=2*self.R@u+2*self.lambda_k*dk.T@(k-self.K@g)
        return value,gradient,g,k

    def source_projected_solve(self,x):
        u=np.concatenate((self.previous[1:],self.previous[-1:]));old_gradient=np.zeros(HORIZON)
        for _ in range(99):
            value,gradient,g,k=self.value_gradient(x,u,self.previous_input)
            if not np.isfinite(value) or not np.isfinite(gradient).all():break
            # Author code uses the same fixed step and prior-gradient momentum.
            u=np.clip(u-.02*(gradient+old_gradient),-10,10);old_gradient=gradient
        value,gradient,g,k=self.value_gradient(x,u,self.previous_input)
        accepted=bool(np.isfinite(value) and np.isfinite(u).all() and np.max(np.abs(u))<=10+1e-10)
        if not accepted:u=np.zeros(HORIZON)
        self.previous=u.copy();self.previous_input=float(u[0])
        def number(value):return float(value) if np.isfinite(value) else None
        return u[:,None],dict(accepted=accepted,status='source counter 1..99, max_iter=100' if accepted else 'nonfinite',
                             objective=value if np.isfinite(value) else None,input_residual=float(max(0.,np.max(np.abs(u))-10)),
                             kernel_residual=number(np.linalg.norm(self.K@g-k)),g_stationarity_residual=number(np.max(np.abs(self.M@g-self.lambda_k*self.K.T@k))),
                             gradient_norm=float(np.linalg.norm(gradient)) if np.isfinite(gradient).all() else None,
                             previous_input_indexing='known actually applied u[k-1]; output is post-control x[k+1]')

    def solve(self,x):
        """Equation 35 with a bounded generic optimizer permitted in paper section IV.

        The public PGD routine remains above for parity. Solver choice is frozen
        before fitting/cases, not changed after inspecting scientific outcomes.
        """
        start=time.monotonic();previous=self.previous_input
        guess=np.concatenate((self.previous[1:],self.previous[-1:]))
        def callback(u):
            if time.monotonic()-start>2.:raise TimeoutError('2 s solve ceiling')
            value,gradient,_,_=self.value_gradient(x,u,previous)
            return (value,gradient) if np.isfinite(value) and np.isfinite(gradient).all() else (1e100,np.zeros(HORIZON))
        try:
            result=minimize(callback,guess,jac=True,method='L-BFGS-B',bounds=[(-10,10)]*HORIZON,options=dict(maxiter=100,maxls=20,ftol=1e-9))
            u=result.x;value,gradient,g,k=self.value_gradient(x,u,previous)
            accepted=bool(np.isfinite(value) and value<1e100 and np.isfinite(u).all() and np.max(np.abs(u))<=10+1e-10)
            if not accepted:u=np.zeros(HORIZON)
            def number(v):return float(v) if np.isfinite(v) else None
            info=dict(accepted=accepted,status=str(result.message),converged=bool(result.success),iterations=int(result.nit),
                      objective=number(value),input_residual=float(max(0.,np.max(np.abs(u))-10)),
                      kernel_residual=number(np.linalg.norm(self.K@g-k)),g_stationarity_residual=number(np.max(np.abs(self.M@g-self.lambda_k*self.K.T@k))),
                      gradient_norm=number(np.linalg.norm(gradient)),acceptance='finite feasible proposal, even at fixed iteration limit; no global optimality claim')
        except TimeoutError as error:
            u=np.zeros(HORIZON);info=dict(accepted=False,status=str(error),converged=False,input_residual=0.)
        self.previous=u.copy();self.previous_input=float(u[0]);return u[:,None],info
