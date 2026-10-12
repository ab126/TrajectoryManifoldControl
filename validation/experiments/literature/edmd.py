"""Controlled finite-dictionary EDMD and bounded QP; task adaptation of Korda–Mezic."""
import numpy as np
from scipy import sparse
import osqp
from sklearn.preprocessing import PolynomialFeatures
from common import Q, HORIZON


class EDMD:
    def __init__(self, trig=False, ridge=1e-6):
        self.trig, self.ridge = trig, ridge
        self.library = PolynomialFeatures(2, include_bias=False)
        self.library.fit(np.zeros((1,4)))

    def features(self, x):
        shape = x.shape[:-1]
        v = self.library.transform(np.asarray(x).reshape(-1,4))
        if self.trig:
            theta = np.asarray(x).reshape(-1,4)[:,2]
            v = np.column_stack((v,np.sin(theta),np.cos(theta)))
        return v.reshape(*shape,-1)

    def lift(self,x):
        v = (self.features(x)-self.mean)/self.std
        return np.concatenate((v,np.ones((*v.shape[:-1],1))),axis=-1)

    def fit(self,X,U):
        x = X[:,:-1].reshape(-1,4)
        y = X[:,1:].reshape(-1,4)
        f = self.features(x)
        self.mean, self.std = f.mean(0), f.std(0).clip(1e-8)
        self.state_scale = x.std(0).clip(1e-8)
        z, zn = self.lift(x), self.lift(y)
        v = np.column_stack((z,U.reshape(-1,1)))
        gram = v.T@v / len(v)
        penalty = self.ridge*np.eye(v.shape[1]); penalty[-2,-2]=0
        weights = np.linalg.solve(gram+penalty,v.T@zn/len(v))
        self.A, self.B = weights[:-1].T, weights[-1:].T
        # Preserve the constant feature exactly; physical state coordinates retained.
        self.A[-1]=0; self.A[-1,-1]=1; self.B[-1]=0
        self.C=np.zeros((4,z.shape[-1])); self.C[:,:4]=np.diag(self.std[:4]); self.C[:,-1]=self.mean[:4]
        self.condition=float(np.linalg.cond(gram+penalty))
        return self

    def predict_inputs(self,x,U):
        z=self.lift(np.asarray(x)); out=[np.asarray(x)]
        for k in range(U.shape[-2]):
            z=z@self.A.T+U[...,k,:]@self.B.T
            out.append(z@self.C.T)
        return np.stack(out,axis=-2)


class QP:
    def __init__(self, model, horizon=HORIZON):
        self.model, self.horizon = model, horizon
        A,B,C=model.A,model.B,model.C
        F,G=[],[]
        power=np.eye(len(A)); effects=np.zeros((len(A),horizon))
        for k in range(horizon+1):
            F.append(C@power); G.append(C@effects)
            power=A@power; effects=A@effects
            if k<horizon: effects[:,k]=B[:,0]
        self.F,self.G=np.concatenate(F),np.concatenate(G)
        W=sparse.block_diag([Q]*horizon+[12*Q]).toarray()
        D=np.diff(np.eye(horizon),axis=0)
        self.P=2*(self.G.T@W@self.G+.03*np.eye(horizon)+.002*D.T@D)
        self.linear=2*self.G.T@W@self.F
        self.solver=osqp.OSQP()
        self.solver.setup(P=sparse.csc_matrix(self.P),q=np.zeros(horizon),A=sparse.eye(horizon,format='csc'),
                          l=np.full(horizon,-10.),u=np.full(horizon,10.),verbose=False,
                          eps_abs=1e-6,eps_rel=1e-6,max_iter=4000,polishing=False)

    def solve(self,x):
        self.solver.update(q=self.linear@self.model.lift(x))
        r=self.solver.solve(raise_error=False)
        accepted=(r.info.status_val==1 and r.x is not None and np.isfinite(r.x).all()
                  and np.max(np.abs(r.x))<=10+1e-5 and r.info.prim_res<=1e-4)
        u=np.clip(r.x,-10,10) if accepted else np.zeros(self.horizon)
        return u[:,None], dict(accepted=bool(accepted),status=r.info.status,
                              primal_residual=float(r.info.prim_res),dual_residual=float(r.info.dual_res))
