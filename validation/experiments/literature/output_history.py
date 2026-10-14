"""Causal position/angle history, data-fitted linear observer and shared readouts."""
import numpy as np


def histories(Y):
    # Five samples, newest first. Initial unavailable history repeats first observation.
    return np.concatenate([Y[:,np.maximum(np.arange(Y.shape[1])-lag,0)] for lag in range(5)],axis=-1)


def fit_observer(X,U):
    Y=X[:,:,[0,2]];H=histories(Y);h=H[:,:-1].reshape(-1,10);following=H[:,1:].reshape(-1,10)
    mean=h.mean(0);scale=h.std(0).clip(1e-6);v=np.column_stack(((h-mean)/scale,U.reshape(-1,1),np.ones(len(h))));target=(following-mean)/scale
    coef=np.linalg.solve(v.T@v/len(v)+1e-5*np.eye(12),v.T@target/len(v));residual=target-v@coef
    return dict(A=coef[:10].T,B=coef[10:11].T,b=coef[11],mean=mean,scale=scale,W=np.cov(residual,rowvar=False)+1e-5*np.eye(10),R=.01*np.eye(2))


class HistoryObserver:
    def __init__(self,model):self.model=model;self.reset()
    def reset(self):self.mu=np.zeros(10);self.P=np.eye(10);self.started=False
    def observe(self,y,u=None,available=True):
        m=self.model;C=np.eye(10)[:2]
        if self.started:
            if u is None:raise ValueError('Previously applied input required')
            self.mu=m['A']@self.mu+m['B'][:,0]*float(np.asarray(u).reshape(-1)[0])+m['b'];self.P=m['A']@self.P@m['A'].T+m['W']
        if available:
            K=np.linalg.solve(C@self.P@C.T+m['R'],(self.P@C.T).T).T
            self.mu+=K@((np.asarray(y)-m['mean'][:2])/m['scale'][:2]-C@self.mu);self.P=(np.eye(10)-K@C)@self.P
        self.started=True;return self.mu*m['scale']+m['mean']


def readout(features,targets):
    v=np.column_stack((features.reshape(-1,features.shape[-1]),np.ones(np.prod(features.shape[:-1],dtype=int))))
    t=targets.reshape(-1,targets.shape[-1]);return np.linalg.solve(v.T@v/len(v)+1e-5*np.eye(v.shape[-1]),v.T@t/len(v))


def apply_readout(features,weights):return features@weights[:-1]+weights[-1]
