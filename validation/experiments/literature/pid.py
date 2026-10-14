"""Data-fitted cascade: position PD reference, angle PID, conditional integration."""
import numpy as np
from common import DT,Q

REFERENCE_BOUND=.25
GAIN_BOUNDS=[(-2.,2.),(-10.,10.),(-200.,200.),(-200.,200.),(-100.,100.)]


def command(x,integral,gains):
    kp,kd,kt,kw,ki=np.asarray(gains)
    reference=np.clip(kp*x[...,0]+kd*x[...,1],-REFERENCE_BOUND,REFERENCE_BOUND)
    error=reference-x[...,2]
    # omega is measured dtheta/d(normalized time); derivative-on-measurement.
    raw=kt*error-kw*x[...,3]+ki*integral
    force=np.clip(raw,-10.,10.)
    increment=DT*error
    # Freeze integration when its force contribution would increase saturation.
    permit=(np.abs(raw)<=10.)|((raw-force)*ki*increment<=0.)
    return force,np.clip(integral+np.where(permit,increment,0.),-10.,10.),reference,raw


def local_matrix(A,B,gains):
    kp,kd,kt,kw,ki=gains
    feedback=np.array([kt*kp,kt*kd,-kt,-kw])
    matrix=np.zeros((5,5));matrix[:4,:4]=A+B@feedback[None]
    matrix[:4,4]=B[:,0]*ki;matrix[4,:4]=DT*np.array([kp,kd,-1.,0.]);matrix[4,4]=1.
    return matrix


def design_cost(gains,A,B,initial,steps=200):
    x=initial.copy();integral=np.zeros(len(x));cost=0.;previous=None
    for _ in range(steps):
        force,integral,_,_=command(x,integral,gains)
        cost+=np.mean(np.einsum('ni,ij,nj->n',x,Q,x)+.03*force**2)
        if previous is not None:cost+=.002*np.mean((force-previous)**2)
        x=x@A.T+force[:,None]*B[:,0];previous=force
        if not np.isfinite(x).all() or np.max(np.abs(x))>1e6:return 1e20
    cost+=np.mean(np.einsum('ni,ij,nj->n',x,12*Q,x))
    return float(cost)


class PID:
    def __init__(self,gains):self.gains=np.asarray(gains);self.reset()
    def reset(self):self.integral=0.
    def solve(self,x):
        force,self.integral,reference,raw=command(np.asarray(x),self.integral,self.gains)
        return np.full((30,1),force),dict(accepted=bool(np.isfinite(force)),integral=float(self.integral),reference=float(reference),unsaturated_force=float(raw),saturated=bool(abs(raw)>10.))
