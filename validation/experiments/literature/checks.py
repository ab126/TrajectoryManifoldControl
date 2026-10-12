"""Bounded independent correctness checks; run before scientific evaluation."""
import argparse
import time
import numpy as np
from scipy.optimize import minimize
from common import OUT, Budget, dump, isolation, load_data, objective, plant_interface
from edmd import EDMD, QP


def main():
    method=argparse.ArgumentParser();method.add_argument('method',choices=['edmd','sindy'])
    method=method.parse_args().method
    isolation(method); budget=Budget('preparation',2);budget.checkpoint()
    rng=np.random.default_rng(481); A=np.diag([.8,.9,.7,.85]);A[0,1]=.05
    B=np.array([[.1],[.2],[-.1],[.08]])
    X=np.empty((500,9,4));U=rng.uniform(-1,1,(500,8,1));X[:,0]=rng.normal(size=(500,4))
    for k in range(8):X[:,k+1]=X[:,k]@A.T+U[:,k]@B.T
    model=EDMD(ridge=0).fit(X,U)
    model_error=float(np.max(np.abs(model.predict_inputs(X[:,0],U)-X)))
    assert model_error<1e-9, model_error
    qp=QP(model,horizon=8);x=np.array([.2,.5,-.3,.1]);u,info=qp.solve(x)
    independent=minimize(lambda v:objective(model.predict_inputs(x,v[:,None]),v[:,None]),np.zeros(8),
                         method='BFGS',options=dict(gtol=1e-8))
    qp_error=float(np.max(np.abs(u[:,0]-independent.x)))
    assert qp_error<1e-5 and info['accepted'], (qp_error,info)
    # Cost calculated literally with loops, including terminal and adjacent-only rates.
    pred=model.predict_inputs(x,u)
    from common import Q
    literal=sum(v@Q@v for v in pred[:-1])+sum(.03*v[0]**2 for v in u)+pred[-1]@(12*Q)@pred[-1]+sum(.002*(u[k+1,0]-u[k,0])**2 for k in range(7))
    assert abs(literal-objective(pred,u))<1e-10
    # Force channel, initialization and zero-input equivalence.
    assert np.linalg.norm(model.predict_inputs(x,np.ones((8,1)))-model.predict_inputs(x,np.zeros((8,1))))>.01
    assert np.max(np.abs(model.lift(x)@model.C.T-x))<1e-12
    z=model.lift(x);zero=model.predict_inputs(x,np.zeros((8,1)))
    for k in range(8):
        z=z@model.A.T
        assert np.max(np.abs(z@model.C.T-zero[k+1]))<1e-12
    result=dict(linear_rollout_max_error=model_error,qp_vs_independent_max_error=qp_error,
                objective_literal_error=float(abs(literal-objective(pred,u))),force_dependence=True,zero_input=True)
    if method=='sindy':
        from sindy import SINDYc
        # Independent sparse nonlinear discrete system with measured applied inputs.
        X=np.empty((600,5,4));U=rng.uniform(-1,1,(600,4,1));X[:,0]=rng.uniform(-1,1,(600,4))
        for k in range(4):
            x=X[:,k];inc=np.column_stack((.04*x[:,1],-.03*x[:,0]+.02*x[:,2]**2+.1*U[:,k,0],.02*x[:,3],-.05*x[:,2]+.02*x[:,0]*U[:,k,0]))
            X[:,k+1]=x+inc
        sparse=SINDYc(threshold=.001).fit(X,U)
        err=float(np.max(np.abs(sparse.predict_inputs(X[:,0],U)-X)))
        assert err<1e-6,err
        assert np.count_nonzero(sparse.coef)==7,np.count_nonzero(sparse.coef)
        result.update(sparse_recovery_max_error=err,sparse_terms=int(np.count_nonzero(sparse.coef)))
    step=plant_interface();data,identity=load_data(0);X,U=data['train'];replay=[]
    for e in range(3):
        for k in range(30):replay.append(np.max(np.abs(step(X[e,k],U[e,k,0])-X[e,k+1])))
    assert max(replay)<2e-6,max(replay)
    result.update(saved_plant_one_step_max_error=float(max(replay)),data_sha256=identity['sha256'],passed=True)
    result['accounting']=budget.checkpoint();dump(OUT/method/'checks.json',result)
    print(result,flush=True)


if __name__=='__main__':main()
