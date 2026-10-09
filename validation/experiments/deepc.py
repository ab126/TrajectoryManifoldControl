"""Regularized mosaic DeePC, with episode boundaries and channel order intact.

The SVD removes null and numerically small directions at the stated cutoff.
Within the retained subspace, Z g = B alpha and the minimum coefficient norm
is ||g||² = ||alpha||² / number_of_columns. This is not a learned dynamics model.
"""
from dataclasses import dataclass
import numpy as np
import scipy.sparse as sparse
import osqp


@dataclass
class BehaviorBasis:
    B: np.ndarray
    singular_values: np.ndarray
    scale: np.ndarray
    columns: int
    layout: dict
    horizon: int
    inputs: int
    tini: int = 0


def windows(X, U, horizon, tini=0, output_indices=None):
    """One column per complete window, never crossing an episode boundary.

    Full state: [x_0,...,x_N; u_0,...,u_(N-1)].
    Output only: [u_past; y_past; y_0,...,y_N; u_future].
    Within each block time precedes channel.
    """
    X=np.asarray(X,dtype=float);U=np.asarray(U,dtype=float)
    if X.ndim!=3 or U.ndim!=3 or X.shape[:2]!=(U.shape[0],U.shape[1]+1):
        raise ValueError('Expected episode × time × channel arrays with one more state than input')
    n,m=X.shape[-1],U.shape[-1];width=horizon+tini
    if horizon<1 or tini<0 or U.shape[1]<width:raise ValueError('Insufficient complete episode length')
    starts=U.shape[1]-width+1
    blocks=[]
    if output_indices is None:
        if tini:raise ValueError('Full-state windows use the observed current state')
        layout={'X':slice(0,(horizon+1)*n),'Uf':slice((horizon+1)*n,(horizon+1)*n+horizon*m)}
        for k in range(starts):
            blocks.append(np.concatenate((X[:,k:k+horizon+1].reshape(len(X),-1),U[:,k:k+horizon].reshape(len(X),-1)),axis=1).T)
    else:
        if tini<1:raise ValueError('Output-only windows require observed history')
        Y=X[:,:,output_indices];p=Y.shape[-1];offset=0;layout={}
        for key,size in [('Up',tini*m),('Yp',tini*p),('Yf',(horizon+1)*p),('Uf',horizon*m)]:
            layout[key]=slice(offset,offset+size);offset+=size
        for k in range(starts):
            blocks.append(np.concatenate((U[:,k:k+tini].reshape(len(X),-1),Y[:,k:k+tini].reshape(len(X),-1),
                Y[:,k+tini:k+width+1].reshape(len(X),-1),U[:,k+tini:k+width].reshape(len(X),-1)),axis=1).T)
    return np.concatenate(blocks,axis=1),layout


def compress(Z,layout,horizon,inputs,tini=0,relative_tolerance=1e-8):
    scale=np.std(Z,axis=1).clip(1e-6)
    left,s,_=np.linalg.svd(Z/scale[:,None],full_matrices=False)
    keep=s>relative_tolerance*s[0]
    return BehaviorBasis(scale[:,None]*left[:,keep]*s[keep]/np.sqrt(Z.shape[1]),s[keep],scale,Z.shape[1],layout,horizon,inputs,tini)


def state_cost(basis,Q,terminal_factor=12.):
    n=Q.shape[0];N=basis.horizon;X=basis.B[basis.layout['X']].reshape(N+1,n,-1)
    return sum(x.T@Q@x for x in X[:-1])+terminal_factor*X[-1].T@Q@X[-1]


def output_cost(basis,Q_positions,Q_velocities,dt,terminal_factor=12.):
    N=basis.horizon;p=len(Q_positions)
    Y=basis.B[basis.layout['Yf']].reshape(N+1,p,-1)
    V=np.diff(Y,axis=0)/dt
    Qp=np.diag(Q_positions);Qv=np.diag(Q_velocities)
    return (sum(y.T@Qp@y for y in Y[:-1])+sum(v.T@Qv@v for v in V)
        +terminal_factor*(Y[-1].T@Qp@Y[-1]+V[-1].T@Qv@V[-1]))


class DeePC:
    """One-step replanning QP; no local feedback or state estimator for IO arm."""
    def __init__(self,basis,Q,R,*,umax=10.,lambda_per_column=.01,lambda_history=1000.,
                 terminal_factor=12.,input_rate_weight=.002,dt=.02*np.sqrt(9.81),
                 output_positions=(.02,100.),output_velocities=(30.,10.)):
        self.basis=basis;self.umax=umax;self.lambda_history=lambda_history
        self.output_only=basis.tini>0;N=basis.horizon;m=basis.inputs;B=basis.B
        self.Uf=B[basis.layout['Uf']].reshape(N,m,-1)
        H=np.einsum('kir,ij,kjs->rs',self.Uf,R,self.Uf)
        if N>1:H+=input_rate_weight*np.einsum('kir,kis->rs',np.diff(self.Uf,axis=0),np.diff(self.Uf,axis=0))
        H+=lambda_per_column*np.eye(B.shape[1])
        if self.output_only:
            H+=output_cost(basis,output_positions,output_velocities,dt,terminal_factor)
            self.Yp=B[basis.layout['Yp']]/basis.scale[basis.layout['Yp'],None]
            H+=lambda_history*self.Yp.T@self.Yp
            self.Yf=B[basis.layout['Yf']].reshape(N+1,len(output_positions),-1)
            self.equalities=np.vstack((B[basis.layout['Up']],self.Yf[0]))
        else:
            H+=state_cost(basis,Q,terminal_factor)
            self.X=B[basis.layout['X']].reshape(N+1,len(Q),-1)
            self.equalities=self.X[0]
        self.H=(H+H.T)/2
        self.A=np.vstack((self.equalities,self.Uf.reshape(N*m,-1)))
        self.count=len(self.equalities)
        bound=np.r_[np.zeros(self.count),np.full(N*m,umax)]
        self.q=np.zeros(B.shape[1]);self.solver=osqp.OSQP()
        self.solver.setup(P=sparse.csc_matrix(np.triu(2*self.H)),q=self.q,A=sparse.csc_matrix(self.A),
            l=-bound,u=bound,verbose=False,eps_abs=1e-6,eps_rel=1e-6,max_iter=10000,
            polishing=True,check_termination=25)
        self.observed_y=[];self.applied_u=[]
        self.diagnostics=dict(terminal_policy_used=[],loss=[],predictions=[],solve_seconds=[],
            qp_status=[],qp_iterations=[],qp_primal_residual=[],qp_dual_residual=[],initial_residual=[],
            solve_steps=[],plan_inputs=[],history_residual=[])
        self.last_plan=None

    def solve(self,initial,*,past_u=None,past_y=None):
        initial=np.asarray(initial,dtype=float).reshape(-1)
        if self.output_only:
            if initial.shape!=self.Yf[0].shape[:1]:raise ValueError('IO controller accepts current output channels only')
            past_u=np.asarray(past_u,dtype=float).reshape(-1);past_y=np.asarray(past_y,dtype=float).reshape(-1)
            if len(past_u)!=self.basis.tini*self.basis.inputs or len(past_y)!=len(self.Yp):raise ValueError('Incorrect history length')
            target=np.r_[past_u,initial]
            normalized=past_y/self.basis.scale[self.basis.layout['Yp']]
            q=-2*self.lambda_history*self.Yp.T@normalized
        else:
            if len(initial)!=len(self.equalities):raise ValueError('Incorrect initial state dimension')
            target=initial;q=np.zeros_like(self.q)
        if not np.isfinite(target).all() or not np.isfinite(q).all():
            return dict(finite=False,status='nonfinite observation',u=np.zeros((self.basis.horizon,self.basis.inputs)))
        l=np.r_[target,np.full(self.basis.horizon*self.basis.inputs,-self.umax)]
        u=np.r_[target,np.full(self.basis.horizon*self.basis.inputs,self.umax)]
        self.solver.update(q=q,l=l,u=u)
        result=self.solver.solve(raise_error=False)
        # Both solved statuses still undergo explicit feasibility checks.
        good=result.info.status_val in (1,2) and result.x is not None and np.isfinite(result.x).all()
        residual=np.inf;history=None
        if good:
            alpha=result.x;plan=np.einsum('kir,r->ki',self.Uf,alpha)
            residual=float(np.max(np.abs(self.equalities@alpha-target)))
            good=residual<=1e-4*max(1.,float(np.max(np.abs(target)))) and np.max(np.abs(plan))<=self.umax+1e-4
            predicted=np.einsum('kir,r->ki',self.Yf if self.output_only else self.X,alpha)
            history=float(np.linalg.norm(self.Yp@alpha-normalized)) if self.output_only else None
        if not good:plan=np.zeros((self.basis.horizon,self.basis.inputs));predicted=np.empty((0,))
        return dict(finite=bool(good),status=result.info.status,u=np.clip(plan,-self.umax,self.umax),prediction=predicted,
            loss=float(result.info.obj_val) if good else np.inf,seconds=float(result.info.run_time),iterations=int(result.info.iter),
            primal_residual=float(result.info.prim_res),dual_residual=float(result.info.dual_res),initial_residual=residual,history_residual=history)

    def __call__(self,k,observation):
        self.diagnostics['terminal_policy_used'].append(False)
        if self.output_only:
            y=np.asarray(observation,dtype=float).reshape(-1)
            if len(y)!=self.Yf.shape[1]:raise ValueError('IO controller receives only measured output channels')
            self.observed_y.append(y.copy())
            if len(self.applied_u)<self.basis.tini:
                value=np.zeros(self.basis.inputs);self.applied_u.append(value);return value
            result=self.solve(y,past_y=self.observed_y[-self.basis.tini-1:-1],past_u=self.applied_u[-self.basis.tini:])
        else:result=self.solve(observation)
        self.last_plan=result
        for key,source in [('loss','loss'),('solve_seconds','seconds'),('qp_status','status'),('qp_iterations','iterations'),
            ('qp_primal_residual','primal_residual'),('qp_dual_residual','dual_residual'),('initial_residual','initial_residual'),('history_residual','history_residual')]:
            self.diagnostics[key].append(result.get(source,np.inf if key!='qp_status' else result['status']))
        self.diagnostics['solve_steps'].append(k)
        self.diagnostics['predictions'].append(result.get('prediction',np.empty(0)))
        self.diagnostics['plan_inputs'].append(result['u'])
        value=result['u'][0]
        if self.output_only:self.applied_u.append(value.copy())
        return value
