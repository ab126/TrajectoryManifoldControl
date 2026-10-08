"""Performance ensemble and validated vectorization of the submitted RK4 plant."""
from common import *
from torch import nn
from src.manifold_control import DataDrivenCEMSolver

class EnsembleCosts(DataDrivenCEMSolver):
    """Mean of five member trajectory costs, not cost of mean state rollout."""
    def __init__(self,*args,members,**kwargs):
        super().__init__(*args,**kwargs)
        self.members=members
    def _cost(self,x,u):
        costs=[super()._cost(x,u)]
        for decoder in self.members[1:]:
            member_x=decoder.rollout(x[:,0],u)
            costs.append(super()._cost(member_x,u))
        return torch.stack(costs).mean(0)

class OracleTransition(nn.Module):
    """Same uniform-rod equations/RK4 as src/inverted_pendulum.py, batched.

    Must pass comparison with the submitted NumPy rk4_step before any oracle use.
    Evaluation always steps the original NumPy plant. No baseline plant import.
    """
    x_dim=4; u_dim=1
    def __init__(self,device='cpu'):
        super().__init__()
        self.register_buffer('xu_std',torch.ones(5,dtype=torch.float64,device=device))
        self.dt=.02/np.sqrt(1/9.81)
    def rhs(self,x,u):
        p,v,th,w=x.unbind(-1)
        b=.5*th.cos(); r=u[...,0]+.5*th.sin()*w*w; s=.5*th.sin()
        det=1.-b*b
        a=(r/3-b*s)/det; angular=(3*s-b*r)/det
        return torch.stack((v,a,w,angular),-1)
    def forward(self,x,u):
        h=self.dt; k1=self.rhs(x,u); k2=self.rhs(x+h*k1/2,u)
        k3=self.rhs(x+h*k2/2,u); k4=self.rhs(x+h*k3,u)
        return x+h*(k1+2*k2+2*k3+k4)/6

def check_oracle(device='cpu'):
    from src.inverted_pendulum import rk4_step
    f=OracleTransition(device); rng=np.random.default_rng(SEED); errors=[]
    for _ in range(100):
        x=rng.uniform(-1,1,4)*[1.5,2.5,1.3,3.5]; u=rng.uniform(-10,10)
        a=f(torch.tensor(x,device=device),torch.tensor([u],dtype=torch.float64,device=device)).cpu().numpy()
        b=rk4_step(x,u,f.dt,2.)
        np.testing.assert_allclose(a,b,atol=1e-12,rtol=1e-12); errors.append(float(max(abs(a-b))))
    return max(errors)
