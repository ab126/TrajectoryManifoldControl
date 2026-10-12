"""torchDFINE embeddings/dynamics with explicitly separate causal/control adapters.

Upstream modules are unchanged apart from package-relative imports in vendor/dfine.
Our B channel is consumed at prediction t→t+1 with the actually applied u[t].
"""
from types import SimpleNamespace as NS
import numpy as np
import torch
from torch import nn
from vendor.dfine.DFINE import DFINE as SourceDFINE


def config(latent=4,regularization=0.,seed=0,observations=4):
    return NS(seed=seed,model=NS(dim_y=observations,dim_a=latent,dim_x=latent,supervise_behv=False,
              hidden_layer_list=[32,32,32],activation='tanh',nn_kernel_initializer='xavier_normal',
              ldm_kernel_initializer='default',init_A_scale=1.,init_C_scale=1.,init_W_scale=.5,
              init_R_scale=.5,init_cov=1.,is_W_trainable=True,is_R_trainable=True),
              loss=NS(scale_l2=regularization,steps_ahead=[1,2,3,4]))


class DFINEAdapter(nn.Module):
    def __init__(self,latent=4,regularization=0.,seed=0,controlled=False,observations=4):
        super().__init__();self.controlled=controlled;self.regularization=regularization
        self.source=SourceDFINE(config(latent,regularization,seed,observations))
        self.B=nn.Parameter(torch.zeros(latent,1),requires_grad=controlled)
        self.register_buffer('mean',torch.zeros(observations));self.register_buffer('scale',torch.ones(observations))
        self.dynamic_dim=latent;self.observations=observations
        # Prior covariance remains the source's fixed positive-definite identity;
        # W/R diagonal log-covariances are jointly fitted with the other parameters.
        self.source.ldm.Lambda_0.requires_grad_(False)
        self.reset()

    def reset(self):self.current_mean=None;self.current_covariance=None;self.previous_input=None

    def filter_encoded(self,a,U=None,mask=None):
        n,t,_=a.shape;ldm=self.source.ldm
        if U is None:U=a.new_zeros(n,t-1,1)
        shared_covariance=mask is None
        if mask is None:mask=a.new_ones(n,t,1)
        W,R=ldm._get_covariance_matrices();A,C=ldm.A,ldm.C
        mu=ldm.mu_0.expand(n,-1);P=ldm.Lambda_0.expand(1 if shared_covariance else n,-1,-1)
        filtered=[];covariances=[];predictions=[]
        for k in range(t):
            S=C@P@C.T+R
            K=torch.linalg.solve(S,(P@C.T).transpose(-1,-2)).transpose(-1,-2)
            if not shared_covariance:K=K*mask[:,k,None]
            residual=a[:,k]-(mu@C.T)
            mu=mu+(K@residual.unsqueeze(-1)).squeeze(-1)
            P=(torch.eye(self.dynamic_dim,device=a.device)-K@C)@P
            filtered.append(mu);covariances.append(P.expand(n,-1,-1))
            force=U[:,k]@self.B.T if self.controlled and k<t-1 else torch.zeros_like(mu)
            mu=mu@A.T+force;P=A@P@A.T+W;predictions.append(mu)
        return torch.stack(filtered,1),torch.stack(covariances,1),torch.stack(predictions,1)

    def filter_sequence(self,y,U=None,mask=None):
        standardized=(y-self.mean)/self.scale
        if mask is not None:standardized=torch.where(mask.bool(),standardized,torch.zeros_like(standardized))
        a=self.source.encoder(standardized.reshape(-1,self.observations)).reshape(*y.shape[:-1],-1)
        return self.filter_encoded(a,U,mask)

    def decode_dynamic(self,z):return self.source.decoder(z@self.source.ldm.C.T)*self.scale+self.mean

    def loss(self,Y,U,mask=None):
        mu,_,_=self.filter_sequence(Y,U,mask);ldm=self.source.ldm
        if mask is None:mask=Y.new_ones(*Y.shape[:-1],1)
        total=Y.new_zeros(())
        for horizon in [1,2,3,4]:
            z=mu[:,:-horizon]
            for j in range(horizon):
                z=z@ldm.A.T+(U[:,j:Y.shape[1]-horizon+j]@self.B.T if self.controlled else torch.zeros_like(z))
            prediction=self.decode_dynamic(z)
            error=((prediction-Y[:,horizon:])/self.scale).square()*mask[:,horizon:]
            total+=error.sum()/(mask[:,horizon:].sum()*self.observations)
        # Source prediction + encoder/decoder L2 loss, no added reconstruction loss.
        regularization=sum(torch.norm(p) for name,p in self.source.named_parameters() if 'weight' in name)
        return total+self.regularization*regularization

    def predict_inputs(self,x,U):
        device=self.mean.device
        with torch.no_grad():
            x=torch.as_tensor(x,dtype=torch.float32,device=device);u=torch.as_tensor(U,dtype=torch.float32,device=device)
            batch=x.shape[:-1];y=x.reshape(-1,1,self.observations)
            mu,_,_=self.filter_sequence(y);z=mu[:,0].reshape(*batch,-1)
            out=[self.decode_dynamic(z)]
            for k in range(u.shape[-2]):
                z=z@self.source.ldm.A.T+(u[...,k,:]@self.B.T if self.controlled else torch.zeros_like(z))
                out.append(self.decode_dynamic(z))
            return torch.stack(out,dim=-2).cpu().numpy()

    def observe(self,y,applied_input=None,available=True):
        """Stateful causal filtering; consumes only the previously applied force."""
        with torch.no_grad():
            ldm=self.source.ldm;device=self.mean.device
            y=torch.as_tensor(y,dtype=torch.float32,device=device)
            if self.current_mean is None:
                mu=ldm.mu_0.clone();P=ldm.Lambda_0.clone()
            else:
                if self.controlled and applied_input is None:raise ValueError('Previous applied input required after initialization')
                mu=self.current_mean@ldm.A.T
                if self.controlled:mu=mu+torch.as_tensor(applied_input,dtype=torch.float32,device=device).reshape(1)@self.B.T
                W,_=ldm._get_covariance_matrices();P=ldm.A@self.current_covariance@ldm.A.T+W
            if available:
                a=self.source.encoder((y-self.mean)/self.scale);_,R=ldm._get_covariance_matrices();C=ldm.C
                K=torch.linalg.solve(C@P@C.T+R,(P@C.T).T).T
                mu=mu+K@(a-C@mu);P=(torch.eye(self.dynamic_dim,device=device)-K@C)@P
            self.current_mean,self.current_covariance=mu,P
            return self.decode_dynamic(mu).cpu().numpy()

    def forecast_filtered(self,U):
        if self.current_mean is None:raise RuntimeError('Filter must observe/reset before planning')
        with torch.no_grad():
            u=torch.as_tensor(U,dtype=torch.float32,device=self.mean.device);z=self.current_mean.expand(*u.shape[:-2],-1)
            out=[self.decode_dynamic(z)]
            for k in range(u.shape[-2]):
                z=z@self.source.ldm.A.T+u[...,k,:]@self.B.T
                out.append(self.decode_dynamic(z))
            return torch.stack(out,dim=-2).cpu().numpy()
