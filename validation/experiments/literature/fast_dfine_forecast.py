"""Condensed controlled latent means and pointwise batched source DFINE decoding."""
import torch
from torch import nn
from common import HORIZON


class FastDFINEForecast(nn.Module):
    __constants__=['latent','manifold','observations','horizon']
    def __init__(self,model):
        super().__init__();assert model.controlled and model.observations==4
        self.decoder=model.source.decoder;self.latent=model.dynamic_dim;self.manifold=model.source.dim_a;self.observations=4;self.horizon=HORIZON
        A=model.source.ldm.A.detach().cpu();B=model.B.detach().cpu();F=[torch.eye(self.latent,dtype=A.dtype)];G=[torch.zeros(self.latent,self.horizon,dtype=A.dtype)]
        for k in range(self.horizon):
            F.append(A@F[-1]);following=A@G[-1];following[:,k]=B[:,0];G.append(following)
        device=model.mean.device
        self.register_buffer('F',torch.stack(F).reshape(-1,self.latent).to(device))
        self.register_buffer('G',torch.stack(G).reshape(-1,self.horizon).to(device))
        self.register_buffer('C',model.source.ldm.C.detach().clone())
        self.register_buffer('mean',model.mean.detach().clone());self.register_buffer('scale',model.scale.detach().clone())

    def forward(self,mu,U):
        latent=(U[:,:,0]@self.G.T+self.F@mu).reshape(-1,self.horizon+1,self.latent)
        manifold=latent@self.C.T
        return self.decoder(manifold.reshape(-1,self.manifold)).reshape(-1,self.horizon+1,self.observations)*self.scale+self.mean
