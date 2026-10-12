"""PyTorch port of Lusch's encoder/decoder and radius-dependent complex blocks.

Original: BethanyL/DeepKoopman networkarch.py/training.py, MIT license.
Explicit task extensions: four observed states, declared dimensions, B u in
forced stages only. Passive weights/data are never reused by controlled stages.
"""
import numpy as np
import torch
from torch import nn
from common import DT


def mlp(dims,activation=nn.ReLU):
    layers=[]
    for k,(a,b) in enumerate(zip(dims[:-1],dims[1:])):
        layers.append(nn.Linear(a,b))
        if k<len(dims)-2:layers.append(activation())
    return nn.Sequential(*layers)


class DeepKoopman(nn.Module):
    def __init__(self,latent=4,controlled=False):
        super().__init__()
        assert latent in (4,8,16)
        self.latent=latent;self.controlled=controlled
        self.encoder=mlp([4,80,80,latent]);self.decoder=mlp([latent,80,80,4])
        self.omega=nn.ModuleList([mlp([1,170,2]) for _ in range(latent//2)])
        self.B=nn.Parameter(torch.zeros(latent,1),requires_grad=controlled)
        self.register_buffer('mean',torch.zeros(4));self.register_buffer('scale',torch.ones(4))

    def autonomous(self,z):
        out=[]
        for j,net in enumerate(self.omega):
            pair=z[...,2*j:2*j+2];frequency=net(pair.square().sum(-1,keepdim=True))
            angle=frequency[...,0]*DT;growth=torch.exp(frequency[...,1]*DT)
            c,s=torch.cos(angle),torch.sin(angle)
            out.append(growth.unsqueeze(-1)*torch.stack((c*pair[...,0]-s*pair[...,1],s*pair[...,0]+c*pair[...,1]),dim=-1))
        return torch.cat(out,dim=-1)

    def latent_step(self,z,u):
        return self.autonomous(z)+(u@self.B.T if self.controlled else torch.zeros_like(z))

    def rollout(self,x,u,return_latent=False):
        z=self.encoder((x-self.mean)/self.scale);states=[self.decoder(z)*self.scale+self.mean];latents=[z]
        for k in range(u.shape[-2]):
            z=self.latent_step(z,u[...,k,:]);states.append(self.decoder(z)*self.scale+self.mean);latents.append(z)
        result=torch.stack(states,dim=-2)
        return (result,torch.stack(latents,dim=-2)) if return_latent else result

    def predict_inputs(self,x,u):
        with torch.no_grad():
            device=self.mean.device
            out=self.rollout(torch.as_tensor(x,dtype=torch.float32,device=device),torch.as_tensor(u,dtype=torch.float32,device=device))
            return out.cpu().numpy()

    def loss(self,X,U,recon_lam=.001):
        normalized=(X-self.mean)/self.scale
        encoded=self.encoder(normalized);reconstructed=self.decoder(encoded)
        prediction,latents=self.rollout(X[:,0],U,return_latent=True)
        reconstruction=(reconstructed[:,0]-normalized[:,0]).square().mean()
        forecast=((prediction[:,1:]-X[:,1:])/self.scale).square().mean()
        consistency=(latents[:,1:]-encoded[:,1:]).square().mean()
        linf=(reconstructed[:,0]-normalized[:,0]).abs().max()+((prediction[:,1]-X[:,1])/self.scale).abs().max()
        l2=sum(p.square().sum()/2 for name,p in self.named_parameters() if 'weight' in name)
        # Source coefficients and absolute losses; inputs/states are train-standardized.
        return recon_lam*(reconstruction+forecast)+consistency+1e-9*linf+1e-14*l2

    def fit_fixed_encoder_input(self,X,U):
        """Separate bounded diagnostic; invoked after independent zero-input training.

        Controlled main training uses forced-only data and its own weights.
        """
        with torch.no_grad():
            z=self.encoder((X-self.mean)/self.scale)
            residual=z[:,1:]-self.autonomous(z[:,:-1])
            u=U.reshape(-1,1);r=residual.reshape(-1,self.latent)
            self.B.copy_(torch.linalg.lstsq(u,r).solution.T)
