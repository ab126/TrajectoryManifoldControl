"""Vectorized evaluation of the unchanged learned Lusch frequency-block weights."""
import torch
from torch import nn
from common import DT,HORIZON


class FastDeepForecast(nn.Module):
    __constants__=['pairs','horizon','dt']
    def __init__(self,model):
        super().__init__();assert model.controlled
        self.encoder=model.encoder;self.decoder=model.decoder
        self.pairs=model.latent//2;self.horizon=HORIZON;self.dt=float(DT)
        self.register_buffer('mean',model.mean.detach().clone());self.register_buffer('scale',model.scale.detach().clone());self.register_buffer('B',model.B.detach().clone())
        self.register_buffer('W1',torch.stack([network[0].weight.T.detach() for network in model.omega]))
        self.register_buffer('b1',torch.stack([network[0].bias.detach() for network in model.omega])[:,None])
        self.register_buffer('W2',torch.stack([network[2].weight.T.detach() for network in model.omega]))
        self.register_buffer('b2',torch.stack([network[2].bias.detach() for network in model.omega])[:,None])

    def advance(self,z,u):
        paired=z.reshape(-1,self.pairs,2).permute(1,0,2)
        radius=paired.square().sum(-1,keepdim=True)
        omega=torch.bmm(torch.relu(torch.bmm(radius,self.W1)+self.b1),self.W2)+self.b2
        angle=omega[...,0]*self.dt;growth=torch.exp(omega[...,1]*self.dt)
        cosine,sine=torch.cos(angle),torch.sin(angle)
        following=growth.unsqueeze(-1)*torch.stack((cosine*paired[...,0]-sine*paired[...,1],sine*paired[...,0]+cosine*paired[...,1]),dim=-1)
        return following.permute(1,0,2).reshape(-1,2*self.pairs)+u@self.B.T

    def forward(self,x,u):
        z=self.encoder((x-self.mean)/self.scale)
        # The known initial measured state is explicit in the shared MPC cost.
        latents=[z]
        for k in range(self.horizon):
            z=self.advance(z,u[:,k]);latents.append(z)
        latent=torch.stack(latents,dim=1)
        states=self.decoder(latent.reshape(-1,2*self.pairs)).reshape(-1,self.horizon+1,4)*self.scale+self.mean
        states[:,0]=x
        return states
