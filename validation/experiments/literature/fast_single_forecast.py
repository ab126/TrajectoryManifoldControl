"""TorchScript evaluation of the unchanged submitted transition network weights."""
import torch
from torch import nn


class FastSingleForecast(nn.Module):
    __constants__=['horizon']
    def __init__(self,transition):
        super().__init__();self.net=transition.net;self.horizon=30
        for name in ['xu_mean','xu_std','dx_mean','dx_std']:self.register_buffer(name,getattr(transition,name).detach().clone())
    def forward(self,x,u):
        states=[x]
        for k in range(self.horizon):
            xu=torch.cat((x,u[:,k]),-1);delta=self.net((xu-self.xu_mean)/self.xu_std);x=x+delta*self.dx_std+self.dx_mean;states.append(x)
        return torch.stack(states,1)
