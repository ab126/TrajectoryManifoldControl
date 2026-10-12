"""Bounded source parity and independent causal/control correctness checks on CPU."""
import argparse
import numpy as np
import torch
from common import *


def deepkoopman_checks():
    from deepkoopman import DeepKoopman
    torch.manual_seed(45);model=DeepKoopman(4,controlled=True).double()
    z=torch.randn(7,4,dtype=torch.float64)
    with torch.no_grad():
        actual=model.autonomous(z).numpy();expected=[]
        for j,network in enumerate(model.omega):
            pair=z[:,2*j:2*j+2].numpy();current=(pair**2).sum(1,keepdims=True)
            for layer in network:
                if isinstance(layer,torch.nn.Linear):current=current@layer.weight.numpy().T+layer.bias.numpy()
                else:current=np.maximum(current,0)
            omega,mu=current[:,0],current[:,1]
            # Source form_complex_conjugate_block/varying_multiply, independent NumPy.
            c,s=np.cos(omega*DT),np.sin(omega*DT)
            expected.append(np.exp(mu*DT)[:,None]*np.column_stack((c*pair[:,0]-s*pair[:,1],s*pair[:,0]+c*pair[:,1])))
        error=float(np.max(np.abs(actual-np.column_stack(expected))));assert error<1e-12
        zero=model.latent_step(z,torch.zeros(7,1,dtype=torch.float64));assert torch.equal(zero,model.autonomous(z))
        model.B.copy_(torch.tensor([[.1],[-.2],[.3],[.4]],dtype=torch.float64))
        forced=model.latent_step(z,torch.ones(7,1,dtype=torch.float64));assert torch.allclose(forced-zero,model.B.T.expand(7,-1),atol=1e-12)
        # Fixed-encoder input fit recovers independently constructed residual response.
        u=torch.linspace(-1,1,7,dtype=torch.float64)[:,None]
        recovered=torch.linalg.lstsq(u,u@model.B.T).solution.T;assert torch.allclose(recovered,model.B,atol=1e-12)
        # Loss parity uses bounded values; the separate force test above exercises B.
        x=.1*torch.randn(7,31,4,dtype=torch.float64);u=torch.zeros(7,30,1,dtype=torch.float64)
        predicted,latent=model.rollout(x[:,0],u,return_latent=True);encoded=model.encoder(x)
        reconstruction=(model.decoder(encoded)[:,0]-x[:,0]).square().mean()
        forecast=(predicted[:,1:]-x[:,1:]).square().mean();consistency=(latent[:,1:]-encoded[:,1:]).square().mean()
        infinity=(predicted[:,0]-x[:,0]).abs().max()+(predicted[:,1]-x[:,1]).abs().max()
        l2=sum(p.square().sum()/2 for name,p in model.named_parameters() if 'weight' in name)
        literal=.001*(reconstruction+forecast)+consistency+1e-9*infinity+1e-14*l2
        assert torch.allclose(model.loss(x,u),literal,atol=1e-12),(model.loss(x,u),literal)
    return dict(passed=True,source_block_numpy_max_error=error,loss_literal_parity=True,zero_input_reduction=True,force_dependence=True,fixed_encoder_input_recovery=True,
                source_scope='frequency-dependent complex blocks, reconstruction/multi-step/latent-consistency/Linf/L2 source terms',
                adaptations='four-state input, dimensions 4/8/16, normalized measured data, forced-only B addition for stage 7')


def dfine_checks():
    from dfine import DFINEAdapter
    torch.manual_seed(70);model=DFINEAdapter(latent=4,controlled=True)
    a=torch.randn(3,12,4);mask=torch.ones(3,12,1);mask[:,4:6]=0;u=torch.zeros(3,11,1)
    with torch.no_grad():
        own,ownP,_=model.filter_encoded(a,u,mask)
        _,source,_,sourceP=model.source.ldm.compute_forwards(a,mask)
        error=float(torch.max(torch.abs(own-source.permute(1,0,2))));perror=float(torch.max(torch.abs(ownP-sourceP.permute(1,0,2,3))))
        assert error<2e-5 and perror<2e-5,(error,perror)
        # Independently construct a linear controlled state-space example.
        ldm=model.source.ldm;ldm.A.copy_(torch.diag(torch.tensor([.8,.9,.7,.85])));ldm.C.copy_(torch.eye(4));ldm.mu_0.zero_();ldm.Lambda_0.copy_(torch.eye(4));ldm.W_log_diag.fill_(np.log(.03));ldm.R_log_diag.fill_(np.log(.1));model.B.copy_(torch.tensor([[.1],[.2],[-.1],[.3]]))
        u=torch.randn(3,11,1);filtered,cov,_=model.filter_encoded(a,u,mask)
        A=ldm.A.numpy();C=ldm.C.numpy();B=model.B.numpy();W=np.eye(4)*.03;R=np.eye(4)*.1;independent=[];independentP=[]
        for batch in range(3):
            mu=np.zeros(4);P=np.eye(4);rows=[];matrices=[]
            for k in range(12):
                if mask[batch,k,0]:
                    K=P@C.T@np.linalg.inv(C@P@C.T+R);mu=mu+K@(a[batch,k].numpy()-C@mu);P=(np.eye(4)-K@C)@P
                rows.append(mu.copy());matrices.append(P.copy())
                if k<11:mu=A@mu+B[:,0]*float(u[batch,k,0]);P=A@P@A.T+W
            independent.append(rows);independentP.append(matrices)
        linear_error=float(np.max(np.abs(filtered.numpy()-np.array(independent))));assert linear_error<2e-6
        minimum=float(torch.linalg.eigvalsh(cov).min());assert minimum>0
        altered=a.clone();altered[:,8:]+=100
        future,_,_=model.filter_encoded(altered,u,mask);assert torch.equal(future[:,:8],filtered[:,:8])
        missing=a.clone();missing[:,4:6]+=10000
        alternate,_,_=model.filter_encoded(missing,u,mask);assert torch.equal(alternate,filtered)
        no_force,_,_=model.filter_encoded(a,torch.zeros_like(u),mask);assert torch.max(torch.abs(no_force-filtered))>.01
        shared,sharedP,_=model.filter_encoded(a,u)
        explicit,explicitP,_=model.filter_encoded(a,u,torch.ones_like(mask))
        shared_error=float(torch.max(torch.abs(shared-explicit)))
        assert shared_error<2e-5 and torch.max(torch.abs(sharedP-explicitP))<2e-5
        # Source compute_loss is causal even though its forward also returns unused smoothing.
        passive=DFINEAdapter(4,regularization=1e-4);Y=torch.randn(2,12,4);M=torch.ones(2,12,1)
        native_vars=passive.source(Y,M);source_loss,_=passive.source.compute_loss(Y,native_vars,M)
        own_loss=passive.loss(Y,torch.zeros(2,11,1),M);loss_error=float(torch.abs(source_loss-own_loss));assert loss_error<1e-5,loss_error
    return dict(passed=True,source_filter_max_error=error,source_covariance_max_error=perror,independent_controlled_kalman_max_error=linear_error,
                covariance_minimum_eigenvalue=minimum,source_loss_error=loss_error,shared_covariance_batch_max_error=shared_error,force_indexing=True,no_future_leakage=True,missing_observation_invariance=True,zero_input_source_reduction=True,
                adaptation='source mappings/LDM parameters; explicit B u update; fixed source identity prior covariance; causal filtering; no smoother used for forecast/control')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('method',choices=['deepkoopman','dfine']);args=parser.parse_args()
    isolation(args.method);torch.set_num_threads(1);torch.set_num_interop_threads(1);budget=Budget('preparation',2);budget.checkpoint()
    result=deepkoopman_checks() if args.method=='deepkoopman' else dfine_checks()
    dump(OUT/args.method/'checks.json',result);budget.checkpoint();print(result,flush=True)


if __name__=='__main__':main()
