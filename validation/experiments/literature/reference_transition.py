"""Load submitted neural classes by AST, without importing batch runners or plant code."""
import ast
import typing
import torch
from torch import nn
from common import ROOT


def classes():
    source=ROOT/'src/manifold_control.py';tree=ast.parse(source.read_text())
    nodes=[n for n in tree.body if isinstance(n,ast.ClassDef) and n.name in ['DataDrivenTransition','CanonicalBehaviorDecoder']]
    namespace=dict(typing.__dict__);namespace.update(torch=torch,nn=nn,TensorLike=torch.Tensor)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),namespace)
    return namespace['DataDrivenTransition'],namespace['CanonicalBehaviorDecoder']


def saved_decoder(seed,device='cuda'):
    Transition,Decoder=classes()
    payload=torch.load(ROOT/f'validation/results_v1/models/episodes3500_seed{seed}/transition.pt',map_location=device,weights_only=False)
    transition=Transition(4,1,hidden_dims=payload['hidden_dims']).to(device)
    transition.load_state_dict(payload['transition_state_dict']);transition.eval()
    return Decoder(transition,30).to(device).eval()
