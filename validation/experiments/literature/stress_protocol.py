"""Exact existing E3 disturbance indexing and opaque physical mismatch evaluation."""
import ast,json
import numpy as np
from common import ROOT,DT,digest

CONDITIONS={**{f'measurement_{v:g}':dict(mode='measurement',level=v) for v in (.01,.03,.1)},**{f'process_{v:g}':dict(mode='process',level=v) for v in (.01,.03,.1)},'mass_minus25':dict(mode='mismatch',mass=1.5,length=1.),'mass_plus25':dict(mode='mismatch',mass=2.5,length=1.),'length_minus25':dict(mode='mismatch',mass=2.,length=.75),'length_plus25':dict(mode='mismatch',mass=2.,length=1.25),'input_delay':dict(mode='delay',samples=1)}
REQUESTED=['measurement_0.01','process_0.01','mass_minus25','mass_plus25','length_minus25','length_plus25','input_delay']


def disturbances(seed,condition,case,scale):
    index=case['index']+(0 if case['set']=='A' else 1000);code=list(CONDITIONS).index(condition)
    return np.random.default_rng(np.random.SeedSequence([20261009,seed,code,index])).standard_normal((700,4))*(scale*CONDITIONS[condition].get('level',0.))


def mismatch_step():
    source=ROOT/'src/inverted_pendulum.py';tree=ast.parse(source.read_text());functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['dynamics_open_loop','rk4_step']];namespace=dict(np=np);exec(compile(ast.Module(body=functions,type_ignores=[]),str(source),'exec'),namespace)
    def step(x,u,condition):
        c=CONDITIONS[condition];length=c.get('length',1.);scale=np.array([1/length,1/np.sqrt(length),1.,np.sqrt(length)])
        return namespace['rk4_step'](np.asarray(x)*scale,float(u),DT/np.sqrt(length),c.get('mass',2.))/scale
    return step
