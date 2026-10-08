"""Run original controller against separate shooting on scripted shared observations."""
from common import *
import argparse
from collections import Counter
from unittest.mock import patch
from independent_equivalence import Shooting
from scripts import run_inverted_pendulum_mc as dev
from src.manifold_control import DataDrivenTransition, CanonicalBehaviorDecoder, DataDrivenCEMSolver

class RecordedSolver(DataDrivenCEMSolver):
    def _cost(self,x,u):
        value=super()._cost(x,u)
        self.costs.append(value.clone())
        return value
    def solve(self,*args,**kwargs):
        self.costs=[]
        self.initial=args[1].clone()
        self.last=super().solve(*args,**kwargs)
        return self.last

def closure(controller):
    return dict(zip(controller.__code__.co_freevars,(c.cell_contents for c in controller.__closure__)))

def state(case,k):
    # Deliberately independent observations: sustained queue; switch boundary;
    # switch interrupts queue; long local visit followed by stale warm start;
    # saturated local gain; ± nominal box corners; nonfinite-cost recovery.
    rng=np.random.default_rng(indexed_seed('equivalence_states',case,k))
    x=rng.uniform(-1,1,4)*[1.5,2.5,1.3,3.5]
    if case%5==0 and k%20<10: x=np.array([0.,0.,.01,0.])
    elif case%5==1 and k%7<2: x=np.array([0.,0.,np.sqrt(.75/100)*(1+(-1 if k%7==0 else 1)*1e-6),0.])
    elif case%5==2 and k%15 in (2,3,4): x=np.array([0.,0.,.03,0.])
    elif case%5==3: x=np.array([1.5,2.5,1.3,3.5])*(-1 if k%10<5 else 1)
    elif case%5==4 and k%23==0: x=np.array([0.,0.,np.nan,0.])
    return x

def run(cases=50,samples=100,device='cpu'):
    model=DataDrivenTransition(4,1,(192,192,192)).to(device).eval()
    # Synthetic weights only for numerical equivalence, never success evidence.
    model.dx_std.fill_(.01)
    torch.save(model.state_dict(),OUT/('equivalence_weights.pt' if device=='cpu' else 'equivalence_weights_cuda.pt'))
    decoder=CanonicalBehaviorDecoder(model,30)
    cfg=dev.make_config('dev',device=device)
    gain=np.array([[.1,.4,-200.,-.3]])
    flags=Counter(); max_errors={key:0. for key in ('action','plan','queue','states','cost','loss','warm_start')}
    solves=0
    def compare(a,b,key):
        a=np.asarray(a); b=np.asarray(b)
        np.testing.assert_allclose(a,b,atol=1e-5,rtol=1e-5,equal_nan=True)
        finite=np.isfinite(a)&np.isfinite(b)
        if finite.any(): max_errors[key]=max(max_errors[key],float(np.max(abs(a[finite]-b[finite]))))
    for c in range(cases):
        with patch.object(dev,'DataDrivenCEMSolver',RecordedSolver): original=dev.make_controller(decoder,cfg,gain)
        independent=Shooting(model,gain=gain,device=device)
        for k in range(samples):
            x=state(c,k)
            before=closure(original)
            do_plan=not (float(x@before['Q'].cpu().numpy()@x)<.75) and len(before['pending'])==0
            draws=normals(c,k,device=device) if do_plan else None
            eps=iter(draws) if do_plan else None
            def replay(shape,**kwargs):
                value=next(eps)
                assert tuple(shape)==tuple(value.shape)
                return value
            with patch('torch.randn',side_effect=replay): a=original(k,x)
            b=independent.control(x,draws)
            after=closure(original)
            solver=after['solver']
            branch=('local' if original.diagnostics['terminal_policy_used'][-1] else
                    ('nonfinite' if do_plan and not solver.last.finite else ('plan' if do_plan else 'queue')))
            assert branch==independent.branch,(c,k,branch,independent.branch)
            assert len(after['pending'])==len(independent.queue)
            compare(a,b,'action'); compare(after['previous'].cpu(),independent.previous.cpu(),'plan')
            compare(after['pending'].cpu(),independent.queue.cpu(),'queue'); flags[branch]+=1
            if do_plan:
                solves+=1
                expected=torch.cat((before['previous'][5:],before['previous'][-5:]))
                compare(solver.initial.cpu(),expected.cpu(),'warm_start')
                compare(solver.last.x.cpu(),independent.last['x'].cpu(),'states')
                compare(solver.last.u.cpu(),independent.last['u'].cpu(),'plan')
                compare(solver.last.loss,independent.last['loss'],'loss')
                for costs,other in zip(solver.costs,independent.last['costs']): compare(costs.cpu(),other.cpu(),'cost')
    assert all(flags[k]>0 for k in ('local','queue','plan','nonfinite'))
    # Branching never advances another method's random stream.
    assert torch.equal(normals(1,99),normals(1,99))
    assert not torch.equal(normals(1,99),normals(1,100))
    result={'passed':True,'cases':cases,'samples_per_case':samples,'version':'full' if cases==50 and samples==100 else 'reduced',
            'solves_per_implementation':solves,'branch_counts':dict(flags),'max_abs_errors':max_errors,
            'atol':1e-5,'rtol':1e-5,'identical_flags_and_queue_lengths':True,
            'device':device,'dtype':'float32','weights':'synthetic shared dev architecture; not trained success evidence',
            'independence':'separate implementation; same author, no fresh-context subagent',
            'nonfinite':'NaN observation induces real nonfinite rollout/cost; comparison includes NaN equality and branch/fallback'}
    save('equivalence.json' if device=='cpu' else 'equivalence_cuda.json',result); print(json.dumps(result,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--cases',type=int,default=50); p.add_argument('--samples',type=int,default=100)
    p.add_argument('--device',default='cpu'); args=p.parse_args(); configure()
    with Meter('ordinary shooting equivalence',reservation=900): run(args.cases,args.samples,args.device)
