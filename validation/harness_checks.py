"""Cheap endpoint/IUT/indexing contract checks; no campaign or power simulation."""
from common import *
from gate_a import strict_physical
from power_rules import crossed_primary_bounds,holm
from src.inverted_pendulum import wrap_physical_F_caller_as_u_caller,wrap_u_caller_as_physical_F_caller

if __name__=='__main__':
    configure()
    with Meter('harness contract checks',reservation=30):
        X=np.zeros((1001,4)); assert strict_physical(X)
        X[0,0]=2.4; assert strict_physical(X)
        X[0,0]=2.400001; assert not strict_physical(X)
        X.fill(0); X[0,2]=np.pi/2+1e-6; assert not strict_physical(X)
        X.fill(0); X[-1,3]=.050001/np.sqrt(9.81); assert not strict_physical(X)
        X.fill(0); X[-1,1]=.050001/np.sqrt(9.81); assert not strict_physical(X)
        X.fill(0); X[-1,0]=.100001; assert not strict_physical(X)
        X.fill(0); X[-1,2]=.020001; assert not strict_physical(X)
        X.fill(0); X[0,0]=np.nan; assert not strict_physical(X)
        a=np.zeros((4,8,3)); a[:,:,0]=1
        assert crossed_primary_bounds(a,100)['primary_reject']
        a[:,:,1]=1
        assert not crossed_primary_bounds(a,100)['primary_reject']
        assert np.array_equal(holm([.001,.01,.2]),[True,True,False])
        before=normals(5,8)
        normals(7,18); disturbances(1,10)
        assert torch.equal(before,normals(5,8))
        assert not torch.equal(before,normals(5,9))
        assert np.array_equal(disturbances(5,8),disturbances(5,8))
        F=lambda t,y:1+.3*t+.2*sum(y)
        norm=wrap_physical_F_caller_as_u_caller(F,.7,8.3,.4)
        roundtrip=wrap_u_caller_as_physical_F_caller(norm,.7,8.3,.4)
        state=np.array([.2,-.3,.4,-.5])
        assert abs(F(.6,state)-roundtrip(.6,state))<1e-12
        torch.save({'case':5,'sample':8,'draws':before},OUT/'indexed_rng_specimen.pt')
        save('harness_checks.json',{'passed':True,'scope':'strict physical scorer boundaries, nonfinite state, IUT requires both contrasts, Holm, branch-independent indexed RNG, callback unit round-trip',
                                    'power_sizing_or_type_I_calibration_run':False})
        print('Harness contracts passed')
