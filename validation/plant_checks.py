"""Independent uniform-rod equations, physical units, and integration checks."""
from common import *
from scipy.integrate import solve_ivp
from src import inverted_pendulum as plant
from scripts import run_inverted_pendulum_mc as dev

def physical_rhs(y, force, cart_mass=2., rod_mass=1., gravity=9.81, length=1.):
    p,v,th,w = y
    d = np.array([[cart_mass+rod_mass, rod_mass*length*np.cos(th)/2],
                  [rod_mass*length*np.cos(th)/2, rod_mass*length**2/3]])
    rhs = [force+rod_mass*length*np.sin(th)*w*w/2, rod_mass*gravity*length*np.sin(th)/2]
    a,b = np.linalg.solve(d,rhs)
    return np.array([v,a,w,b])

def energy(y):
    p,v,th,w = np.asarray(y).T
    return 1.5*v*v + .5*np.cos(th)*v*w + w*w/6 + .5*np.cos(th)

def run():
    rng = np.random.default_rng(SEED)
    errors=[]
    for _ in range(100):
        y = rng.uniform(-2,2,4); u=rng.uniform(-10,10)
        t0,scale,mg=plant._physical_state_scale(1.,9.81,1.)
        expected=physical_rhs(y*scale,u*mg)
        actual=plant.dynamics_open_loop(y,u,2.)*scale/t0
        errors.append(float(np.max(np.abs(expected-actual))))
        np.testing.assert_allclose(actual,expected,atol=1e-11,rtol=1e-11)
        np.testing.assert_allclose((y*scale)/scale,y,atol=1e-14)
    eig = [np.linalg.eigvalsh([[3,.5*np.cos(th)],[.5*np.cos(th),1/3]])[0]
           for th in np.linspace(-np.pi,np.pi,1001)]
    assert min(eig)>0
    for th in (0., np.pi, -np.pi):
        np.testing.assert_allclose(plant.dynamics_open_loop([2.,0.,th,0.],0.,2.),0.,atol=1e-15)
    f=plant.dynamics_open_loop([0,0,0,0],1,2.)
    assert f[1]>0 and f[3]<0
    f=plant.dynamics_open_loop([0,0,.001,0],0,2.)
    assert f[1]<0 and f[3]>0
    force=plant.wrap_u_caller_as_physical_F_caller(lambda t,y:10.,1,9.81,1)(0,np.zeros(4))
    assert abs(force-98.1)<1e-12
    dt=dev.make_config('dev').dt
    assert abs(dt*t0-.02)<1e-15
    x=np.array([0.,.2,.7,-.3]); energies=[float(energy(x))]
    for _ in range(1000):
        x=plant.rk4_step(x,0.,dt,2.)
        energies.append(float(energy(x)))
    drift=max(abs(np.array(energies)-energies[0]))
    assert drift<1e-4, f'20 physical seconds zero-force drift {drift}'
    initial=np.array([.3,-.2,.7,.4]); duration=1./t0; u=1.7
    ref=solve_ivp(lambda t,y:plant.dynamics_open_loop(y,u,2.),(0,duration),initial,
                  method='DOP853',rtol=1e-12,atol=1e-13).y[:,-1]
    convergence=[]
    for n in (50,100,200):
        x=initial.copy()
        for _ in range(n): x=plant.rk4_step(x,u,duration/n,2.)
        convergence.append(float(np.max(abs(x-ref))))
    ratios=[convergence[i]/convergence[i+1] for i in range(2)]
    assert all(r>10 for r in ratios), ratios
    assert convergence[-1]<1e-6
    result={'passed':True,'physical_rhs_max_abs_error':max(errors), 'minimum_mass_matrix_eigenvalue':min(eig),
            'pivot_inertia':'m*l^2/3; COM inertia m*l^2/12', 'force_10_N':force,
            'dt_normalized':dt,'dt_physical':dt*t0,'energy_drift_20s':drift,
            'energy_drift_threshold':1e-4,'RK4_errors_50_100_200':convergence,'convergence_ratios':ratios}
    save('plant_checks.json',result)
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    configure()
    save('determinism.json',determinism_manifest())
    with Meter('plant checks',reservation=120): run()
