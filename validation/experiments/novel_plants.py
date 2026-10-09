"""Minimal plant/dimension/cost adapters around the submitted runner."""
from campaign import *
from distribution_ablation import collector
from dataclasses import dataclass

@dataclass
class Plant:
    name: str
    n: int
    m: int
    low: np.ndarray
    high: np.ndarray
    Q: np.ndarray
    R: np.ndarray
    step: object
    angular_indices: tuple=()
    umax: float=10.


def parameterized(function,constants,extra_globals=None):
    """Replace explicitly hard-coded dimensions/costs; preserve the algorithm."""
    code=function.__code__
    values=[]
    for value in code.co_consts:
        replacement=value
        for before,after in constants:
            if type(value) is type(before) and value==before:replacement=after;break
        values.append(replacement)
    namespace=dict(function.__globals__)
    if extra_globals:namespace.update(extra_globals)
    return types.FunctionType(code.replace(co_consts=tuple(values)),namespace,argdefs=function.__defaults__,closure=function.__closure__)


def collect(plant,config):
    if plant.m!=1:raise ValueError('Selected novel-system experiments use one input')
    fn=parameterized(runner.collect_trajectory_data,
        [(4,plant.n),((-1.5,-2.5,-1.3,-3.5),tuple(float(v) for v in plant.low)),
         ((1.5,2.5,1.3,3.5),tuple(float(v) for v in plant.high))],{'_plant_step':plant.step})
    return fn(config)


def train(plant,config,data,checkpoint):
    fn=parameterized(runner.train_transition,[(4,plant.n)])
    return fn(data['train'],data['validation'],config,checkpoint)


def make_control(plant,decoder,config,gain,arm):
    if arm=='L':return controller(decoder,config,gain,'L')
    constants=[((.02,30.,100.,10.),tuple(float(v) for v in np.diag(plant.Q))),(.03,float(plant.R[0,0]))]
    if arm=='M1':constants.append((5,1))
    fn=parameterized(runner.make_controller,constants)
    return fn(decoder,config,gain if arm=='H' else None)


def rk4(rhs,x,u,dt):
    k1=rhs(x,u);k2=rhs(x+.5*dt*k1,u);k3=rhs(x+.5*dt*k2,u);k4=rhs(x+dt*k3,u)
    return x+dt*(k1+2*k2+2*k3+k4)/6


def acrobot_rhs(x,u):
    q1,q2,w1,w2=np.asarray(x)
    cosine=np.cos(q2);sine=np.sin(q2)
    mass=np.array([[5/3+cosine,1/3+.5*cosine],[1/3+.5*cosine,1/3]])
    coriolis=np.array([-.5*sine*(2*w1*w2+w2*w2),.5*sine*w1*w1])
    gravity=np.array([1.5*np.sin(q1)+.5*np.sin(q1+q2),.5*np.sin(q1+q2)])
    acceleration=np.linalg.solve(mass,np.array([0.,float(np.asarray(u).reshape(-1)[0])])+gravity-coriolis)
    return np.r_[w1,w2,acceleration]


def acrobot_energy(x):
    q1,q2,w1,w2=x
    mass=np.array([[5/3+np.cos(q2),1/3+.5*np.cos(q2)],[1/3+.5*np.cos(q2),1/3]])
    return .5*np.array([w1,w2])@mass@np.array([w1,w2])+1.5*np.cos(q1)+.5*np.cos(q1+q2)


def make_plant(name):
    if name=='linear':
        A=np.diag([1.06,1.02,.96,.90]);B=np.array([.2,.2,.1,.1])[:,None]
        def step(x,u,dt):return A@x+B@np.asarray(u).reshape(1)
        return Plant(name,4,1,-np.ones(4),np.ones(4),np.eye(4),R.copy(),step)
    if name=='acrobot':
        def step(x,u,dt):
            result=np.asarray(x)
            for _ in range(8):result=rk4(acrobot_rhs,result,u,dt/8)
            return result
        # Rule: the authored 100-per-angle, 10-per-angular-rate, R=.03 weights.
        return Plant(name,4,1,np.array([-1.3,-1.3,-1.,-1.]),np.array([1.3,1.3,1.,1.]),
                     np.diag([100.,100.,10.,10.]),R.copy(),step,(0,1),umax=1.)
    raise ValueError(name)


def endpoint(plant,t,X,U,local):
    tail=t>=t[-1]-3.-1e-9
    finite=bool(np.isfinite(X).all() and np.isfinite(U).all())
    if plant.name=='linear':
        good=np.all(np.abs(X)<=.02,axis=1);no_fall=finite
    else:
        absolute=np.stack((X[:,0],X[:,0]+X[:,1]),axis=1)
        speeds=np.stack((X[:,2],X[:,2]+X[:,3]),axis=1)*np.sqrt(9.81)
        good=np.all(np.abs(absolute)<=.02,axis=1)&np.all(np.abs(speeds)<=.05,axis=1)
        no_fall=finite and bool(np.all(np.abs(absolute)<np.pi/2))
    primary=bool(no_fall and np.all(good[tail]))
    def number(value):return float(value) if np.isfinite(value) else None
    return dict(primary_success=primary,balanced=bool(np.all(good[tail])),no_fall=no_fall,
        fell=not no_fall,nonfinite_state=not finite,settle_seconds=settle_time(t,good),
        max_state_norm=number(np.max(np.linalg.norm(X,axis=1))),final_state_norm=number(np.linalg.norm(X[-1])),
        max_abs_u=number(np.max(np.abs(U))),input_limit_seconds=float(np.sum(np.abs(U[:,0])>=plant.umax-1e-6)*.02),
        local_fraction=float(np.mean(local)),switch_toggles=int(np.sum(np.asarray(local)[1:]!=np.asarray(local)[:-1])))


def simulate(plant,ctl,state,config,seconds=None):
    # Parameterize the dimension and opaque plant; remove only the cart-specific
    # progress label so a two-state plant is also valid.
    import ast,inspect,textwrap
    tree=ast.parse(textwrap.dedent(inspect.getsource(runner.simulate_plant)))
    class Parameters(ast.NodeTransformer):
        def visit_Constant(self,node):
            if type(node.value) is int and node.value==4:return ast.copy_location(ast.Constant(plant.n),node)
            return node
        def visit_Call(self,node):
            if isinstance(node.func,ast.Name) and node.func.id=='tqdm':node.keywords=[]
            return self.generic_visit(node)
    Parameters().visit(tree);ast.fix_missing_locations(tree)
    namespace=dict(runner.simulate_plant.__globals__);namespace['_plant_step']=plant.step
    exec(compile(tree,'<plant-parameterized submitted simulator>','exec'),namespace)
    return namespace['simulate_plant'](ctl,state,config,seconds=seconds)
