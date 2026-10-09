"""Optional five-member episode-bootstrap mean-cost NN-MPC (PETS-lite)."""
from campaign import *
from models import EnsembleCosts


def bootstrap_indices(seed,member,episodes):
    return np.random.default_rng(np.random.SeedSequence([20261009,seed,member,13579])).integers(episodes,size=episodes)


def train_members(path,train,validation,config):
    X,U=train
    for member in range(5):
        checkpoint=path/f'ensemble_{member}.pt'
        if checkpoint.exists():continue
        indices=bootstrap_indices(config.seed,member,len(X))
        member_seed=10000+5*config.seed+member
        np.random.seed(member_seed);torch.manual_seed(member_seed)
        member_config=runner.ExperimentConfig(**{**asdict(config),'seed':member_seed})
        runner.train_transition((X[indices],U[indices]),validation,member_config,checkpoint)
    dump(path/'ensemble_diagnostics.json',dict(members=5,bootstrap='Training episodes with replacement; validation/test never bootstrapped',
        member_seeds=[10000+5*config.seed+i for i in range(5)],base_training_episodes=len(X),
        member_training_transitions=len(X)*U.shape[1],prediction_horizon=30,online_replan_interval=1,
        cost='Mean of member trajectory costs, not cost of the mean trajectory; no local switch',
        limitations='Deterministic ensemble: no aleatoric variance or PETS particle propagation'))


def load_members(path,device):
    return [CanonicalBehaviorDecoder(runner.load_transition(path/f'ensemble_{i}.pt',device)[0],30).to(device).eval() for i in range(5)]


def make_ensemble_controller(members,config):
    # Same imported CEM factory, replacing only its solver class and hold literal.
    code=runner.make_controller.__code__
    indices=[i for i,value in enumerate(code.co_consts) if type(value) is int and value==5]
    if len(indices)!=1:raise RuntimeError('Inspect changed submitted replan constant')
    constants=list(code.co_consts);constants[indices[0]]=1
    namespace=dict(runner.make_controller.__globals__)
    namespace['DataDrivenCEMSolver']=lambda *args,**kwargs:EnsembleCosts(*args,members=members,**kwargs)
    factory=types.FunctionType(code.replace(co_consts=tuple(constants)),namespace)
    return factory(members[0],config,None)
