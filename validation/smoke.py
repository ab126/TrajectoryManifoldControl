"""Tiny execution of original dev functions, with all output paths isolated."""
from common import *
import argparse
from unittest.mock import patch
from scripts import run_inverted_pendulum_mc as dev

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--device',default='cpu'); cli=parser.parse_args()
    configure()
    assert json.loads((OUT/'plant_checks.json').read_text())['passed']
    assert json.loads((OUT/('equivalence.json' if cli.device=='cpu' else 'equivalence_cuda.json')).read_text())['passed']
    args=dev.build_parser().parse_args(['--profile','dev','--episodes','120','--epochs','3','--device',cli.device,'--seed','234'])
    smoke_dir='smoke_pipeline' if cli.device=='cpu' else 'smoke_pipeline_cuda'
    original_config=dev.make_config
    def tiny_config(profile,**overrides):
        return original_config(profile,**overrides,simulation_seconds=2.)
    components={}
    def timed(name,fn):
        def call(*a,**kw):
            sync(cli.device); t=time.perf_counter(); value=fn(*a,**kw); sync(cli.device)
            components[name]=components.get(name,0.)+time.perf_counter()-t
            return value
        return call
    start=time.perf_counter()
    with Meter('dev pipeline smoke only',reservation=300):
        # Only artifact root and finite smoke duration are patched. All collection,
        # splitting, training, local fitting, control, and plant functions are original.
        with (patch.object(dev,'__file__',str(ROOT/'validation'/smoke_dir/'scripts/run_inverted_pendulum_mc.py')),
              patch.object(dev,'make_config',side_effect=tiny_config),
              patch.object(dev,'collect_trajectory_data',side_effect=timed('collection',dev.collect_trajectory_data)),
              patch.object(dev,'train_transition',side_effect=timed('training',dev.train_transition)),
              patch.object(dev,'fit_local_feedback',side_effect=timed('local_fit',dev.fit_local_feedback)),
              patch.object(dev,'simulate_plant',side_effect=timed('simulation',dev.simulate_plant))):
            result=dev.run(args)
        diags=result['solver_diagnostics']
        steps=sum(len(v['terminal_policy_used']) for v in diags.values())
        solves=sum(len(v['loss']) for v in diags.values())
        local=sum(sum(v['terminal_policy_used']) for v in diags.values())
        save('smoke.json' if cli.device=='cpu' else 'smoke_cuda.json',{'label':'SMOKE ONLY; no scientific validation', 'configuration':result['config'],
                          'original_modulo_collection':True,'original_70_15_15_split':result['split_sizes'],
                          'steps':steps,'solves':solves,'replan_rate':solves/steps,'local_steps':local,
                          'wall_s':time.perf_counter()-start,
                          'component_seconds':components,
                          'projection_limit':'120 episodes, 3 epochs, six angle-only cases, 2s each; rate is not a 20s nominal estimate'})
