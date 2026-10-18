"""Matched objective/decoder static optimizer comparison on validation states."""
import argparse
from dataclasses import asdict
import importlib.util
from pathlib import Path
import sys
import time

from validation_benchmark import load_upstream,ROOT,write_json,sha256,PIN


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-dir',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output=args.output.resolve()
    if args.output.is_relative_to(ROOT/'external'):p.error('output must be outside external')
    if (args.output/'comparison.json').exists():p.error('output already exists')
    args.output.mkdir(parents=True,exist_ok=True)
    c,_,_=load_upstream(args.output);np,torch=c.np,c.torch
    spec=importlib.util.spec_from_file_location('src.manifold_experiments',ROOT/'src/manifold_experiments.py')
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    model,payload=c.runner.load_transition(args.model_dir/'transition.pt','cpu')
    decoder=c.CanonicalBehaviorDecoder(model,30).eval()
    with np.load(args.model_dir/'data.npz') as data:split=c.runner.split_episodes(data['X'],data['U'],payload['config']['seed'])
    rows=[]
    for index,state in enumerate(split['validation'][0][:4,0]):
        for solver in ('original_cem','safe_cem','adam'):
            config=module.NativeConfig(solver=solver,iterations=20,seed=payload['config']['seed'])
            native=module.NativeSolver(decoder,torch.as_tensor(c.Q,dtype=torch.float32),torch.as_tensor(c.R,dtype=torch.float32),config)
            before=time.perf_counter();result=native.solve(torch.from_numpy(state))
            rows.append(dict(index=index,solver=solver,config=asdict(config),loss=result.loss,finite=bool(result.finite),
                seconds=time.perf_counter()-before,iterations=result.iterations,**native.diagnostics))
    write_json(args.output/'comparison.json',dict(upstream=PIN,checkpoint_sha256=sha256(args.model_dir/'transition.pt'),
        code_sha256=sha256(ROOT/'src/manifold_experiments.py'),script_sha256=sha256(__file__),
        states='First four validation episodes, zero initial plan for all, unchanged quadratic objective',
        rows=rows,limitation='Forward candidate counts exclude backward-pass cost; wall latency is also recorded.'))
    print([(r['index'],r['solver'],r['loss'],r['seconds']) for r in rows])


if __name__=='__main__':main()
