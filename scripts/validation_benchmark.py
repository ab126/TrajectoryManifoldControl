"""Isolated adapter to the pinned validation project; never imports local src.

Run this entry point in a fresh process. All benchmark physics, collection,
splitting, baseline policies, initial conditions and legacy scoring are upstream.
Only additional scoring is loaded from this repository by an explicit file path.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / 'external/TrajectoryManifoldControl'
PIN = 'e66a2b79f27f347286576f678f424625f60787b4'


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def load_upstream(output):
    actual = subprocess.check_output(['git', '-C', str(UPSTREAM), 'rev-parse', 'HEAD'], text=True).strip()
    if actual != PIN:
        raise RuntimeError(f'upstream pin changed: {actual}')
    if subprocess.check_output(['git', '-C', str(UPSTREAM), 'status', '--porcelain'], text=True).strip():
        raise RuntimeError('upstream working tree must be clean')
    if any(key == 'src' or key.startswith('src.') or key == 'scripts' or key.startswith('scripts.') for key in sys.modules):
        raise RuntimeError('run adapter as a fresh process to avoid import collisions')
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    sys.dont_write_bytecode = True
    os.environ['MPLCONFIGDIR'] = str(output / '.mpl')
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[key] = '1'
    sys.path[:0] = [str(UPSTREAM), str(UPSTREAM/'validation'), str(UPSTREAM/'validation/experiments')]
    import campaign
    campaign.torch.set_num_threads(1)
    campaign.runner.tqdm = lambda values, **kwargs: values
    for module in (campaign.runner, sys.modules['src.manifold_control'], sys.modules['src.inverted_pendulum']):
        if not Path(module.__file__).resolve().is_relative_to(UPSTREAM):
            raise RuntimeError(f'wrong upstream module: {module.__file__}')
    if not (campaign.OUT/'initial_conditions.json').exists():
        raise RuntimeError('missing upstream saved initial states; refuse to create inside submodule')
    sets = campaign.ic_sets()  # Reads the upstream saved identities, never regenerates.
    campaign.OUT = output  # Prevent later campaign writes into the submodule.
    spec = importlib.util.spec_from_file_location('tmc_validation_metrics', ROOT/'src/validation_metrics.py')
    metrics = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(metrics)
    return campaign, sets, metrics


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, choices=(0, 1, 2, 234), default=0)
    parser.add_argument('--arms', nargs='+', choices=('H', 'L', 'M', 'M1'), default=['H', 'L'])
    parser.add_argument('--set', choices=('A', 'C'), default='A')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--seconds', type=float, default=14.)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--model-dir', type=Path, help='Directory containing transition.pt and data.npz')
    parser.add_argument('--train', action='store_true', help='Reconstruct original dev training, never an original checkpoint replay')
    parser.add_argument('--verify', action='store_true', help='Run adapter contract checks only')
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.output.is_relative_to(UPSTREAM):
        parser.error('output must be outside the submodule')
    args.output.mkdir(parents=True, exist_ok=True)
    c, sets, scoring = load_upstream(args.output)
    np, torch = c.np, c.torch
    cfg = c.runner.make_config('dev', seed=args.seed, device=args.device)
    if args.verify:
        from src.inverted_pendulum import rk4_step
        assert np.asarray(sets['A']).shape == (6, 4)
        expected = np.random.default_rng(c.IC_SEED).uniform([-1.5,-2.5,-1.3,-3.5], [1.5,2.5,1.3,3.5], (200,4))
        np.testing.assert_array_equal(expected, sets['C'])
        x = np.array([.1, .2, .3, .4]); u = np.array([1.2])
        ctl = lambda k, state: u
        t, X, U = c.runner.simulate_plant(ctl, x, cfg, seconds=.04)
        np.testing.assert_array_equal(t, [0., .02, .04])
        np.testing.assert_array_equal(U, np.tile(u, (2,1)))
        np.testing.assert_allclose(X[1], rk4_step(x, u[0], cfg.dt, 2.), rtol=0, atol=0)
        assert np.isclose(cfg.dt, .02*np.sqrt(9.81))
        ids = np.repeat(np.arange(20, dtype=np.float32)[:,None,None], 31, axis=1)
        split = c.runner.split_episodes(ids, np.zeros((20,30,1), np.float32), args.seed)
        groups = [set(v[0][:,0,0]) for v in split.values()]
        assert sum(len(g) for g in groups) == len(set.union(*groups)) == 20
        result = dict(passed=True, pin=PIN, cvxpy_workaround=c.CVXPY_WORKAROUND,
                      initial_conditions_sha256=sha256(UPSTREAM/'validation/results_v1/initial_conditions.json'),
                      imports={name: sys.modules[name].__file__ for name in ('src.manifold_control','src.inverted_pendulum','scripts.run_inverted_pendulum_mc')})
        write_json(args.output/'adapter_checks.json', result)
        print(json.dumps(result, indent=2)); return
    np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model_dir = args.model_dir.resolve() if args.model_dir else args.output/'model'
    if model_dir.is_relative_to(UPSTREAM) and args.train:
        parser.error('training artifacts must be outside the submodule')
    decoder = None
    artifact_kind = 'archived_gain_only'
    archived_diagnostics = UPSTREAM/f'validation/results_v1/models/episodes3500_seed{args.seed}/diagnostics.json'
    if args.train:
        model_dir.mkdir(parents=True, exist_ok=True)
        if (model_dir/'data.npz').exists():
            with np.load(model_dir/'data.npz') as data:
                X, U = data['X'], data['U']
        else:
            X,U = c.runner.collect_trajectory_data(cfg)
            np.savez_compressed(model_dir/'data.npz', X=X, U=U)
        splits = c.runner.split_episodes(X,U,args.seed)
        if (model_dir/'transition.pt').exists():
            transition,payload = c.runner.load_transition(model_dir/'transition.pt',args.device)
            expected = asdict(cfg); saved = payload['config'].copy()
            # Device can change when replaying a checkpoint; scientific settings cannot.
            saved['device'] = expected['device']
            if saved != expected:
                raise RuntimeError('checkpoint configuration does not match requested reconstruction')
        else:
            transition,_ = c.runner.train_transition(splits['train'],splits['validation'],cfg,model_dir/'transition.pt')
        gain,local = c.runner.fit_local_feedback(splits['train'],c.Q,c.R)
        decoder = c.CanonicalBehaviorDecoder(transition,30).to(args.device).eval()
        write_json(model_dir/'diagnostics.json', dict(config=asdict(cfg), prediction=c.prediction_rmse(decoder,splits['test']),local=c.fit_diagnostics(splits['train'],local)))
        artifact_kind = 'reconstructed_training'
    elif args.model_dir:
        transition,payload = c.runner.load_transition(model_dir/'transition.pt',args.device)
        saved = payload['config'].copy(); expected=asdict(cfg); saved['device']=expected['device']
        if saved != expected:
            raise RuntimeError('checkpoint configuration mismatch')
        decoder=c.CanonicalBehaviorDecoder(transition,30).to(args.device).eval()
        with np.load(model_dir/'data.npz') as data:
            splits = c.runner.split_episodes(data['X'],data['U'],args.seed)
        gain,_=c.runner.fit_local_feedback(splits['train'],c.Q,c.R)
        artifact_kind='provided_checkpoint'
    else:
        gain=np.asarray(json.loads(archived_diagnostics.read_text())['local']['gain'])
    if decoder is None and set(args.arms) != {'L'}:
        parser.error('H/M/M1 require --model-dir or --train; original checkpoints are absent upstream')
    identities = sets[args.set][:args.limit] if args.limit is not None else sets[args.set]
    if args.limit is not None and args.limit <= 0:
        parser.error('--limit must be positive')
    spec = dict(upstream=PIN, config=asdict(cfg), arms=args.arms, ic_set=args.set,
                initial_states=identities, seconds=args.seconds, artifact_kind=artifact_kind,
                torch=str(torch.__version__), numpy=str(np.__version__), python=sys.version,
                checkpoint_sha256=sha256(model_dir/'transition.pt') if decoder else None,
                data_sha256=sha256(model_dir/'data.npz') if decoder else None,
                archived_diagnostics_sha256=sha256(archived_diagnostics) if archived_diagnostics.exists() else None,
                cvxpy_workaround=c.CVXPY_WORKAROUND,
                adapter_sha256=sha256(__file__), scoring_sha256=sha256(ROOT/'src/validation_metrics.py'))
    manifest=args.output/'manifest.json'
    if manifest.exists() and json.loads(manifest.read_text()) != spec:
        raise RuntimeError('output already contains a different experiment; use a new directory')
    write_json(manifest,spec)
    for index,state in enumerate(identities):
        for arm in args.arms:
            case=args.output/'cases'/f'{arm}_{index:04d}.json'
            if case.exists():
                continue
            ctl=c.controller(decoder,cfg,gain,arm)
            start=time.perf_counter()
            with np.errstate(over='ignore',invalid='ignore'):
                t,X,U=c.runner.simulate_plant(ctl,np.asarray(state),cfg,seconds=args.seconds)
            diagnostics=ctl.diagnostics
            upstream=c.metrics(t,X,U,diagnostics['terminal_policy_used'],losses=diagnostics['loss'])
            extended=scoring.score_trace(t,X,U,upstream=upstream,
                         fallback_events=int(np.sum(~np.isfinite(diagnostics['loss']))))
            times=np.asarray(diagnostics['solve_seconds'])
            result=dict(seed=args.seed,arm=arm,ic_set=args.set,ic_index=index,initial_state=state,
                        wall_seconds=time.perf_counter()-start,upstream=upstream,**extended,
                        solver_latency=dict(n=len(times),median=float(np.median(times)) if len(times) else None,
                            p95=float(np.quantile(times,.95)) if len(times) else None,
                            deadline_miss_fraction=float(np.mean(times>.02)) if len(times) else None))
            raw=args.output/'raw'/f'{arm}_{index:04d}.npz'; raw.parent.mkdir(exist_ok=True)
            np.savez_compressed(raw,t=t,X=X,U=U,loss=diagnostics['loss'],solve_seconds=times,
                                local=diagnostics['terminal_policy_used'])
            write_json(case,result)
            print(json.dumps({k:result[k] for k in ('arm','ic_index','legacy_uninterrupted','terminal_recovery','periodic_terminal_recovery','wall_seconds')}),flush=True)


if __name__ == '__main__':
    main()
