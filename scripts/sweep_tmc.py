"""Reproduce legacy three-seed fixed-cart convergence map (or compare standalone TMC)."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys
import time

from validation_benchmark import ROOT, VALIDATION, load_upstream, sha256, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--models-root', type=Path, default=Path('saves/validation/models'))
    p.add_argument('--seeds', type=int, nargs='+', default=[0,1,2], choices=[0,1,2])
    p.add_argument('--controller', choices=['legacy-hybrid','standalone'], default='legacy-hybrid')
    p.add_argument('--train', action='store_true', help='Collect/train missing matched artifacts, potentially expensive')
    p.add_argument('--plot-only', action='store_true')
    p.add_argument('--seconds', type=float, default=14.)
    p.add_argument('--max-wall-seconds', type=float, default=3600.)
    p.add_argument('--limit', type=int, help='First N saved D states for pipeline checks only')
    p.add_argument('--device', choices=['cpu','cuda'], default='cpu')
    args = p.parse_args()
    if len(set(args.seeds))!=len(args.seeds) or args.seconds<=0 or args.max_wall_seconds<=0 or (args.limit is not None and args.limit<1):
        p.error('seeds must be distinct and durations/limit positive')
    output = args.output.resolve()
    if output.is_relative_to(VALIDATION):
        p.error('output must be outside frozen validation snapshot')
    sys.path.insert(0, str(ROOT))
    if args.plot_only:
        from src.validation_sweep import plot_sweep
        print(json.dumps(plot_sweep(output), indent=2))
        plot_sweep(output, endpoint='primary_success'); return
    if args.train:
        for seed in args.seeds:
            subprocess.run([sys.executable, str(ROOT/'scripts/validation_benchmark.py'),
                '--train', '--seed', str(seed), '--device', args.device,
                '--model-dir', str(args.models_root/f'reconstructed-seed{seed}'),
                '--output', str(output/'training'/f'seed{seed}'),
                '--arms', 'L', '--set', 'A', '--limit', '1', '--seconds', '.04'], check=True)
    # Fail before importing simulation code if any requested artifacts are missing.
    artifacts = {}
    for seed in args.seeds:
        model = args.models_root/f'reconstructed-seed{seed}'
        for name in ('transition.pt','data.npz'):
            if not (model/name).is_file():
                p.error(f'missing {model/name}; provide matched artifacts or use --train')
        artifacts[str(seed)] = {name:sha256(model/name) for name in ('transition.pt','data.npz')}
    output.mkdir(parents=True, exist_ok=True)
    c, sets, scoring = load_upstream(output)
    np, torch = c.np, c.torch
    from src.validation_sweep import primary_success, terminal_recovery, plot_sweep, UPSTREAM
    from src.manifold_experiments import NativeConfig
    from src.validation_controllers import trajectory_controller, ControllerFailure
    # Pin the complete original grid even in limited pipeline checks.
    states = sets['D']
    arms = ['H','L'] if args.controller=='legacy-hybrid' else ['TMC','L']
    manifest = dict(version=1, upstream_inspected=UPSTREAM, seeds=args.seeds, arms=arms,
        controller=args.controller, states=states, seconds=args.seconds, device=args.device, plot_endpoint='terminal_recovery',
        artifacts=artifacts, endpoint='terminal recovery: finite + final3s abs(theta)<=.02 and abs(omega_physical)<=.05; crossings allowed, no cart criterion; primary_success additionally forbids abs(theta)>=pi/2',
        source_sha256={s:sha256(ROOT/s) for s in ['scripts/sweep_tmc.py','src/validation_sweep.py',
            'scripts/run_inverted_pendulum_mc.py','scripts/validation_benchmark.py',
            'src/manifold_control.py','src/inverted_pendulum.py','src/validation_controllers.py','src/manifold_experiments.py']},
        torch=str(torch.__version__), numpy=str(np.__version__),
        standalone_config=dict(iterations=20, learning_rate=.15, starts=3, replan_interval=1))
    path = output/'manifest.json'
    if path.exists() and json.loads(path.read_text())!=manifest:
        p.error('output protocol differs; use a new output directory')
    write_json(path, manifest)
    start = time.perf_counter()
    budget_exhausted = False
    for seed in args.seeds:
        model = args.models_root/f'reconstructed-seed{seed}'
        transition, payload = c.runner.load_transition(model/'transition.pt', args.device)
        cfg = c.runner.make_config('dev', seed=seed, device=args.device)
        saved = payload['config'].copy(); saved['device'] = args.device
        if saved!=asdict(cfg):
            p.error(f'seed{seed} checkpoint differs from original dev configuration')
        with np.load(model/'data.npz') as data:
            splits = c.runner.split_episodes(data['X'], data['U'], seed)
        gain, _ = c.runner.fit_local_feedback(splits['train'], c.Q, c.R)
        decoder = c.CanonicalBehaviorDecoder(transition, cfg.horizon).to(args.device).eval()
        torch.manual_seed(seed); np.random.seed(seed)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        for i, state in enumerate(states[:args.limit] if args.limit else states):
            for arm in arms:
                key = f'seed{seed}_{arm}_{i:04d}'
                case = output/'cases'/f'{key}.json'
                if case.exists():
                    prior = json.loads(case.read_text())
                    if sha256(output/'raw'/f'{key}.npz') != prior['raw_sha256']:
                        raise RuntimeError(f'saved trace hash mismatch: {key}')
                    continue
                if time.perf_counter()-start >= args.max_wall_seconds:
                    budget_exhausted = True; break
                ctl = c.controller(decoder, cfg, gain, arm) if arm in ('H','L') else trajectory_controller(
                    decoder, NativeConfig(solver='adam', seed=seed, iterations=20, learning_rate=.15, starts=3),
                    x_ref=[0.]*4, u_ref=[0.], umax=cfg.umax)
                before = time.perf_counter()
                x, u, latencies = [np.asarray(state, dtype=float)], [], []
                failure = None
                for k in range(round(args.seconds/.02)):
                    tick = time.perf_counter()
                    try:
                        action = np.asarray(ctl(k, x[-1]), dtype=float).reshape(1)
                    except ControllerFailure as exc:
                        failure = str(exc); break
                    latencies.append(time.perf_counter()-tick)
                    if not np.isfinite(action).all():
                        failure = 'nonfinite input'; break
                    with np.errstate(over='ignore', invalid='ignore'):
                        next_x = c.runner._plant_step(x[-1], action, cfg.dt)
                    u.append(action); x.append(next_x)
                    if not np.isfinite(next_x).all():
                        failure = 'nonfinite plant state'; break
                t = np.arange(len(x))*.02
                x, u = np.asarray(x), np.asarray(u).reshape(-1,1)
                flags = getattr(ctl, 'diagnostics', {}).get('terminal_policy_used', [arm=='L']*len(u))
                diagnostics = getattr(ctl, 'diagnostics', {})
                metrics = c.metrics(t,x,u,flags, losses=diagnostics.get('loss', [])) if len(u) else {}
                success = failure is None and primary_success(t,x,u)
                if args.seconds>=3 and failure is None and success!=metrics['primary_success']:
                    raise RuntimeError('legacy endpoint parity failed')
                if args.seconds>=3 and failure is None and terminal_recovery(t,x,u)!=metrics['balanced']:
                    raise RuntimeError('legacy terminal recovery parity failed')
                raw = output/'raw'/f'{key}.npz'; raw.parent.mkdir(exist_ok=True)
                np.savez_compressed(raw, t=t, X=x, U=u, controller_seconds=latencies)
                row = dict(seed=seed, arm=arm, ic_index=i, initial_state=state,
                    primary_success=success, terminal_recovery=failure is None and terminal_recovery(t,x,u), failure=failure, wall_seconds=time.perf_counter()-before,
                    metrics=metrics, raw_sha256=sha256(raw), controller_diagnostics=dict(
                        solve_seconds=diagnostics.get('solve_seconds', []),
                        losses=[float(v) if np.isfinite(v) else None for v in diagnostics.get('loss', [])]))
                write_json(case,row)
                print(json.dumps({k:row[k] for k in ('seed','arm','ic_index','primary_success','terminal_recovery','failure','wall_seconds')}), flush=True)
            if budget_exhausted: break
        if budget_exhausted: break
    print(json.dumps(plot_sweep(output), indent=2))
    plot_sweep(output, endpoint='primary_success')


if __name__=='__main__':
    main()
