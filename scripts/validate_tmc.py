"""New resumable paired TMC/L validation; TMC has no fallback of any kind."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

from validation_benchmark import ROOT, VALIDATION, load_upstream, sha256, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--data', type=Path, help='Training data.npz for matched fitted L gain')
    p.add_argument('--archived-gain-seed', type=int, choices=(0, 1, 2),
                   help='Explicit alternative to --data; reported as unmatched archived gain')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--set', choices=('tuning', 'A', 'C', 'fresh'), default='tuning')
    p.add_argument('--limit', type=int)
    p.add_argument('--seconds', type=float, default=14.)
    p.add_argument('--max-wall-seconds', type=float, default=300.)
    p.add_argument('--arms', nargs='+', choices=('TMC', 'L', 'hybrid'), default=['TMC', 'L'])
    p.add_argument('--x-ref', type=float, nargs=4, default=[0.] * 4)
    p.add_argument('--u-ref', type=float, nargs=1, default=[0.])
    p.add_argument('--solver', choices=('adam', 'safe_cem'), default='adam')
    p.add_argument('--iterations', type=int, default=20)
    p.add_argument('--learning-rate', type=float, default=.15)
    p.add_argument('--starts', type=int, default=3)
    p.add_argument('--replan-interval', type=int, default=1)
    p.add_argument('--terminal-window', type=int, default=1)
    p.add_argument('--periodic', action='store_true')
    p.add_argument('--include-previous-input', action='store_true')
    p.add_argument('--population', type=int, default=256)
    p.add_argument('--elite', type=int, default=32)
    p.add_argument('--cem-iterations', type=int, default=3)
    p.add_argument('--switch-cost', type=float, default=.75)
    p.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    args = p.parse_args()
    if (args.data is None) == (args.archived_gain_seed is None):
        p.error('provide exactly one of --data or --archived-gain-seed')
    if args.seconds <= 0 or args.max_wall_seconds <= 0 or (args.limit is not None and args.limit < 1):
        p.error('durations and limit must be positive')
    if args.iterations < 1 or args.starts < 1 or args.learning_rate <= 0 or args.terminal_window < 1:
        p.error('optimizer sizes, rate and terminal window must be positive')
    if not 0 < args.elite <= args.population or args.cem_iterations < 1:
        p.error('invalid CEM sizes')
    if len(set(args.arms)) != len(args.arms):
        p.error('arms must be distinct')
    args.output = args.output.resolve()
    if args.output.is_relative_to(VALIDATION):
        p.error('output must be outside frozen validation snapshot')
    args.output.mkdir(parents=True, exist_ok=True)
    c, sets, scoring = load_upstream(args.output)
    np, torch = c.np, c.torch
    from src.manifold_experiments import NativeConfig
    from src.validation_controllers import validation_lqr, trajectory_controller, ControllerFailure
    transition, payload = c.runner.load_transition(args.checkpoint, args.device)
    cfg = c.runner.ExperimentConfig(**{**payload['config'], 'device': args.device})
    if payload['x_dim'] != 4 or payload['u_dim'] != 1:
        p.error('validation plant requires four states and one input')
    if not np.isclose(cfg.dt, .02 / np.sqrt(1 / 9.81), rtol=0, atol=1e-14):
        p.error('checkpoint sampling time differs from validation plant')
    if not 1 <= args.replan_interval <= payload['horizon']:
        p.error('replan interval must lie within checkpoint horizon')
    if not np.isfinite(args.x_ref + args.u_ref).all() or not np.isfinite(args.switch_cost) or args.switch_cost < 0:
        p.error('references and nonnegative switch cost must be finite')
    decoder = c.CanonicalBehaviorDecoder(transition, payload['horizon']).to(args.device).eval()
    if args.data:
        with np.load(args.data) as data:
            split = c.runner.split_episodes(data['X'], data['U'], cfg.seed)
        gain, _ = c.runner.fit_local_feedback(split['train'], c.Q, c.R)
        gain_source = dict(kind='fitted_training_data', path=str(args.data.resolve()), sha256=sha256(args.data))
    else:
        path = VALIDATION / f'results_v1/models/episodes3500_seed{args.archived_gain_seed}/diagnostics.json'
        gain = np.asarray(json.loads(path.read_text())['local']['gain'])
        gain_source = dict(kind='unmatched_archived_gain', seed=args.archived_gain_seed, sha256=sha256(path))
    config = NativeConfig(solver=args.solver, seed=cfg.seed, iterations=args.iterations,
        learning_rate=args.learning_rate, starts=args.starts, replan_interval=args.replan_interval,
        terminal_window=args.terminal_window, periodic=args.periodic,
        include_previous_input=args.include_previous_input, population=args.population,
        elite=args.elite, cem_iterations=args.cem_iterations)
    if args.set == 'tuning':
        states = np.random.default_rng(20261017).uniform([-.5,-.8,-.6,-1.], [.5,.8,.6,1.], (8,4)).tolist()
    elif args.set == 'fresh':
        states = json.loads((VALIDATION/'results_literature/inventory/fresh_200_states.json').read_text())['states']
    else:
        states = sets[args.set]
    if args.limit is not None:
        states = states[:args.limit]
    protocol = dict(version=1, checkpoint_sha256=sha256(args.checkpoint), gain_source=gain_source,
        gain=gain.tolist(), config=asdict(config), plant_config=asdict(cfg), states=states,
        state_set=args.set, seconds=args.seconds, arms=args.arms, x_ref=args.x_ref, u_ref=args.u_ref,
        switch_cost=args.switch_cost, Q=c.Q.tolist(), R=c.R.tolist(),
        sources={f:sha256(ROOT/f) for f in ('src/validation_controllers.py', 'src/manifold_experiments.py',
            'src/manifold_control.py', 'src/inverted_pendulum.py', 'src/validation_metrics.py',
            'scripts/validate_tmc.py', 'scripts/run_inverted_pendulum_mc.py', 'scripts/validation_benchmark.py')},
        torch=str(torch.__version__), numpy=str(np.__version__),
        timing='Ideal feedback, solver latency recorded, no computational delay injected.',
        failure_policy='TMC terminates case on invalid plan; no substituted action.',
        references='Absolute initial states; reference-relative metrics, normalized coordinates.')
    manifest = args.output/'manifest.json'
    if manifest.exists() and json.loads(manifest.read_text()) != json.loads(json.dumps(protocol)):
        raise ValueError('output protocol differs; use a new output directory')
    write_json(manifest, protocol)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    start = time.perf_counter()
    rows = []
    for index, state in enumerate(states):
        for arm in args.arms:
            case = args.output/'cases'/f'{arm}_{index:04d}.json'
            if case.exists():
                rows.append(json.loads(case.read_text()))
                continue
            if time.perf_counter() - start >= args.max_wall_seconds:
                break
            linear = validation_lqr(gain, x_ref=args.x_ref, u_ref=args.u_ref, umax=cfg.umax)
            ctl = linear if arm == 'L' else trajectory_controller(decoder, config,
                x_ref=args.x_ref, u_ref=args.u_ref, umax=cfg.umax,
                linear=linear if arm == 'hybrid' else None, switch_cost=args.switch_cost)
            X, U, times = [np.asarray(state, dtype=float)], [], []
            failure = None
            for k in range(round(args.seconds/.02)):
                before = time.perf_counter()
                try:
                    action = np.asarray(ctl(k, X[-1]), dtype=float).reshape(1)
                except ControllerFailure as exc:
                    times.append(time.perf_counter()-before)
                    failure = str(exc)
                    break
                times.append(time.perf_counter()-before)
                if not np.isfinite(action).all() or (np.abs(action) > cfg.umax).any():
                    failure = 'invalid controller action'
                    break
                with np.errstate(over='ignore', invalid='ignore'):
                    next_state = c.runner._plant_step(X[-1], action, cfg.dt)
                U.append(action)
                X.append(next_state)
                if not np.isfinite(next_state).all():
                    failure = 'nonfinite plant state'
                    break
            x, u = np.asarray(X), np.asarray(U).reshape(-1, 1)
            t = np.arange(len(x))*.02
            metrics = {}
            if len(x) >= 2:
                error = x - np.asarray(args.x_ref)
                upstream = c.metrics(t, error, u, [arm == 'L']*len(u), losses=[])
                metrics = scoring.score_trace(t, error, u, upstream=upstream,
                    fallback_events=len(getattr(ctl, 'diagnostics', {}).get('fallback_events', [])))
                metrics['state_rmse'] = [float(v) if np.isfinite(v) else None
                    for v in np.sqrt(np.mean(error**2, axis=0))]
                metrics['final_state_error'] = [float(v) if np.isfinite(v) else None for v in error[-1]]
            success = failure is None and metrics.get('terminal_recovery', False)
            d = getattr(ctl, 'diagnostics', {})
            row = dict(arm=arm, ic_index=index, completed=failure is None, failure=failure,
                terminal_recovery=bool(success), tracking_metrics=metrics,
                linear_steps=d.get('linear_steps', []), fallback_events=d.get('fallback_events', []),
                failure_steps=d.get('failure_steps', []),
                controller_latency=dict(n=len(times), median=float(np.median(times)) if times else None,
                    p95=float(np.quantile(times,.95)) if times else None,
                    deadline_miss_fraction=float(np.mean(np.asarray(times)>.02)) if times else None))
            raw = args.output/'raw'/f'{arm}_{index:04d}.npz'
            raw.parent.mkdir(exist_ok=True)
            np.savez_compressed(raw, t=t, X=x, U=u, controller_seconds=times,
                solve_seconds=d.get('solve_seconds', []), solve_steps=d.get('solve_steps', []))
            write_json(case, row)
            rows.append(row)
            print(json.dumps(row), flush=True)
        else:
            continue
        break
    baseline = {r['ic_index']:r for r in rows if r['arm']=='L'}
    rates, paired = [], []
    for arm in args.arms:
        group = [r for r in rows if r['arm']==arm]
        rates.append(dict(arm=arm, n=len(group), successes=sum(r['terminal_recovery'] for r in group),
                          failures=sum(not r['completed'] for r in group)))
        shared = [r for r in group if r['ic_index'] in baseline]
        paired.append(dict(arm=arm, n=len(shared),
            wins=sum(r['terminal_recovery'] and not baseline[r['ic_index']]['terminal_recovery'] for r in shared),
            losses=sum(not r['terminal_recovery'] and baseline[r['ic_index']]['terminal_recovery'] for r in shared)))
    summary = dict(complete=len(rows)==len(states)*len(args.arms), rates=rates, paired_recovery=paired)
    write_json(args.output/'summary.json', summary)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
