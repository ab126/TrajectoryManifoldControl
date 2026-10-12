"""Post-E4 CPU-only sliding-feedback comparison; saved data and cases are reused."""
from campaign import *
from smc_baseline import surface, calibrate, SlidingFeedback, effort
from e3_robustness import CONDITIONS, normal_sequence, physical_mismatch_step, applied_inputs
from scipy.stats import binomtest
import itertools

EXPERIMENT = 'E6_SMC'
NEW_ARMS = ('SMC', 'SMC_linear')
MODELS = {}
NOMINAL_STEP = runner._plant_step


def model(seed, condition='nominal'):
    process = condition in CONDITIONS and CONDITIONS[condition]['mode'] == 'process'
    key = (seed, condition if process else 'nominal')
    if key in MODELS:
        return MODELS[key]
    base = OUT/'E3/models'/condition if process else OUT/'models'
    path = base/f'episodes3500_seed{seed}'
    saved = json.loads((path/'diagnostics.json').read_text())
    A, B = np.asarray(saved['local']['A']), np.asarray(saved['local']['B'])
    C, P, equivalent = surface(A, B, Q, R)
    with np.load(path/'data.npz') as data:
        splits = runner.split_episodes(data['X'], data['U'], seed)
    calibration = calibrate(A, B, C, splits['train'], splits['validation'])
    with np.load(OUT/'models'/f'episodes3500_seed{seed}'/'data.npz') as data:
        nominal = runner.split_episodes(data['X'], data['U'], seed)['train'][0]
    std = nominal[:, :-1].reshape(-1, 4).std(axis=0).clip(1e-6)
    metadata = dict(seed=seed, condition=key[1], source_data=str(path/'data.npz'),
        source_diagnostics=str(path/'diagnostics.json'), A=A.tolist(), B=B.tolist(),
        C=C.tolist(), P=P.tolist(), equivalent_spectral_radius=float(max(abs(np.linalg.eigvals(equivalent)))),
        calibration=calibration, rho=.9, beta=.45*calibration['phi'],
        nominal_state_std=std.tolist(), guarantees='Exact fitted unsaturated linear model only')
    dump(OUT/EXPERIMENT/'models'/f'{key[1]}_seed{seed}.json', metadata)
    MODELS[key] = (A, C, calibration['phi'], std, runner.make_config('dev', seed=seed, device='cpu'))
    return MODELS[key]


def make_control(A, C, phi, arm):
    law = SlidingFeedback(A, C, phi, correction=arm == 'SMC')
    diagnostics = dict(terminal_policy_used=[], loss=[], predictions=[], solve_seconds=[], evaluation_seconds=[])
    def control(k, x):
        start = time.perf_counter()
        value, _, _ = law.command(x)
        diagnostics['evaluation_seconds'].append(time.perf_counter()-start)
        diagnostics['terminal_policy_used'].append(False)  # No use of submitted LQR branch.
        return value
    control.diagnostics = diagnostics
    return control


def evaluate(task):
    seed, condition, arm, name, index, state, seconds = task
    out = OUT/EXPERIMENT
    label = condition+'_'+name
    key = f'seed{seed}_{arm}_{label}_{index:04d}_{seconds:g}s'
    destination = out/'cases'/(key+'.json')
    if destination.exists():
        row = json.loads(destination.read_text())
        if row.get('smc_recorded'):
            return row
    A, C, phi, std, config = model(seed, condition)
    spec = CONDITIONS.get(condition, dict(mode='nominal'))
    steps = round(seconds/.02)
    noise = (normal_sequence(seed, condition, index+(0 if name == 'A' else 1000), steps, std*spec.get('level', 0))
             if condition != 'nominal' else np.zeros((steps, 4)))
    sample, previous = 0, np.zeros(1)
    def plant(x, u, dt):
        nonlocal sample, previous
        value = np.asarray(u).copy()
        if spec['mode'] == 'delay':
            value, previous = previous, value
        xn = (physical_mismatch_step(x, value, dt, spec['mass'], spec['length'])
              if spec['mode'] == 'mismatch' else NOMINAL_STEP(x, value, dt))
        if spec['mode'] == 'process':
            xn = xn+noise[sample]
        sample += 1
        return xn
    ctl = make_control(A, C, phi, arm)
    observe = (lambda k, x: x+noise[k]) if spec['mode'] == 'measurement' else None
    runner._plant_step = plant
    try:
        row = run_case(out, None, None, config, arm, label, index, state, seconds, observe=observe, control=ctl)
    finally:
        runner._plant_step = NOMINAL_STEP
    raw = ROOT/row['raw_path']
    with np.load(raw) as data:
        arrays = {k: data[k] for k in data.files}
    arrays['U_applied'] = applied_inputs(arrays['U'], spec['mode'] == 'delay')
    observed = arrays['X'][:-1]+noise if spec['mode'] == 'measurement' else arrays['X'][:-1]
    s = (observed@C.T)[:, 0]
    beta = .45*phi if arm == 'SMC' else 0.
    raw_commands = -(observed@A.T@C.T)[:, 0]+.9*s-beta*np.clip(s/phi, -1, 1)
    arrays.update(smc_surface=s, smc_raw_command=raw_commands)
    if spec['mode'] in ('measurement', 'process'):
        arrays[spec['mode']+'_noise'] = noise
    if ctl.diagnostics['evaluation_seconds']:
        arrays['controller_evaluation_seconds'] = np.asarray(ctl.diagnostics['evaluation_seconds'])
    temporary = raw.with_suffix('.tmp.npz')
    np.savez_compressed(temporary, **arrays)
    temporary.replace(raw)
    def number(v):
        return float(v) if np.isfinite(v) else None
    actual = arrays['U_applied']
    durations = arrays.get('controller_evaluation_seconds', np.array([]))
    row.update(effort(actual, arrays['U']), smc_recorded=True, phi=phi, beta=beta,
        feedback_fraction=1., max_abs_u=number(np.max(np.abs(actual))),
        input_limit_seconds=float(np.sum(np.abs(actual[:, 0]) >= 10-1e-6)*.02),
        max_command_u=number(np.max(np.abs(arrays['U']))),
        surface_abs_q95=number(np.quantile(np.abs(s), .95)),
        tail_surface_rms=number(np.sqrt(np.mean(s[-150:]**2))),
        controller_evaluation_seconds=number(np.sum(durations)) if len(durations) else None,
        controller_evaluation_samples=len(durations), reused_reference=False)
    dump(destination, row)
    return row


def reference(seed, condition, arm, name, index, state, seconds):
    if condition == 'nominal':
        experiment, source_name = ('E0', 'A') if name == 'A' else ('E1', 'C')
        source_label = source_name
    else:
        experiment, source_label = 'E3', condition+'_'+name
    source = OUT/experiment/'cases'/f'seed{seed}_{arm}_{source_label}_{index:04d}_{seconds:g}s.json'
    row = json.loads(source.read_text())
    if not np.allclose(row['initial_state'], state, rtol=0, atol=1e-12):
        raise ValueError(f'Unmatched reference IC: {source}')
    label = condition+'_'+name
    key = f'seed{seed}_{arm}_{label}_{index:04d}_{seconds:g}s'
    row.update(case=key, ic_set=label, reused_reference=True, source_case=str(source),
               source_case_wall_seconds=row['wall_seconds'], wall_seconds=0.)
    raw = ROOT/row['raw_path']
    with np.load(raw) as data:
        applied = data['U_applied'] if 'U_applied' in data else data['U']
        row.update(effort(applied, data['U']))
    dump(OUT/EXPERIMENT/'cases'/(key+'.json'), row)


def paired_feedback(out, rows):
    groups = {}
    for row in rows:
        key = (row['seed'], row['ic_set'], row['seconds'])
        groups.setdefault(key, {}).setdefault(row['arm'], {})[row['ic_index']] = row
    comparisons = []
    for key, arms in sorted(groups.items()):
        for reference_arm in ('L', 'SMC_linear'):
            for arm in NEW_ARMS:
                if reference_arm == arm:
                    continue
                a, b = arms.get(reference_arm, {}), arms.get(arm, {})
                shared = sorted(a.keys() & b.keys())
                if not shared:
                    continue
                a_only = sum(a[i]['primary_success'] and not b[i]['primary_success'] for i in shared)
                b_only = sum(b[i]['primary_success'] and not a[i]['primary_success'] for i in shared)
                comparisons.append(dict(seed=key[0], ic_set=key[1], seconds=key[2], reference=reference_arm, arm=arm,
                    n_paired=len(shared), reference_only=a_only, arm_only=b_only,
                    success_rate_difference=(a_only-b_only)/len(shared),
                    exact_mcnemar_p=float(binomtest(a_only, a_only+b_only, .5).pvalue) if a_only+b_only else 1.))
    csv_write(out/'feedback_paired_comparisons.csv', comparisons)


def summarize_smc():
    out = OUT/EXPERIMENT
    rows, rates = summarize(out)
    paired_feedback(out, rows)
    from figures import figures
    figures(EXPERIMENT)
    distributions = []
    for key, items in itertools.groupby(sorted(rows, key=lambda r: (r['seed'], r['ic_set'], r['seconds'], r['arm'])),
                                       key=lambda r: (r['seed'], r['ic_set'], r['seconds'], r['arm'])):
        items = list(items)
        for field in ('input_total_variation', 'command_total_variation', 'input_rms', 'surface_abs_q95', 'tail_surface_rms', 'controller_evaluation_seconds'):
            values = [r[field] for r in items if r.get(field) is not None]
            distributions.append(dict(seed=key[0], ic_set=key[1], seconds=key[2], arm=key[3], field=field,
                n=len(items), n_observed=len(values), median=float(np.median(values)) if values else None,
                q95=float(np.quantile(values, .95)) if values else None))
    csv_write(out/'feedback_distributions.csv', distributions)
    text = ['SMC feedback comparison\n', 'Empirical same-data comparison. Guarantees concern the fitted unsaturated linear model only.\n',
        'H and L cases are reused from E0/E1/E3; their original elapsed times are not charged again.\n',
        'See rates.csv, paired_comparisons.csv (H reference), feedback_paired_comparisons.csv (L and zero-correction references), distributions.csv and feedback_distributions.csv.\n',
        '| Seed | Condition | Seconds | Arm | Successes/n |\n|---|---|---|---|---|\n']
    text += [f"| {r['seed']} | {r['ic_set']} | {r['seconds']} | {r['arm']} | {r['successes']}/{r['n']} |\n" for r in rates]
    (out/'RESULTS.md').write_text('\n'.join(text))


def tasks():
    sets = ic_sets()
    nominal, stress = [], []
    for seed in (0, 1, 2):
        for name, states in (('A', sets['A']), ('C50', sets['C'][:50])):
            for index, state in enumerate(states):
                for arm in NEW_ARMS:
                    nominal.append((seed, 'nominal', arm, name, index, state, 14.))
                for condition in CONDITIONS:
                    for arm in NEW_ARMS:
                        stress.append((seed, condition, arm, name, index, state, 14.))
        for index in (0, 2, 5):
            for arm in NEW_ARMS:
                nominal.append((seed, 'nominal', arm, 'A', index, sets['A'][index], 60.))
    # Four prespecified pilot cases estimate direct-feedback runtime; these also
    # belong to the final nominal sample, and are never repeated.
    pilot = [t for t in nominal if t[0] == 0 and t[3:5] in (('A', 0), ('C50', 0)) and t[-1] == 14.]
    return pilot+[t for t in nominal if t not in pilot], stress


def main(max_wall_seconds, max_cpu_seconds):
    import fcntl
    from queued_smc import prerequisites_ready
    if not prerequisites_ready():
        raise RuntimeError('SMC cannot run before required E2–E4 manifests are DONE')
    out = OUT/EXPERIMENT
    out.mkdir(exist_ok=True)
    lock = (out/'.run.lock').open('w')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print('SMC already running; no duplicate launch.', flush=True)
        return
    path = out/'manifest.json'
    prior = json.loads(path.read_text()) if path.exists() else {}
    if prior.get('status') == 'DONE':
        return
    nominal, stress = tasks()
    manifest = dict(prior or dict(experiment=EXPERIMENT, git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        seeds=[0,1,2], device='cpu', wall_seconds=0., cpu_seconds=0.,
        config=dict(arms=['H','L',*NEW_ARMS], new_arms=list(NEW_ARMS), ic_sets=['A','C50'], physical_sample_seconds=.02,
            input_bounds=[-10.,10.], rho=.9, beta_rule='.45*phi; zero for SMC_linear',
            calibration='max(.001,4*validation residual abs q95), inside training-scale nearest-6000 fit radius',
            process_models='Reuse E3 process-trained fits; freeze nominal calibration for measurement/mismatch/delay',
            pilot='seed0 nominal A[0] and C50[0], both new arms, included in final outcomes',
            nominal_cases=len(nominal), stress_cases=len(stress), neural_training=False,
            reference_cohort='E0/E1/E3 development collection: 3500 x 30; E4 recollects different data, so no direct paired DeePC claim',
            limitations='Same-data quasi-SMC adaptation; no certified nonlinear uncertainty bounds or saturation guarantee')))
    prior_wall, prior_cpu = manifest['wall_seconds'], manifest.get('cpu_seconds', 0.)
    wall_start, cpu_start = time.perf_counter(), time.process_time()
    manifest.update(status='RUNNING', requested_max_wall_seconds=max_wall_seconds, max_cpu_seconds=max_cpu_seconds)
    def checkpoint():
        manifest.update(wall_seconds=prior_wall+time.perf_counter()-wall_start,
                        cpu_seconds=prior_cpu+time.process_time()-cpu_start)
        dump(path, manifest)
    def within_budget():
        wall = prior_wall+time.perf_counter()-wall_start
        cpu = prior_cpu+time.process_time()-cpu_start
        if wall >= max_wall_seconds or cpu >= max_cpu_seconds:
            raise TimeoutError('SMC reached its one-hour or remaining-campaign cap')
        estimate = manifest.get('initial_estimate_seconds')
        if estimate and wall > 3*estimate:
            raise TimeoutError('SMC exceeded three times its measured pilot estimate')
    try:
        checkpoint()
        selected = nominal+stress if manifest.get('include_stress', True) else nominal
        for index, task in enumerate(selected):
            within_budget()
            seed, condition, arm, name, ic_index, state, seconds = task
            row = evaluate(task)
            for reference_arm in ('H', 'L'):
                reference(seed, condition, reference_arm, name, ic_index, state, seconds)
            manifest['completed_new_cases'] = index+1
            if index == 3 and not manifest.get('initial_estimate_seconds'):
                costs = [json.loads(p.read_text())['wall_seconds'] for p in (out/'cases').glob('*.json')
                         if not json.loads(p.read_text()).get('reused_reference')]
                equivalent_cases = sum(t[-1]/14 for t in nominal+stress)
                estimate = 30+1.5*max(costs)*equivalent_cases
                manifest.update(initial_estimate_seconds=estimate, include_stress=estimate <= max_wall_seconds-prior_wall)
                if not manifest['include_stress']:
                    selected[len(nominal):] = []
                    manifest['deviation'] = 'Pilot projects full stress scope above remaining allowance; nominal comparison retained'
                    manifest['initial_estimate_seconds'] = 30+1.5*max(costs)*sum(t[-1]/14 for t in nominal)
            manifest['total_new_cases'] = len(selected)
            checkpoint()
        within_budget()
        summarize_smc()
        manifest['status'] = 'DONE'
    except TimeoutError as error:
        manifest.update(status='PARTIAL', error=str(error))
        summarize_smc()
    except BaseException as error:
        manifest.update(status='ERROR', error=repr(error))
        raise
    finally:
        checkpoint()
        lock.close()


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--max-wall-seconds',type=float,default=3600.)
    parser.add_argument('--max-cpu-seconds',type=float,default=3600.)
    args=parser.parse_args()
    torch.set_num_threads(1)
    runner.tqdm=lambda iterable,**kwargs: iterable
    main(args.max_wall_seconds,args.max_cpu_seconds)
