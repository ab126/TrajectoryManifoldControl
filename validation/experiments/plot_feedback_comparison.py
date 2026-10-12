"""Completed same-data feedback comparison and independently checked local gains."""
from campaign import *
from local_stability import nominal_linearization, spectral_radius, checks
from smc_baseline import SlidingFeedback
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    start, cpu_start = time.perf_counter(), time.process_time()
    out = OUT/'E6_SMC'
    manifest = json.loads((out/'manifest.json').read_text())
    assert manifest['status'] == 'DONE'
    rows = list(csv.DictReader((out/'summary.csv').open()))
    assert len(rows) == 8100
    assert sum(r['reused_reference'] == 'False' for r in rows) == 4050
    rates = list(csv.DictReader((out/'rates.csv').open()))
    assert len(rates) == 300
    arms = ('H', 'L', 'SMC', 'SMC_linear')
    labels = ('H', 'L', 'SMC', 'Zero\ncorrection')
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for ax, (condition, seconds, title) in zip(axes, (
            ('nominal_C50', 14, 'Nominal IC-C50, 14 s; n=50'),
            ('input_delay_C50', 14, '20 ms delay, IC-C50; n=50'),
            ('nominal_A', 60, 'Nominal IC-A subset, 60 s; n=3'))):
        for a, arm in enumerate(arms):
            for seed in range(3):
                r = next(r for r in rates if r['seed'] == str(seed) and r['arm'] == arm
                         and r['ic_set'] == condition and float(r['seconds']) == seconds)
                value = float(r['rate'])
                ax.errorbar(a+.07*(seed-1), value,
                    yerr=[[max(0., value-float(r['wilson95_low']))], [max(0., float(r['wilson95_high'])-value)]],
                    fmt='o', color=f'C{seed}', capsize=2, markersize=4, linewidth=.7,
                    label=f'seed {seed}' if a == 0 else None)
        ax.set(xticks=range(4), xticklabels=labels, ylim=(-.03, 1.03), title=title)
    axes[0].set_ylabel('Primary success; Wilson 95% intervals')
    axes[0].legend(fontsize=8)
    fig.suptitle('Original 3,500 × 30 collection; each whole-run seed remains separate')
    fig.tight_layout()
    fig.savefig(out/'feedback_success_comparison.png', dpi=150)
    plt.close(fig)
    effort_rows = list(csv.DictReader((out/'feedback_distributions.csv').open()))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, condition, title in zip(axes, ('nominal_A', 'input_delay_A'), ('Nominal IC-A', '20 ms delay, IC-A')):
        for a, arm in enumerate(arms):
            for seed in range(3):
                r = next(r for r in effort_rows if r['seed'] == str(seed) and r['arm'] == arm
                         and r['ic_set'] == condition and float(r['seconds']) == 14
                         and r['field'] == 'input_total_variation')
                ax.scatter(a+.07*(seed-1), float(r['median']), color=f'C{seed}',
                           label=f'seed {seed}' if a == 0 else None)
        ax.set(xticks=range(4), xticklabels=labels, yscale='log', title=title)
    axes[0].set_ylabel('Normalized-input total variation, median within seed')
    axes[0].legend(fontsize=8)
    fig.suptitle('Six authored ICs, 14 s; applied inputs, including delay')
    fig.tight_layout()
    fig.savefig(out/'feedback_effort_comparison.png', dpi=150)
    plt.close(fig)
    timing = []
    groups = {}
    for r in rows:
        if r['arm'] in ('SMC', 'SMC_linear'):
            groups.setdefault((r['seed'], r['arm'], r['ic_set'], r['seconds']), []).append(r)
    for key, items in sorted(groups.items()):
        values = [float(r['controller_evaluation_seconds'])/int(r['controller_evaluation_samples'])*1e6
                  for r in items if r['controller_evaluation_seconds'] and int(r['controller_evaluation_samples']) > 0]
        timing.append(dict(zip(('seed', 'arm', 'ic_set', 'seconds'), key), n=len(items), n_observed=len(values),
            median_per_case_mean_microseconds=float(np.median(values)) if values else None,
            q95_per_case_mean_microseconds=float(np.quantile(values, .95)) if values else None))
    csv_write(out/'feedback_timing.csv', timing)
    dt = runner.make_config('dev').dt
    Ad, Bd = nominal_linearization(dt)
    verification = checks(dt)
    stability, law_errors = [], []
    for seed in range(3):
        d = json.loads((out/'models'/f'nominal_seed{seed}.json').read_text())
        A, B, C = np.asarray(d['A']), np.asarray(d['B']), np.asarray(d['C'])
        for arm in ('SMC', 'SMC_linear'):
            law = SlidingFeedback(A, C, d['calibration']['phi'], correction=arm == 'SMC', rho=d['rho'])
            gain = C@A-(law.rho-law.beta/law.phi)*C
            numerical = np.array([(law.command(1e-6*axis)[0][0]-law.command(-1e-6*axis)[0][0])/(2e-6)
                                  for axis in np.eye(4)]).reshape(1, 4)
            error = float(np.max(np.abs(numerical+gain)))
            assert error < 1e-8
            law_errors.append(error)
            stability.append(dict(seed=seed, arm=arm,
                fitted_closed_loop_radius=spectral_radius(A-B@gain),
                true_closed_loop_radius=spectral_radius(Ad-Bd@gain),
                phi=d['calibration']['phi'], validation_samples=d['calibration']['validation_samples']))
    csv_write(out/'feedback_local_stability.csv', stability)
    dump(out/'feedback_local_stability_checks.json', dict(
        plant_linearization_checks=verification, law_derivative_max_errors=law_errors,
        interpretation='At the nominal upright origin, inside the boundary and before saturation; no finite-IC or noisy-plant guarantee'))
    elapsed, cpu = time.perf_counter()-start, time.process_time()-cpu_start
    manifest['comparison_postprocess_wall_seconds'] = manifest.get('comparison_postprocess_wall_seconds', 0.)+elapsed
    manifest['wall_seconds'] += elapsed
    manifest['cpu_seconds'] += cpu
    dump(out/'manifest.json', manifest)
    print(stability, flush=True)


if __name__ == '__main__':
    main()
