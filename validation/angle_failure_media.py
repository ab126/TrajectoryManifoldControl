"""Add saved angle-nonsettling plots and animations to the existing notebook.

Reads archived trajectories only. No plant transitions, controllers or training.
The full dashboard archive is not regenerated. Re-running replaces this section.
"""
import base64
import html
import json
from pathlib import Path

import numpy as np

from trajectory_media import HERE, RUN_NAME, animation_document, animation_iframe, digest, load_case
import matplotlib.pyplot as plt


def angle_summaries(run_dir, rows):
    summaries = []
    for case, row in sorted(rows.items()):
        data = load_case(run_dir, case)
        angle, omega = data['physical'][-151:, 2], data['physical'][-151:, 3]
        angle_failed = bool(np.any(np.abs(angle) > .02))
        omega_failed = bool(np.any(np.abs(omega) > .05))
        assert angle_failed == ('terminal_angle' in row['failure_flags'])
        assert omega_failed == ('terminal_angular_speed' in row['failure_flags'])
        summaries.append({'case': case, 'angle_failed': angle_failed, 'omega_failed': omega_failed,
                          'angle_violation_ever': 'angle' in row['failure_flags'],
                          'tail_angle_rms_rad': float(np.sqrt(np.mean(angle ** 2))),
                          'tail_max_abs_angle_rad': float(np.max(np.abs(angle))),
                          'tail_max_abs_omega_rad_s': float(np.max(np.abs(omega))),
                          'final_angle_rad': float(angle[-1])})
    return summaries


def select_examples(summaries):
    failed = [row for row in summaries if row['angle_failed']]
    rank = lambda row: (row['tail_angle_rms_rad'], -row['case'])
    first = max((row for row in failed if not row['angle_violation_ever']), key=rank)
    second = max((row for row in failed if abs(row['final_angle_rad']) <= .02), key=rank)
    third = max((row for row in failed if row['angle_violation_ever'] and row['case'] not in
                 (first['case'], second['case'])), key=rank)
    return [
        (first, 'Persistent oscillation without crossing the fall limit',
         'Largest final-window angle RMS among angle failures that never crossed the ±π/2 limit.'),
        (second, 'Ends near zero, but does not settle',
         'Largest final-window angle RMS among angle failures whose final sample is inside ±0.02 rad.'),
        (third, 'Still oscillating after an earlier fall',
         'Largest final-window angle RMS among remaining angle failures that crossed the ±π/2 limit.'),
    ]


def decorate(axis, ylabel, limit=None, final_only=False):
    axis.set_xlabel('Physical time (s)')
    axis.set_ylabel(ylabel)
    axis.set_xlim(17 if final_only else 0, 20)
    axis.axhline(0, color='black', lw=.7)
    axis.axvspan(17, 20, color='gray', alpha=.10)
    if limit is not None:
        axis.axhspan(-limit, limit, color='#18854b', alpha=.16)
        for sign in (-1, 1):
            axis.axhline(sign * limit, color='#18854b', ls='--', lw=.9)
    axis.grid(alpha=.25)


def case_figure(data, case):
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    time, state = data['time_s'], data['physical']
    axes[0, 0].plot(time, state[:, 2], lw=1)
    decorate(axes[0, 0], 'Raw angle θ (rad)', .02)
    axes[0, 0].set_title('Full recovery; shaded interval is the settling window')
    axes[0, 1].plot(time[-151:], state[-151:, 2], lw=1.2)
    decorate(axes[0, 1], 'Raw angle θ (rad)', .02, True)
    axes[0, 1].set_title('17–20 s: angle must remain within the green band')
    axes[1, 0].plot(time[-151:], state[-151:, 3], lw=1.2)
    decorate(axes[1, 0], 'Angular speed ω (rad/s)', .05, True)
    axes[1, 0].set_title('Angular-speed settling is a separate requirement')
    axes[1, 1].step(data['control_time_s'], data['force_N'], where='post', lw=.7)
    decorate(axes[1, 1], 'Applied force (N)')
    for bound in (-98.1, 98.1):
        axes[1, 1].axhline(bound, color='#ad2442', ls='--', lw=.8)
    axes[1, 1].set_title('Original bounded control; no new controller run')
    fig.suptitle(f'Case {case:03d}: pendulum angle does not satisfy final-window settling\n'
                 'Original saved true-model Gate A trajectory; raw angle is not wrapped')
    return fig


def main():
    run_dir = HERE / 'overnight_runs' / RUN_NAME
    output = HERE / 'figures' / RUN_NAME / 'angle_nonsettling'
    output.mkdir(exist_ok=True)
    audit = json.loads((run_dir / 'calibration/independent_audit.json').read_text())
    assert audit['audit_passed'] and audit['calibration_complete']
    for name, expected in json.loads((run_dir / 'checksums.json').read_text()).items():
        assert digest(run_dir / name) == expected, name
    rows = {row['case']: row for row in audit['case_diagnostics']}
    summaries = angle_summaries(run_dir, rows)
    failed = [row for row in summaries if row['angle_failed']]
    omega_only = [row['case'] for row in summaries if not row['angle_failed'] and row['omega_failed']]
    assert len(failed) == 52 and omega_only == [51]
    examples = select_examples(summaries)
    colors = ('#1768ac', '#a04592', '#d08012')
    fig, axes = plt.subplots(1, 2, figsize=(13, 4), constrained_layout=True)
    for row in failed:
        data = load_case(run_dir, row['case'])
        for ax in axes:
            ax.plot(data['time_s'], data['physical'][:, 2], color='gray', alpha=.18, lw=.6)
    for (summary, title, reason), color in zip(examples, colors):
        data = load_case(run_dir, summary['case'])
        for ax in axes:
            ax.plot(data['time_s'], data['physical'][:, 2], color=color, lw=1.2,
                    label=f'Case {summary["case"]:03d}')
    for index, ax in enumerate(axes):
        decorate(ax, 'Raw angle θ (rad)', .02, index == 1)
        if index == 1:
            extent = max(row['tail_max_abs_angle_rad'] for row in failed)
            ax.set_ylim(-1.08 * extent, 1.08 * extent)
        ax.legend(fontsize=8)
    axes[0].set_title('All 52 angle-settling failures: full 20 s')
    axes[1].set_title('Final 3 s: ±0.02 rad target highlighted')
    fig.savefig(output / 'cohort.png', dpi=140)
    fig.savefig(output / 'cohort.pdf')
    plt.close(fig)

    cells = []
    def markdown(identifier, text):
        cells.append({'cell_type': 'markdown', 'id': identifier, 'metadata': {}, 'source': text.splitlines(True)})
    def code(identifier, text, mime, content):
        cells.append({'cell_type': 'code', 'id': identifier, 'metadata': {}, 'source': text.splitlines(True),
                      'execution_count': None, 'outputs': [{'output_type': 'display_data', 'metadata': {},
                                                          'data': {mime: content}}]})
    def image(identifier, path):
        code(identifier, f'display(Image(filename=str(figures / {str(path.relative_to(output.parent))!r})))\n',
             'image/png', base64.b64encode(path.read_bytes()).decode())

    markdown('anglefail-intro', '## Cases where the angle does not settle near zero\n\n'
             '**52/200** trajectories leave the **±0.02 rad (±1.15°)** angle tolerance during the final '
             '**17–20 s** window. An angle near zero at the last sample alone is insufficient. '
             'There are **53/200** combined angle/angular-speed failures: case **051** passes the angle '
             'band but fails the separate ±0.05 rad/s speed band. Earlier rail/fall failures and cart '
             'recentering are different criteria. These are original true-model development traces, '
             'not fully trained neural-controller results.\n\n'
             'Gray curves show all 52 angle failures; colored examples use the recorded selection rules '
             'below (ties: lowest case ID). Angles are **raw, unwrapped radians**. Animations open at '
             '**17 s** to show persistent motion; use the slider or Reset to inspect the whole recovery.\n\n'
             f'Angle-failure case IDs: `{[row["case"] for row in failed]}`.\n')
    image('anglefail-cohort', output / 'cohort.png')
    exported = []
    for summary, title, reason in examples:
        case = summary['case']
        data = load_case(run_dir, case)
        fig = case_figure(data, case)
        png, pdf = output / f'case_{case:03d}.png', output / f'case_{case:03d}.pdf'
        fig.savefig(png, dpi=140)
        fig.savefig(pdf)
        plt.close(fig)
        document = animation_document(data, rows[case], title, start_sample=850)
        (output / f'case_{case:03d}.html').write_text(document)
        markdown(f'anglefail-case-{case}', f'### Case {case:03d}: {title}\n\n{reason}\n\n'
                 f'Final-window max |θ| = **{summary["tail_max_abs_angle_rad"]:.6f} rad**; '
                 f'angle RMS = **{summary["tail_angle_rms_rad"]:.6f} rad**; '
                 f'final θ = **{summary["final_angle_rad"]:.6f} rad**. '
                 f'Max |ω| in the window = **{summary["tail_max_abs_omega_rad_s"]:.6f} rad/s**.\n'
                 f'Full failure flags: `{rows[case]["failure_flags"]}`.\n')
        image(f'anglefail-plot-{case}', png)
        code(f'anglefail-animation-{case}', f'case = {case}\n'
             f'document = animation_document(load_case(run_dir, case), rows[case], {title!r}, start_sample=850)\n'
             'display(HTML(animation_iframe(document)))\n', 'text/html', animation_iframe(document))
        exported.append({**summary, 'title': title, 'selection_rule': reason})

    path = HERE / 'notebooks/gate_a_examples.ipynb'
    notebook = json.loads(path.read_text())
    notebook['cells'] = [cell for cell in notebook['cells'] if not cell['id'].startswith('anglefail-')]
    index = next(i for i, cell in enumerate(notebook['cells']) if cell['id'] == 'cell028')
    notebook['cells'][index:index] = cells
    for cell in notebook['cells']:
        if cell['id'] == 'localdiag-summary':
            cell['source'] = ''.join(cell['source']).replace('Early safety/search failures still need the GPU diagnostic.',
                'The CUDA diagnosis is now complete; see diagnostics/cem_findings.md. Early safety/search failures remain unresolved.').splitlines(True)
        if cell['id'] == 'cell030':
            cell['source'] = ''.join(cell['source']).replace('Continue only with bounded oracle diagnosis before another calibration;',
                'Both bounded diagnostic phases are complete; specify a versioned oracle amendment before another calibration;').splitlines(True)
    notebook['metadata']['trajectory_validation']['angle_nonsettling_section'] = {
        'angle_failure_cases': 52, 'combined_angular_failure_cases': 53, 'omega_only_case': 51,
        'examples': [row['case'] for row in exported], 'animation_start_s': 17,
        'new_experimental_jobs': 0}
    path.write_text(json.dumps(notebook, indent=1) + '\n')
    links = ''.join(f'<li>Case {row["case"]:03d}: {html.escape(row["title"])} — '
                    f'<a href="case_{row["case"]:03d}.html">Animation (starts at 17 s)</a> · '
                    f'<a href="case_{row["case"]:03d}.png">Angle plots</a></li>' for row in exported)
    (output / 'index.html').write_text('<!doctype html><meta charset="utf-8">'
        '<title>Angle nonsettling</title><style>body{font:15px system-ui;max-width:1100px;margin:25px auto}'
        'img{width:100%}li{margin:12px 0}</style><h1>Angle does not settle near zero</h1>'
        '<p>52/200 original saved Gate A cases leave the ±0.02 rad angle band at least once during 17–20 s. '
        '53 fail angle or angular speed; case 051 is angular-speed-only. These are development '
        'true-model traces, not new controller runs.</p><img src="cohort.png"><ul>' + links + '</ul>')
    metadata = {'scope': 'saved-data visualization only', 'source_run': RUN_NAME,
                'source_code_sha256': digest(__file__), 'angle_failure_cases': 52,
                'combined_angular_failure_cases': 53, 'omega_only_case': 51,
                'examples': exported, 'all_angle_failures': failed,
                'notebook_sha256': digest(path), 'new_experimental_jobs': 0,
                'checksums': {p.name: digest(p) for p in output.iterdir() if p.is_file() and p.name != 'manifest.json'}}
    (output / 'manifest.json').write_text(json.dumps(metadata, indent=2) + '\n')
    archive = output.parent
    archive_metadata = json.loads((archive / 'manifest.json').read_text())
    archive_metadata['notebook_SHA256'] = digest(path)
    archive_metadata['angle_nonsettling'] = {'manifest': 'angle_nonsettling/manifest.json',
                                           'angle_failure_cases': 52, 'examples': [row['case'] for row in exported]}
    (archive / 'manifest.json').write_text(json.dumps(archive_metadata, indent=2) + '\n')
    index = (archive / 'index.html').read_text()
    link = '<h2>Angle nonsettling</h2><p><a href="angle_nonsettling/index.html">52 angle failures: zoom plots and three animations starting at 17 s</a>.</p>'
    if link not in index:
        (archive / 'index.html').write_text(index + link)
    readme = (archive / 'README.md').read_text()
    readme = readme.replace('and eight offline animations', 'and eleven offline animations (eight original, three angle-nonsettling)')
    if 'angle_nonsettling/index.html' not in readme:
        readme += '\nAngle-nonsettling section: angle_nonsettling/index.html; 52 angle failures, three selected examples with final-window zoom plots and animations starting at 17 s.\n'
    (archive / 'README.md').write_text(readme)
    hashes = {str(p.relative_to(archive)): digest(p) for p in sorted(archive.rglob('*'))
              if p.is_file() and p.name != 'checksums.json'}
    (archive / 'checksums.json').write_text(json.dumps(hashes, indent=2) + '\n')
    from notebooks.export_source import main as export_notebook_source
    export_notebook_source()
    print(json.dumps({'angle_failure_cases': 52, 'examples': [row['case'] for row in exported],
                      'new_cells': len(cells), 'output': str(output), 'new_experimental_jobs': 0}))


if __name__ == '__main__':
    main()
