"""Add local-diagnostic figures and configure the existing display-only kernel.

No controller execution. Re-run after trajectory_media.py if regenerating assets.
"""
import base64
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).parent/'artifacts/mpl'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from trajectory_media import digest, HERE, RUN_NAME


def main():
    output = HERE/'figures'/RUN_NAME
    local_dirs = [p.parent for p in (HERE/'diagnostics').glob('*/job.json')
                  if json.loads(p.read_text())['mode']=='local' and json.loads(p.read_text())['returncode']==0]
    if len(local_dirs)!=1:
        raise ValueError('Expected exactly one completed local diagnostic')
    local = local_dirs[0]
    result = json.loads((local/'results.json').read_text())
    cases = sorted(set(row['label'] for row in result['local']))
    target = output/'diagnostics/local_feedback'; target.mkdir(parents=True, exist_ok=True)
    t = np.arange(1001)*.02
    for label in cases:
        fig, axes = plt.subplots(5, 1, figsize=(11, 12), constrained_layout=True)
        for recipe, color in [('submitted_weights', '#b24452'), ('fixed_position_candidate', '#18854b')]:
            with np.load(local/f'local_{recipe}_{label}.npz') as trace:
                Y = trace['X']*[1., np.sqrt(9.81), 1., np.sqrt(9.81)]
                force = trace['U'][:, 0]*9.81
                for j in range(4):
                    axes[j].plot(t, Y[:, j], color=color, label=recipe, lw=1.)
                axes[4].step(t[:-1], force, color=color, label=recipe, lw=.8, where='post')
        for j, title in enumerate(['p (m)', 'v (m/s)', 'theta (rad)', 'omega (rad/s)', 'Force (N)']):
            axes[j].set_ylabel(title); axes[j].set_xlabel('Physical time (s)'); axes[j].set_xlim(0, 20)
            axes[j].grid(alpha=.2)
            if j<4:
                bound = [.1, .05, .02, .05][j]
                axes[j].fill_between(t, -bound, bound, where=t>=17, color='gray', alpha=.2)
        for j, bound in [(0, 2.4), (2, np.pi/2), (4, 98.1)]:
            for sign in (-1, 1): axes[j].axhline(sign*bound, ls='--', color='black', lw=.7)
        axes[0].legend()
        fig.suptitle(f'Local-only diagnostic: {label}\nRed: original Qp=.02; green: fixed oracle-only Qp=2; not a calibration or H change')
        fig.savefig(target/f'{label}.png', dpi=125); fig.savefig(target/f'{label}.pdf'); plt.close(fig)
    counts = {recipe: sum(row['strict_physical_success'] for row in result['local'] if row['recipe']==recipe)
              for recipe in ('submitted_weights', 'fixed_position_candidate')}
    usage = json.loads((local/'job.json').read_text())['worker_seconds']
    summary = f'''# Bounded local-feedback diagnostic

Original plant, known-model local feedback throughout, 20 s, seven fixed probes.
Original Qp=0.02: **{counts['submitted_weights']}/7 strict successes**.
Predeclared oracle-only Qp=2: **{counts['fixed_position_candidate']}/7**.
All 14 trajectories remained within rail and angle safety limits. The original
gain failed only position on every probe. These are selected diagnostic states,
not IID calibration outcomes; the local law was applied outside the original
switch region to isolate it. The candidate is not adopted in H or Gate A.

For p=±1 m, the original final-window maximum |p| was 0.270384 m, versus about
1.03e-7 m for the fixed candidate. This confirms the nonlinear slow-recentering
problem on those probes. Early broad-box safety/search failures remain unresolved.

Actual supervised worker usage: {usage:.6f} s ({usage/3600:.6f} h); 14000 diagnostic
plant transitions, separate from training/acquisition and the 200-case calibration.
The shared diagnostic cap is 900 s, including imports, failure and future CEM work.

GPU CEM comparison is **prepared, not executed**: 20 fixed states, five common-
prefix starts, zero/local-rollout warm starts, 200 solves. This session has no
GPU nodes. Run in the GPU-enabled host terminal:

```bash
bash TrajectoryManifoldControl/validation/run_diagnostics.sh cem --execute
```

See `validation/oracle_diagnostic_spec.md` for its fixed scope and limitations.
After reviewing that result, specify and freeze an oracle amendment before a
new calibration. No box shrink, tuning or dependent stage has been launched.
'''
    (HERE/'diagnostics/local_findings.md').write_text(summary)
    links = ''.join(f'<li><a href="{label}.png">{label}: PNG</a> · <a href="{label}.pdf">PDF</a></li>' for label in cases)
    (target/'index.html').write_text('<!doctype html><meta charset="utf-8"><h1>Local-feedback diagnosis</h1><p>Original: 0/7; fixed oracle-only position candidate: 7/7. Both safe. Selected probes, not calibration.</p><ul>'+links+'</ul>')
    index = (output/'index.html').read_text()
    addition = '<h2>New bounded local-feedback diagnosis</h2><p>Separate from the original 200 cases: <a href="diagnostics/local_feedback/index.html">seven paired trajectory figures</a>.</p>'
    if addition not in index: (output/'index.html').write_text(index+addition)
    notebook_path = HERE/'notebooks/gate_a_examples.ipynb'
    notebook = json.loads(notebook_path.read_text())
    notebook['cells'] = [cell for cell in notebook['cells'] if not cell['id'].startswith('localdiag')]
    notebook['cells'].append({'cell_type': 'markdown', 'id': 'localdiag-summary', 'metadata': {},
                             'source': ('## Additional bounded local-feedback diagnosis\n\nOriginal weights: **0/7** strict successes; fixed oracle-only Qp=2: **7/7**, both safe. Four pure-position probes and cases 1,2,9, with local feedback throughout to isolate the gain. This confirms slow nonlinear position recovery on selected probes; it is not a 200-case attainability result or an H modification. Early safety/search failures still need the GPU diagnostic.\n').splitlines(True)})
    representative = 'pure_p_+1'
    notebook['cells'].append({'cell_type': 'code', 'id': 'localdiag-figure', 'metadata': {}, 'execution_count': None,
        'source': [f"display(Image(filename=str(figures / 'diagnostics/local_feedback/{representative}.png')))\n"],
        'outputs': [{'output_type': 'display_data', 'metadata': {}, 'data': {'image/png': base64.b64encode((target/f'{representative}.png').read_bytes()).decode()}}]})
    kernel_python = Path('/home/ap60/miniconda3/envs/overseeg/bin/python')
    kernel_dir = HERE/'notebooks/kernels/trajectory-visualization'; kernel_dir.mkdir(parents=True, exist_ok=True)
    kernel = {'argv': [str(kernel_python), '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
              'display_name': 'Trajectory visuals — saved data', 'language': 'python',
              'env': {'MPLCONFIGDIR': str(HERE/'artifacts/mpl'), 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'}}
    (kernel_dir/'kernel.json').write_text(json.dumps(kernel, indent=2)+'\n')
    env_script = "import importlib.metadata as m; print('\\n'.join(sorted(d.metadata['Name']+'=='+d.version for d in m.distributions())))"
    freeze = subprocess.check_output([str(kernel_python), '-c', env_script], text=True)
    (HERE/'notebooks/display-environment-freeze.txt').write_text(freeze)
    notebook['metadata']['kernelspec'] = {'display_name': kernel['display_name'], 'language': 'python', 'name': 'trajectory-visualization'}
    notebook['metadata']['language_info']['version'] = '3.12.0'
    notebook['metadata']['trajectory_validation']['display_environment'] = 'existing overseeg Python 3.12; NumPy 2.4.6; Matplotlib 3.10.8; saved-data display only; scientific CPU/CUDA environments unchanged'
    notebook_path.write_text(json.dumps(notebook, indent=1)+'\n')
    commands = '''#!/usr/bin/env bash
set -euo pipefail
validation_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export JUPYTER_PATH="$validation_dir/notebooks${JUPYTER_PATH:+:$JUPYTER_PATH}"
export MPLCONFIGDIR="$validation_dir/artifacts/mpl"
export JUPYTER_DATA_DIR="$validation_dir/notebooks/.jupyter-data"
export JUPYTER_RUNTIME_DIR="$validation_dir/notebooks/.jupyter-runtime"
export IPYTHONDIR="$validation_dir/notebooks/.ipython"
mkdir -p "$JUPYTER_DATA_DIR" "$JUPYTER_RUNTIME_DIR"
/home/ap60/miniconda3/envs/overseeg/bin/python -m jupyter trust "$validation_dir/notebooks/gate_a_examples.ipynb"
exec /home/ap60/miniconda3/envs/overseeg/bin/python -m jupyter notebook "$validation_dir/notebooks/gate_a_examples.ipynb" "$@"
'''
    (HERE/'notebooks/open_notebook.sh').write_text(commands)
    metadata = json.loads((output/'manifest.json').read_text())
    metadata['local_diagnostic'] = {'directory': str(local), 'original_successes': counts['submitted_weights'],
                                  'candidate_successes': counts['fixed_position_candidate'], 'probes': 7,
                                  'plant_transitions': 14000, 'worker_seconds': usage, 'calibration': False}
    metadata['notebook_SHA256'] = digest(notebook_path)
    (output/'manifest.json').write_text(json.dumps(metadata, indent=2)+'\n')
    hashes = {str(path.relative_to(output)): digest(path) for path in sorted(output.rglob('*')) if path.is_file() and path.name!='checksums.json'}
    (output/'checksums.json').write_text(json.dumps(hashes, indent=2)+'\n')
    print(json.dumps({'diagnostic_comparison_figures': len(cases), 'notebook': str(notebook_path), 'local_result': counts}))


if __name__ == '__main__': main()
