"""Saved Gate A trajectories: physical figures, offline animations and notebook.

No transition models, controller solves or new acquisition. Animation geometry
uses the submitted full-length uniform rod, theta measured rightward from upright.
"""
import argparse
import base64
from collections import Counter
import csv
import hashlib
import html
import json
import os
from pathlib import Path
import resource
import time

os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).parent / 'artifacts/mpl'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
RUN_NAME = '20261007T054549.695689Z'
SCALE = np.array([1., np.sqrt(9.81), 1., np.sqrt(9.81)])
LIMITS = np.array([.1, .05, .02, .05])
COLORS = {'plan': '#1768ac', 'queue': '#b07913', 'local': '#18854b', 'nonfinite': '#ad2442'}


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as file:
        for block in iter(lambda: file.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def load_case(run_dir, case):
    with np.load(Path(run_dir) / 'calibration' / f'case_{case:03d}.npz', allow_pickle=False) as trace:
        data = {key: trace[key].copy() for key in trace.files}
    X, U = data['X'], data['U'][:, 0]
    if X.shape != (1001, 4) or U.shape != (1000,):
        raise ValueError('Expected a complete 20-second saved trajectory')
    Y = X * SCALE
    p, v, theta, omega = Y.T
    kinetic = 1.5 * v*v + .5 * np.cos(theta)*v*omega + omega*omega/6
    potential = 4.905 * np.cos(theta)
    energy = kinetic + potential
    force = 9.81 * U
    work = np.r_[0., np.cumsum(force * np.diff(p))]
    cost_components = X*X * np.array([.02, 30., 100., 10.])
    return {**data, 'normalized': X, 'physical': Y, 'force_N': force,
            'time_s': np.arange(1001)*.02, 'control_time_s': np.arange(1000)*.02,
            'kinetic_J': kinetic, 'potential_J': potential, 'energy_J': energy,
            'work_J': work, 'energy_work_residual_J': energy-energy[0]-work,
            'power_interval_W': force*np.diff(p)/.02,
            'cart_acceleration_interval_m_s2': np.diff(v)/.02,
            'angular_acceleration_interval_rad_s2': np.diff(omega)/.02,
            'tracking_cost_components': cost_components, 'tracking_cost': cost_components.sum(1),
            'input_running_cost': .03*U*U,
            'force_slew_N_s': np.r_[np.nan, np.diff(force)/.02],
            'endpoint_ratios': np.abs(Y)/LIMITS,
            'rail_margin_fraction': 1-np.abs(p)/2.4,
            'angle_margin_fraction': 1-np.abs(theta)/(np.pi/2),
            'rod_COM_xy_m': np.c_[p+.5*np.sin(theta), .5*np.cos(theta)],
            'rod_tip_xy_m': np.c_[p+np.sin(theta), np.cos(theta)]}


def category(row):
    if row['success']:
        return 'success'
    failures = row['failure_flags']
    rail, angle = 'rail' in failures, 'angle' in failures
    if rail or angle:
        return 'failure/safety/' + ('rail_and_angle' if rail and angle else 'rail' if rail else 'angle')
    return 'failure/settling/' + ('position_only' if failures == ['terminal_position'] else 'position_and_speed')


def exemplars(rows, run_dir):
    examples = []
    successes = [row for row in rows if row['success']]
    successes.sort(key=lambda row: -float(load_case(run_dir, row['case'])['tracking_cost'][0]))
    for rank, row in enumerate(successes):
        examples.append({'case': row['case'], 'label': f'Success {rank+1}: initial tracking-cost rank {rank+1}/3',
                         'reason': 'All three observed successes are included, ranked by initial tracking cost; this is not a broad-box performance claim.'})
    both = sorted((row for row in rows if category(row) == 'failure/safety/rail_and_angle'),
                  key=lambda row: (row['first_angle_sample'], row['case']))
    choices = [
        (next(row for row in rows if row['case'] == 1), 'Safe recovery, slow recentering',
         'Case 1 passes every condition except terminal position.'),
        (min((row for row in rows if 'angle' in row['failure_flags']), key=lambda row: (row['first_angle_sample'], row['case'])),
         'Earliest angle-limit crossing', 'Minimum first angle-crossing sample, with case ID as tie-breaker.'),
        (min((row for row in rows if category(row) == 'failure/safety/rail'), key=lambda row: (row['first_rail_sample'], row['case'])),
         'Early rail violation without angle violation', 'Earliest crossing among rail-only safety failures.'),
        (both[len(both)//2], 'Rail and angle failure',
         'Median first-angle-crossing rank among combined safety failures, with case ID as tie-breaker.'),
        (min((row for row in rows if category(row) == 'failure/settling/position_and_speed'), key=lambda row: row['case']),
         'Safe but position and speed do not settle', 'Lowest case ID among safe position-and-speed failures.'),
    ]
    for row, label, reason in choices:
        if row['case'] not in [item['case'] for item in examples]:
            examples.append({'case': row['case'], 'label': label, 'reason': reason})
    return examples


def dashboard(data, row, png, pdf):
    fig, axes = plt.subplots(4, 3, figsize=(16, 12), constrained_layout=True)
    axes = axes.ravel()
    t, tu, Y = data['time_s'], data['control_time_s'], data['physical']
    labels = ['Cart position (m)', 'Cart speed (m/s)', 'Rod angle (degrees)', 'Angular speed (degrees/s)']
    for j in range(4):
        scale = 180/np.pi if j >= 2 else 1
        axes[j].plot(t, Y[:, j]*scale, lw=1.2)
        axes[j].fill_between(t, -LIMITS[j]*scale, LIMITS[j]*scale, where=t >= 17, color='#18854b', alpha=.18)
        if j in (0, 2):
            bound = 2.4 if j == 0 else 90
            for sign in (-1, 1):
                axes[j].axhline(sign*bound, color='#ad2442', ls='--', lw=.8)
        axes[j].set_title(labels[j])
    axes[4].step(tu, data['force_N'], where='post', lw=.8)
    axes[4].set_title('Applied cart force (N); right axis: normalized u')
    for bound in (-98.1, 98.1):
        axes[4].axhline(bound, color='#ad2442', ls='--', lw=.8)
    axes[4].secondary_yaxis('right', functions=(lambda f: f/9.81, lambda u: u*9.81))
    axes[5].plot(tu, data['force_slew_N_s'], lw=.7)
    axes[5].set_title('Discrete force slew (N/s); first value undefined')
    axes[6].plot(t, data['energy_J']-4.905, label='energy relative to upright')
    axes[6].plot(t, data['work_J'], label='work relative to initial state', alpha=.8)
    axes[6].set_title('Mechanical energy / applied work (J)')
    axes[6].legend(fontsize=7)
    axes[7].plot(t, data['energy_work_residual_J'], lw=.8)
    axes[7].set_title('Energy change minus held-force work (J)')
    axes[8].semilogy(t, np.maximum(data['tracking_cost'], 1e-12), label='normalized state cost')
    axes[8].axhline(.75, color='#18854b', ls='--', lw=.8, label='local switch threshold')
    axes[8].set_title('State tracking cost; not full MPC objective')
    axes[8].legend(fontsize=7)
    for j, name in enumerate(('position', 'speed', 'angle', 'angular speed')):
        axes[9].semilogy(t, np.maximum(data['endpoint_ratios'][:, j], 1e-8), label=name, lw=.8)
    axes[9].axhline(1., color='#ad2442', ls='--', lw=.8)
    axes[9].set_title('Absolute state / terminal tolerance; enforce at 17–20 s')
    axes[9].legend(fontsize=7, ncol=2)
    axes[10].plot(t, data['rail_margin_fraction'], label='rail')
    axes[10].plot(t, data['angle_margin_fraction'], label='angle')
    axes[10].axhline(0., color='#ad2442', ls='--', lw=.8)
    axes[10].set_title('Fractional safety margin; negative = violation')
    axes[10].legend(fontsize=7)
    codes = {'local': 0, 'queue': 1, 'plan': 2, 'nonfinite': 3}
    flags = data['branches']
    axes[11].step(tu[:len(flags)], [codes[flag] for flag in flags], where='post', lw=.7)
    axes[11].set_yticks(list(codes.values()), list(codes))
    latency = axes[11].twinx()
    plan_times = tu[np.isin(flags, ['plan', 'nonfinite'])]
    latency.scatter(plan_times, data['solve_seconds']*1000, color='#ad2442', s=3, alpha=.6)
    latency.set_ylabel('Solve ms', fontsize=8)
    axes[11].set_title('Controller branch / solve latency; Gate A timing exempt')
    for ax in axes:
        ax.set_xlim(0, 20)
        ax.axvspan(17, 20, color='gray', alpha=.07)
        ax.set_xlabel('Physical time (s)', fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=.2)
    status = 'STRICT SUCCESS' if row['success'] else 'FAIL: ' + ', '.join(row['failure_flags'])
    fig.suptitle(f'Gate A development case {row["case"]:03d} — {status}\nOriginal uniform-rod plant; physical units; all 20 seconds retained', fontsize=12)
    png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png, dpi=125)
    fig.savefig(pdf)
    plt.close(fig)


def animation_document(data, row, description, start_sample=0):
    if not isinstance(start_sample, int) or not 0 <= start_sample <= 1000:
        raise ValueError('Animation start sample must be an integer from 0 to 1000')
    values = np.c_[data['physical'], np.r_[data['force_N'], data['force_N'][-1]]].round(9).tolist()
    payload = json.dumps({'values': values, 'branches': data['branches'].tolist()}, separators=(',', ':'), allow_nan=False)
    title = html.escape(f'Case {row["case"]:03d}: {description}')
    outcome = 'Strict success' if row['success'] else 'Failed: ' + ', '.join(row['failure_flags'])
    template = '''<!doctype html><html><head><meta charset="utf-8"><title>__TITLE__</title>
<style>body{font:14px system-ui;margin:12px;color:#183047}canvas{width:100%;max-width:1000px;background:#f5f8fa;border:1px solid #ccd6df}button,select{padding:6px}input{width:45%}.muted{font-size:12px;color:#526475}</style></head><body>
<b>__TITLE__</b><div>__OUTCOME__ · Original saved development trajectory</div>
<p><button id="play">Play</button> <button id="reset">Reset</button> <input id="seek" type="range" min="0" max="1000" value="0"><select id="speed"><option value=".5">0.5×</option><option value="1" selected>1× physical time</option><option value="2">2×</option></select> <span id="time"></span></p>
<canvas id="scene" width="1000" height="350"></canvas><canvas id="plots" width="1000" height="260"></canvas>
<p class="muted">Rod length 1 m, COM at half length. Positive angle tips right from upright. Dashed red lines are rail limits. Scene follows the cart when it leaves the central viewport; plotted values retain the full trace. Final 3 s are shaded. Gate A timing is exempt. Select examples for explanation, not performance inference.</p>
<script>const D=__DATA__;const rows=D.values;const scene=document.getElementById('scene'),g=scene.getContext('2d');const plots=document.getElementById('plots'),q=plots.getContext('2d');const seek=document.getElementById('seek'),play=document.getElementById('play');let pos=__START_SAMPLE__,running=false,last=0;
function line(c,x1,y1,x2,y2,color,width=1){c.strokeStyle=color;c.lineWidth=width;c.beginPath();c.moveTo(x1,y1);c.lineTo(x2,y2);c.stroke()}
function draw(){const i=Math.min(1000,Math.floor(pos)),[p,v,th,w,F]=rows[i];seek.value=i;document.getElementById('time').textContent=(i*.02).toFixed(2)+' s';g.clearRect(0,0,1000,350);const cam=Math.abs(p)>1.7?p:0,scale=130,X=x=>500+(x-cam)*scale,base=180;
line(g,0,205,1000,205,'#60788a',2);for(const rail of [-2.4,2.4]){g.setLineDash([5,4]);line(g,X(rail),35,X(rail),205,'#bd314c',2);g.setLineDash([]);g.fillStyle='#bd314c';g.fillText('rail '+rail+' m',Math.max(8,Math.min(915,X(rail)+4)),30)}
g.fillStyle=Math.abs(p)>2.4?'#bd314c':'#1768ac';g.fillRect(X(p)-38,base-8,76,23);for(const dx of [-24,24]){g.beginPath();g.arc(X(p)+dx,base+19,8,0,2*Math.PI);g.fill()}
const tx=X(p+Math.sin(th)),ty=base-Math.cos(th)*scale;line(g,X(p),base,tx,ty,Math.abs(th)>Math.PI/2?'#bd314c':'#d08012',7);g.fillStyle='#183047';g.beginPath();g.arc(X(p+.5*Math.sin(th)),base-.5*Math.cos(th)*scale,5,0,2*Math.PI);g.fill();
const arrow=Math.max(-75,Math.min(75,F/98.1*75));line(g,X(p),base+45,X(p)+arrow,base+45,'#18854b',3);if(Math.abs(arrow)>1){const sign=Math.sign(arrow);line(g,X(p)+arrow,base+45,X(p)+arrow-sign*10,base+40,'#18854b',2);line(g,X(p)+arrow,base+45,X(p)+arrow-sign*10,base+50,'#18854b',2)}
g.fillStyle='#183047';g.fillText('p='+p.toFixed(3)+' m   v='+v.toFixed(3)+' m/s   theta='+(th*180/Math.PI).toFixed(2)+' deg   omega='+(w*180/Math.PI).toFixed(2)+' deg/s',10,16);g.fillText('F='+F.toFixed(2)+' N   u='+(F/9.81).toFixed(2)+'   branch='+D.branches[Math.min(i,999)],10,235);if(cam!==0)g.fillText('Following cart · full position shown below',700,235);
q.clearRect(0,0,1000,260);const specs=[[0,'p (m)',2.4,1],[2,'theta (deg)',90,180/Math.PI],[4,'F (N)',98.1,1]];for(let r=0;r<3;r++){const [col,label,bound,mult]=specs[r],top=r*85+12,bottom=top+60;const vals=rows.map(z=>z[col]*mult),lo=Math.min(-bound,...vals),hi=Math.max(bound,...vals),Y=y=>bottom-(y-lo)/(hi-lo||1)*60;const T=k=>80+k/1000*900;q.fillStyle='#e4ebdf';q.fillRect(T(850),top,135,60);for(const b of [-bound,bound])line(q,80,Y(b),980,Y(b),'#d19a9f');line(q,80,bottom,980,bottom,'#a1aeb8');q.strokeStyle='#1768ac';q.lineWidth=1;q.beginPath();vals.forEach((y,k)=>k?q.lineTo(T(k),Y(y)):q.moveTo(T(k),Y(y)));q.stroke();line(q,T(i),top,T(i),bottom,'#183047');q.fillStyle='#183047';q.fillText(label,4,top+12);q.fillText(hi.toFixed(1),4,top+28);q.fillText(lo.toFixed(1),4,bottom);q.fillText('0 s',80,bottom+12);q.fillText('20 s',945,bottom+12)}}
function tick(now){if(!running)return;if(last)pos=Math.min(1000,pos+(now-last)/1000*Number(document.getElementById('speed').value)/.02);last=now;draw();if(pos>=1000){running=false;play.textContent='Play'}else requestAnimationFrame(tick)}
play.onclick=()=>{running=!running;play.textContent=running?'Pause':'Play';if(running){if(pos>=1000)pos=0;last=0;requestAnimationFrame(tick)}};document.getElementById('reset').onclick=()=>{running=false;play.textContent='Play';pos=0;draw()};seek.oninput=()=>{pos=Number(seek.value);last=0;draw()};draw();</script></body></html>'''
    return template.replace('__TITLE__', title).replace('__OUTCOME__', html.escape(outcome)).replace('__DATA__', payload).replace('__START_SAMPLE__', str(start_sample))


def animation_iframe(document):
    encoded = base64.b64encode(document.encode()).decode()
    return f'<iframe src="data:text/html;base64,{encoded}" width="100%" height="780" style="border:0" sandbox="allow-scripts"></iframe>'


def build(run_dir):
    start = time.perf_counter()
    run_dir = Path(run_dir).resolve()
    audited = json.loads((run_dir/'calibration/independent_audit.json').read_text())
    if not audited['audit_passed'] or not audited['calibration_complete']:
        raise ValueError('Complete, audited saved trajectories required')
    for name, sha in json.loads((run_dir/'checksums.json').read_text()).items():
        if digest(run_dir/name) != sha:
            raise ValueError(f'Saved evidence changed: {name}')
    output = HERE/'figures'/run_dir.name
    output.mkdir(parents=True, exist_ok=True)
    rows = audited['case_diagnostics']
    examples = exemplars(rows, run_dir)
    inventory, curves = [], []
    for count, row in enumerate(rows, 1):
        case, group = row['case'], category(row)
        data = load_case(run_dir, case)
        relative = Path('cases')/group/f'case_{case:03d}'
        dashboard(data, row, output/relative.with_suffix('.png'), output/relative.with_suffix('.pdf'))
        (output/'data').mkdir(exist_ok=True)
        np.savez_compressed(output/'data'/f'case_{case:03d}.npz', **data)
        inventory.append({'case': case, 'success': row['success'], 'category': group,
                          'failure_flags': row['failure_flags'], 'initial_physical': data['physical'][0].tolist(),
                          'dashboard_png': str(relative.with_suffix('.png')),
                          'dashboard_pdf': str(relative.with_suffix('.pdf')),
                          'data_npz': f'data/case_{case:03d}.npz'})
        curves.append((row['success'], data['physical'], data['force_N']))
        if count % 10 == 0:
            print(f'Figures {count}/{len(rows)}; {time.perf_counter()-start:.1f}s', flush=True)
    fig, axes = plt.subplots(5, 1, figsize=(12, 13), constrained_layout=True)
    labels = ['p (m)', 'v (m/s)', 'theta (deg)', 'omega (deg/s)', 'Force (N)']
    for success in (False, True):
        for passed, states, force in curves:
            if passed != success:
                continue
            for j, ax in enumerate(axes):
                values = states[:, j]*(180/np.pi if j in (2, 3) else 1) if j < 4 else force
                ax.plot(np.arange(len(values))*.02, values, color='#18854b' if passed else '#b24452',
                        alpha=.9 if passed else .12, lw=1.5 if passed else .5)
    for j, ax in enumerate(axes):
        ax.set_ylabel(labels[j]); ax.set_xlabel('Physical time (s)'); ax.set_xlim(0, 20)
        ax.axvspan(17, 20, color='gray', alpha=.12); ax.grid(alpha=.2)
    for j, bound in [(0, 2.4), (2, 90), (4, 98.1)]:
        for sign in (-1, 1):
            axes[j].axhline(sign*bound, color='black', ls='--', lw=.7)
    fig.suptitle('All 200 Gate A trajectories — green: 3 strict successes; red: 197 failures\nFull ranges retained; development calibration, not confirmation')
    fig.savefig(output/'all_trajectories.png', dpi=125); fig.savefig(output/'all_trajectories.pdf'); plt.close(fig)
    animations = output/'animations'; animations.mkdir(exist_ok=True)
    for example in examples:
        row = next(row for row in rows if row['case'] == example['case'])
        path = animations/f'case_{row["case"]:03d}.html'
        path.write_text(animation_document(load_case(run_dir, row['case']), row, example['label']))
        example['animation_html'] = str(path.relative_to(output))
    metadata = {'source_run': str(run_dir), 'source_checksums_SHA256': digest(run_dir/'checksums.json'),
                'source_code_SHA256': digest(__file__), 'examples': examples, 'cases': inventory,
                'group_counts': dict(Counter(item['category'] for item in inventory)),
                'definitions': {'time_s': 'physical seconds, dt=0.02', 'physical_state_order': ['p m', 'v m/s', 'theta rad', 'omega rad/s'],
                    'normalized_action': 'u=F/(m*g), m=1 kg, g=9.81 m/s2',
                    'energy': 'uniform rod, full length 1m, mass 1kg, cart 2kg; potential zero at pivot height',
                    'work_J': 'cumulative F[k]*(p[k+1]-p[k]); exact held-force work along saved displacement',
                    'derived_not_sensor_measurements': True,
                    'interval_acceleration': 'finite interval velocity difference / 0.02 s, not instantaneous model acceleration'},
                'new_experimental_jobs': 0, 'render_wall_seconds': time.perf_counter()-start,
                'render_peak_RSS_MiB': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                'packages': {'numpy': np.__version__, 'matplotlib': matplotlib.__version__}}
    (output/'manifest.json').write_text(json.dumps(metadata, indent=2)+'\n')
    with (output/'cases.csv').open('w', newline='') as file:
        writer = csv.writer(file); writer.writerow(['case', 'success', 'category', 'failure_flags', 'png', 'pdf', 'data'])
        for item in inventory:
            writer.writerow([item['case'], item['success'], item['category'], ';'.join(item['failure_flags']), item['dashboard_png'], item['dashboard_pdf'], item['data_npz']])
    links = ''.join(f'<li><a href="{item["animation_html"]}">{html.escape(item["label"])} (case {item["case"]:03d})</a> — {html.escape(item["reason"])}</li>' for item in examples)
    table = ''.join(f'<tr><td>{item["case"]:03d}</td><td>{html.escape(item["category"])}</td><td>{html.escape(", ".join(item["failure_flags"]))}</td><td><a href="{item["dashboard_png"]}">PNG</a> · <a href="{item["dashboard_pdf"]}">PDF</a> · <a href="{item["data_npz"]}">Data</a></td></tr>' for item in inventory)
    (output/'index.html').write_text(f'<!doctype html><meta charset="utf-8"><title>Gate A trajectory archive</title><style>body{{font:15px system-ui;max-width:1200px;margin:30px auto}}td,th{{padding:7px;border-bottom:1px solid #ddd;text-align:left}}img{{width:100%}}</style><h1>Gate A: all 200 trajectories</h1><p>3 successes, 197 failures. Development calibration; timing-exempt. All examples are saved original-plant trajectories. Measurables are derived quantities, not noisy sensor observations.</p><h2>Animated examples</h2><ul>{links}</ul><h2>All trajectories</h2><a href="all_trajectories.pdf">Overview PDF</a><img src="all_trajectories.png"><h2>Per-case states, controls and derived measurables</h2><p>Green shading: terminal tolerances during 17–20 s. Red lines: safety/input limits. No trajectory values clipped.</p><table><tr><th>Case</th><th>Category</th><th>Failures</th><th>Files</th></tr>{table}</table>')
    make_notebook(output, metadata)
    (output/'README.md').write_text('Open index.html for all 200 cases and eight offline animations. Each case has a 12-panel PNG/PDF and NPZ with normalized/physical states, controls, energies, work, power, interval accelerations, rod geometry, cost components, safety margins, branches and solve times. Original source trajectories were not edited. All figures retain the complete 20 seconds. Energy/work/cost are derived, not measured sensor data. Notebook: ../../notebooks/gate_a_examples.ipynb. Its outputs are prepopulated; trust the notebook to run embedded animation scripts or open the standalone HTML examples. Re-running cells needs NumPy, Matplotlib and IPython in a notebook kernel; the private scientific environment does not contain a Jupyter kernel.\n')
    hashes = {str(path.relative_to(output)): digest(path) for path in sorted(output.rglob('*')) if path.is_file() and path.name != 'checksums.json'}
    (output/'checksums.json').write_text(json.dumps(hashes, indent=2)+'\n')
    print(json.dumps({'cases': len(inventory), 'animations': len(examples), 'output': str(output), 'wall_s': time.perf_counter()-start,
                      'peak_RSS_MiB': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024}), flush=True)


def make_notebook(output, metadata):
    cells = []
    def md(text):
        cells.append({'cell_type': 'markdown', 'id': f'cell{len(cells):03d}', 'metadata': {}, 'source': text.splitlines(True)})
    def code(text, outputs=None):
        cells.append({'cell_type': 'code', 'id': f'cell{len(cells):03d}', 'metadata': {}, 'source': text.splitlines(True),
                      'execution_count': None, 'outputs': outputs or []})
    def rich(mime, data):
        return [{'output_type': 'display_data', 'metadata': {}, 'data': {mime: data}}]
    md('# Gate A: successes and failure mechanisms\n\n200 completed development cases; **3 strict successes**, not confirmation. Each animation uses saved true-plant states, with the original uniform rod and physical units. All three successes are shown, ranked by initial tracking cost. Failures illustrate different endpoint mechanisms. No model is trained or solved by this notebook.\n\nOutputs are prepopulated from saved evidence. **Trust this notebook** for embedded JavaScript, or open the standalone HTML animations in the figure folder. Notebook regeneration needs NumPy, Matplotlib and IPython; figures/animations need no server or GPU.\n')
    code("from pathlib import Path\nimport sys, json\nfrom IPython.display import display, HTML, Image\n\n# Locate validation from a workspace, validation, or notebooks working directory.\nvalidation = next(p for root in [Path.cwd(), *Path.cwd().parents]\n                  for p in [root, root / 'TrajectoryManifoldControl' / 'validation']\n                  if (p / 'trajectory_media.py').exists())\nsys.path.insert(0, str(validation))\nfrom trajectory_media import load_case, animation_document, animation_iframe\n"+f"figures = validation / 'figures' / '{output.name}'\nrun_dir = validation / 'overnight_runs' / '{output.name}'\nmanifest = json.loads((figures / 'manifest.json').read_text())\naudit = json.loads((run_dir / 'calibration' / 'independent_audit.json').read_text())\nrows = {{r['case']: r for r in audit['case_diagnostics']}}\nassert len(rows) == 200 and audit['audit_passed']\nprint('3/200 strict successes; original-plant replay audit passed.')\n", [{'output_type': 'stream', 'name': 'stdout', 'text': ['Precomputed saved-data outputs: 3/200 strict successes; original-plant replay audit passed.\n']}])
    md('## Entire cohort\n\nRed: failures; green: successes. Full state/control ranges retained. Inspect individual figures for small settling errors. Required endpoint: rail ±2.4 m and angle ±90° throughout; final 3 seconds |p|≤0.1 m, |v|≤0.05 m/s, |θ|≤0.02 rad, |ω|≤0.05 rad/s.\n')
    code("display(Image(filename=str(figures / 'all_trajectories.png')))\n", rich('image/png', base64.b64encode((output/'all_trajectories.png').read_bytes()).decode()))
    for example in metadata['examples']:
        case = example['case']
        item = next(item for item in metadata['cases'] if item['case'] == case)
        md(f'## {example["label"]}: case {case:03d}\n\n{example["reason"]}\n\nInitial physical state (p, v, θ, ω): `{item["initial_physical"]}`. Outcome: **'+('strict success' if item['success'] else ', '.join(item['failure_flags']))+'. Play/pause, scrub time, or change playback speed.\n')
        document = (output/example['animation_html']).read_text()
        code(f"case = {case}\ndocument = animation_document(load_case(run_dir, case), rows[case], {example['label']!r})\ndisplay(HTML(animation_iframe(document)))\n", rich('text/html', animation_iframe(document)))
        code(f"display(Image(filename=str(figures / {item['dashboard_png']!r})))\n", rich('image/png', base64.b64encode((output/item['dashboard_png']).read_bytes()).decode()))
    md('## Explore any case\n\nSet CASE_ID from 0 to 199 below. All cases already have figures and derived-data NPZs. Control is held between samples; force 10 normalized = 98.1 N. Energies/work, interval accelerations and tracking cost are **derived measurables**, not sensor measurements. The state cost is not the full predictive objective. Gate A latency is exempt.\n')
    code("CASE_ID = 1\nitem = next(r for r in manifest['cases'] if r['case'] == CASE_ID)\ndisplay(HTML(animation_iframe(animation_document(load_case(run_dir, CASE_ID), rows[CASE_ID], 'Custom case'))))\ndisplay(Image(filename=str(figures / item['dashboard_png'])))\n")
    md('## Interpretation\n\n27 safe cases failed only position, 5 safe cases failed position and speed, and 165 cases violated a rail or angle limit. More search and local settling are distinct questions. The slow local pole explains a concern, not a physical unattainability proof. Continue only with bounded oracle diagnosis before another calibration; Stage 1 remains stopped. See `validation/gate_a_review.md` and the figure archive `manifest.json` for provenance and selection rules.\n')
    notebook = {'cells': cells, 'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                'language_info': {'name': 'python', 'version': '3.11'},
                'trajectory_validation': {'source_run': str(output.name), 'precomputed_saved_data_outputs': True, 'new_experimental_jobs': 0}},
                'nbformat': 4, 'nbformat_minor': 5}
    (HERE/'notebooks').mkdir(exist_ok=True)
    (HERE/'notebooks/gate_a_examples.ipynb').write_text(json.dumps(notebook, indent=1)+'\n')


def refresh_examples(run_dir):
    run_dir = Path(run_dir).resolve(); output = HERE/'figures'/run_dir.name
    metadata = json.loads((output/'manifest.json').read_text())
    for name, sha in json.loads((run_dir/'checksums.json').read_text()).items():
        if digest(run_dir/name) != sha:
            raise ValueError('Saved run evidence changed')
    rows = json.loads((run_dir/'calibration/independent_audit.json').read_text())['case_diagnostics']
    examples = exemplars(rows, run_dir)
    for example in examples:
        row = next(row for row in rows if row['case'] == example['case'])
        path = output/'animations'/f'case_{row["case"]:03d}.html'
        path.write_text(animation_document(load_case(run_dir, row['case']), row, example['label']))
        example['animation_html'] = str(path.relative_to(output))
    metadata['examples'] = examples
    metadata['render_source_code_SHA256'] = metadata.get('render_source_code_SHA256', metadata['source_code_SHA256'])
    metadata['source_code_SHA256'] = digest(__file__)
    (output/'manifest.json').write_text(json.dumps(metadata, indent=2)+'\n')
    index = (output/'index.html').read_text()
    begin, end = index.index('<ul>'), index.index('</ul>')+5
    links = ''.join(f'<li><a href="{item["animation_html"]}">{html.escape(item["label"])} (case {item["case"]:03d})</a> — {html.escape(item["reason"])}</li>' for item in examples)
    (output/'index.html').write_text(index[:begin]+'<ul>'+links+'</ul>'+index[end:])
    make_notebook(output, metadata)
    hashes = {str(path.relative_to(output)): digest(path) for path in sorted(output.rglob('*')) if path.is_file() and path.name != 'checksums.json'}
    (output/'checksums.json').write_text(json.dumps(hashes, indent=2)+'\n')
    print(json.dumps({'refreshed_animations': len(examples), 'figures_rerendered': 0}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=HERE/'overnight_runs'/RUN_NAME)
    parser.add_argument('--refresh-examples-only', action='store_true')
    args = parser.parse_args()
    refresh_examples(args.run_dir) if args.refresh_examples_only else build(args.run_dir)
