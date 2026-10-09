"""Detached continuation from E0 to E1 and the required user checkpoint."""
import csv
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'validation/results_v1'


def wait_finished(experiment,pid):
    path=OUT/experiment/'manifest.json'
    while True:
        if path.exists():
            manifest=json.loads(path.read_text())
            if manifest['status']=='DONE':return manifest
            if manifest['status']=='ERROR':raise RuntimeError(f'{experiment}: '+manifest.get('error','error'))
        try:os.kill(pid,0)
        except ProcessLookupError:raise RuntimeError(f'{experiment} PID {pid} stopped; inspect JOB and resume saved cases')
        time.sleep(30)


def save_report(experiment):
    with (OUT/experiment/'rates.csv').open() as f:rates=list(csv.DictReader(f))
    report=OUT/'RESULTS.md'
    text=report.read_text()
    first=f'<!-- {experiment}_AUTO_START -->';last=f'<!-- {experiment}_AUTO_END -->'
    body='\n'+first+f'\n\n### {experiment} measured results\n\n'
    body+=f'Sources: [{experiment} rates]({experiment}/rates.csv), [matched comparisons]({experiment}/paired_comparisons.csv), [case metrics]({experiment}/summary.csv), [distributions]({experiment}/distributions.csv), [success figure]({experiment}/success_rates.png), [manifest]({experiment}/manifest.json).\n\n'
    body+='| Seed | IC set | Arm | Successes/n (14 s, no fall) | Wilson 95% CI |\n|---|---|---|---|---|\n'
    for r in rates:
        if float(r['seconds'])!=14:continue
        body+=f"| {r['seed']} | {r['ic_set']} | {r['arm']} | {r['successes']}/{r['n']} | {float(r['wilson95_low']):.3f}–{float(r['wilson95_high']):.3f} |\n"
    body+='\nSettling, excursion, input saturation and local fractions are distributions, not extra acceptance criteria. The 60 s runs remain separate in the CSVs.\n'
    if experiment=='E0':
        by_arm={}
        for r in rates:
            if float(r['seconds'])==14:by_arm.setdefault(r['arm'],[]).append(int(r['successes']))
        body+='\nAcross the ten whole-run seeds, authored-set primary successes per six cases: '+', '.join(f"{arm} {min(v)}–{max(v)}/6" for arm,v in sorted(by_arm.items()))+'. These are repeated fixed ICs; the per-seed counts, not a pooled population interval, are the reproducibility evidence.\n'
    body+='\n'+last+'\n'
    if first in text:text=text[:text.index(first)]+body.strip()+text[text.index(last)+len(last):]
    else:text+=body
    report.write_text(text)


def commit(experiment):
    paths=[f'validation/results_v1/{experiment}','validation/results_v1/models','validation/results_v1/RESULTS.md','validation/results_v1/STATUS','validation/results_v1/.gitignore','validation/experiments']
    subprocess.run(['git','add',*paths],cwd=ROOT,check=True)
    subprocess.run(['git','commit','-m',f'Complete {experiment} matched controller validation'],cwd=ROOT,check=True)


def main(e0_pid):
    wait_finished('E0',e0_pid)
    save_report('E0');commit('E0')
    with (OUT/'E0/summary.csv').open() as f:rows=list(csv.DictReader(f))
    # Independent case wall times include contention. Divide by eight workers to
    # project elapsed device occupancy; H/H* use maximum 14 s observed case cost.
    costs={arm:max(float(r['wall_seconds']) for r in rows if r['arm']==arm and float(r['seconds'])==14) for arm in ('H','L','M','M1','Hstar')}
    estimate=sum(costs.values())*3*(171+200+1681)/8
    estimate+=sum(costs.values())*3*9*(60/14)/8
    prior=sum(json.loads(p.read_text()).get('wall_seconds',0) for p in OUT.glob('*/manifest.json'))
    prior+=json.loads((OUT/'E0/authored_manifest.json').read_text()).get('wall_seconds',0)
    remaining=40*3600-prior
    if estimate>remaining:
        with (OUT/'RESULTS.md').open('a') as f:f.write(f'\n2026-10-09 deviation: full E1 projected {estimate/3600:.2f} h exceeds the remaining cap {remaining/3600:.2f} h. No E1 job launched; reduce scope at resume.\n')
        return
    e1=OUT/'E1';e1.mkdir(exist_ok=True)
    cmd=[str(ROOT/'validation/.venv-gpu/bin/python'),str(ROOT/'validation/experiments/parallel_campaign.py'),'E1','--workers','8']
    timeout=min(int(3*estimate),int(remaining))
    cmd=['timeout',str(timeout),*cmd]
    with (e1/'run.log').open('a') as log:
        process=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
    with (e1/'JOB').open('a') as f:f.write(f'PID {process.pid}\nCOMMAND {shlex.join(cmd)}\nESTIMATED_WALL_SECONDS {estimate:.1f}\nTIMEOUT_SECONDS {timeout}\n')
    (OUT/'STATUS').write_text('RUNNING\nE0 completed; E1 matched IC-B/C/D and 60 s subsets are detached.\nNext: E0+E1 user checkpoint, then E2 data efficiency.\n')
    manifest=wait_finished('E1',process.pid)
    save_report('E1')
    (OUT/'STATUS').write_text('CHECKPOINT_USER\nE0 and E1 completed; see RESULTS.md for reproduction, L attribution and operating-region maps.\nNext: report this checkpoint, then resume E2–E4 under the superseding handoff.\n')
    commit('E1')
    print('E0 and E1 complete; CHECKPOINT_USER written.',flush=True)

if __name__=='__main__':
    try:main(int(sys.argv[1]))
    except BaseException as error:
        with (OUT/'RESULTS.md').open('a') as f:f.write(f'\nDetached continuation needs inspection: {error!r}. Saved cases remain resumable.\n')
        raise
