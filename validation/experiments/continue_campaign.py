"""Detached sequential continuation after the reviewed E0+E1 checkpoint."""
import csv
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'validation/results_v1'
PYTHON=ROOT/'validation/.venv-gpu/bin/python'
ENV=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1',
         LD_LIBRARY_PATH=str(PYTHON.parent.parent/'lib'),LD_PRELOAD=str(PYTHON.parent.parent/'lib/libstdc++.so.6'))


def remaining():
    elapsed=sum(json.loads(p.read_text()).get('wall_seconds',0.) for p in OUT.glob('*/manifest.json'))
    elapsed+=json.loads((OUT/'E0/authored_manifest.json').read_text()).get('wall_seconds',0.)
    return 40*3600-elapsed


def note(message):
    with (OUT/'RESULTS.md').open('a') as f:f.write('\n'+time.strftime('%Y-%m-%d')+': '+message+'\n')
    print(message,flush=True)


def report(experiment):
    rows=list(csv.DictReader((OUT/experiment/'rates.csv').open()))
    groups={}
    for r in rows:groups.setdefault((r.get('episodes','3500'),r['ic_set'],r['seconds'],r['arm']),[]).append(r)
    body=f'\n<!-- {experiment}_AUTO_START -->\n\n### {experiment} completed measurements\n\n'
    body+=f'Sources: [per-seed counts and Wilson intervals]({experiment}/rates.csv), [matched comparisons]({experiment}/paired_comparisons.csv), [case metrics]({experiment}/summary.csv), [figure]({experiment}/success_rates.png), [manifest]({experiment}/manifest.json). Raw-array paths are recorded in the case metrics.\n\n'
    body+='Counts below are ranges across whole-run seeds; seeds are not pooled. The common cart-pole endpoint remains angle stabilization with no fall. Novel-plant endpoints are stated in their manifests.\n\n'
    body+='| Episodes | IC condition | Seconds | Arm | Success-count range / n |\n|---|---|---|---|---|\n'
    for (episodes,ic,seconds,arm),items in sorted(groups.items()):
        counts=[int(r['successes']) for r in items];ns={int(r['n']) for r in items}
        n=str(next(iter(ns))) if len(ns)==1 else ','.join(map(str,sorted(ns)))
        span=str(min(counts)) if min(counts)==max(counts) else f'{min(counts)}–{max(counts)}'
        body+=f'| {episodes} | {ic} | {seconds} | {arm} | {span}/{n} |\n'
    body+=f'\n<!-- {experiment}_AUTO_END -->\n'
    path=OUT/'RESULTS.md';text=path.read_text();first=f'<!-- {experiment}_AUTO_START -->';last=f'<!-- {experiment}_AUTO_END -->'
    if first in text:text=text[:text.index(first)]+body.strip()+text[text.index(last)+len(last):]
    else:
        anchor='## Protocol and resume' if experiment=='E4' else '## Q2 — Errors in the implementation note'
        text=text.replace(anchor,body+'\n'+anchor,1)
    path.write_text(text)


def committed(experiment):
    result=subprocess.run(['git','show',f'HEAD:validation/results_v1/{experiment}/manifest.json'],cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,universal_newlines=True)
    return result.returncode==0 and json.loads(result.stdout).get('status')=='DONE'


def commit(experiment):
    paths=[str((OUT/experiment).relative_to(ROOT)),'validation/results_v1/RESULTS.md','validation/results_v1/STATUS',
           'validation/results_v1/CONTINUE_JOB','validation/experiments/continue_campaign.py','validation/experiments/figures.py']
    if experiment=='E2':paths.append('validation/results_v1/models')
    subprocess.run(['git','add',*paths],cwd=ROOT,check=True)
    figures=sorted(str(p.relative_to(ROOT)) for p in (OUT/experiment).glob('*.png'))
    if figures:subprocess.run(['git','add','-f',*figures],cwd=ROOT,check=True)
    if subprocess.run(['git','diff','--cached','--quiet'],cwd=ROOT).returncode:
        subprocess.run(['git','commit','-m',f'Complete {experiment} validation measurements'],cwd=ROOT,check=True)


def finish(experiment,pid,estimate,started,child=None):
    path=OUT/experiment/'manifest.json'
    last_projection=0.
    while True:
        alive=child.poll() is None if child is not None else Path(f'/proc/{pid}').exists()
        value=json.loads(path.read_text()) if path.exists() else {}
        if not alive:
            if value.get('status')!='DONE':raise RuntimeError(f'{experiment} stopped without DONE; inspect JOB/run.log and resume saved cases')
            break
        # Estimate remaining work from observed per-arm costs, avoiding the bias
        # from cheap L cases finishing before CEM cases. Long cases count as 60/14.
        now=time.time();summary=OUT/experiment/'summary.csv'
        if now-last_projection>=60 and summary.exists():
            rows=list(csv.DictReader(summary.open()));observed={}
            for row in rows:
                count,cost=observed.get(row['arm'],(0.,0.))
                observed[row['arm']]=(count+float(row['seconds'])/14,float(row['wall_seconds'])+cost)
            config=value.get('config',{});arms=config.get('arms',[])
            target=1680 if experiment=='E2' else 560 if experiment=='E2_distribution' else 1848 if experiment=='E3' else 3*(56+3*60/14)
            if arms and all(observed.get(arm,(0,0))[0]>=10 for arm in arms):
                projected=now-started+sum(max(0,target-observed[arm][0])*observed[arm][1]/observed[arm][0] for arm in arms)/8
                print(f'{experiment}: {len(rows)} cases saved; projected wall {projected/3600:.2f} h; initial estimate {estimate/3600:.2f} h',flush=True)
                if projected>3*estimate:
                    os.killpg(pid,signal.SIGTERM)
                    raise RuntimeError(f'{experiment} projected {projected/3600:.2f} h exceeds three times its initial estimate')
            last_projection=now
        # The timeout also enforces this bound if the continuation is interrupted.
        if now-started>3*estimate:
            os.killpg(pid,signal.SIGTERM)
            raise RuntimeError(f'{experiment} exceeded three times its initial estimate')
        time.sleep(30)
    if experiment=='E2' and not value.get('postprocess_wall_seconds'):
        # Fresh import includes the per-budget/per-seed distribution correction.
        start=time.perf_counter()
        with (OUT/experiment/'run.log').open('a') as log:
            subprocess.run([str(PYTHON),str(ROOT/'validation/experiments/figures.py'),experiment],cwd=ROOT,env=ENV,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=max(1,int(remaining())))
        value['postprocess_wall_seconds']=time.perf_counter()-start
        value['wall_seconds']+=value['postprocess_wall_seconds']
        path.write_text(json.dumps(value,indent=2)+'\n')
    report(experiment)
    (OUT/'STATUS').write_text(f'RUNNING\n{experiment} complete; summaries and figures saved.\nNext: remaining E2–E4 stages, then final interpretation.\n')
    commit(experiment)


def launch(experiment,args,estimate):
    path=OUT/experiment/'manifest.json'
    if path.exists() and json.loads(path.read_text()).get('status')=='DONE':
        if not committed(experiment):report(experiment);commit(experiment)
        return
    job=OUT/experiment/'JOB'
    if job.exists():
        lines=job.read_text().splitlines()
        pid=int(next(x.split()[1] for x in reversed(lines) if x.startswith('PID ')))
        if Path(f'/proc/{pid}').exists():
            original=float(next(x.split()[1] for x in reversed(lines) if x.startswith('ESTIMATED_WALL_SECONDS ')))
            print(f'Reattaching {experiment} PID {pid}; no duplicate launch.',flush=True)
            finish(experiment,pid,original,job.stat().st_mtime)
            return
    available=remaining()
    if estimate>available:raise RuntimeError(f'{experiment} estimate {estimate/3600:.2f} h exceeds remaining compute cap {available/3600:.2f} h; reduce scope at resume')
    out=OUT/experiment;out.mkdir(exist_ok=True)
    command=['timeout',str(int(min(available,3*estimate))),str(PYTHON),str(ROOT/'validation/experiments'/args[0]),*args[1:]]
    with (out/'run.log').open('a') as log:child=subprocess.Popen(command,cwd=ROOT,env=ENV,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    (out/'JOB').write_text(f'PID {child.pid}\nCOMMAND '+ ' '.join(shlex.quote(x) for x in command)+f'\nESTIMATED_WALL_SECONDS {estimate:.1f}\nTIMEOUT_SECONDS {command[1]}\nNOTE Detached, saved-case resume; estimate based on E1 mean case costs plus model preparation and stated plant/baseline multipliers.\n')
    (OUT/'STATUS').write_text(f'RUNNING\n{experiment} detached; saved cases are resumable.\nNext: finish remaining stages and interpret results.\n')
    finish(experiment,child.pid,estimate,time.time(),child)


def main(e2_pid):
    job=(OUT/'E2/JOB').read_text().splitlines()
    estimate=float(next(x.split()[1] for x in job if x.startswith('ESTIMATED_WALL_SECONDS')))
    started=(OUT/'E2/JOB').stat().st_mtime
    if not committed('E2'):finish('E2',e2_pid,estimate,started)
    rows=list(csv.DictReader((OUT/'E1/summary.csv').open()))
    costs={arm:sum(float(r['wall_seconds']) for r in rows if r['arm']==arm and r['seconds']=='14.0' and r['ic_set']=='C')/600 for arm in ('H','L','M1')}
    total=sum(costs.values())
    launch('E2_distribution',['e2_distribution.py'],560*total/8+900)
    launch('E3',['e3_robustness.py'],1848*(costs['H']+costs['L'])/8+900)
    effective=3*(56+3*60/14)
    launch('E3_linear',['e3_novel.py','linear'],effective*total/8+900)
    # Nonlinear integration is more expensive; reserve the mandatory DeePC comparison.
    e4_estimate=effective*(costs['H']+costs['L']+6*costs['M1']+6*35)/8+1800
    acrobot_estimate=2*effective*total/8+1800
    if remaining()<acrobot_estimate+e4_estimate:
        note('Deviation: omit acrobot before cutting optional E5, to preserve matched DeePC within the 40-hour cap (handoff cut order).')
    else:launch('E3_acrobot',['e3_novel.py','acrobot'],acrobot_estimate)
    args=['e4_deepc.py']
    if e4_estimate>remaining():
        args+=['--skip-ensemble'];e4_estimate-=effective*5*costs['M1']/8
        note('Deviation: optional E5 omitted to preserve mandatory matched DeePC within the compute cap.')
    launch('E4',args,e4_estimate)
    (OUT/'STATUS').write_text('RUNNING\nDetached E2–E4 measurements completed and committed.\nNext: inspect figures, synthesize Q1–Q3, and write the final verdict before DONE.\n')
    print('Measurements complete; final interpretation remains.',flush=True)

if __name__=='__main__':
    try:main(int(sys.argv[1]))
    except BaseException as error:
        note(f'Detached continuation stopped: {error!r}. Saved cases remain resumable; inspect JOB files on the next turn.')
        (OUT/'STATUS').write_text('RUNNING\nDetached continuation needs inspection; saved cases remain resumable.\nNext: inspect JOB/run.log, resolve the stopped stage, and continue.\n')
        raise
