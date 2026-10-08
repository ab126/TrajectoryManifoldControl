"""Host-side CUDA readiness only, sharing the existing one-worker-hour ledger.

Never starts Gate A, tuning, confirmatory sweeps or transfer. No CPU fallback.
All measurement jobs are sequential; concurrency alone launches bounded workers.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE=Path(__file__).resolve().parent
OUT=HERE/'artifacts'

def used():
    return sum(json.loads(s)['worker_seconds'] for s in (OUT/'usage.jsonl').read_text().splitlines())

def main():
    for name in ('plant_checks.json','equivalence.json'):
        if not json.loads((OUT/name).read_text())['passed']: raise SystemExit('Correctness gate failed')
    jobs=[('gpu_probe',[]),
          ('equivalence',['--device','cuda','--cases','50','--samples','100']),
          ('smoke',['--device','cuda']),
          ('benchmark',['--device','cuda','--kind','control','--members','1']),
          ('benchmark',['--device','cuda','--kind','control','--members','5']),
          ('benchmark',['--device','cuda','--kind','training','--members','1']),
          ('benchmark',['--device','cuda','--kind','training','--members','5']),
          ('benchmark',['--device','cuda','--kind','oracle']),
          ('concurrency',['--device','cuda','--workers','1','2','4'])]
    for task,args in jobs:
        command=[sys.executable,str(HERE/'launch.py'),task,*args]
        with (OUT/'commands.log').open('a') as f:
            f.write(json.dumps({'GPU_readiness_command':command,'charged_seconds_before':used()})+'\n')
        log=OUT/('gpu_readiness_'+task+'_'+('_'.join(args).replace('/','_') or 'probe')+'.log')
        print(f'Running {task} {args}; remaining {(3600-used())/3600:.3f} worker-h',flush=True)
        with log.open('w') as output:
            result=subprocess.run(command,stdout=output,stderr=subprocess.STDOUT,env=os.environ.copy())
        if result.returncode:
            raise SystemExit(f'{task} failed ({result.returncode}); stop, inspect {log}. No downstream benchmarks launched.')
    subprocess.run([sys.executable,str(HERE/'estimate_stages.py'),'--device','cuda'],check=True)
    print('GPU readiness completed. Review device-specific artifacts before allocating Gate A; no calibration launched.')

if __name__=='__main__': main()
