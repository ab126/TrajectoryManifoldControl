"""Dedicated reproducible method environments; never activate/borrow environments."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'validation/results_literature/environments'
METHODS=['pets','deepkoopman','dfine']


def main():
    OUT.mkdir(exist_ok=True)
    assert set(os.sched_getaffinity(0))<=set(range(80,128))
    env=os.environ.copy()
    for name in ['PYTHONHOME','PYTHONPATH','CONDA_PREFIX','CONDA_DEFAULT_ENV','CONDA_PROMPT_MODIFIER','CONDA_SHLVL']:env.pop(name,None)
    env.update(PYTHONNOUSERSITE='1',PIP_CACHE_DIR='/tmp/trajectory-literature-work/pip-cache')
    for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:env[name]='1'
    for method in METHODS:
        prefix=Path('/tmp/trajectory-literature-envs')/method
        interpreter=str(prefix/'bin/python')
        commands=[['taskset','-c','80-127',sys.executable,'-I','-m','venv',str(prefix)],
                  ['taskset','-c','80-127',interpreter,'-I','-m','pip','install','torch==2.5.1','--index-url','https://download.pytorch.org/whl/cu124'],
                  ['taskset','-c','80-127',interpreter,'-I','-m','pip','install','numpy==2.2.6','scipy==1.15.3','scikit-learn==1.6.1','osqp==1.0.4','tqdm==4.67.1','yacs==0.1.8','matplotlib==3.10.8','tensorboard==2.20.0']]
        record=dict(method=method,watchdog_pid=os.getpid(),state='setup',commands=commands,started_at=time.time())
        (OUT/f'JOB_setup_{method}.json').write_text(json.dumps(record,indent=2)+'\n')
        for command in commands:subprocess.run(command,env=env,check=True)
        freeze=subprocess.check_output(['taskset','-c','80-127',interpreter,'-I','-m','pip','freeze'],env=env,text=True)
        (OUT/f'{method}.lock').write_text(freeze)
        paths=json.loads(subprocess.check_output(['taskset','-c','80-127',interpreter,'-I','-c','import sys,site,json; print(json.dumps(dict(interpreter=sys.executable,prefix=sys.prefix,base_prefix=sys.base_prefix,user_site=site.ENABLE_USER_SITE,sys_path=sys.path)))'],env=env,text=True))
        assert not paths['user_site']
        assert all('site-packages' not in p or Path(p).is_relative_to(prefix) for p in paths['sys_path'])
        paths.update(reconstruction_commands=commands,lock=f'validation/results_literature/environments/{method}.lock',clear_environment=['PYTHONHOME','PYTHONPATH','CONDA_PREFIX','CONDA_DEFAULT_ENV','CONDA_PROMPT_MODIFIER','CONDA_SHLVL'],library_threads=1)
        (OUT/f'{method}.json').write_text(json.dumps(paths,indent=2)+'\n')
        record.update(state='complete',finished_at=time.time());(OUT/f'JOB_setup_{method}.json').write_text(json.dumps(record,indent=2)+'\n')


if __name__=='__main__':main()
