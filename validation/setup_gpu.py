"""Build a private CUDA venv offline; never inherit system-site-packages."""
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

HERE=Path(__file__).resolve().parent
CPU=HERE/'.venv'
CUDA=Path('/home/ap60/venvs/mtbi-torch-cu12')
TARGET=HERE/'.venv-gpu'
SITE='lib/python3.11/site-packages'
if TARGET.exists():
    raise SystemExit('GPU environment already exists; inspect it rather than overwriting it')
subprocess.run([str(CPU/'bin/python'),'-m','venv',str(TARGET)],check=True)
exclude={'torch','torchgen','functorch','sympy','isympy.py'}
def copied(src,dst):
    if src.is_dir():
        shutil.copytree(src,dst,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__'))
    else:
        shutil.copy2(src,dst)
for src in (CPU/SITE).iterdir():
    if src.name in exclude or src.name.startswith(('torch-','sympy-')) or src.name=='__pycache__': continue
    copied(src,TARGET/SITE/src.name)
for src in (CUDA/SITE).iterdir():
    if src.name in {'torch','torchgen','functorch','sympy','isympy.py','nvidia','triton'} or src.name.startswith(('torch-','sympy-','nvidia_','triton-')):
        print('CUDA dependency:',src.name,flush=True)
        copied(src,TARGET/SITE/src.name)
for src in (CPU/'lib').iterdir():
    if src.is_file() and not src.name.startswith('libtorch'):
        shutil.copy2(src,TARGET/'lib'/src.name,follow_symlinks=True)
manifest={'source_cuda_environment':str(CUDA),'common_dependency_source':str(CPU),
          'system_site_packages':False,'torch_expected':'2.5.1+cu121','CUDA_build_expected':'12.1',
          'method':'private copies of installed wheel/runtime dependencies; no network or source-env modifications',
          'runtime_check_pending':'CUDA allocation requires NVIDIA device nodes in execution context'}
(HERE/'artifacts/gpu_environment_setup.json').write_text(json.dumps(manifest,indent=2)+'\n')
