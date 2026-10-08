"""Freeze and fingerprint the private CUDA environment without GPU experiments."""
import hashlib
import importlib.metadata as md
import json
import os
import subprocess
import sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
TARGET=HERE/'.venv-gpu'
OUT=HERE/'artifacts'
assert Path(sys.prefix)==TARGET, 'Use the private CUDA Python'
assert 'include-system-site-packages = false' in (TARGET/'pyvenv.cfg').read_text()
freeze=subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True)
(HERE/'gpu-pip-freeze.txt').write_text(freeze)
versions={dist.metadata['Name']:dist.version for dist in md.distributions()}
assert versions['torch']=='2.5.1+cu121'
assert versions['sympy']=='1.13.1'
(HERE/'gpu-requirements.lock').write_text('\n'.join(f'{name}=={version}' for name,version in sorted(versions.items(),key=lambda item:item[0].lower()))+'\n')
result=subprocess.run([sys.executable,'-m','pip','check'],capture_output=True,text=True)
(OUT/'gpu_pip_check.txt').write_text(result.stdout+result.stderr)
assert result.returncode==0, result.stdout+result.stderr
hashes={}
site=TARGET/'lib/python3.11/site-packages'
paths=[TARGET/'bin/python',TARGET/'lib/libstdc++.so.6',site/'torch/_C.cpython-311-x86_64-linux-gnu.so']
paths+=list((site/'torch/lib').glob('*.so*'))
paths+=list((site/'nvidia').glob('*/lib/*.so*'))
for path in paths:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''): h.update(block)
    hashes[str(path.relative_to(HERE))]=h.hexdigest()
(OUT/'gpu_native_fingerprints.json').write_text(json.dumps(hashes,indent=2)+'\n')
(OUT/'gpu_environment.json').write_text(json.dumps({'python':sys.version,'executable':sys.executable,
    'prefix':sys.prefix,'system_site_packages':False,'distributions':versions,
    'pip_check_passed':True,'native_fingerprint_files':len(hashes),
    'LD_LIBRARY_PATH':os.environ.get('LD_LIBRARY_PATH'),'LD_PRELOAD':os.environ.get('LD_PRELOAD'),
    'GPU_execution_verified':False,'note':'Installed dependency closure; CUDA allocation result is in gpu_probe.json'},indent=2)+'\n')
print('Private CUDA dependency check passed;',len(hashes),'native files fingerprinted')
