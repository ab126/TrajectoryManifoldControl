"""Copy the resolved missing native closure from the same offline source env."""
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent
SOURCE=Path('/home/ap60/miniconda3/envs/mtbi_detection0/lib')
target=HERE/'.venv/lib'
site=target/'python3.11/site-packages'
env=os.environ.copy(); env['LD_LIBRARY_PATH']=str(target)+':'+str(SOURCE)
env['LD_PRELOAD']=str(target/'libstdc++.so.6')
deps={}
for package,pattern in [('PIL','*.so'),('matplotlib','*.so'),('contourpy','*.so'),('kiwisolver','*.so')]:
    for binary in (site/package).glob(pattern):
        output=subprocess.run(['ldd',str(binary)],env=env,capture_output=True,text=True,check=False).stdout
        assert 'not found' not in output, output
        for match in re.finditer(r'=> (/\S+)',output):
            path=Path(match.group(1))
            if str(path).startswith(str(SOURCE)+'/'):
                deps[path.name]=str(path)
for name,path in deps.items():
    dst=target/name
    if not dst.exists(): shutil.copy2(path,dst,follow_symlinks=True)
(HERE/'artifacts/native_completion.json').write_text(json.dumps(deps,indent=2))
print('Copied native graphics closure',list(deps))
