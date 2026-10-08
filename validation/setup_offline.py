"""Copy an explicit installed dependency closure into a private venv, offline.

No system-site-packages, no modification of the source environment. Retain
distribution metadata for pip freeze. Conda shared libraries are copied locally.
"""
import importlib.metadata as md
import json
import shutil
from pathlib import Path

HERE=Path(__file__).resolve().parent
SOURCE=Path('/home/ap60/miniconda3/envs/mtbi_detection0')
SITE=SOURCE/'lib/python3.11/site-packages'
TARGET=HERE/'.venv/lib/python3.11/site-packages'
names=['numpy','scipy','matplotlib','tqdm','pytest','torch','filelock','fsspec',
       'jinja2','markupsafe','networkx','sympy','mpmath','typing_extensions',
       'contourpy','cycler','fonttools','kiwisolver','pillow','pyparsing','python-dateutil',
       'six','packaging','pluggy','iniconfig','pygments']
manifest=[]
roots=set()
def copy_missing(src,dst):
    src,dst=Path(src),Path(dst)
    if dst.exists() and dst.stat().st_size==src.stat().st_size: return str(dst)
    return shutil.copy2(src,dst)
for name in names:
    d=md.distribution(name)
    manifest.append({'name':d.metadata['Name'],'version':d.version,'source':str(d.locate_file(''))})
    for f in d.files or []:
        first=str(f).split('/')[0]
        if first not in ('.','..') and (SITE/first).exists(): roots.add(first)
for first in sorted(roots):
    src=SITE/first; dst=TARGET/first
    print('copy',first,flush=True)
    if src.is_dir(): shutil.copytree(src,dst,dirs_exist_ok=True,copy_function=copy_missing,
                                    ignore=shutil.ignore_patterns('__pycache__','include'))
    else: copy_missing(src,dst)
# Conda package RPATHs resolve to the venv's lib. Copy runtime dependencies,
# including ISA-specific MKL dispatch libraries, without inheriting other envs.
lib=HERE/'.venv/lib'
patterns=['libmkl*.so*','libiomp*.so*','libstdc++.so*','libgcc_s.so*','libsleef.so*',
          'libtorch_python.so*',
          'libprotobuf.so*','libabsl*.so*','libuv.so*','libgomp.so*','libz.so*',
          'libopenblas*.so*','libgfortran*.so*','libquadmath*.so*']
copied=[]
for pattern in patterns:
    for src in (SOURCE/'lib').glob(pattern):
        dst=lib/src.name
        shutil.copy2(src,dst,follow_symlinks=True); copied.append(src.name)
(HERE/'artifacts/offline_environment.json').write_text(json.dumps({'distributions':manifest,
    'shared_libraries':copied,'system_site_packages':False,
    'missing_dependency':'cvxpy; unused plant imports isolated by validation import hook'},indent=2))
