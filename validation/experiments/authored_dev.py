"""Run the original dev entry point, recording elapsed time and configuration."""
from bootstrap import *
import json
import subprocess
import time
from dataclasses import asdict
out = ROOT / 'validation/results_v1/E0'
manifest_path = out / 'authored_manifest.json'
start = time.time()
manifest = dict(git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                config=asdict(runner.make_config('dev',device='cuda')),seeds=[234],device='cuda',
                cvxpy_unused_import_workaround=CVXPY_WORKAROUND,status='RUNNING', wall_seconds=0)
manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
try:
    result = runner.run(runner.build_parser().parse_args(['--profile','dev','--device','cuda']))
    (out/'authored_summary.json').write_text(json.dumps(result,indent=2)+'\n')
    manifest['status']='DONE'
except BaseException as error:
    manifest.update(status='ERROR',error=repr(error))
    raise
finally:
    manifest['wall_seconds']=time.time()-start
    manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
