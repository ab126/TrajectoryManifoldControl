import hashlib
import json
from pathlib import Path
import subprocess
import sys


def test_freeze_requires_completed_screen_and_unchanged_sources(tmp_path):
    root=Path(__file__).resolve().parents[1]
    model=tmp_path/'model';model.mkdir()
    checkpoint=model/'transition.pt';checkpoint.write_bytes(b'fixture, not a trained model')
    (model/'data.npz').write_bytes(b'dataset fixture')
    screen=tmp_path/'screen';screen.mkdir()
    manifest=dict(code_sha256='wrong',checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                  variants={'adam1':{'solver':'adam'}})
    (screen/'manifest.json').write_text(json.dumps(manifest))
    (screen/'summary.json').write_text(json.dumps({'complete':False}))
    command=[sys.executable,str(root/'scripts/frozen_manifold_validation.py'),'freeze',
        '--screen',str(screen),'--variant','adam1','--models',str(model),'--bundle',str(tmp_path/'freeze.json')]
    result=subprocess.run(command,capture_output=True,text=True)
    assert result.returncode!=0 and 'complete the declared' in result.stderr
    (screen/'summary.json').write_text(json.dumps({'complete':True}))
    result=subprocess.run(command,capture_output=True,text=True)
    assert result.returncode!=0 and 'source changed' in result.stderr
    manifest['code_sha256']=hashlib.sha256((root/'src/manifold_experiments.py').read_bytes()).hexdigest()
    (screen/'manifest.json').write_text(json.dumps(manifest))
    result=subprocess.run(command,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    frozen=json.loads((tmp_path/'freeze.json').read_text())
    assert frozen['models'][0]['checkpoint_sha256']==manifest['checkpoint_sha256']
    assert frozen['configuration']=={'solver':'adam'}
