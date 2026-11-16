"""Exercise collection/training/evaluation and guards with a deliberately tiny model."""
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]


def test_candidate_training_and_resumable_screen(tmp_path):
    model=tmp_path/'model';output=tmp_path/'screen'
    train=[sys.executable,'scripts/train_tmc_candidate.py','--output',str(model),
           '--episodes','40','--epochs','1','--hidden-dims','16','16','--multistep-epochs','0']
    done=subprocess.run(train,cwd=ROOT,capture_output=True,text=True)
    assert done.returncode==0,done.stderr
    manifest=json.loads((model/'training_manifest.json').read_text())
    assert manifest['config']['episodes']==40
    assert manifest['config']['hidden_dims']==[16,16]
    assert (model/'data.npz').is_file() and (model/'transition.pt').is_file()
    screen=[sys.executable,'scripts/screen_tmc_candidate.py','--model-dir',str(model),
            '--output',str(output),'--limit','1','--seconds','.04','--max-wall-seconds','60']
    done=subprocess.run(screen,cwd=ROOT,capture_output=True,text=True)
    assert done.returncode==0,done.stderr
    case=output/'cases'/'hybrid_0000.json';before=case.stat().st_mtime_ns
    assert not json.loads(case.read_text())['terminal_recovery']
    assert json.loads((output/'summary.json').read_text())['saved']==2
    done=subprocess.run(screen,cwd=ROOT,capture_output=True,text=True)
    assert done.returncode==0,done.stderr
    assert case.stat().st_mtime_ns==before
    changed=subprocess.run(screen+['--cem-iterations','4'],cwd=ROOT,capture_output=True,text=True)
    assert changed.returncode!=0 and 'protocol differs' in changed.stderr
