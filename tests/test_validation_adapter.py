"""The adapter must run in a fresh interpreter: both repos have a src package."""
import json
from pathlib import Path
import subprocess
import sys


def test_upstream_adapter_contract(tmp_path):
    root=Path(__file__).resolve().parents[1]
    subprocess.run([sys.executable,str(root/'scripts/validation_benchmark.py'),
                    '--output',str(tmp_path),'--verify'],cwd=root,check=True,capture_output=True,text=True)
    result=json.loads((tmp_path/'adapter_checks.json').read_text())
    assert result['passed']
    assert all('/external/TrajectoryManifoldControl/' in path for path in result['imports'].values())
