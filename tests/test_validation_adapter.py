"""Validate root-level reuse and immutable upstream provenance."""
import json
from pathlib import Path
import shutil
import subprocess
import sys


def test_upstream_adapter_contract(tmp_path):
    root=Path(__file__).resolve().parents[1]
    subprocess.run([sys.executable,str(root/'scripts/validation_benchmark.py'),
                    '--output',str(tmp_path),'--verify'],cwd=root,check=True,capture_output=True,text=True)
    result=json.loads((tmp_path/'adapter_checks.json').read_text())
    assert result['passed']
    assert result['source_manifest_sha256']
    for path in result['imports'].values():
        assert Path(path).is_relative_to(root)
        assert '/external/' not in path


def test_snapshot_integrity_rejects_changed_helpers(tmp_path, monkeypatch):
    import pytest
    from scripts import validation_benchmark as adapter
    snapshot=tmp_path/'validation'
    shutil.copytree(adapter.VALIDATION,snapshot)
    monkeypatch.setattr(adapter,'VALIDATION',snapshot)
    adapter.verify_snapshot()
    with (snapshot/'compat.py').open('a') as stream:
        stream.write('\n# altered fixture\n')
    with pytest.raises(RuntimeError,match='validation snapshot changed: compat.py'):
        adapter.verify_snapshot()


def test_packed_hl_reference_coverage():
    from scripts.validation_benchmark import load_reference_cases
    cases=load_reference_cases()
    assert len(cases)==1236
    for seed,successes in enumerate((122,130,128)):
        rows=[cases[f'E1/cases/seed{seed}_L_C_{index:04d}_14s.json'] for index in range(200)]
        assert sum(row['primary_success'] for row in rows)==successes
