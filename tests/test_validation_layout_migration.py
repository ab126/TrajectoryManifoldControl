import importlib.util
from pathlib import Path

import pytest


def test_layout_migration_only_accepts_integration_hash_changes(monkeypatch):
    root=Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root/'scripts'))
    spec=importlib.util.spec_from_file_location('layout_migration',root/'scripts/migrate_validation_layout.py')
    migration=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(migration)
    old=dict(configuration={'solver':'adam','iterations':20},models=[{'checkpoint_sha256':'unchanged'}],
             source_sha256='native',scoring_sha256='metrics',adapter_sha256='old',evaluator_sha256='old')
    new={**old,'adapter_sha256':'new','evaluator_sha256':'new','validation_manifest_sha256':'snapshot'}
    migration.check_equivalence(old,new)
    with pytest.raises(ValueError,match='scientific configuration changed'):
        migration.check_equivalence(old,{**new,'configuration':{'solver':'adam','iterations':21}})
    with pytest.raises(ValueError,match='scientific configuration changed'):
        migration.check_equivalence(old,{**new,'scoring_sha256':'different'})
