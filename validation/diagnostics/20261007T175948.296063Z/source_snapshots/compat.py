"""Narrow offline workaround for two UNUSED CVXPY imports in the plant.

This is not a CVXPY stub. Only those import statements are omitted in memory.
All functions, source files, plant equations and controller code are unchanged.
Fail closed if either imported identifier is referenced anywhere in the AST.
Remove this hook when CVXPY can be installed; report its use with every result.
"""
import ast
import importlib.abc
import importlib.util
import sys
from pathlib import Path

class PlantLoader(importlib.abc.Loader):
    def create_module(self,spec): return None
    def exec_module(self,module):
        path=Path(__file__).resolve().parents[1]/'src/inverted_pendulum.py'
        tree=ast.parse(path.read_text(),filename=str(path))
        removed=[n for n in tree.body if isinstance(n,ast.ImportFrom) and n.module in
                 ('cvxpy.expressions.expression','cvxpy.constraints.constraint')]
        assert len(removed)==2
        identifiers={alias.asname or alias.name for node in removed for alias in node.names}
        assert not any(isinstance(n,ast.Name) and n.id in identifiers for n in ast.walk(tree)), \
            'CVXPY identifier is used; cannot safely omit import'
        tree.body=[n for n in tree.body if n not in removed]
        module.__file__=str(path)
        exec(compile(tree,str(path),'exec'),module.__dict__)

class PlantFinder(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path,target=None):
        if fullname=='src.inverted_pendulum':
            return importlib.util.spec_from_loader(fullname,PlantLoader())

def install():
    if importlib.util.find_spec('cvxpy') is None:
        sys.meta_path.insert(0,PlantFinder())
        return True
    return False
