"""Parameterize only hard-coded collection box/policy in the imported runner."""
from campaign import *
import ast
import inspect
import textwrap


def collector(box_scale=1.,uniform_only=False):
    # Read the submitted function, not a separately maintained implementation.
    tree=ast.parse(textwrap.dedent(inspect.getsource(runner.collect_trajectory_data)))
    changes=[]
    class Parameters(ast.NodeTransformer):
        def visit_Assign(self,node):
            if len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id in ('state_low','state_high'):
                node.value=ast.BinOp(node.value,ast.Mult(),ast.Constant(float(box_scale)))
                changes.append(node.targets[0].id)
            return self.generic_visit(node)
        def visit_If(self,node):
            if ast.unparse(node.test)=='episode % 2':
                if uniform_only:node.test=ast.Constant(True)
                changes.append('policy')
            return self.generic_visit(node)
    Parameters().visit(tree)
    if sorted(changes)!=['policy','state_high','state_low']:
        raise RuntimeError('Submitted collection structure changed')
    ast.fix_missing_locations(tree)
    namespace=dict(runner.collect_trajectory_data.__globals__)
    exec(compile(tree,'<parameterized submitted collector>','exec'),namespace)
    return namespace['collect_trajectory_data']


if __name__=='__main__':
    print('Collection adapters: half box and odd/uniform policy only. Imported function is parameterized in memory.')
