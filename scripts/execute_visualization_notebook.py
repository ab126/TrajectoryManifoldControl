"""Execute only the appended visualization cells, preserving older experiments.

Run with a scientific environment containing nbclient, nbformat and ipykernel.
The default python3 kernelspec must point to that environment (or use --kernel).
Executed outputs go to saves/, not into the source notebook.
"""
from pathlib import Path
import argparse


def main():
    import nbformat
    from nbclient import NotebookClient

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kernel', default='python3')
    parser.add_argument('--notebook', type=Path, default=Path('manifold_control.ipynb'))
    parser.add_argument('--output', type=Path, default=Path('saves/visualizations/trajectory_manifold_notebook/executed.ipynb'))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    source = nbformat.read(root/args.notebook, as_version=4)
    notebook = nbformat.v4.new_notebook(
        cells=[cell for cell in source.cells if cell.get('id', '').startswith('tmc-viz-')])
    if len(notebook.cells) != 4:
        raise ValueError('expected visualization introduction and three code cells')
    NotebookClient(notebook, kernel_name=args.kernel, timeout=300,
                   resources={'metadata': {'path': str(root)}}).execute()
    output = args.output if args.output.is_absolute() else root/args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, output)
    print(f'Executed {len(notebook.cells)-1} visualization code cells: {output}')


if __name__ == '__main__':
    main()
