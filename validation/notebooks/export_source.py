"""Export the saved-data notebook without generated outputs for version control.

The populated notebook and all scientific artifacts are left untouched.
Run with Python 3 from any working directory; no notebook kernel is started.
"""
import json
from pathlib import Path


def main():
    directory = Path(__file__).resolve().parent
    notebook = json.loads((directory / "gate_a_examples.ipynb").read_text())
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
        cell["metadata"].pop("execution", None)
    notebook["metadata"].pop("widgets", None)
    notebook["metadata"]["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    target = directory / "gate_a_examples_source.ipynb"
    target.write_text(json.dumps(notebook, indent=1) + "\n")


if __name__ == "__main__":
    main()
