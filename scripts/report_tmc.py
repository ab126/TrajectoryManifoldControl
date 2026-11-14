"""Generate convergence figures and a Markdown report from saved TMC validation."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.validation_report import generate_report

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path, help='Validation output containing manifest/cases/raw')
    parser.add_argument('--output', type=Path, help='Default: DIRECTORY/report')
    parser.add_argument('--reference-image', type=Path, help='Optional historical screenshot, labeled separately')
    args = parser.parse_args()
    print(generate_report(args.directory, args.output or args.directory/'report', args.reference_image))
