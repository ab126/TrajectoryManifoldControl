"""Render the screenshot-format figure from an original E1 summary.csv archive."""
import argparse
import ast
import csv
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.validation_sweep import plot_sweep


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--summary', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True, help='New directory, separate from simulated sweeps')
    p.add_argument('--endpoint', choices=['primary_success','terminal_recovery'], default='terminal_recovery')
    args = p.parse_args()
    # Never mix imported archived outcomes with newly simulated cases.
    args.output.mkdir(parents=True, exist_ok=False)
    rows = [r for r in csv.DictReader(args.summary.open()) if r['ic_set']=='D'
            and float(r['seconds'])==14 and r['arm'] in ('H','L')]
    indexed = {}
    states = {}
    for r in rows:
        key = (int(r['seed']),r['arm'],int(r['ic_index']))
        if key in indexed:
            raise ValueError(f'duplicate archived case {key}')
        if key[0] not in (0,1,2):
            raise ValueError('expected original seeds 0,1,2')
        state = ast.literal_eval(r['initial_state'])
        if state[:2] != [0.,0.]:
            raise ValueError('original grid requires zero cart position/velocity')
        if key[2] in states and states[key[2]]!=state:
            raise ValueError('mismatched seed/arm initial states')
        states[key[2]] = state
        indexed[key] = dict(seed=key[0],arm=key[1],ic_index=key[2],
            primary_success=r['primary_success'].lower()=='true',
            terminal_recovery=r['balanced'].lower()=='true' and r['nonfinite_state'].lower()=='false')
    if sorted(states)!=list(range(1681)) or len(indexed)!=10086:
        raise ValueError('archive must contain all 1681 x 3 seeds x 2 arms')
    manifest = dict(states=[states[i] for i in range(1681)], seeds=[0,1,2], arms=['H','L'],
                    artifact_kind='archived_outcomes_not_new_simulations', seconds=14, plot_endpoint=args.endpoint,
                    source_sha256=hashlib.sha256(args.summary.read_bytes()).hexdigest())
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    cases = args.output/'cases'; cases.mkdir()
    for (seed,arm,i), row in indexed.items():
        (cases/f'seed{seed}_{arm}_{i:04d}.json').write_text(json.dumps(row)+'\n')
    print(json.dumps(plot_sweep(args.output),indent=2))


if __name__=='__main__':
    main()
