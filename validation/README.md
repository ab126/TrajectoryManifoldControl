# Minimal independent validation snapshot

This directory contains only the dependencies used by our benchmark adapters,
not a second TrajectoryManifoldControl repository. The plant, decoder, training
and original controller are imported from this repository's `src/` and `scripts/`.
Their hashes must match the independently evaluated source before baseline runs.

Source: [Roy Phillips's validation repository](https://github.com/aroyphillips/TrajectoryManifoldControl),
commit `e66a2b79f27f347286576f678f424625f60787b4`.
[SOURCE.json](SOURCE.json) records individual upstream paths and SHA256 hashes.
The imported Python helpers, reports, initial states and fitted diagnostics are
unchanged. `reference_cases.json` packs the full JSON objects for 36 authored and
1200 random H/L reference cases, without changing values. Original per-case
source hashes are retained in the provenance manifest.

Roy's [collaborator brief](results_v1/COLLABORATOR_BRIEF.md) and
[results report](results_v1/RESULTS.md) are the reader entry points. His checked-in
guidance keeps the submitted plant/controller unchanged and new harness work
under root `validation/`. Historical campaigns described by those reports remain
completed or stopped; importing them does not restart them. Links from retained
reports to omitted media, notebooks and other campaigns refer to the
[full pinned upstream tree](https://github.com/aroyphillips/TrajectoryManifoldControl/tree/e66a2b79f27f347286576f678f424625f60787b4/validation).

## Kept and omitted

- Kept: the six runtime helpers (including CVXPY compatibility),
  saved initial conditions (including fresh200), seed0/1/2 fitted diagnostics,
  H/L references, four relevant reports, and the implementation note.
- Omitted: duplicated source, root notebooks/logs, unrelated literature/oracle
  campaigns, per-case filesystem duplication, generated media, environments and
  caches. Full history remains at the source commit; no experiment data under
  our `saves/validation/` was removed.

The supported entry points are our `scripts/validation_benchmark.py`, diagnostic,
ablation and frozen-evaluation scripts. The retained upstream campaign module is
used as a library; its optional standalone launchers require the full upstream
archive and are not part of this minimal snapshot. New outputs must go outside
this immutable directory, normally to `saves/validation/`.

```bash
python scripts/validation_benchmark.py --verify --output saves/validation/root-layout-checks
python -m pytest -q
```

To reconstruct the snapshot into a new/empty destination from an upstream clone:

```bash
python scripts/vendor_validation_snapshot.py --git-dir /path/to/upstream/.git --output /tmp/validation-snapshot
```

The extractor rejects overwrites. It uses only Git and the Python standard
library. It does not run training or evaluation. The prior submodule's Git store
is also retained locally under `.git/modules/external/TrajectoryManifoldControl`.

No project-level upstream LICENSE/COPYING was found at this pin. Attribution and
source provenance are retained; no new redistribution license is claimed.
See [our report](../docs/validation/REPORT.md) for reproduction results and limits.
See [the layout migration](../docs/validation/LAYOUT.md) to resume the preserved
frozen run with its new integration-only bundle.
