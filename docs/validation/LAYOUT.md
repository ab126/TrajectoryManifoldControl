# Root validation layout (2026-10-18)

The user's request to remove redundancy replaces the full upstream submodule
with a **2.2 MB, 19-file snapshot** in root `validation/`. No duplicate plant,
decoder, controller, root notebooks or environment is imported. No model was
retrained and no controller, objective, initial state or scoring rule changed.

## Guidance and retained content

Roy's checked-in validation plan places new harness work under `validation/`
and preserves the submitted source. His [collaborator brief](../../validation/results_v1/COLLABORATOR_BRIEF.md)
is the reader entry point; it distinguishes completed original evidence from the
stopped literature extension. Public commit/issue comment endpoints contained no
additional comments when checked; the guidance used here is the checked-in text.

[SOURCE.json](../../validation/SOURCE.json) records the upstream URL, commit
`e66a2b79f27f347286576f678f424625f60787b4`, exact imported-file hashes and
baseline-source hashes. Manifest SHA256:
`7d97e6bd3a8f082b5b87c0c9c6372545e203e6cde06643c695ee347cbb913dea`.

Retained are six runtime helpers, original/fresh saved starts, fitted seed0/1/2
diagnostics, four reports and the implementation note. The1236 H/L reference
objects (36 authored,1200 random) are stored in one lossless JSON pack with their
original source paths/hashes. All16 copied files are byte-identical to upstream;
all1236 packed objects were compared with the original JSON values before removal.
The adapter checks these hashes and reuses our identical `src/` and `scripts/`.
It has no runtime dependency on a second Git checkout.

The full submodule was deinitialized and removed from the index, including its
sole `.gitmodules` entry. Its local Git object store remains recoverable under
`.git/modules/external/TrajectoryManifoldControl`; no local experiment data or
trained checkpoint under `saves/validation/` was removed. Excluded upstream
campaigns/media remain retrievable from the recorded commit. No upstream
project-level license was found; attribution is preserved without asserting new
redistribution rights.

## Verification commands

Use the established environment, or rebuild it from `requirements-cpu.txt`:

```bash
export TMC_PY=/tmp/tmc-validation-env/bin/python
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "$TMC_PY" -m pytest -q
"$TMC_PY" scripts/validation_benchmark.py --verify --output saves/validation/root-layout-checks
"$TMC_PY" scripts/validation_benchmark.py --arms L --set A --output saves/validation/root-layout-L-A
"$TMC_PY" scripts/validation_benchmark.py --arms L --set C --limit 3 --output saves/validation/root-layout-L-C3
"$TMC_PY" scripts/summarize_validation.py saves/validation/root-layout-L-A saves/validation/root-layout-L-C3 --output saves/validation/root-layout-parity
```

The nine replays match archived endpoint outcomes and compared final-angle,
final-cart-position and maximum-cart-excursion values exactly. Hash/packing and
layout-migration tests supplement the existing regression suite:50tests pass
in5.83s; syntax/whitespace checks pass. No large
evaluation or training was launched during this refactor.

## Continuing the existing frozen run

Historical manifests and `docs/validation/EVIDENCE.json` remain unchanged as
records of their original source version. Their old absolute paths are historical
provenance. Old source-hash guards have not been relaxed. A new layout release
changes integration hashes only; scientific settings, model/data, native source,
scoring, selection evidence and evaluation protocol must remain identical.

Old bundle:`saves/validation/frozen-native.json`, SHA256
`97a3d77c12d69a30a8ef799f8703c77f425f0dc94f762c88a0403c7a01ad9f41`.
New bundle:`saves/validation/frozen-native-root-validation.json`, SHA256
`01e556c2f7100e95bb121482bbb76ff596b2bd04d854e340df6ef16e80e04767`.

The following **one-time** preparation was executed (it rejects overwrites):

```bash
"$TMC_PY" scripts/frozen_manifold_validation.py freeze --screen saves/validation/tuning-screen-v1 --variant adam1 --models saves/validation/models/reconstructed-seed0 --bundle saves/validation/frozen-native-root-validation.json
"$TMC_PY" scripts/frozen_manifold_validation.py evaluate --bundle saves/validation/frozen-native-root-validation.json --arms L H native --set C --output saves/validation/final-root-layout-seed0-C --verify-only
"$TMC_PY" scripts/migrate_validation_layout.py --old-bundle saves/validation/frozen-native.json --new-bundle saves/validation/frozen-native-root-validation.json --old-output saves/validation/final-seed0-C --new-output saves/validation/final-root-layout-seed0-C
```

Migration authenticates the old integration against committed source
`f41056d670cdb60a03ca73e118be9517fff987cd`, rejects scientific or protocol changes,
and copies completed case/raw files byte-for-byte. Six controller cases (two
paired initial states), six raw traces and the summary were retained in the new
output; originals remain untouched. `LAYOUT_MIGRATION.json` records all copied
hashes. These are reused measurements, not six new experiments.

Current continuation command (not run during the refactor):

```bash
"$TMC_PY" -u scripts/frozen_manifold_validation.py evaluate --bundle saves/validation/frozen-native-root-validation.json --arms L H native --set C --output saves/validation/final-root-layout-seed0-C --max-wall-seconds 180
```

This skips the preserved cases and resumes the remaining original C200 starts.
It does not select new settings. Current paired outcomes remain native1/2,
H1/2,L2/2; the research target is still unachieved. Fresh200, remaining fitted
seeds and native60s remain pending. Use new output directories for new benchmark
or ablation runs because their integration-source hashes changed.

## Rebuilding the imported files

Normal use needs no submodule. To reconstruct into a new/empty directory from
the retained Git store (or an upstream clone's `.git`):

```bash
"$TMC_PY" scripts/vendor_validation_snapshot.py --git-dir .git/modules/external/TrajectoryManifoldControl --output /tmp/tmc-validation-snapshot
```

The extractor uses the pinned revision, never modifies the source Git store,
and rejects destination overwrites. `validation/README.md` is our local guide;
the other18 files are generated by the importer. Optional upstream campaign
launchers are outside the minimal dependency set; use this repository's supported
entry points rather than relaunching historical upstream campaigns.
