# Investigation and candidate launch commands

Run from the repository root with `micromamba activate control311`.

## Reproduce the investigation

```sh
python scripts/investigate_tmc.py \
  --run saves/validation/grid-hybrid-3seed-v1 \
  --models-root saves/validation/models \
  --output saves/validation/investigation-replay-v1
```

This audits all saved case JSON and runs 24 independent-tuning seed/start probes
per solver (CEM3, CEM10, Adam40), plus fixed-input model replay. It does not train
or claim new coverage. Expect about a minute. Model/data hashes must match the
completed grid. Use a new output for each evidence snapshot.

## First test the solver without changing the model

These two commands differ only in CEM search rounds. Both use the current hybrid,
with the same local-law switch and failed-plan fallback, and the same matched L.
The historical controller returns zero on invalid plans; none were observed in
the completed grid. The current hybrid uses L on invalid plans.

```sh
for rounds in 3 10; do
  python scripts/screen_tmc_candidate.py \
    --model-dir saves/validation/models/reconstructed-seed0 \
    --output "saves/validation/tune-cem${rounds}-seed0-v2" \
    --set tuning --solver original_cem --cem-iterations "$rounds" \
    --population 256 --replan-interval 5 \
    --seconds 14 --max-wall-seconds 1800
done
```

Omit `--limit` for all eight independent tuning starts. For a quick first pass,
add `--limit 2`; those are the two screens launched during the investigation.
Then compare `--replan-interval 1` in a **new output directory**, keeping the
remaining settings fixed. Replan1 offers feedback every 20 ms instead of every
100 ms; compute latency is recorded, not injected. CEM10 is roughly 0.3 s per
solve on this CPU, so expect several minutes per start with replan1. A lower
learned or replay cost alone is not evidence of better 14-second recovery.

Both launchers are resumable between cases. Repeat the exact command until all
requested cases are saved; the budget is per invocation and checked between
cases. Keep scientific settings/output fixed. `--limit` and wall budget may
change on resume. In the tuning screen, full completion is 16 saved cases.

The earlier Adam20 hypothesis is separately checked with:

```sh
python scripts/validate_tmc.py \
  --checkpoint saves/validation/models/reconstructed-seed0/transition.pt \
  --data saves/validation/models/reconstructed-seed0/data.npz \
  --output saves/validation/investigation-adam20-seed0-v1 \
  --set tuning --limit 2 --seconds 14 --max-wall-seconds 900 \
  --solver adam --iterations 20 --learning-rate .15 --starts 3 \
  --replan-interval 1 --arms hybrid L
```

This was launched during the investigation. The original tuning starts were
already used in earlier work; these are screening evidence, not an untouched
final test or a fair optimizer-only ablation (Adam also changes replanning).

## Proposed stronger model: hypothesis, not a promised winner

Candidate: 14,000 measured episodes, 30 transitions each, 256×256×256 network,
200 one-step epochs plus 20 multistep epochs, window10. Collection covers initial
angles ±85°, with 25% of episodes near upright and lower-amplitude inputs. This
addresses observed horizon error and target-region accuracy, rather than simply
making every layer much larger. The original initial-angle collection domain was
±1.3 rad (±74.5°), smaller than the evaluation grid; subsequent measured states
may nevertheless cover larger angles.

```sh
python scripts/train_tmc_candidate.py \
  --output saves/validation/models/candidate-seed0-v1 \
  --seed 0 --episodes 14000 --epochs 200 \
  --hidden-dims 256 256 256 \
  --angle-limit-deg 85 --local-fraction .25 \
  --multistep-epochs 20 --window 10 --device cpu
```

Expect hours on CPU; this large training job was **not** launched. `--device cuda`
requires a CUDA PyTorch environment. Completed dataset/checkpoints are reused;
training does not checkpoint partial epochs. Use new outputs for changed settings.
A partially created fine-tuning directory without its summary requires a new
training output (the fine-tuner intentionally refuses a nonempty partial output).

Compare against the one-step control arm created from the same candidate:

```sh
for variant in one_step multistep; do
  python scripts/screen_tmc_candidate.py \
    --model-dir "saves/validation/models/candidate-seed0-v1/finetuned/${variant}" \
    --output "saves/validation/candidate-${variant}-tuning-v1" \
    --set tuning --solver original_cem --cem-iterations 10 \
    --population 256 --replan-interval 1 \
    --seconds 14 --max-wall-seconds 1800
done
```

Each candidate's L is refitted from that candidate's training split. This is the
fair matched-data comparison. If testing only model improvements against a fixed
historical gain, label that as an additional unmatched comparison.

For attribution, train separate outputs with only one changed factor: episodes,
width, collection mixture, then multistep loss. The combined candidate is a
practical proposal, not evidence identifying which factor helps.

## Confirm on the dense grid after locking tuning settings

```sh
for seed in 0 1 2; do
  python scripts/train_tmc_candidate.py \
    --output "saves/validation/models/candidate-seed${seed}-v1" \
    --seed "$seed" --episodes 14000 --epochs 200 \
    --hidden-dims 256 256 256 --angle-limit-deg 85 --local-fraction .25 \
    --multistep-epochs 20 --window 10 --device cpu || break
  python scripts/screen_tmc_candidate.py \
    --model-dir "saves/validation/models/candidate-seed${seed}-v1/finetuned/multistep" \
    --output "saves/validation/candidate-grid-seed${seed}-v1" \
    --set D --solver original_cem --cem-iterations 10 \
    --population 256 --replan-interval 1 \
    --seconds 14 --max-wall-seconds 172800 || break
done
```

The solver parameters here are provisional: replace them with the settings
selected on tuning **before** starting the grid, then keep them locked. Expect
days for three dense grids with CEM10/replan1. Each seed has 3,362 cases; reissue
the same screen command for any budget-limited seed until `complete: true`.
Do not use `sweep_tmc.py` for the changed-width/data candidate: that launcher
intentionally enforces the original dev configuration.

## Repeat the already observed winning seed

The existing seed-2 recipe already beat its matched linear baseline on this
specific grid: 1,225 versus 1,085 recoveries. To repeat it with the existing
checkpoint, without claiming independent validation:

```sh
python scripts/sweep_tmc.py --seeds 2 --controller legacy-hybrid \
  --models-root saves/validation/models --seconds 14 \
  --output saves/validation/grid-observed-seed2-repeat-v1 \
  --max-wall-seconds 172800
```

Choosing seed2 because it won this grid is post hoc selection. It is not a
parameter guarantee for other seeds, references, model fits or fresh states.
