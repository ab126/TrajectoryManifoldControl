# Core TMC versus LQR: launch and tuning

Run from the repository root with `micromamba activate control311`.
New comparisons use **core TMC**: no LQR switch or fallback. Invalid plans fail
that case. LQR is fitted on the same model's training split. Hybrid results in
[the investigation](REPORT.md) remain historical evidence, not core-TMC results.
Wait for the active grid before starting the tuning plan below.

## Active grid / resume

```sh
python scripts/sweep_tmc.py --seeds 0 1 2 --controller standalone \
  --models-root saves/validation/models --seconds 14 \
  --output saves/validation/grid-standalone-3seed-v1 \
  --max-wall-seconds 172800
```

Already running; do not start a second writer. This uses Adam20, learning rate
.15, three starts, replan1; 1,681 initial states per seed, 10,086 paired cases.
The former hybrid used CEM, so this comparison also changes the optimizer.

## Tune the solver on existing models

Screen CEM3 versus CEM10, keeping every other setting fixed:

```sh
for rounds in 3 10; do
  python scripts/screen_tmc_candidate.py \
    --model-dir saves/validation/models/reconstructed-seed0 \
    --output "saves/validation/core-cem${rounds}-seed0-v1" \
    --arms TMC L --set tuning --solver original_cem \
    --cem-iterations "$rounds" --population 256 --replan-interval 1 \
    --seconds 14 --max-wall-seconds 1800
done
```

Compare Adam20 against the active grid's solver settings:

```sh
python scripts/screen_tmc_candidate.py \
  --model-dir saves/validation/models/reconstructed-seed0 \
  --output saves/validation/core-adam20-seed0-v1 \
  --arms TMC L --set tuning --solver adam --iterations 20 \
  --learning-rate .15 --starts 3 --replan-interval 1 \
  --seconds 14 --max-wall-seconds 1800
```

All eight tuning starts produce 16 cases. `--limit 2` is a pipeline check only.
Test replan5 in a new output, holding other settings fixed. Choose settings from
14-second recovery and latency, not learned cost alone. Compute delay is recorded
but not injected into the plant. CEM10/replan1 can take minutes per start.

## Candidate model: unverified hypothesis

14,000 episodes, 256×256×256 network, 200 one-step epochs, then 20 multistep
epochs with window10; ±85° collection and 25% near-upright/small-input episodes.
Expect hours on CPU. Test changes separately if attributing improvements.

```sh
python scripts/train_tmc_candidate.py \
  --output saves/validation/models/candidate-seed0-v1 \
  --seed 0 --episodes 14000 --epochs 200 --hidden-dims 256 256 256 \
  --angle-limit-deg 85 --local-fraction .25 \
  --multistep-epochs 20 --window 10 --device cpu

for variant in one_step multistep; do
  python scripts/screen_tmc_candidate.py \
    --model-dir "saves/validation/models/candidate-seed0-v1/finetuned/${variant}" \
    --output "saves/validation/core-candidate-${variant}-tuning-v1" \
    --arms TMC L --set tuning --solver original_cem --cem-iterations 10 \
    --population 256 --replan-interval 1 --seconds 14 --max-wall-seconds 1800
done
```

CEM10/replan1 is provisional; replace it with the selected tuning settings.
Each variant gets its own matched LQR. Completed training artifacts are reused;
partial epochs are not checkpointed. Partial fine-tuning requires a new output.

## Confirm after locking settings

Repeat training above for seeds 0, 1, 2 in separate model directories. Evaluate
each chosen checkpoint with the locked solver, changing `--set tuning` to
`--set D` and using a new output, for example:

```sh
python scripts/screen_tmc_candidate.py \
  --model-dir saves/validation/models/candidate-seed0-v1/finetuned/multistep \
  --output saves/validation/core-candidate-grid-seed0-v1 \
  --arms TMC L --set D --solver original_cem --cem-iterations 10 \
  --population 256 --replan-interval 1 --seconds 14 --max-wall-seconds 172800
```

Each seed has 3,362 cases; three grids can take days. The candidate launcher
supports changed widths/data; `sweep_tmc.py` enforces the original model recipe.
No tuning choice or larger model guarantees improved recovery.

## Resume and evidence

Budgets apply per invocation, between cases. Repeat the exact command until
`summary.json` reports `complete: true`; only limit/budget may change. Use a new
output for changed scientific settings or launcher versions. Never share an
output between processes. Existing hybrid screens cannot resume as core TMC.

To audit the completed historical hybrid grid without launching new control:

```sh
python scripts/investigate_tmc.py \
  --run saves/validation/grid-hybrid-3seed-v1 \
  --models-root saves/validation/models \
  --output saves/validation/investigation-replay-v2
```
