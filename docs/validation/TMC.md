# Reference-aware TMC and hybrid validation

`src/validation_controllers.py` implements validation's fitted linear law as
`clip(u_ref - K @ (x - x_ref), -umax, umax)`. The hybrid and new linear arm use
this same callable, including identical floating-point output on linear steps.
The hybrid switches when `(x-x_ref).T @ Q @ (x-x_ref) < .75`, and uses L on
invalid manifold plans. Switching clears queued manifold inputs. Both manifold
cost and warm-start input initialization use the requested references.

TMC means standalone trajectory-manifold control in this launcher. It has no
linear branch or failure substitution. An invalid solve terminates that case,
records failure, and allows other cases to run. Optimizer warm starts and
scheduled plan execution are normal MPC operation, not failure fallbacks.
Default replanning is every sample. The old frozen controllers, snapshots,
bundles and launchers remain unchanged; this is a new experiment.

All references are in normalized plant coordinates: position/l, cart velocity/
sqrt(l*g), angle, angular velocity/sqrt(g/l), and force/(m*g). Any finite reference
can be supplied; arbitrary reference states and inputs need not form a physically
attainable equilibrium. Constant cart displacement with upright angle, zero
velocities and zero input is an example of an attainable shifted equilibrium.
Initial states remain the absolute saved states, not shifted by the reference.

## Launch from repository root

Activate the project environment first:

```sh
micromamba activate control311
```

### Clean checkout: generate artifacts first

`validate_tmc.py` loads a trained checkpoint; it does not train one. `saves/`
is ignored by Git, so a clean clone has no saved model or training data.
The old example assumed a locally generated
`saves/saved_models/canonical_decoder_dev.pt`. `--archived-gain-seed 0` supplies
only a linear gain; it cannot supply or replace a missing learned model.

From the repository root, reconstruct a seed-0 model and matching dataset:

```sh
python scripts/validation_benchmark.py --train --seed 0 --device cpu \
  --model-dir saves/validation/models/reconstructed-seed0 \
  --output saves/validation/reconstruction-seed0 \
  --arms L --set A --limit 1 --seconds .04
```

This collects 3,500 episodes and trains for 100 epochs, which can take substantial
time. `.04` limits only the subsequent baseline rollout, not training. There is
no training wall-time limit. This creates `transition.pt`, `data.npz` and
`diagnostics.json` in the model directory. This is newly reconstructed training,
not a replay of an archived checkpoint. `--train` reuses existing data and a
compatible checkpoint; choose a new model directory for fresh training.

Check the required artifacts:

```sh
python -c "from pathlib import Path; p = Path('saves/validation/models/reconstructed-seed0'); assert (p/'transition.pt').is_file(), 'missing checkpoint'; assert (p/'data.npz').is_file(), 'missing training data'; print('Artifacts present')"
```

First run a pipeline smoke check in its own output directory:

```sh
python scripts/validate_tmc.py \
  --checkpoint saves/validation/models/reconstructed-seed0/transition.pt \
  --data saves/validation/models/reconstructed-seed0/data.npz \
  --output saves/validation/tmc-smoke-seed0-v1 \
  --set tuning --limit 1 --seconds .04 --max-wall-seconds 600 \
  --solver adam --iterations 20 --learning-rate .15 --starts 3 \
  --replan-interval 1 --arms TMC L
```

Check `summary.json` for `complete: true`, and each case JSON for
`completed: true` and `failure: null`. Short smoke runs cannot establish recovery.
Then launch the requested 14-second tuning screen in a separate output directory:

```sh
python scripts/validate_tmc.py \
  --checkpoint saves/validation/models/reconstructed-seed0/transition.pt \
  --data saves/validation/models/reconstructed-seed0/data.npz \
  --output saves/validation/tmc-screen-seed0-v1 \
  --set tuning --limit 2 --seconds 14 --max-wall-seconds 600 \
  --solver adam --iterations 20 --learning-rate .15 --starts 3 \
  --replan-interval 1 --arms TMC L
```

L is fitted only on the training split using the checkpoint's seed and validation
Q/R, never on evaluation starts. Keep the checkpoint and its original dataset
together: `--data` must point to that model's training data. After selecting
settings on tuning starts, lock them, choose a new output path and use `--set C`
without `--limit` for all 200 starts. Repeat across training seeds.

### Existing development artifacts

If you already ran `python scripts/run_inverted_pendulum_mc.py --profile dev`,
replace the checkpoint/data arguments above with:

```sh
--checkpoint saves/saved_models/canonical_decoder_dev.pt \
--data saves/datasets/canonical_trajectories_dev.npz
```

If only the checkpoint exists and its original training data is unavailable,
replace `--data ...` with `--archived-gain-seed 0` for an explicitly **unmatched**
screen. This does not fix a missing checkpoint or provide a matched comparison.
Supply exactly one of these gain-source options.

### Resume or start a fresh evaluation

Repeat the exact command to resume between cases. The wall budget is checked
between cases; it does not interrupt an active rollout. Use a new, unused output path
for a clean evaluation or changed settings; existing case files are skipped,
not rerun. Keep smoke and 14-second outputs separate. A budget-limited run can
exit normally with `summary.json` reporting `complete: false`; repeat the same
command until complete. The wall budget is per invocation. `--set fresh` uses saved fresh-200 starts; `--set A` uses
the six authored starts; `--set tuning` uses the eight independent tuning starts.
Use tuning starts for parameter screening before locking a final protocol.
The manifest pins sources, checkpoint, gain/data, references, solver settings,
initial states and library versions; incompatible resumes fail.

To include the prepared hybrid and test a displaced target, use a new output
path and append:

```sh
--arms TMC L hybrid --x-ref 1 0 0 0 --u-ref 0
```

The default paired run contains only TMC and L. Raw measured states/inputs,
controller/solver latencies and solve steps are saved under `raw/`; case results
include explicit failure, linear-use and fallback steps. `summary.json` reports
recovery counts and paired wins/losses on shared starts, including failed cases.
Recovery retains validation's angle/rate endpoint relative to the supplied
reference over a full final three seconds; it does not require cart recentering.
Per-state RMSE/final errors are also recorded. Short smoke runs cannot establish
recovery. Simulation uses ideal feedback: measured compute latency is reported
against the 20 ms deadline, not injected as plant delay.

## Suggested settings from the recorded screen

The recorded evidence is in [REPORT.md](REPORT.md), “Switchable experiments.”
Only two tuning starts were completed: Adam/plain 2/2, CEM 0/2, periodic Adam
1/2, periodic plus terminal-window-5 2/2, L 2/2. Native solves took about .2 s on
the historical CPU and did not beat L in the partial independent evaluation.

Start with Adam, 20 iterations, learning rate .15, three starts, feedback every
sample, plain quadratic objective and terminal window 1 (launcher defaults).
Keep Q=diag(.02,30,100,10), R=.03, terminal weight 12Q and horizon from the
checkpoint. These are a starting protocol, not newly validated improvements.

On tuning starts only, compare 10/20/40 iterations for runtime versus recovery,
and learning rates .05/.10/.15 if convergence is unstable. Keep replan interval
1 initially; changing it to 5 reduces solves but weakens feedback and the old
CEM queue-5 screen failed. Screen terminal window 5 separately; its earlier
angular improvement accompanied .247 m/s final cart speed and no recovery-count
improvement. Periodic cost is optional, not the recommended default. Do not
increase horizon without checking learned prediction error at the added steps.
Select settings on tuning data, then lock them before C/fresh testing. The tiny
screen and historical final outcomes do not establish superiority over linear.
