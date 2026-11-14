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

A runnable local screen using the existing seed-234 checkpoint and an explicitly
unmatched archived seed-0 linear gain:

```sh
python scripts/validate_tmc.py \
  --checkpoint saves/saved_models/canonical_decoder_dev.pt \
  --archived-gain-seed 0 \
  --output saves/validation/tmc-screen-v1 \
  --set tuning --limit 2 --seconds 14 --max-wall-seconds 600 \
  --solver adam --iterations 20 --learning-rate .15 --starts 3 \
  --replan-interval 1 --arms TMC L
```

For matched scientific validation, use a checkpoint and its original training
`data.npz`. L is fitted only on the training split, with the validation recipe
and Q/R, never on evaluation starts. If those artifacts need reconstruction,
the existing training command is available (potentially expensive):

```sh
python scripts/validation_benchmark.py --train --seed 0 \
  --model-dir saves/validation/models/reconstructed-seed0 \
  --output saves/validation/reconstruction-seed0 \
  --arms L --set A --limit 1 --seconds .04
```

Then launch the new paired evaluation:

```sh
python scripts/validate_tmc.py \
  --checkpoint saves/validation/models/reconstructed-seed0/transition.pt \
  --data saves/validation/models/reconstructed-seed0/data.npz \
  --output saves/validation/tmc-c200-seed0-v1 \
  --set C --seconds 14 --max-wall-seconds 600 \
  --solver adam --iterations 20 --learning-rate .15 --starts 3 \
  --replan-interval 1 --arms TMC L
```

Repeat the exact command to resume between cases. The wall budget is checked
between cases; it does not interrupt an active rollout. Use a new output path
for changed settings. `--set fresh` uses saved fresh-200 starts; `--set A` uses
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
