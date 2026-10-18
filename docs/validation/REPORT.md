# Validation integration and native manifold-controller investigation

Dates: 2026-10-17–18. Baseline integration and bounded research screens are
complete. The bounded independent run completed two paired starts and remains
partial against the full200 protocol. Standalone gradient-based manifold control recovered both
independent tuning starts, versus original standalone M 0/2 and H 1/2, but has not beaten
L on a completed independent benchmark and takes about 0.2 s per solve on this CPU.

## Provenance and method identity

Upstream: https://github.com/aroyphillips/TrajectoryManifoldControl.git,
minimal root `validation/` snapshot, pinned at
`e66a2b79f27f347286576f678f424625f60787b4`.
The original full submodule was replaced on October18 at the user's request;
[current layout and continuation commands](LAYOUT.md) supersede its path-dependent
integration instructions below. Scientific results and historical artifacts are
unchanged. `validation/SOURCE.json` pins every imported file and the reused
local baseline source. The snapshot is2.2MB,19files, including1236 H/L reference
records packed losslessly into one JSON file.
Original local `src/manifold_control.py` and
`scripts/run_inverted_pendulum_mc.py` are byte-identical to upstream.
The implementation specification in upstream
`validation/handoff/sources/trajectory_manifold_implementation.tex` was read
alongside the source. Upstream has no tracked project-level license; vendor
licenses apply to their vendors only. Required validation helpers and metadata
are copied verbatim with attribution; controller/plant files are reused locally
and checked against upstream hashes. Redistribution terms remain unresolved.

The active model is a normalized residual transition network
`f_hat(x,u)=x+mu_dx+std_dx*h((col(x,u)-mu_xu)/std_xu)`. Recursing it gives the
trajectory decoder `Phi_hat_N(col(x0,u))`. Input coordinates and measured x0 pass
through exactly; the active encoder is coordinate extraction. Older unconstrained
latent decoders/encoders exist but are not the reported H. With n=4,m=1,N=30,
the coordinates have dimension34, fixed-state free coordinates30, and packed
behavior154. Thus the active implementation does not compress the34 coordinates.
This says nothing about exact feasibility under the real plant: future states
are approximate learned predictions.

The original network has 76,036 parameters, hidden widths 192/192/192, shared
across 30 recursive steps. Original training: 3500 episodes, 30 transitions/episode,
2450/525/525 episode split, 73,500 training transitions, 100 epochs, batch 2048.
Scaling uses training data only. Original H uses CEM 256 candidates × 3 iterations,
32 elites, five-action execution and data-fitted L near the origin. L fits one
fixed A,B and gain per training seed. No simulator derivatives, equations or
hidden parameters enter learning or any new optimization code.

## Benchmark contracts and endpoints

The adapter runs in a fresh interpreter. It calls the identical local collection,
splitting, simulation and original controller, and the unchanged imported
initial-state, controller-selection and legacy-metric helpers directly.
It checks module paths and all pinned snapshot/baseline hashes. Outputs go outside
the snapshot. Upstream CUDA-only campaign orchestration is bypassed; its
scientific functions run unchanged, on an explicitly recorded device.

State order is normalized cart position, cart speed, angle, angular speed:
`[p/l, p_dot/sqrt(l*g), theta, theta_dot/sqrt(g/l)]`. Physical speed reporting
uses l=1, g=9.81. The benchmark is a uniform rod (mass-matrix rotational term 1/3,
coupling 1/2), not a point mass. Held normalized input is `u=F/(m_p*g)`, bounded
at ±10, equivalent to ±98.1 N for benchmark m_p=1. The physical sample is 0.02 s,
normalized RK4 step 0.02*sqrt(9.81), duration 14 s/700 actions; horizon 30=0.6 s.
Actions apply before each plant step. State angle stays continuous and unwrapped.

| Endpoint | Prior crossing allowed? | Final full3s conditions |
|---|---|---|
| A `legacy_uninterrupted` | No: abs(theta)<pi/2 throughout | Delegates exactly to upstream primary endpoint |
| B `terminal_recovery` | Yes | abs(theta)<=0.02rad and physical abs(omega)<=0.05rad/s |
| C `periodic_terminal_recovery` | Yes | abs(atan2(sin(theta),cos(theta)))<=0.02rad and same speed bound |

All rules apply to all arms. B/C fail incomplete 3 s traces. Settling requires at
least 3 s of remaining good samples. Full-state settling delegates the upstream
angular/cart-speed/one-second position-drift predicate: no hidden recentering.
Additional outputs include terminal maxima, final errors, cart displacement,
normalized effort/variation/saturation, sampled crossings of all horizontal
orientations, rotation range and signed net turns, numerical failures and
fallback events. Crossings are sampled counts (possible between-sample reversals
are unresolved); rotation range is floor((max(theta)-min(theta))/(2pi)), not a
claim about total traveled angular distance.

## Verified reproduction

Adapter contracts passed: saved C states exactly equal IC_SEED 20261009, six
authored angles[-60,-30,-15,15,30,60]degrees, upstream import provenance,
held-input application and timing, episode separation. Genuine CVXPY is installed;
the upstream unused-import workaround is inactive.

First replay: seed 0 L on all six A cases and first 3 C states matched archived
binary outcomes and compared numerical values exactly. Then full L replay:

| Fitted seed | Cases | A legacy | B recovery | C periodic |
|---|---:|---:|---:|---:|
|0|200|122|122|122|
|1|200|130|130|130|
|2|200|128|128|128|

All 600 per-case A/B outcomes match archives. Final theta, final cart position and
maximum cart excursion differences are exactly 0 for all 600. Two initial launches
omitted `--seed` despite seed1/2 directory names; the summary's duplicate guard
caught the error. Those seed0 duplicates remain preserved but excluded. Only
explicit `*-corrected` seed1/2 directories contribute to this table.

Upstream excludes raw arrays and trained checkpoints from Git. One original dev
model was reconstructed on CPU, seed 0; it is not an exact original CUDA
checkpoint replay. Test N-step RMSE differs from archived seed 0 by at most
2.84e-7 across states, and reconstructed L agrees to floating-point rounding.
H matches all 9 authored/subset binary outcomes, though continuous trajectories
differ, as expected with separate training and CPU versus CUDA CEM randomness.
This verifies small-suite outcomes, not full H/C200 reproducibility.

| Reconstructed seed0 | n | H A/B/C successes | matched L A/B/C successes | H-only / L-only |
|---|---:|---:|---:|---:|
| Six authored starts |6|4|6|0 /2|
| First3 random starts |3|1|2|0 /1|

The subsets are reproducibility checks, not tuning data or independent evidence
of improvement. Do not pool them as independent fitted-model replications.

Reconstruction hashes (SHA256):

- Checkpoint:`c3f6032f5e35b6bb36395a4eec23813ac41e44a4c72a04037b03629f5cbf4b4f`
- Dataset:`0e825cda7821de4afbe839f5da022b5942c6f8b14dd5208a040f05278cbde119`

## Decoder and optimizer diagnosis before changes

525 untouched test episodes, excluding exact x0 from forecasting statistics:

| Quantity | Cart position | Normalized cart speed | Angle | Normalized angular speed |
|---|---:|---:|---:|---:|
| Rollout NRMSE (target population std) |0.01018|0.01695|0.00910|0.01839|
| R² |0.999896|0.999713|0.999917|0.999662|
| 30-step RMSE |0.04086|0.07289|0.05556|0.11521|
| 30-step95th absolute error |0.07748|0.10865|0.12165|0.20975|

At 30 steps, angular RMSE 0.0556 rad exceeds the endpoint angle tolerance 0.02 rad.
Angular-speed RMSE is 0.361 rad/s physically, versus 0.05 rad/s terminal tolerance.
High global R² therefore does not establish precision adequate for balancing.
Full horizon-by-coordinate tables are in `diagnostics-seed0/diagnostics.json`.

Initial-state, input-coordinate and predicted-prefix residuals are 0. A finite
difference of the full decoder has rank 34; identity coordinate projection
guarantees this in exact arithmetic. There is no evidence of coordinate collapse.
Future angle/rate sensitivities and singular spectra are saved separately:
full decoder rank alone cannot validate control-relevant state sensitivity.
The four learned angle/rate input-Jacobian spectra have smallest singular values
0.0196,0.0195,0.0203,0.00178 and largest2.16,2.16,2.44,0.437, respectively.
They show no complete learned input collapse at these points, but were not
compared with simulator Jacobians (which are excluded from this work).
Learned one-step zero-input equilibrium bias is approximately
[-0.000498,-0.000755,0.000180,-0.001645] in stored coordinates; its physical
angular-speed increment is -0.00515 rad/s per sample. Equilibrium drift is a
plausible precision bottleneck, not established as the sole cause.

Four first validation states were optimized from zero warm starts. The exact
chosen input sequences were replayed through the upstream plant; terminal angle
prediction errors were 0.0289, 0.0713, 0.1236, 0.0223 rad. No plant prediction entered
optimization. In one case chosen CEM cost 762.67 exceeded a feasible L-generated
sequence's learned cost 702.23: a concrete optimizer-suboptimality witness. That
sequence was diagnostic only and never used as a manifold-control action.
Chosen plan costs agree with best sampled costs up to float32 batch-versus-single
rollout roundoff (about0.004 absolute); no global convergence claim follows.
All four CEM solves took about 83 ms and missed 20 ms on this CPU.

Source audit: original CEM lacks nonfinite-cost masking; original warm starts
always shift by 5 even after intervening local-mode samples. The unused original
Adam solver stores a post-update coordinate snapshot against a pre-update loss.
These remain preserved in originals. Dominant observed limits are prediction
accuracy at balancing tolerances and search quality; five-step feedback and
short-horizon terminal objectives are hypotheses addressed by separate ablations.

## Switchable experiments

`src/manifold_experiments.py` adds standalone control without a linear branch:

- Safe CEM masks invalid costs/states before ranking and handles fewer finite
  elites; failure stays explicit.
- Projected Adam optimizes all30 bounded free inputs with three starts: previous
  feasible plan, zero, independent random. It retains the best evaluated
  input/state/loss snapshot and evaluates the final iterate. Decoder weights freeze.
  The batched optimizer therefore carries90 scalar variables for three separate
  30-coordinate candidates, not a different90-dimensional trajectory manifold.
- Replanning1 versus5; warm starts advance by actual sample age and expire after
  missed measurements or horizon age. An invalid solve uses the shifted previous
  feasible action or zero, with a recorded fallback event. It never calls L.
- Optional smooth2(1-cos(theta)) angle cost retains local quadratic curvature;
  optional five-state terminal window and previous-applied-input smoothness term.
  No hard horizontal constraint or cart recentering requirement is added.

The fixed-model screen uses independently saved seed20261017 tuning states in
[-.5,.5]×[-.8,.8]×[-.6,.6]×[-1,1]. It is a small local recovery screen, not a
substitute for broad final C200/fresh200 testing. Runtime has a resumable300s
between-case budget; a case already running is allowed to finish.

Completed 16-case tuning screen (two starts, one independently fitted model):

| Variant | A legacy | B recovery | C periodic |
|---|---:|---:|---:|
| Original standalone M, queue5 |0/2|0/2|0/2|
| Original H |1/2|1/2|1/2|
| Original L |2/2|2/2|2/2|
| Safe CEM, queue5 |0/2|0/2|0/2|
| Safe CEM, feedback1 |0/2|0/2|0/2|
| Projected Adam, original objective |2/2|2/2|2/2|
| Adam, periodic objective |1/2|1/2|1/2|
| Adam, periodic + terminal window5 |2/2|2/2|2/2|

Plain Adam was frozen using the recorded tie rule. The window variant improved
second-start angular settling to0.64s, but its final cart speed was0.247m/s and
it did not improve recovery counts. No inference of superiority over L follows
from two paired ties. Plain Adam versus M has two wins/no losses on these tuning
starts; versus L it has zero wins/zero losses. These are not independent fitted
model replications or final test results.

Safe CEM5 versus safe CEM1 isolates feedback frequency within the new wrapper;
their warm-start tail holds the last planned action. Original M versus safe CEM5
also changes masking/warm-start handling and is not a single-factor comparison.
The original arms' `solver_evaluations` counter in this first screen is an
uninstrumented zero; it must not be read as zero compute. The evidence exporter
derives their actual forward counts as768×saved CEM solves. Native Adam records
its own candidate count; backward-pass computation is additional.

The training screen starts both arms from the original seed0 checkpoint, uses
the same measured training episodes, scaling, minibatch orders and window starts,
and fine-tunes10epochs at1e-4. One arm adds a5-step state-normalized rollout MSE
to the unchanged dx-normalized one-step loss. Both select using the same
validation one-step+rollout criterion, never test starts. Values0.00397709
(one-step) and0.00398455(multistep) show no benefit for this tested configuration.
With original standalone M optimizer/objective, both fine-tuned models recover
0/2 tuning starts versus L2/2. This configuration is not promoted.

The matched static optimizer test used four first-validation states, identical
zero warm starts and the original quadratic objective:

| Validation state | Original/safe CEM cost | Adam20-step cost |
|---|---:|---:|
|0|1421.24|1398.56|
|1|762.67|1751.57|
|2|2723.11|4714.50|
|3|36629.13|62912.87|

Adam improves only1/4 static solves at this budget; safe CEM agrees with original
CEM on these finite cases. CEM uses768 forward candidates; Adam uses63 forward
candidates plus60 backward paths, so these are not equal-compute experiments.
After startup, measured times are about0.08sCEM versus0.21sAdam. Runtime includes
host contention from other bounded diagnostics and is not a hardware benchmark.

On the first closed-loop tuning state, Adam recovers with terminal maximum angle
0.00321rad and physical speed0.00983rad/s, no fallback. Original M, H, safe CEM5
and safe CEM1 fail; L recovers. Adam's final cart position is4.304m and its median
solve is0.200s, with100%20ms deadline misses. This preliminary feedback result
does not contradict the static test: repeated warm starts and feedback differ
from zero-initialized single solves. It is not final held-out superiority.

On the second tuning state, quadratic Adam again recovers (terminal maximum
angle 0.00622 rad, physical angular speed 0.0110 rad/s). Its energy is 3.612 in
normalized effort units, final cart position 3.917 m, and settling time 3.82 s.
Periodic Adam fails the joint terminal endpoint on that state despite meeting
the angle tolerance. Thus the periodic objective is not promoted. Selection
rule recorded before the remaining objective results: maximize tuning recovery;
on ties retain the original objective and terminal weight. This screens a
candidate, not a claim of generalization from two starts.

A direct independent-horizon decoder was not substituted: current recursion has
exact predicted-prefix consistency, and the diagnostics do not identify coordinate
compression as a problem. A direct model would need explicit cross-time feasibility
and prefix validation. Geometry penalties are likewise deferred without evidence
that their extra computation addresses the measured errors.

## Historical reproduction (original submodule layout)

The commands in this section record the original experiment, at repository
commit `f41056d`. Source-hash guards intentionally prevent reusing those old
manifests with changed integration code. Use [LAYOUT.md](LAYOUT.md) for the
current root layout, migrated frozen continuation and fresh output directories.
The controller/model/scoring settings have not changed.

Run from the repository root. Current tested interpreter is
`/tmp/tmc-validation-env/bin/python`; this temporary environment is not portable.
To rebuild an environment, use Python3.11, install torch from the CPU wheel
index, then the pinned `docs/validation/requirements-cpu.txt`. Existing Conda
`environment.yml` is also retained; different versions must be recorded.

```bash
git submodule update --init external/TrajectoryManifoldControl
export TMC_PY=/tmp/tmc-validation-env/bin/python
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "$TMC_PY" -m pytest -q
"$TMC_PY" scripts/validation_benchmark.py --verify --output saves/validation/adapter-checks
"$TMC_PY" scripts/validation_benchmark.py --seed 0 --arms L --set A --output saves/validation/L-seed0-A
"$TMC_PY" scripts/validation_benchmark.py --seed 0 --arms L --set C --limit 3 --output saves/validation/L-seed0-C3
"$TMC_PY" scripts/validation_benchmark.py --seed 0 --arms L --set C --output saves/validation/L-seed0-C200
"$TMC_PY" scripts/validation_benchmark.py --seed 1 --arms L --set C --output saves/validation/L-seed1-C200-corrected
"$TMC_PY" scripts/validation_benchmark.py --seed 2 --arms L --set C --output saves/validation/L-seed2-C200-corrected
"$TMC_PY" scripts/summarize_validation.py saves/validation/L-seed0-C200 saves/validation/L-seed1-C200-corrected saves/validation/L-seed2-C200-corrected --output saves/validation/L-full-summary --figures

# Reuses checkpoint/data if they exist; new CPU reconstruction otherwise.
"$TMC_PY" -u scripts/validation_benchmark.py --seed 0 --train --model-dir saves/validation/models/reconstructed-seed0 --arms H L --set A --output saves/validation/reconstructed-seed0-A
"$TMC_PY" scripts/validation_benchmark.py --seed 0 --model-dir saves/validation/models/reconstructed-seed0 --arms H L --set C --limit 3 --output saves/validation/reconstructed-seed0-C3
# Diagnostics refuse to overwrite their output: choose a new directory on repeat.
"$TMC_PY" scripts/diagnose_validation_decoder.py --model-dir saves/validation/models/reconstructed-seed0 --output saves/validation/diagnostics-repeat
"$TMC_PY" scripts/compare_manifold_solvers.py --model-dir saves/validation/models/reconstructed-seed0 --output saves/validation/static-solvers-repeat

# Resume: same invocation and output, unchanged model and source hashes.
"$TMC_PY" -u scripts/run_manifold_ablation.py --model-dir saves/validation/models/reconstructed-seed0 --output saves/validation/tuning-screen-v1 --limit 2 --seconds 14 --adam-iterations 20 --max-wall-seconds 300
"$TMC_PY" scripts/train_multistep_ablation.py --model-dir saves/validation/models/reconstructed-seed0 --output saves/validation/multistep-repeat --epochs 10 --window 5
"$TMC_PY" scripts/run_manifold_ablation.py --model-dir saves/validation/multistep-screen-v1/one_step --output saves/validation/model-one-step-control --variants M L --limit 2 --seconds 14 --max-wall-seconds 90
"$TMC_PY" scripts/run_manifold_ablation.py --model-dir saves/validation/multistep-screen-v1/multistep --output saves/validation/model-multistep-control --variants M L --limit 2 --seconds 14 --max-wall-seconds 90
```

The completed screen selected plain Adam and it was frozen before final native
scores were read. Freeze/verification and bounded nominal/robustness commands
below were executed. C200 stopped after two paired starts and is resumable. Include separately
trained seeds1/2 in a future bundle before starting those evaluations:

```bash
# Freeze once: existing bundles are protected against overwrite.
"$TMC_PY" scripts/frozen_manifold_validation.py freeze --screen saves/validation/tuning-screen-v1 --variant adam1 --models saves/validation/models/reconstructed-seed0 --bundle saves/validation/frozen-native.json
"$TMC_PY" scripts/frozen_manifold_validation.py evaluate --bundle saves/validation/frozen-native.json --arms L H native --set C --output saves/validation/final-seed0-C --verify-only
# Reissue the same command to resume; budget is checked between complete cases.
"$TMC_PY" -u scripts/frozen_manifold_validation.py evaluate --bundle saves/validation/frozen-native.json --arms L H native --set C --output saves/validation/final-seed0-C --max-wall-seconds 180
"$TMC_PY" -u scripts/frozen_manifold_validation.py evaluate --bundle saves/validation/frozen-native.json --arms L H native --set fresh --output saves/validation/final-seed0-fresh --max-wall-seconds 300
# Executed baseline-only long-run checks; native long-run command follows.
"$TMC_PY" -u scripts/frozen_manifold_validation.py evaluate --bundle saves/validation/frozen-native.json --arms L H --set A --indices 2 3 --seconds 60 --output saves/validation/final-seed0-A60 --max-wall-seconds 30
# Not executed: three arms including native at60s.
"$TMC_PY" -u scripts/frozen_manifold_validation.py evaluate --bundle saves/validation/frozen-native.json --arms L H native --set A --indices 2 3 --seconds 60 --output saves/validation/final-seed0-native-A60 --max-wall-seconds 300
# Limited robustness: fixed authored index2, original upstream intervention.
"$TMC_PY" -u scripts/frozen_manifold_validation.py evaluate --bundle saves/validation/frozen-native.json --arms L H native --set A --indices 2 --condition input_delay --output saves/validation/final-seed0-delay --max-wall-seconds 300
"$TMC_PY" -u scripts/frozen_manifold_validation.py evaluate --bundle saves/validation/frozen-native.json --arms L H native --set A --indices 2 --condition mass_plus25 --output saves/validation/final-seed0-mass --max-wall-seconds 300
```

The frozen evaluator checks source/scoring/checkpoint/data hashes, applies all
three endpoints identically and saves within-seed exact paired binary tests.
It measures every controller call including L. Computation latency is recorded
separately from ideal simulation time; it is not silently modeled as zero-delay
hardware performance. At the observed0.2s/solve, native200×700 steps alone is
approximately7.8CPU wall hours per fitted seed, before other arms and holdouts.
Mismatch/delay interventions call upstream `e3_robustness` directly, only during
evaluation. Delay starts with zero applied input and shifts commands one sample;
raw files distinguish commands from applied force, and effort scores use applied
force. Mismatch uses upstream nominal-coordinate conversions; the controller
receives no true mismatch parameters. Original H/L and native use the same
intervention. Optional indices explicitly identify limited subsets.

## Frozen independent and robustness results

Freeze SHA256:
`97a3d77c12d69a30a8ef799f8703c77f425f0dc94f762c88a0403c7a01ad9f41`.
The nominal C200 run is partial and resumable. Its first native case failed all
three endpoints while L succeeded; settings are unchanged after seeing this.
Do not report this unfinished run as a200-state comparison.
That native case met the angle tolerance (terminal maximum0.01271rad) but missed
physical angular speed (0.06899rad/s versus0.05). It had no numerical failure or
fallback; median controller-call time was0.2115s. This remains a failure under
every specified endpoint, rather than a reason to weaken the tolerance.

The bounded allocation finished two complete paired starts (original C indices
0/1), spending306.2s because the between-case limit permits a started case to
finish. All three endpoints give H1/2, native1/2, L2/2. Native versus L: zero
paired wins, one loss, exact McNemar p=1.0 with only one discordant pair. The
observed gap is one recovery of two starts (50percentage points); this is not
an estimate of the full200-state gap and provides no superiority evidence.

| Original C index | H recovery | Native recovery | L recovery | Native terminal max angle / speed | Native effort | Final native cart p |
|---|---|---|---|---|---:|---:|
|0|fail|fail|pass|0.01271rad /0.06899rad/s|22.817|−6.213m|
|1|pass|pass|pass|0.00715rad /0.02587rad/s|1.794|4.259m|

Native angular settling on index1 is2.40s versus L0.98s (H1.26s); index0 never
settles under the terminal predicate. Native median call latencies are0.2115s
and0.2035s, with100%20ms misses. L medians are8.46/8.28microseconds with no
misses. H's median is dominated by queue/local calls; its p95 on index0 is
86.9ms and20%of calls miss20ms. These are CPU measurements under concurrent
evaluation, not certified real-time hardware results. No controller fallback or
numerical failure occurred in these two native cases.

Completed limited checks, all A/B/C endpoints agree:

| Test (predeclared authored starts) | Original H | Native Adam | L |
|---|---:|---:|---:|
| One-sample input delay,−15° |0/1|1/1|0/1|
| +25% cart mass,−15° |1/1|1/1|1/1|
|60s,−15°/+15° |2/2|not run|2/2|

The delay result is one paired native-only win versus L, not evidence of broad
robustness superiority. Models and controller settings were frozen before these
scores. No changes were selected using this result. Raw delayed runs store both
commands and applied inputs; metrics and force plots use applied inputs.

Result manifests contain configurations, initial identities, model/data/source
SHA256 and versions. Historical artifacts stay under ignored `saves/validation/`;
this report and checkpoint notes are versionable. Figures (five panels with angle,
physical rate, cart position/speed and normalized force, plus terminal tolerances)
are in `initial-replay/`, `L-full-summary/`, and `reconstructed-seed0-summary/`.
Generate them with `scripts/summarize_validation.py ... --figures`.
For native/frozen case directories, the tested figure-only command is:

```bash
"$TMC_PY" scripts/summarize_validation.py saves/validation/tuning-screen-v1 saves/validation/final-seed0-C saves/validation/final-seed0-delay saves/validation/final-seed0-mass saves/validation/final-seed0-A60 --output saves/validation/frozen-figures --figures-only
"$TMC_PY" scripts/export_validation_evidence.py --output docs/validation/EVIDENCE.json
```

The versionable `EVIDENCE.json` includes compact tables, per-case secondary
metrics, frozen configuration, artifact hashes and source hashes. Raw arrays and
checkpoints stay outside Git. Forty-seven tests pass; adapter import/units/split,
periodic scoring, finite-candidate handling, gradient-snapshot acceptance,
actual-age warm starts and freeze guards are covered. Syntax/whitespace checks
also pass; upstream remains clean. No commit or push was made.

## Remaining work and inference limits

The original-model/quadratic-objective Adam candidate is frozen after the small
validation screen. Full original H evaluation, complete improved-controller
comparisons for seeds0/1/2, fresh200 and native60s are not claimed complete. The original
report's broad H/L gap remains historical evidence; the reproduced small-suite
gap is2/6 authored and1/3 random-subset recoveries. Native tuning results cannot
be presented as held-out improvement. Do not use the already viewed final-state
scores to select new weights. Further experiments must continue on the saved
independent tuning starts and validation episodes.

The target to exceed L on independent terminal recovery is not achieved.
The strongest existing bottleneck evidence predates final evaluation: horizon
prediction errors exceed balancing tolerances, equilibrium bias is nonzero,
and cold-start Adam loses to CEM on3/4 broad validation states at20iterations.
The successful tuning suite covers a smaller local box and cannot establish
broad recovery. Next investigate these using training/validation data and new
independent tuning starts, alongside latency reduction; preserve this frozen
run and never select changes to pass its observed final cases. Completing the
unchanged frozen evaluation is a separate resumable task. No larger network,
direct decoder, longer horizon or geometry penalty is validated by this session.
