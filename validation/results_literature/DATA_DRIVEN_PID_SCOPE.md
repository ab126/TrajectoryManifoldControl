# Data-driven PID comparison

User-approved additional comparator, October 12, 2026. This supplements stages
1–8; it does not replace controlled Deep Koopman or controlled DFINE.

## Controller and data privileges

Add one main-table controller, `DD-PID`: an outer cart-position PD loop producing
an angle reference and an inner angle PID loop producing force. Use the measured
position, velocity, angle and angular velocity available to the other full-state
controllers. Include explicit reference limiting, force saturation, anti-windup
and per-episode integrator reset. Differentiate the measured angle rather than
the reference; document the derivative sign and physical/normalized units.

The cascade is a declared structural prior. Position/angle loop separation is
also used in the [MathWorks cart-pendulum example](https://www.mathworks.com/help/control/ug/control-of-an-inverted-pendulum-on-a-cart.html).
This comparator is our data-fitted PID adaptation, not a reproduction of that
example's model-based tuning or a claim of model-free identification.

Identify the tuning model exclusively from each seed's saved original training
episodes and measured inputs. Reuse the verified data-fitted local A/B where its
training identity matches. Do not use analytic plant equations, true parameters,
test trajectories, or simulator-in-the-loop gain optimization. The simulator is
for final evaluation only. No L catch or neural fallback is included in DD-PID.

Tune the five gains on the identified model with the shared physical state/input
cost. Use training-only initial conditions, and freeze the tuning starts, gain
bounds, reference bound, anti-windup rule, simulation duration, optimization
limits and selection rule in a manifest before tuning. Limit to six deterministic
optimization starts per seed and 100 iterations per start. Verify unsaturated
local closed-loop stability on the identified augmented model; distinguish that
check from a guarantee for the nonlinear plant. Do not infer gain signs from
an assumed analytic plant convention. Report unsuccessful tuning as such.

The PID tuning objective is a controller-design objective; the common 30-step
forecast metric applies to the fitted predictor and is reported separately.
Do not tune the controller using final endpoint counts. Preserve all candidate
scores and the selected gains, integrator/anti-windup settings and model hashes.

## Matched evaluation and evidence

Use fitting seeds 0–2, the existing whole-episode splits, 20 ms sampling, force
bounds, primary angular-tail/no-fall endpoint and original reference identities.
Run the same bounded pilot, six authored plus fifty random states at 14 s and
the three prescribed 60 s cases. Then fill the shared original C200 column and
the separate fresh 200-state holdout. Apply the same delay, mismatch and noise
tests and optional maps/data-efficiency tests when the allocation permits.
Reuse nominal cases only after verifying every state/model/config/endpoint identity.

Report per-seed successes/denominator, paired discordant counts against H/L,
falls, angle/rate tails, cart displacement/settling, saturation, effort, latency
and integral-state behavior. Include DD-PID in controller-only summary tables
and common-axis figures; keep the introductory map H/L-only. An E4 comparison
requires fitting PID on E4 training data and belongs in the separate E4 cohort.

Before evaluating, check force sign and units, discrete integral and anti-windup
updates, saturation, reset, derivative response and an independent literal
controller calculation. Save selected raw trajectories for endpoint replay.

## Execution and accounting

The existing supervised worker owns implementation and launches. Add durable
code under experiments/literature and results under results_literature/pid.
Use a dedicated /tmp/trajectory-literature-envs/pid environment, explicit taskset
80–127, single-thread libraries and the existing usage/credit guards.

Allocate at most 1 aggregate CPU core-hour and zero GPU-hours from the unspent
core reserve, after checking prior charges and outstanding commitments. This
suballocation does not increase the reserve or the 30 GPU-hour / 84 aggregate
CPU core-hour ceilings and does not displace the approved mandatory stages.
If less reserve remains, use only that remainder and mark cap-limited work
explicitly. Pilot and project the full requested matrix before expansion; retain
exact denominators at the cap. Do not interrupt or duplicate active jobs.

At the next checkpoint, acknowledge this scope through reviews/WORKER_ACK.md,
freeze the implementation manifest, and queue the bounded PID work alongside
the remaining approved comparisons. No further research preference is pending.
