# OA-1 trace contract v1

This contract specifies review data; it authorizes no execution. I03 produces
records; I04 independently recomputes their numerical/branch meaning. Source,
scientific config, gain, this contract and any interpretation addendum must be
externally frozen and hashed before adoption. Same-author replay is not an
independent-author certificate.

## Representation

Each case has normalized float64 `X` (1001,4), float64 `U` (1000,1), indexed
action records, terminal status and original case/seed identifiers. Incomplete
cases retain exactly their observed prefix; no padding can manufacture success.
Nonfinite suffixes are explicitly marked. Raw numeric arrays retain their
binary dtype/shape; nonfinite diagnostics are represented as array values plus
flags, never ambiguous JSON NaN/Infinity. Compact headers use finite JSON.
Case arrays, solver arrays and checkpoints belong in the separate lab archive.
Git keeps code, compact manifests/decisions and the output-free notebook.

The archive serializer and destination must be separately pinned before launch.
An absent file is unavailable evidence; a wrong hash/value is a failed check.
I04 must not regenerate missing experimental artifacts to complete an audit.
Reconstructing candidates from **available** seed/trace inputs is audit replay,
not a new calibration or certification of attainability.

## Planning record

Required fields: `scientific_sha256`, `evidence_kind` (`scientific` or
`synthetic_fixture`), integer `case`, integer physical `sample`, RNG `key`,
`seed`, float32 CPU draw SHA256, `warm_controls`, `warm_states`, fresh
`local_controls`, `local_states`, `iterations`, `selected`.

RNG payload is compact JSON `[20261006,"cem",["oracle_v1",case,0],sample]`;
SHA256 first eight bytes, little endian, modulo `2**63-1`. A local CPU Torch
generator draws the **full** `(3,256,60,1)` float32 tensor before any slot
replacement, cast to float64 on the prediction device. Global RNG state and
branch/call count do not choose noise. The full draw hash includes deterministic
slots' discarded draws. A fixture noise provider must mark fixture evidence.

Each iteration records mean/std before and after; per-candidate `first_seen`,
`arrays_finite`, `input_bounded`, `arithmetic_finite`, `eligible`, `feasible`,
`D`, `W`, `J`, four objective components; elite indices, eligible/feasible
counts, and whether moments updated. Global first-seen indices are 0..767.
The original shifted warm plan remains slot 3 every iteration; fresh local
plan is constructed once per call, not subject to entry eligibility.

`selected` contains full controls/predictions, first-seen index, eligibility,
feasibility, cost components and rank `(infeasible,D,W,J,first_seen)`. The
incumbent is retained across iterations using the same ranking as elites.
Partial solver records survive exceptions; fewer than 32 eligible candidates
must show no moment update and an exception fallback.

## Action record

Required fields: case/sample/config hash/evidence kind, observation, mode before
and after, branch, input (null only for a nonfinite stop), local guard, queue
length before/after, previous-plan sample and actual warm shift, invalidation
reasons, queue guard, planning record or null, planning-call count (0 or 1),
zero-input prediction where relevant, latched safety/exception flags, first
violation sample, error code/cause where relevant, and control/solve timing.

Local guard records include eligibility threshold, predicted nodes/controls,
every V/decrease/safety value, maximum saturation, minimum rail/angle margins,
and dimensionless minimum margin. A threshold/safety rejection before rollout
is recorded explicitly; it must not be represented as a passed rollout.

Queue guards record expected node, allclose result, independently computed next
node and safety result. Actual age selects the saved node/control. Entry into
local, local exit, infeasibility, exception and interruption invalidate both
queue and previous plan. Ordinary queued-input guard failures also clear both
queue and previous plan, and replan once from zero, per the user's October 9
clarification. See [implementation decisions](oracle_implementation_decisions_v1.md).

Only the nonfinite plant state stops a finite-horizon case early. Finite safety
violations remain latched while the trajectory continues. Solver/local errors
latch exception failure and use zero for one sample. No-feasible-plan fallback
alone does not fail the physical endpoint. Both log the zero next-node safety
prediction; zero is not a safe-rescue certificate. New-plan guard failure is an
implementation error, not permission for a second solve on the same sample.

## Audit and storage bounds

I04 independently checks frozen config/seed/box, plant replay, objective/ranking,
RNG/slots/moments/incumbent, local predicates, queue history/action/fallback,
and the originally specified strict endpoint as a historical reference, plus
nested balancing/cart-rest diagnostics. The user has requested a revised Gate-A
endpoint; its exact rule and relation to OA-1 are pending and require a new
versioned config. It imports no I02/I03 numerical/policy helpers. It must
reject mutations and any purported passing 199-case or interrupted run.
Primary IUT and secondary Holm are unaffected by this oracle-development trace.

At 200000 worst-case calls there are 153600000 candidate rows. Seven float64
cost/excess components plus int64 index and five Boolean flags occupy at least
10.5984 GB uncompressed, before moments, guard records and metadata. Selected
and warm plan/state arrays alone add 0.9728 GB. JSON materialization would add
substantial space and control-step overhead. These are byte-count forecasts,
not storage/latency benchmarks. Choose/pin a lossless binary array archive,
capacity reserve and checkpoint format before execution; measure amended
full-step timing with logging only under a later explicit allocation.
