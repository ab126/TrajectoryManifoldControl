# I02 arithmetic contract v1

[oracle_math_v1.py](oracle_math_v1.py) implements the I02 portion of
[OA-1](oracle_amendment_proposal_v1.md), consuming I01's immutable
`ScientificConfig`. It does not import the SUT, `common.py`, old diagnostics,
the plant or a launcher. Scientific libraries are lazy imports. No device
setup, prediction rollout, case sampling, CEM search or job execution occurs.

The [I02 decision record](reviews/i02_v1/decision.json),
[test results](reviews/i02_v1/test_results.json), and
[manifest](reviews/i02_v1/manifest.json) distinguish implemented/tested
arithmetic from pending independent review and scientific validation. Full
OA-1 remains specified, unadopted and unexecuted.

## Scoring and selection interface

All arrays must be float64. Matching NumPy ndarrays or Torch tensors are
accepted; mixed backends/dtypes/devices fail with `MathError`. Torch operations
remain on the input device. This interface does not establish the caller's
GPU UUID/runtime identity; I05 must verify that against the reviewed release.

| Function | Contract |
|---|---|
| `objective_components(config, X, U)` | Shapes `(...,T+1,4)`, `(...,T,1)`, any positive T; returns `stage_state`, `stage_input`, `terminal`, `slew`, and `total` |
| `score_candidates(config, X, U, first_seen_offset=0)` | Shapes `(M,61,4)`, `(M,60,1)`, M>0; returns components, `arrays_finite`, `input_bounded`, `arithmetic_finite`, `eligible`, `feasible`, `D`, `W`, `first_seen` |
| `rank_candidates(scores)` | Stable ordered indices: eligible candidates before invalid, then `(infeasible,D,W,J,first_seen)` |
| `elite_moments(config, scores, U)` | Same rank, first 32 eligible candidates; returns indices, unsmoothed mean, unbiased std floored at .02 |

Stage cost sums nodes `0..T-1`; terminal uses node T and **12Q**; slew uses
adjacent inputs only. Q/R/Qf and rate weight come from the frozen config.
There is no dt multiplier, angle wrapping or first-input slew term. The
planning objective remains separate from common H/L/E evaluation cost.

D and W include **all 61 nodes**, including initial and terminal. Safety and
input bounds are inclusive. Raw nonfinite arrays, out-of-bound inputs, or
nonfinite cost/excess arithmetic are ineligible. The last case remains visible
as `arrays_finite=true, arithmetic_finite=false`; overflow is not silently
clipped into a valid score. Invalid entries have infinity sort keys and never
enter moments. Internal nonfinite diagnostic arrays must be explicitly encoded
by I03's trace contract, rather than written as invalid JSON numbers.

I03 must use this same rank for both elites and the incumbent, preserve the
original first-seen index across all three iterations (offsets 0/256/512), and
retain the incumbent across iterations. The arithmetic functions do not keep
history. I03 enforces the locked 256 population/three iterations/one start,
deterministic slots, RNG contract, smoothing and action/fallback state machine.

Fewer than 32 eligible candidates raises `SearchError`, code
`insufficient_finite_candidates`, with `finite_count` and `required`. No moment
update occurs. I03 must clear queue/previous plan, latch exception failure,
apply zero for one sample, and log the exception. Ordinary absence of a feasible
plan with sufficient eligible candidates remains a distinct search outcome.

There are no population-to-CPU transfers in scoring/ranking. Elite moments
need one scalar eligible-count read on Torch; their arrays remain on device.
This structural property and CPU backend parity are tested. CUDA correctness
and amended latency, including sorting/logging overhead, are **not measured**
by this batch and remain prerequisites before adoption.

## Gain artifact and local arithmetic

`upright_rk4_matrices(config)` returns the exact held-input RK4 derivative
A/B from OA-1's uniform-rod upright Ac/Bc. `construct_gain(config)` calls
SciPy's stabilizing discrete DARE with frozen Q/R, computes K, and validates
A/B/P/K, positive definiteness, residual and linear closed-loop radius.
It compares K/P to the byte-pinned original proposal with `atol=rtol=1e-10`.
No ridge, retry, weight change or alternative gain is permitted on failure.

The resulting immutable `GainArtifact` provides `canonical_bytes()`, `sha256`
and a fresh `to_dict()` copy. Its exact fields are `artifact_version`,
`scientific_sha256`, `specification_sha256`, `method`, `A`, `B`, `P`, `K`,
`spectral_radius`, `dare_relative_residual`, `reference_verified`.
Full binary64 matrices are serialized in sorted compact ASCII JSON without
a newline; reference verification means numerical agreement, not independent
author certification. The residual check is infinity-norm residual divided by
`max(1, ||P||_inf) <= atol+rtol`; closed-loop spectral radius must be <1.

`load_gain(config, path, expected_sha256=...)` requires an externally supplied
byte hash and then repeats numerical/structural validation. Missing files are
`MathError('unavailable', ...)`; wrong hash, malformed/duplicate/nonfinite JSON,
wrong config/reference/matrices/diagnostics or noncanonical bytes are rejected.
I03 must load/bind the independently reviewed artifact rather than approve its
own output. This batch saves a same-author candidate artifact for that review;
it does not fill or adopt the I01 release metadata.

`LocalArithmetic(config, verified_gain)` caches Q/P/K once. It accepts CPU
float64 states of shape `(4,)`, and exposes `safe`, `state_cost`, `V`, `input`,
`transition`, and `predicate`. `input` returns `(clipped_input, raw_input)`.
`transition(x,u,next_x)` requires exactly the computed clipped local input and
records both inequality sides, V/V-next, cost, saturation, safety and decrease.
Nonfinite state, feedback or predicate arithmetic raises `MathError`; I03 must
use the exception fallback, not treat it as an ordinary local exit.

`predicate(x,predicted_states,mode='entry'|'stay')` requires exactly 61/2 nodes
respectively, with first node equal to x. It checks the inclusive .75/1.0
initial-cost threshold, all nodes' safety, and every specified decrease test.
It returns every transition record, maximum saturation, minimum rail margin
in metres, minimum angle margin in radians, and minimum dimensionless safety
margin (each margin divided by its bound before taking the minimum).
It does **not** generate or certify the supplied local-feedback rollout.
I03 must supply it from the known model; I04 independently recomputes it.

The local tests use synthetic contraction predictions and isolated arithmetic
fixtures. The only computed gain evidence is fixed linear algebra. These do
not establish nonlinear entry-region validity, physical recovery, invariant
safety, hybrid success, finite-search completeness or global optimality.

## Verification and remaining boundary

[31 I02 unit tests](tests/test_oracle_math_v1.py) cover hand costs, all-node
safety, inclusive bounds, lexicographic ordering/ties, invalid exclusion,
unbiased/floored moments, insufficient-candidate errors, CPU NumPy/Torch parity,
direct linear RK4 slope parity, fixed K/P and residual/stability, artifact
rejection, clipped feedback, and entry/stay/decrease arithmetic. I01's 32 tests
remain the regression dependency; the old 29-test scientific campaign is not
rerun here.

Independent gain/cost/rank/predicate review, nonlinear guard checks, CUDA/runtime
checks, amended timing, archive contract, feasible allocation and OA-1 adoption
remain open. I03/I04 follow I02; I05 follows those batches. Completed diagnoses,
H/plant, old source snapshots/runs, endpoint/box/device/precision, primary IUT
and secondary Holm remain as previously locked. No new experimental allocation
or downstream execution follows from these software tests.
