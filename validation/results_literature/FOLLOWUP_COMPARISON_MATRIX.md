# Matched 200-state controller comparison

Evaluate closed-loop controllers on `results_v1/initial_conditions.json`, `sets.C`:
the original 200 four-dimensional states, unchanged and in order, with fitting seeds
0, 1 and 2, nominal dynamics, 14-second horizon, original input limit and primary
endpoint. These are 200 shared states, not 200 training seeds. Retain the independent
fresh 200-state holdout already specified in the literature plan; report it separately.

## SMC reference

The main summary uses `SMC`, the existing sliding-mode feedback with its correction
enabled: ρ=0.9, β=0.45φ, original saved data-fitted A/B, surface and calibration.
Do not select between it and `SMC_linear` using observed evaluation results.
The zero-correction arm remains an ablation in detailed reporting, not a second
main-table entry. Do not retune using the 200 test states.

Use the original collection and fitted seed identities, as in E6 nominal results.
Reuse E6 nominal C50 cases only after verifying exact state/seed/model/configuration
and endpoint identity; evaluate remaining states and any unmatched cases under the
supervised worker. Save the new aggregate/case records under
`results_literature/smc_C200/`; preserve completed E6 files. Compare against the
matched E1 H/L cases on all 200 states. Per seed, require 200 unique indexed cases,
success counts with denominator 200, paired discordant counts versus H and L,
fall counts, and a reproducible source/config/data manifest. Inspect saved endpoints
on selected newly evaluated cases. Save a concise reviewer packet for existing
overseer. Allow at most 1 aggregate CPU core-hour for this reference extension,
charged to the existing core reserve, without increasing the overall 30 GPU-hour /
84 CPU core-hour ceilings. Pilot before expanding; report incomplete work at the cap.

## Remaining controller rows

Include `DD-PID` using [the approved scope](DATA_DRIVEN_PID_SCOPE.md): gains fitted
from measured training data, native feedback without common-L catch, the same
C200 states/seeds/endpoints, and separately fitted E4 cohorts if evaluated.

Fill the original 200-state column for each completed literature controller within
its stage allocation; prediction-only models have forecast tables instead.
The original every-step neural planner (`E1`, `M1`, `C`, 14 seconds) already has
200 cases per seed and zero successes in each seed; reuse those matched results
rather than repeating them.
Prioritize the SMC reference after an existing job checkpoint; do not interrupt or
duplicate a running job. Legacy E4 controllers have a different training collection:
to extend their 50-state results, evaluate their own H/L references and the same
200 states using the E4 model identities. Put those scores in a separately labeled
E4 cohort, not beside original-collection H/L as a matched ranking. Pilot cost and
use only remaining declared allocations; leave missing entries explicit when the
extension cannot fit. Do not increase budgets or invent a common-data claim.

## Presentation and diagnostic evidence

The introductory angle/rate map has only H and L, using saved 14-second results.
The main table reports controllers only and one SMC design. Keep diagnostic planner
variants and SMC ablations in detailed figures/reports where needed for attribution.
Explain that L's A/B and gain use only measured state/input transitions, with full
state observed and a known quadratic control objective; exact equations are used
by the simulator and separate stability/replay audits, not its fit or decisions.
Explain DeePC's feasible optimization versus invalid nonlinear predictions using
`results_v1/E4/deepc_matched_plan_replays.json`; do not claim solver failure or a
family-wide impossibility. Research summaries should explain the current methods,
evidence and limitations without recounting document revision conversations.
