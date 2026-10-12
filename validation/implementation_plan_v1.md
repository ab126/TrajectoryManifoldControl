# Validation implementation plan v1

Prepared October 8, 2026 against HEAD
`4c9de6db91d1c2306c0cb1318c8b309f4da390df` and the uncommitted OA-1 review package.
This is a plan to build the remaining validation machinery, not authorization
to execute new experiments. The original planning session produced documents
only. Later I01 implementation/status is recorded separately in
[oa1_decisions.md](oa1_decisions.md); that work does not authorize experiments.

The first implementation target is **an auditable, configuration-driven oracle
reference**, not another controller run. The original reference failed Gate A;
the proposed OA-1 policy has not been implemented or validated. Build small
reviewable changes, establish independent checks, and resolve allocation before
any new calibration. Leave dependent experimental stages stopped.

Sources: [evidence assessment](oracle_review_assessment_v1.md),
[OA-1 specification](oracle_amendment_proposal_v1.md),
[proposed manifest](oa1_config_proposed.json),
[validation handoff](validation_handoff.md), and the full
[campaign](handoff/plans/stress_test_plan_current.md) and
[critique](handoff/plans/critique_plan.md) plans. The
[collaborator memo](collaborator_memo_v1.md) is the short handoff.

## Nonnegotiable scope and evidence rules

All new code, fixtures, reports and manifests belong under `validation/`.
Preserve submitted H and the original uniform-rod plant at source commit
`325beae718ab093d3ca48b18a59ae61fd93798a3`; pin their existing SHA256 hashes.
Do not port unaudited `inverted_pendulum/` candidates or treat `external_repos/`
as validated baselines. Keep original run directories, execution snapshots and
usage ledgers immutable except for properly logged new occupancy.

OA-1 is an oracle-only proposed variant. Its changes must not reach H or L.
Keep the strict sampled 20-s endpoint, full box, L4 prediction device, float64
oracle/plant, N=60 and one CEM start as specified. Learned H/E remain float32.
The timing-agnostic-primary amendment is pending. Pure unit fixtures may use
CPU arithmetic and synthetic transitions; label them separately from scientific
device validation, numerical plant tests and measured controller performance.

Do not rerun the two completed diagnoses, repeat the old readiness campaign,
or spend their unused balance on new questions. Missing archived raw data is
`unavailable`, not `failed` or permission to regenerate it. Git retains source,
compact reports/manifests and the output-free notebook; raw arrays, checkpoints,
media and environments need a separate checksummed lab archive.

## First implementation batch: prepare OA-1 without launching it

Use the following PR-sized sequence. I01/I02 now exist; their exact interfaces
are [oracle_config_contract_v1.md](oracle_config_contract_v1.md) and
[oracle_math_contract_v1.md](oracle_math_contract_v1.md). Remaining filenames
are **planned additions**, not existing runnable tools. No experimental launch
command is ready yet.

| Order | Change and intended files | Meaningful verification | Acceptance / dependency |
|---|---|---|---|
| I01 | `oracle_config.py`, `tests/test_oracle_config.py`, `oa1_decisions.md`: strict schema, typed numeric contract, reviewed-config hash, source pins, adoption state and archive availability | Reject unknown fields, wrong source/reference/gain/Q/Qf/seed/box/endpoint/device/precision; reject proposed config for execution; unavailable archives remain a distinct status | Pure config loading never initializes CUDA, imports the SUT, launches work or writes into old runs. Review exact spec-to-schema mapping first |
| I02 | `oracle_math_v1.py`, `tests/test_oracle_math_v1.py`: objective components, finite flags, safety excess/ranking, gain construction and local predicate arithmetic | Hand-computable short plans; inclusive bounds; a cheaper unsafe plan loses to a feasible plan; stable ties; NaNs cannot enter moments; fewer than 32 finite candidates latches search-error failure | Exactly match OA-1 costs/indexing/rank and fixed gain within its tolerances; P is a local diagnostic, not Qf. Depends on I01 |
| I03 | `oracle_policy_v1.py`, `tests/test_oracle_policy_v1.py`: explicit planning/local/queue state machine, deterministic CEM slots and forensic records | Synthetic transition/solver fixtures exercise local entry/exit, actual shifts 0/1/5/60/>60, stale-plan invalidation, guard-triggered replanning, one-call-per-sample limit, infeasibility/exception fallbacks and latched failures | Every emitted action and branch has an auditable reason; RNG keys depend on case/physical sample, not branch history. Depends on I01–I02 |
| I04 | `oracle_audit_v1.py`, `tests/test_oracle_audit_v1.py`, `trace_contract_v1.md`: independent config/trace scoring and decision replay | Mutate seed, input, cost, feasibility rank, switch, queue age, failure flags or final-window sample; auditor must reject. A 199-case or interrupted run cannot pass | Auditor reads an externally frozen reviewed config, implements its own numerical scoring/ranking/predicate logic, and does not import the policy or its scoring helpers. Depends on the trace contract, not policy internals |
| I05 | `gate_a_v1.py`, `oracle_queue_v1.py`, `tests/test_oracle_queue_v1.py`: explicit config-selecting runner, preflight and capped lifecycle | Fake subprocess/clock fixtures: occupied lock, stale active job, insufficient reservation, mismatched HEAD/config/timing, interruption, audit failure and refusal to overwrite | Preflight cannot initialize a scientific run. Execution requires adoption, source/runtime checks, feasible cumulative reservation and fresh output directory. No downstream jobs are enqueued |

Start with **I01 alone**: make the proposed manifest reviewable as typed data,
add its rejection fixtures, and produce the decision record. Do not implement
all five changes in one large patch. Use the current `gate_a.py`,
`overnight_audit.py` and `overnight_queue.py` as documented historical references;
preserve their original behavior and snapshots while introducing explicit v1
entry points. Avoid importing `common.py` merely to read metadata: it initializes
the scientific dependency path. Reuse the documented RNG contract without
silently coupling lightweight preflight to Torch or CUDA.

The proposed JSON contains descriptive formulas and strings. I01 must map these
to a fixed typed implementation contract; **do not evaluate arbitrary formula
strings**. Any changed numerical choice requires a new proposal version. Keep
execution adoption metadata separate from scientific values; the runner must
compare against the externally reviewed config, not approve its own output.

For I03/I04, define records before implementation: true and observed states;
applied input; branch/mode; plan sample and elapsed shift; queue lengths and
invalidation reason; local entry/stay/decrease values; selected and warm controls
and predictions; per-candidate finite/feasible flags, D/W/J/rank; RNG keys;
solve/control-step times; fallback/exception/first-violation flags; complete and
partial prefixes. Candidate rank summaries suffice; full trajectories for every
sampled candidate are not required. Forecast worst-case storage for 200000
planning calls before choosing an archive format. Any logging reduction must
preserve the independent replay contract and be explicitly reviewed.

The collaborator should adversarially review I02–I04 or supply the independent
auditor; author identity and shared code/dependencies must be disclosed. Passing
the existing same-author equivalence does not approve the amended policy. No
subagents are requested or launched by this plan.

## Before a new Gate A: decisions and bounded empirical checks

Complete these prerequisites in order; software preparation does not itself
authorize the following numerical checks or measurements.

1. **Review the reference meaning.** Resolve and record whether OA-1's proposed
   Qp=2 planning, .75/1.0 entry/stay hysteresis, local decrease test, feasibility
   ranking and zero fallback are the intended reference. A short-horizon feasible
   candidate is not an invariant safe controller; zero fallback may be unsafe.
   Preserve the distinction between original balancing and reviewer-added rail/
   recentering requirements. Keep the timing-primary decision separate.
2. **Freeze a bounded entry/transition verification specification.** Include
   origin, deterministic states just inside/on/outside the two switch surfaces,
   coupled state directions and input saturation, plus the existing seven local
   probe states as documented context. Test the *new guard/hybrid transitions*,
   not another copy of the completed local-only diagnosis. State construction,
   number of checks, integration reference/tolerances, interaction counts and
   worker-time cap must be declared before execution. No threshold/gain search.
   The p=±1 probes start outside OA-1's new local entry region and are essential
   to explaining why the prior 7/7 does not validate the new hybrid.
3. **Obtain independent nonlinear guard and implementation review.** Compare
   the unchanged plant and oracle, check clipping/decrease and entry/exit
   behavior. Any sampled phase/decrease map is empirical; do not call it a
   certified four-dimensional safe region. A failed check returns the proposal
   for revision, without a quiet threshold adjustment or repeated runs until pass.
4. **Measure the amended workload under a separate allocation.** Pin host L4
   UUID/software and one worker; use synchronized end-to-end timing of the
   worst branches, including entry plus planning, fresh local warm construction,
   transfers, plant and logging. Save startup/shutdown/audit cost and archive
   footprint. The original solve-only timings and 0.08064 observed replan rate
   do not reserve this policy. Do not silently move prediction to CPU or float32.
5. **Freeze adoption, execution source and allocation.** The decision record
   identifies reviewed config/source hashes, independent reviewer, evidence,
   adopted versus pending choices, raw archive destination and allowed jobs.
   Only then may the revised runner expose a concrete Gate-A-only invocation.

Current Stage-0 usage is **1.385377/15 worker-hours**; **13.614623 h** remain.
Diagnosis is complete at **78.986003/900 s**. Even the unchanged one-start CUDA
reservation is **14.279505 h**, so current allocation does not fit. OA-1's
proposed **20 h total** Stage-0 cap is pending, with prospective slots of
1 h support/verification/timing, 17 h Gate-A-only queue, 0.5 h archive/review,
and about 0.114623 h reserve after prior usage. These are caps, not permission
or performance estimates. Support checks must themselves fit their frozen scope.

The prospective queue acceptance condition is
`200*1000*tau_amended*1.2/3600 + 1/3 <= 17 h`, i.e. tau≤0.25 s, with all costs
accounted once. Mean plus 20% is a budget model; enforce the actual cap and mark
interrupted work incomplete. If this condition or an actual measured overhead
reservation fails, hold for another explicit decision. Parallelism cannot reduce
summed worker-hours. Future Stage-1/2 feasibility is a separate problem; their
existing forecasts do not fit the original caps.

## New Gate A and failure decision

Once prerequisites are satisfied and execution separately authorized, run one
fresh OA-1 200-case development calibration from its declared case stream,
full box and unchanged strict thresholds. Audit complete traces and provenance;
require ≥196/200 and ≥95% in each nonempty angle-sign half. Report the nested
settling counts, all-time safety, exceptions, fallbacks and sign-half denominators,
not just a pass flag. Save every failure in the denominator and archive all runs.

If it fails, stop. Distinguish candidate infeasibility, gain/entry failure,
objective preference, optimizer implementation, queue/fallback effects,
numerical error and possible physical nonrecoverability. A failure to find a
safe sampled plan is not proof no safe control exists. Use targeted coupled
recoverability analysis only if needed and separately scoped; neither a
cart-only braking approximation nor more random starts certifies viability.
Half-box then quarter-box fallback requires a new reviewed version/allocation,
fixed thresholds and retained broad-box stress results. Never shrink automatically.

If it passes, independently review and freeze the nominal task. A Gate A pass
does not establish H's performance or authorize Stage 1 by itself.

## Other open questions: implement after the reference machinery

The following packages complete the campaign rather than repair the oracle.
Offline interface/protocol work can be prepared while experiments are held;
collection, training, tuning, pilot, sizing simulations and confirmation require
their own reviewed gates and allocations. Keep each package independently
reviewable instead of launching the entire backlog.

| Package / dependency | Concrete implementation | What it resolves; completion evidence |
|---|---|---|
| D01 data access, before training | Fixed 30-transition collector independent of planning horizon; nested episode IDs; 80/20 episode split; train-only scaling/local ID; member bootstrap manifests; indexed case/CEM/noise streams; interaction ledger | Prevent leakage, cross-episode windows and accidental RNG drift. Synthetic fixtures verify split/bootstrap/window boundaries and count attempted/failed interactions; no truth leakage to H/L |
| D02 H/L/E adapters, after nominal-task review | H retains submitted policy; L uses its same identified local fit; E has five independent episode-bootstrap networks, mean trajectory cost and no local switch. Add explicit operational deadline holds/failure logging | Defines the mandatory fair comparison. Original dev 70/15/15 reproduction is separate from shared 80/20 evaluation; E is not full PETS. Full reproduction and real training remain unexecuted |
| D03 training/model diagnostics, after allocated datasets/checkpoints | Held-out 1/5/30/60-step errors, visited-state coverage, nearest-6000 fit radius/excitation/conditioning, fitted equilibrium/Jacobians, saturation/energy defects and trained-E disagreement/coverage | Separates learned error/local identification from task/search effects. Explicitly specify longer targets for 60-step tests; 30-transition episodes cannot silently provide them. Untrained timing networks cannot establish uncertainty or control performance |
| S01 inference machinery, before final lock | Paired crossed bundle/case resampling; primary/absolute-target rules; gated key-secondary ratio and Holm; crossed-effects Bernoulli simulation, censoring-aware monotone curves and sensitivity fits | Keep BOTH one-sided 95% H−L and H−E lower bounds >.05, H estimate≥.95/LCB≥.90, ratio upper≤.80, then six-test Holm. Boundary/mixed-null calibration and all-fail/all-pass cases remain required; scaffold tests are not power evidence |
| S02 pilot, sizing and campaign orchestration, after Gate A/task/recipe/allocation decisions | Plan's exactly 12 E recipes on independent tuning data, frozen checkpoint recipe; Gate B 3 bundles×50 cases at full acquisition budget, declared one expansion; pilot-derived power, hidden final-case lock and stage reservations | Preserve full-budget 105000 acquisition transitions, D=31500 tuning charge per method and crossed seed/case uncertainty. Gate B kill cuts confirmation; mandatory powered H/L/E cannot be weakened to fit a cap |
| X01 conditional exploratory work, after an informative pilot | Explicit switch/queue variants, model-fidelity/counterfactual studies and at most the three prioritized noise/delay/parameter stresses | Explain component effects without silently changing H. Transfer and optional comparator families remain deferred; transfer currently has zero committed allocation |

For S01/S02, retain the full plan's sizing grid, 2000 simulations/design,
1000 sizing/5000 final bootstrap draws, conservative heterogeneity scenarios,
80% power at a 10-point gain and target-attainment requirement. Implement and
review before running these computations; reserve statistical compute too.
More case states do not replace independent training bundles. If a powered
mandatory design cannot fit, report insufficient feasible allocation/power rather
than lowering the margin, removing E/L or discarding censored outcomes.

## Document critique can proceed independently

Prepare the C01–C12 claim ledger, proof-to-code map and gap/contribution matrix
from the critique plan. Acquire/hash the exact original PDFs before a
page-by-page audit; the reconstructed TeX is not a substitute for source PDFs.
Check graph/open-domain and exact rank hypotheses, fixed-state slices,
sampled-flow/RK4 scope, constraints/corners and process-noise augmentation.
Structural coordinate copying is not learned fidelity or a reduced shooting
decision space. Review closest primary literature by matched assumptions and
publication date; mark unresolved leads explicitly. No proof, novelty or
data-efficiency verdict follows from the failed controller calibration.

A collaborator's independent proof/code audit may strengthen this track without
waiting for a successful control run. Keep document claims separate from verbal
claims and from reviewer-defined benchmark requirements. This plan schedules
the work; it does not commission agents or external communication.

## Visualization sequence and handoff checklist

First add an archive-reading CEM comparison view: all 40 pairs, one versus
best-five cost components alongside rail/angle margins and warm-start labels.
Then add OA-1 entry/exit phase projections and hybrid timelines when new guard
evidence exists. After a new Gate A, show full cohort outcomes, first violations
and the entire 17–20-s settling window. Only after actual training/pilot data
exist add error/coverage/uncertainty panels, learning curves and power maps.
Use physical seconds/units, raw angle, both u and newtons, safety/tolerance
bands and recorded exemplar rules; no invented uncertainty bands or attraction
certificates. Preserve the existing gallery instead of regenerating it to read it.

Before handing an implementation batch back for review, provide:

- A small diff confined to `validation/`, immutable SUT hashes and a decision
  record distinguishing specified, implemented, tested, independently audited,
  adopted and executed states.
- The meaningful positive/rejection checks relevant to that batch, with actual
  results and explicit untested paths; no duplicate completed experiments.
- Configuration/RNG/trace contracts, independent-auditor limitations and archive
  availability. Historical manifests describe snapshots, not current HEAD.
- A dry preflight and cumulative allocation for any proposed experiment, without
  `--execute`; a concrete execution command only after machinery and decisions
  are ready. No downstream stages selected by default.

Immediate next action: implement and review **I01 configuration/provenance
support**. Follow with I02 arithmetic/ranking and I03/I04 policy/audit. Keep
nonlinear guard validation, timing measurement, OA-1 adoption and amended Gate A
visibly open until their separate prerequisites are met.
