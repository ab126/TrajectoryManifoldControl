# Validation implementation plan v2

October 9, 2026. This current overlay supersedes stale status/task wording in
[v1](implementation_plan_v1.md), preserving v1 and every frozen proposal/run.
Source contracts remain [I01](oracle_config_contract_v1.md),
[I02](oracle_math_contract_v1.md), and [trace v1](trace_contract_v1.md).
New work stays under `validation/`; submitted H, the original uniform-rod plant,
historical snapshots and usage ledgers remain unchanged.

## Scientific review track: paper, specification and evidence

This track can proceed without experimental authorization. Claude is to perform
the independent review; this update supplies scope and source pointers without
undertaking the arXiv review itself. The full original
[critique plan](handoff/plans/critique_plan.md) remains the detailed specification.

| Order | Planned deliverable (not yet produced) | Acceptance |
|---|---|---|
| R01 | `reviews/paper_review_v1/source_manifest.json`, `claim_ledger.csv`: retrieve exact [Arda arXiv v1](https://arxiv.org/html/2609.19079v1), map its claims to the implementation note and active code | Pin version, retrieval time and source hashes. Every ledger row identifies source/location, exact claim, assumptions, status and supporting evidence. Separate paper, note and optional verbal claims |
| R02 | `reviews/paper_review_v1/proof_audit.md`, `paper_annotations.md`: independent proof/assumption audit | Re-derive graph/domain/rank and sampled-map arguments; audit other claims actually present in the paper. Each adverse verdict names an exact claim and supplies the missing hypothesis or counterexample. No verdict is inferred from control outcomes |
| R03 | `reviews/paper_review_v1/literature_matrix.csv`: primary-source, assumption-matched comparison | For each contribution record nearest prior construction, setting, assumptions, guarantee, algorithmic difference and residual gap; pin sources. Cover prior work before the submission; label later work. Leads/abstracts alone do not count as a completed review |
| R04 | `reviews/paper_review_v1/assessment.md`: separate correctness, novelty, code fidelity and practical-benefit judgments | Credit valid results and underclaims; retain unresolved labels. Link every conclusion to R01–R03 or empirical evidence. No comparative performance verdict without trained comparisons |

The [annotated TeX v2](reviews/implementation_tex_v2/trajectory_manifold_implementation_reviewed_v2.tex)
is a bounded correspondence review, **not** R01–R04. Keep annotations relevant
to the method: actual errors/qualifications, accurately disclosed limitations,
structural consequences and unmeasured implementation concerns. Internal
validation repairs belong in the [log](review_log_v1.md), not the method critique.

## Engineering order and current state

Retain I01 first, then I02–I04 as separate review batches, then I05. Component
states are distinct: specified → implemented → tested → independently audited
→ adopted → executed. A tested component does not advance the whole policy to
adopted/executed. No fresh-context independent audit has occurred.

| Batch | State / next concrete work | Acceptance |
|---|---|---|
| I01 configuration | Implemented/tested, 32 tests; [record](reviews/i01_v1/manifest.json) | Preserve strict typed scientific schema, separate release/adoption metadata, canonical config hash, exact source/runtime comparisons and proposed-config execution refusal. No CUDA/SUT import or launch in parsing |
| I02 arithmetic | Implemented/tested, 31 tests; [record](reviews/i02_v1/manifest.json) | Preserve exact costs/indexing/rank, finite rejection and gain tolerances; P is separate from Qf. Independent review remains required |
| I03 policy | Implemented/tested, 17 synthetic tests; [record](reviews/i03_v1/manifest.json) | Actual elapsed shift; indexed full noise draws; deterministic slots; clear queue **and plan** on guard failure; ≤1 solve/sample; explicit local/queue/zero branches and latched exceptions. CUDA/plant performance remains untested for this implementation |
| I04 audit | Next unblocked engineering batch: add `oracle_audit_v1.py`, `tests/test_oracle_audit_v1.py` against the trace contract | Import no I02/I03 numerical or policy helpers; implement independent costs/ranks/gain/predicates/branch replay. Mutation tests reject wrong seed/input/cost/rank/switch/age/flags/window; incomplete/199-case runs cannot pass. Missing archive returns `unavailable`, corrupt evidence `fail`. No experimental regeneration |
| I05 runner/queue | After I02–I04: planned `gate_a_v1.py`, `oracle_queue_v1.py`, `tests/test_oracle_queue_v1.py`; future task-specific version/name depends on endpoint decision | Fake-process checks reject occupied lock, stale job, unadopted config, mismatched reviewed sources/runtime, insufficient reservation and overwrite; interruption preserves partial evidence. No downstream enqueue |

I04 may implement the preserved strict-reference replay and descriptive
balancing/cart-rest components now. It must not select a revised success rule.
Its mathematical independence is separate from independent-author review.
Dependencies may share I01 parsing/hashing and disclosed NumPy/Torch/SciPy
libraries, but not the numerical calculations being checked. Experimental
plant/CUDA/guard checks and amended timing remain separately gated.

Future launch wiring requires a **new versioned scientific config** after the
endpoint/reference decision. Do not reinterpret the old canonical hash or
mutate the v1 proposal/schema to make an old release look adopted. Release pins
must bind scientific config, exact implementation/audit/launcher bytes, gain,
trace contract, source snapshots, environment/device/precision and trusted
adoption/allocation records. The runner compares externally reviewed pins to
observed bytes/runtime; recording its own hash is not approval.

## Decisions required before amended reference execution

1. Freeze the revised endpoint: balancing alone or balancing plus cart rest;
   state excursions and displacement are separate diagnostics. Retain the
   authored input limit. Decide whether a task-aligned balancing oracle
   supersedes OA-1 or OA-1 remains a separately named stricter reference.
   Historical 3/200 is a strict-task outcome, not a method-failure verdict.
2. Freeze any remaining changes to reference cost/gain/entry/safety/fallback;
   audit exact config, math, policy and traces independently. Then specify and
   separately authorize bounded nonlinear guard checks and amended end-to-end
   CUDA timing including logging. No threshold/gain search or diagnosis rerun.
3. Pin local lossless archive format/checkpointing/capacity, with checksums for
   later lab transfer. User accepts local staging, not silent deletion or
   regeneration of historical evidence.
4. Adopt a feasible cumulative allocation and an exact Gate-A-only invocation
   after the checks. The user directs execution; collaborator re-verification
   is not presumed to have occurred. No OA-1 execution is authorized here.

Stage-0 usage: **1.385376803372749/15 h**; remaining **13.614623196627251 h**.
Completed diagnosis: **78.98600335605443/900 s**. Unused diagnostic balance is
not additional scope. The unchanged one-start reservation **14.27950460670723 h**
does not fit. The user accepts a **20 h total target**, pending adopted allocation:
3600 s support + 61200 s queue + 1800 s archive/review leaves **412.643508 s**
after recorded usage. The queue model
`200000 * tau_amended * 1.2 + 1200 <= 61200` requires `tau_amended <= .25 s`.
This is a budget model, not measured capacity; fresh full-step timing is required.
Full independent candidate replay may exceed the **unmeasured** 1200 s overhead.
Measure/reserve audit and startup costs explicitly without double-counting
logging or the 20% allowance. If they do not fit, return for an allocation or
reviewed audit-design decision; do not omit required audit coverage.

Preserve 200 cases, 20 s / 1000 control samples, full declared box, original
plant, L4 prediction, float64 oracle/plant and one start unless a new explicit
version changes them. Historical gate thresholds are ≥196/200 and ≥95% in each
nonempty angle-sign half; their reuse with the revised endpoint must be explicit.
The exact new endpoint is pending, not executed by this plan.

## Comparative validation after task/reference lock

The [v1 D01–X01 packages](implementation_plan_v1.md#other-open-questions-implement-after-the-reference-machinery)
and [full campaign plan](handoff/plans/stress_test_plan_current.md) remain pending:
data interfaces → H/L/E adapters → trained-model diagnostics → inference and
pilot sizing → separately allocated comparisons. Literature comparison R03 is
different from an empirical baseline comparison. Neither is complete.

First compare the unchanged submitted hybrid H with its identified local L and
the specified five-model E on common cases/bundles with matched acquisition and
tuning accounting. Do not use unaudited `inverted_pendulum/` code/results as
baseline evidence; `external_repos/` remains reference-only. Distinguish original
70/15/15 reproduction from campaign 80/20 splitting. Assess model fidelity,
local recovery, objective preference, finite search, state excursions and
physical recoverability separately. No finite failure proves nonrecoverability.

Keep primary IUT: **both one-sided 95% H−L and H−E lower difference bounds >.05**;
retain secondary Holm. Timing-agnostic primary remains pending. An endpoint
change for development must not silently revise the later primary estimand.
Full training/tuning/benchmarking, sweeps, transfer and downstream stages remain
stopped; no subagents or experiments are authorized by this update.
