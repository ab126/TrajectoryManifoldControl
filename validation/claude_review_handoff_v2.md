# Claude review handoff: trajectory-manifold theory and implementation

Updated October 9, 2026. Review the scientific work and the adequacy of the
validation in progress. This handoff is a source index and bounded status report,
not certification. Do not launch experiments or alter the submitted method.

## Assessment to establish

The work has two connected objects: **Arda Bayer's foundational paper**, exact
[arXiv v1](https://arxiv.org/html/2609.19079v1), and the **learned-rollout/CEM/local-feedback
implementation**. Critique both, and check whether the implementation supports
the paper's intended bridge to computation. The linked paper has not been
independently reviewed in this update; the earlier critique plan contains
preliminary reasoning and literature leads, not a completed proof/novelty audit.

Largest unresolved concerns, in order:

1. **Scientific contribution and correctness remain open.** Independently check
   the paper's actual claims, hypotheses, proofs and references. Distinguish
   valid geometric identities, missing hypotheses or false statements, prior
   constructions, conceptual clarification and algorithmic novelty. Do not
   infer a literature verdict from code equivalence or controller outcomes.
2. **Practical comparative benefit has not been tested.** No full development
   reproduction, trained H/L/E comparison, completed literature review or
   baseline study exists. Plant checks and synthetic equivalence are narrower
   evidence. H is the submitted hybrid; L is its same data-identified local
   feedback; E is the planned five-model bootstrap mean-cost controller without
   a local switch, not full PETS. Candidate code is not a validated baseline.
3. **Evaluation requirements must match the claim being assessed.** The
   submitted controller clips normalized inputs to ±10 and has soft state costs;
   it does not enforce hard state constraints. Cart rail/recentering requirements
   came from validation. Misses under those added requirements describe that
   task, not a demonstrated violation of the authors' design. A revised
   balancing endpoint is requested but its exact definition is pending.
4. **Learning, local identification and finite search remain distinct
   uncertainties.** One-step fit does not establish rollout fidelity; a fitted
   DARE solution does not certify saturated nonlinear/hybrid stability; finite
   CEM does not certify optimality or recoverability. Test these explanations
   separately before attributing control outcomes to the geometric formulation.

## Read in this order

| Source | Purpose and evidentiary status |
|---|---|
| [Collaborator memo v2](collaborator_memo_v2.md) | Short, newcomer-facing assessment; definitive checks versus open work |
| [Arda's arXiv v1](https://arxiv.org/html/2609.19079v1) | Primary scientific source for **your** independent critique; retrieve exact version and record retrieval/hash |
| [Annotated implementation TeX v2](reviews/implementation_tex_v2/trajectory_manifold_implementation_reviewed_v2.tex), [annotation ledger](reviews/implementation_tex_v2/annotations.json), [manifest](reviews/implementation_tex_v2/manifest.json) | Preserved transcription plus method-focused comments; same-author bounded correspondence/elementary reasoning, not paper certification |
| [Unannotated transcription](handoff/sources/trajectory_manifold_implementation.tex) | Original prose/equations; reconstructed typography, not the author's original TeX; original PDFs currently unavailable locally |
| [Original critique plan](handoff/plans/critique_plan.md) | Initial C01–C12 reasoning; proof/literature/gap-matrix deliverables still unfinished |
| [Initial stress-test plan](handoff/plans/stress_test_plan_original.md), [GPU-amended campaign plan](handoff/plans/stress_test_plan_current.md) | Full initial design, inference contracts and allocation history; historical task definitions are not current author claims |
| [Current implementation plan v2](implementation_plan_v2.md), [earlier v1](implementation_plan_v1.md) | Current staged work and acceptance criteria; v2 identifies requested task revision without changing frozen configs |
| [Source runner](../scripts/run_inverted_pendulum_mc.py), [active controller](../src/manifold_control.py), [original plant](../src/inverted_pendulum.py) | Inspect actual active path; distinguish legacy classes and oracle-only variants |
| [Evidence assessment](oracle_review_assessment_v1.md), [saved-evidence verification](reviews/oracle_amendment_v1/evidence_final.json) | Existing raw-data/hash rechecks and limitations; not a new experiment |
| [Plant review](plant_review.md), [readiness](server_readiness.md), [CEM findings](diagnostics/cem_findings.md), [CEM review JSON](diagnostics/cem_review.json), [local findings](diagnostics/local_findings.md) | Narrow completed checks and selected diagnostics |

**Requested review outputs:** an evidence-linked claim ledger; proof/error
annotations with exact paper locations and counterexamples where applicable;
an assumption-matched primary-source literature/gap matrix; and separate
verdicts for correctness, novelty, specification fidelity and demonstrated
practical benefit. Mark `unresolved` where evidence is insufficient. Include
positive results and possible underclaims. Attribute claims to the paper,
implementation note or optional verbal context separately. Do not invent a
guarantee the documents explicitly defer. Suggest the smallest informative
remaining comparisons without running them.

The TeX comments intentionally exclude internal validation repairs and stage
history. A stale warm-start concern remains because it is behavior of the
submitted controller; its performance effect is unmeasured. No confirmed false
equation or prose/code mismatch was identified in that bounded correspondence
pass; independently recheck rather than treating this as an all-clear.

## What the completed evidence supports

* CPU/CUDA plant/equivalence checks and the 29 relevant existing tests passed,
  subject to the documented unused-CVXPY-import/C++-runtime workaround.
  Equivalence was authored by the same reviewer and used shared specification,
  weights, randomness and numerical dependencies. It establishes checked
  computational correspondence, not independent-author certification.
* The 200-case true-model reference completed. It met the **reviewer-defined
  strict recovery task in 3/200 cases**. Ignoring earlier added safety checks,
  147 settled angle/angular speed, 67 also settled cart speed, and 12 met all
  final tolerances. These are nested diagnostic counts from that reference,
  **not trained-H/E outcomes**, a new endpoint score or a maximal operating range.
* Selected local probes passed 0/7 with original Qp=.02 and 7/7 with oracle-only
  Qp=2. They do not validate a hybrid switch/entry region. No amendment is adopted.
* The completed CUDA diagnosis contains 200 finite solves at selected states.
  Best-of-five improved cost in 22/40 state/warm pairs; combined predicted safety
  remained 18/40. Pure-position zero-control cost 1.44 was below the tested local
  feedback plan's 2.105506. These are selected 1.2-second open-loop predictions,
  not independent calibration cases, global optima or attainability certificates.

## Status and audit trail for the validation itself

Current HEAD: `4c9de6db91d1c2306c0cb1318c8b309f4da390df`. Compared with the previously
reviewed `17d42cd83dea810af5b2979e657f0074244f2cc4`, committed differences are confined
to validation. The working tree also contains uncommitted validation documents,
code and tests; review the exact bytes recorded in the
[handoff manifest](reviews/claude_handoff_v2/manifest.json), not HEAD alone.
Submitted source commit: `325beae718ab093d3ca48b18a59ae61fd93798a3`.
The runner/controller/plant hash pins still match; preserve the original H and
uniform-rod plant. `inverted_pendulum/` is unaudited candidate material;
`external_repos/` is reference-only.

| Work | Current state |
|---|---|
| Paper proof/literature/novelty critique | Preliminary plan only; independent review requested from Claude |
| Method-focused implementation annotations | v2 produced; original prose preserved; no PDF compilation (pdflatex unavailable) |
| Full original reproduction / trained baseline comparison | Not executed |
| Oracle configuration (I01) / arithmetic (I02) / policy (I03) | Implemented and unit-tested: 32 / 31 / 17 tests; latest combined run 80 passed; synthetic CPU engineering evidence |
| Independent numerical trace auditor (I04) / runner and capped queue (I05) | Planned; not implemented; no amended experimental launch command ready |
| OA-1 strict-reference proposal | Specified; unadopted, unexecuted; independent audit and amended timing pending |
| Revised balancing endpoint / aligned reference / timing-primary decision | Open; must be explicitly versioned and frozen |
| Later experimental stages | Stopped; engineering progress does not authorize execution |

Records to inspect:

* [Current review log](review_log_v1.md), [current implementation decisions](oracle_implementation_decisions_v1.md),
  [I01 decision record](oa1_decisions.md), and [proposal](oracle_amendment_proposal_v1.md)
  with its [frozen proposed JSON](oa1_config_proposed.json).
* Configuration/arithmetic contracts: [I01](oracle_config_contract_v1.md),
  [I02](oracle_math_contract_v1.md), [policy/audit trace contract](trace_contract_v1.md).
  Reports and source snapshots: [I01](reviews/i01_v1/manifest.json),
  [I02](reviews/i02_v1/manifest.json), [I03](reviews/i03_v1/manifest.json).
* Historical execution: [commands log](artifacts/commands.log),
  [readiness usage](artifacts/usage.jsonl), [diagnostic usage](diagnostics/usage.jsonl),
  [completed reference run](overnight_runs/20261007T054549.695689Z/run.json),
  [morning review](overnight_runs/20261007T054549.695689Z/morning_review.md).
* [Pre-update handoff](reviews/claude_handoff_v2/validation_handoff_before_update.md)
  and [original handoff manifest](handoff/manifest.json) preserve detailed history.
  Their then-current statuses and strict-task language are historical.

The user directs validation; their collaborator will reverify. Current checks
are same-author evidence, with no independent-author signoff. Missing ignored
raw artifacts mean `unavailable` evidence; mismatched hashes or values mean a
failed check. Do not regenerate absent evidence. Git carries source, reports,
manifests and the output-free notebook; raw arrays/checkpoints/media/environments
need the separate lab archive. Local staging is accepted, with user transfer at
the end; archive format/capacity/integrity checks remain prerequisites to a run.

Stage-0 occupancy is 1.385376803372749/15 worker-hours; 13.614623196627251 remain.
Diagnosis used 78.98600335605443/900 seconds and is complete. The user accepts
a **20-hour cumulative target**, not execution/adoption. Unchanged one-start
reservation 14.27950460670723 h exceeds the present remainder. The prospective
20-hour allocation and full independent-audit overhead still need verification;
the proposed 1200-second fixed overhead is not measured audit capacity.

Keep primary IUT: **both** one-sided 95% H−L and H−E difference lower bounds
must exceed .05; retain Holm for the secondary family. Timing-agnostic primary
is pending. Revising the development reference endpoint must not silently
change the later primary estimand, case distribution, source, device or precision.

This review is read-only with respect to experiments and historical evidence:
no rerun of either completed diagnosis, no calibration/training/tuning/sweeps/
transfer, and no OA-1 execution. No message or automatic invocation has been
sent to Claude; this document is ready for the user to hand over.
