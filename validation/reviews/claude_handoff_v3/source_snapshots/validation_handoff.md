# TrajectoryManifoldControl validation handoff

Updated October 9, 2026. **Start with the
[Claude review handoff v2](claude_review_handoff_v2.md)** for the source index,
current status, initial plans, evidence and logs. The
[collaborator memo v2](collaborator_memo_v2.md) is the short scientific summary.
The [implementation plan v2](implementation_plan_v2.md) records the remaining
staged work; [review log](review_log_v1.md) records progress.

The review must assess Arda Bayer's exact
[arXiv v1 submission](https://arxiv.org/html/2609.19079v1) and the implementation
built on it. The paper's independent proof/assumption/literature review and a
trained baseline comparison remain unfinished. This update prepares that review
for Claude; it does not perform the paper review or invoke Claude.

The [reviewed implementation TeX v2](reviews/implementation_tex_v2/trajectory_manifold_implementation_reviewed_v2.tex)
preserves original prose/equations and contains method-focused annotations.
No confirmed equation or prose/code error was found in the bounded correspondence
pass; mathematical qualifications and open evidence needs are explicitly marked.
This is same-author evidence, not paper or implementation certification.

The completed true-model reference met the reviewer-defined strict recovery
task in 3/200 cases. Added rail/recentering misses must not be represented as
failures of the authors' stated design. The submitted normalized input limit
±10 is a design bound. The requested balancing endpoint revision is pending an
exact definition; historical scores, configs and runs remain unchanged. These
reference outcomes are not trained-H/E results or a recoverability certificate.

Current HEAD is `4c9de6db91d1c2306c0cb1318c8b309f4da390df`, plus uncommitted validation
work. The submitted controller/plant hashes still match the originally evaluated
`325beae718ab093d3ca48b18a59ae61fd93798a3`; the
[current manifest](reviews/claude_handoff_v2/manifest.json) records exact review
bytes and differences from the previous reviewed HEAD. I01–I03 are implemented
and unit-tested (80 synthetic engineering tests); I04/I05 and independent audits
remain open. No amendment is adopted or executed. Full reproduction, training,
tuning, comparisons and dependent experimental stages remain stopped.

Original plans remain available in full:

* [Initial campaign](handoff/plans/stress_test_plan_original.md).
* [GPU-amended campaign](handoff/plans/stress_test_plan_current.md).
* [Initial critique plan](handoff/plans/critique_plan.md).

The [pre-update handoff](reviews/claude_handoff_v2/validation_handoff_before_update.md)
preserves the full historical inventory, timing tables and strict-task history;
its then-current status statements are historical. The original
[handoff manifest](handoff/manifest.json), both completed diagnoses and old runs
are preserved. Do not rerun diagnoses or regenerate missing ignored artifacts.
Missing raw artifacts are unavailable evidence, not failed tests. Raw arrays,
checkpoints, media and environments require the separate lab archive; code,
reports, manifests and the output-free notebook belong in Git.

Stage-0 usage remains 1.385376803372749/15 worker-hours. The user accepts a 20-hour
total target, with timing, audit overhead and feasible adopted allocation still
required. No new scientific execution is authorized. Preserve the primary
intersection-union test (both one-sided 95% difference lower bounds >.05),
secondary Holm, and all source/plant/device/precision/box choices unless explicitly
versioned. The timing-agnostic-primary proposal remains pending.
