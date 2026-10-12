# Trajectory-manifold validation in progress

October 9, 2026. The assessment concerns Arda Bayer's
[arXiv v1 paper](https://arxiv.org/html/2609.19079v1) and its learned-rollout,
sampled predictive controller with separately identified local feedback.

**The largest gaps are scientific and comparative.** An independent proof and
literature review is unfinished. No full original-budget reproduction or trained
baseline comparison has been completed, so correctness of the paper, novelty,
data efficiency and practical advantage remain open. Preliminary geometric
reasoning and code correspondence do not settle those questions.

**What is established within the checked scope:** plant/unit and CPU/CUDA
equivalence checks passed, along with 29 existing tests, under a documented
dependency workaround. The active code corresponds to the implementation note
in the bounded pass; the note explicitly defers learning, optimization and
stability guarantees. These are same-author checks awaiting independent review.
The submitted controller and uniform-rod plant have been preserved.

**Interpret the control evidence carefully.** The completed 200-case true-model
reference met an added strict recovery task in 3 cases. Its nested terminal
diagnostics found 147 with settled angle/angular speed and 67 also with settled
cart speed. This was not a trained-controller comparison. Hard rails and cart
recentering were validation requirements, not authored design constraints;
misses describe that task and tested trajectories. The submitted input limit
±10 is a design bound. No maximum recoverable range has been established.

Selected local and open-loop search probes suggest objective, local-recovery
and finite-search questions worth separating, but do not certify a revised
controller or establish physical nonrecoverability. The balancing endpoint is
being revised explicitly; old runs and scores remain intact.

**In progress:** configuration, arithmetic, policy, numerical trace replay and
runner/queue/archive support have **130 passing engineering tests**, using
synthetic CPU fixtures and fake processes. Independent-author review, original
plant/CUDA checks for the new implementation, a frozen revised task and amended
runtime/allocation evidence remain unfinished. Proposed-release preflight
correctly refuses execution. No new scientific run or amended reference is authorized or adopted.

Please review the paper and method on their own terms, flag exact errors and
missing hypotheses, credit valid results, and distinguish novelty from faithful
implementation and measured benefit. The
[Claude handoff](claude_review_handoff_v3.md) links the initial plans, annotated
TeX, submitted source, evidence, logs and open decisions.
