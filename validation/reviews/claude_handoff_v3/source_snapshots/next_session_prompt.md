# Continuation prompt after completed diagnosis

**October 9 update:** use [Claude review handoff v2](claude_review_handoff_v2.md)
for the requested paper/method review and [implementation plan v2](implementation_plan_v2.md)
for current engineering work. The earlier prompt below is historical: an oracle
proposal and I01–I03 now exist, the user requests a versioned balancing-endpoint
revision, and independent paper/literature review and trained comparisons remain
open. Its strict-task language does not establish an authored-method failure.
No new experimental execution is authorized.

Copy the following into a new agentic session:

```text
Review TrajectoryManifoldControl's validation evidence and prepare the next concrete, reviewable oracle-amendment proposal. Work from /home/ap60/github/data-driven-control; the Git repository is TrajectoryManifoldControl/.

Read applicable repository instructions, validation/README.md, validation/validation_handoff.md, validation/diagnostics/cem_findings.md and cem_review.json, validation/oracle_amendment_proposal.md, and the full plans under validation/handoff/plans/. Inspect source and raw evidence where available; independently check conclusions instead of treating the handoff as certification.

Verify actual HEAD/status. Latest reviewed HEAD is 17d42cd83dea810af5b2979e657f0074244f2cc4 (validation commit). The originally evaluated submitted source is 325beae718ab093d3ca48b18a59ae61fd93798a3; its controller/plant hashes still match. Report new differences.

SUT: scripts/run_inverted_pendulum_mc.py + the active src/manifold_control.py path. Preserve submitted H and its original uniform-rod plant. inverted_pendulum/ contains unaudited baseline candidates; external_repos/ is reference only. Put new work under validation/.

Completed evidence to verify:
- CPU/CUDA plant/equivalence checks and 29 existing tests passed, with the documented dependency workaround and same-author equivalence limitation.
- Original true-model Gate A failed: 3/200 strict successes. Ignoring earlier safety, 147 settled angle/angular speed, 67 also settled cart speed, and 12 met all final tolerances. This is not a trained-H/E result. Rail +/-2.4 m and recentering were reviewer-added requirements; active MPC/LQR enforce input bounds only, with no hard state constraints.
- Selected local probes: original Qp=.02 passed 0/7, fixed oracle-only Qp=2 passed 7/7; no amendment adopted.
- CUDA diagnosis diagnostics/20261008T190703.853434Z completed 200 finite solves. Best-five improved cost in 22/40 state/warm pairs, but combined predicted safety stayed 18/40. Pure-position zero-control cost 1.44 was lower than the tested local-feedback plan's 2.105506. These are selected 1.2-second open-loop predictions, not attainability or global-optimality certificates.

Do not rerun either completed diagnosis. Total Stage-0 usage is 1.385377/15 worker-hours; diagnosis used 78.986003/900 seconds. Unused time is not permission to expand scope. The unchanged one-start CUDA reservation (~14.28 h) exceeds the remaining 13.614623 h. Dependent stages remain stopped.

Review from control, mathematics, ML and statistics perspectives. Separate objective mismatch, local recovery, finite search, safety and physical recoverability. Keep the primary IUT: BOTH one-sided 95% difference bounds above .05; retain Holm for the secondary family. The timing-agnostic-primary proposal is pending.

Deliver an evidence-linked assessment, prioritized remaining tests/visualizations, and a versioned oracle proposal specifying exact costs, gain, switch/entry conditions, safety handling, fallback/queue behavior and uncertainties. Identify the configuration/independent-audit/launcher changes needed before amended Gate A, and a feasible allocation decision. Do not silently change H, the plant, endpoint, box, device or precision.

This session is review/specification work: no new calibration, training, tuning, benchmarks, sweeps, transfer or subagents. Missing ignored raw artifacts mean unavailable evidence, not failed tests; do not regenerate them silently. Preserve old runs and source snapshots. Code/reports/manifests and the output-free notebook are in Git; raw arrays/checkpoints/media/environments require the separate lab archive.
```
