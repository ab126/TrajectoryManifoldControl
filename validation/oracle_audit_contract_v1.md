# I04 numerical replay contract

`oracle_audit_v1.py` imports no I02/I03 arithmetic, policy, SUT or launcher.
It shares I01's validated configuration, strict JSON parsing, hashes and report
types; NumPy/SciPy/Torch are disclosed shared numerical dependencies, imported
only inside operations. This is independent code by the same author, awaiting
independent-author audit. Import does not draw randomness or initialize CUDA.

`Auditor(config, gain_path, gain_sha256=...)` verifies externally pinned gain
bytes and independently reconstructs A/B via an augmented RK4 polynomial and
P/K via DARE. It recomputes diagnostics. Production accepts no transition/noise
override and uses CUDA for candidate prediction without CPU fallback. The
independent nonlinear uniform-rod implementation is **not scientifically
validated by these synthetic tests**; its equivalence and CUDA/runtime audit
remain prerequisites before execution/adoption.

`fixture=True` permits explicit synthetic CPU transitions/noise and labels
outputs `synthetic_fixture`. It cannot provide scientific evidence or adoption.

| Interface | Contract |
|---|---|
| `read_pinned(path, sha256)` | Missing evidence raises `AuditIssue(status='unavailable')`; wrong bytes raise `fail`; never regenerates artifacts |
| `verify_gain(config, record)` | Checks exact fields/types, numerical matrices, DARE residual, positive definiteness, radius and config/spec bindings |
| `scores(config,X,U,offset)` / `rank(row)` | Independent float64 cost/excess/eligibility/rank computation; all 61 nodes, original first-seen tie break |
| `Auditor.plan(trace,x,warm,case,sample)` | Reconstructs full indexed float32 draw, deterministic slots, predictions, scores, elites/moments, all iterations and incumbent; returns verified selected plan or reproduced insufficient-population exception |
| `Auditor.actions(records)` | Replays local entry/stay/exit, queue age/guards, clear-both invalidation, actual warm shift, ≤1 solve/sample, fallback and latched flags; checks record fields, inputs and nonnegative timing |
| `Auditor.case(data,expected_initial)` | Exact case/header fields, float64 prefix shapes, action-observation/input correspondence, independent plant replay, branch audit and endpoint components; never pads prefixes |
| `endpoint_components(...)` | Historical strict-reference score plus balancing, cart-rest, recentering, safety, finite/complete/input/exception diagnostics; final nodes 850–1000 inclusive, physical velocity units |
| `audit_run(auditor,cases,case_callback=None)` | Reconstructs frozen case stream, checks each case and exact 0–199 coverage; optional callback checks producer rows against verified CaseResults without replaying twice; reports overall ≥196 and ≥95% in each nonempty angle-sign half |
| `check(callable)` | Produces I01 `Report`: `pass`, `fail`, or `unavailable`; only all-pass reports pass |

Numeric arrays require exact shape/dtype. Boolean/integer/string/null metadata
and dictionary key sets are exact; float64 arithmetic uses atol=rtol=1e-10.
NaN equality applies only to diagnostic arrays whose finite flags are separately
checked. It cannot make a nonfinite case pass. Declared timing is checked for
internal consistency, not certified from saved clocks. Full hardware latency
requires separately allocated runtime measurement.

Backend runtime failures and unreproducible exceptions are **unavailable audit evidence**, not verified
failures of the method or permission to rerun. Known insufficient-population
partial traces are replayed and must latch exception failure. Other truncated
or malformed traces cannot pass; preserve them and report the exact failed or
unavailable audit scope. No run with incomplete coverage can pass.

The revised balancing endpoint/reference remains undecided. This auditor
implements the preserved stricter reference and descriptive components only.
It does not adopt OA-1, change H/plant/box/device/precision or the primary IUT/
secondary Holm, launch experiments, or certify nonlinear recovery/global search.
