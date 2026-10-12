# I03 policy engineering evidence

Implemented `oracle_policy_v1.py` against I01/I02 and the trace contract. The
combined I01/I02/I03 run passed 80 tests (32/31/17); see `test_results.json` and
`unit_tests.txt`. Fixtures use CPU arithmetic and synthetic transitions. There
was no original-plant, CUDA, calibration, training or timing experiment.

Review the recorded source snapshots. Required independent-author review,
independent numerical trace replay, nonlinear guard checks and amended runtime
timing remain pending. This tested component does not adopt or execute OA-1.
The user-requested new balancing endpoint/reference is unresolved; the policy
represents the preserved stricter proposal with the explicit queue clarification.

The ordinary queue guard clears both queue and previous plan, then plans once
from zero. Local entry precedes queue use; actual physical age selects the warm
shift. Indexed full float32 CPU noise is cast to the required prediction type;
production fixture-noise overrides are refused. Exceptions preserve partial
traces and latch failure; infeasibility alone uses one zero sample without
claiming safety. Tests do not validate empirical recovery or global optimality.
