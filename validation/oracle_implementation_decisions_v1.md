# Implementation decisions, October 9, 2026

These decisions do not authorize experiments or adopt OA-1/allocation.

* User clarification: ordinary queued-input state mismatch or next-node-safety
  failure clears **both queue and previous plan**, then replans once from zero.
  This resolves ambiguous wording in the pinned OA-1 proposal without editing
  that historical specification/config. The clarified trace contract and
  implementation must be included in any future externally reviewed release.
* Keep the proposed **20-hour total cumulative Stage-0 target**. It is still a
  target, not an adopted increase from the existing 15-hour cap. Fresh amended
  timing and a fitting cumulative reservation remain required before execution.
* The user directs validation; their collaborator will subsequently reverify.
  Current implementation/tests remain same-author evidence. No collaborator
  certification is inferred, and no message has been sent to the collaborator.
* User permits raw evidence to be staged on this machine, with later transfer
  to the lab archive. Future preflight must check a fresh staging destination,
  sufficient free capacity and a lossless checksummed/checkpointed format. No
  files are moved, deleted or regenerated in this implementation session.
* User requests **revision of Gate A**: reviewer-added state requirements must
  not be treated as failures of the authors' design. The submitted normalized
  input bounds are actually ±10 and remain design requirements. Historical
  strict scores/configs/runs remain immutable. The picker asks for the exact new
  balancing/cart-rest endpoint and whether OA-1 remains a stricter named
  reference or is superseded by an aligned balancing-oracle proposal. Those
  answers are pending; no new Gate-A success rule is silently selected.
* Timing-agnostic primary remains pending. BOTH one-sided 95% difference bounds
  above .05 and secondary Holm remain unchanged. A Gate-A endpoint change must
  not silently change the later primary estimand.

I03/I04 can implement and test the preserved OA-1 reference and descriptive
endpoint components. Future Gate-A launch/config wiring depends on the new
endpoint/oracle decision. None of this establishes a maximum recoverable
range, invariant region, global optimum, or trained-H/E performance.
