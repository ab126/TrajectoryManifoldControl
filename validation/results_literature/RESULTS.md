# Literature extension: implementation and results

User-approved stages1–8 follow [the plan](../LITERATURE_COMPARISON_PLAN.md).
Compute ceilings:30 GPU-hours and84 aggregate CPU core-hours. Numerical processes
must remain on CPUs80–127. This is a new campaign; the original results remain DONE.

## Current state

The isolated supervisor adapter and host launcher are prepared. No scientific
experiment or new model session has started in the interactive container.
The original usage guard remains unchanged. The host must launch the task because
this container cannot write the Codex session directory or see NVIDIA device nodes.

Independent review routing to the user-named `overseer` is pending; see TASK.md and
reviews/ROUTING.json. The first scientific stages are EDMD and SINDy implementation,
correctness checks, timed pilots and matched evaluations.

Environment readiness was checked on October12. The launcher created the dedicated
stdlib-only venv `/tmp/trajectory-literature-envs/supervisor`; user site-packages
are disabled and no other environment's site-packages appear in its import paths.
See `environment_readiness.json`. The scientific stages must create separate,
recorded method environments under the same scratch directory. Borrowing `overseeg`
or other projects' environments is forbidden. Existing project environments were
not modified. The supervisor check reported62% five-hour usage,49% weekly usage,
unchanged credit balance1758.873136, and decision `run now`. Actual launch refused
because this container cannot write `~/.codex`; STATUS remains READY.

The user selected the existing overseer, with no replacement agent. A repeated
delivery attempt could not find `/root/overseer`; independent review is pending.

## Execution checkpoint — 2026-10-12T19:36:28.191684+00:00

The supervised host session has now started. Baseline tracked diff/status and protected hashes are saved in `inventory/`; all protected hashes matched. Three saved 3500 × 30 forced datasets and neural checkpoints are present, with SHA256 identities. Dedicated Python 3.11 EDMD and SINDy environments are frozen under `environments/`. No shared environment or submitted source was changed.

EDMD passed exact linear prediction, independent QP solution, literal cost, force dependence, zero-input and saved plant replay checks. PySINDy recovered all seven terms of an independently defined sparse controlled system, maximum rollout error 1.6e-10. See method `checks.json` files. A misread endpoint-only selection metric was corrected to the approved full-rollout metric; initial artifacts remain excluded under `invalid_endpoint_selection/`. See `inventory/selection_correction.json`. The long-horizon subset is the existing indices 0, 2, 5 (-60, -15, +60 degrees).

| Method | Seed | Validation rollout MSE | Test rollout MSE | Selected configuration |
|---|---:|---:|---:|---|
| edmd | 0 | 0.157547 | 0.160445 | {'trig': False, 'ridge': 0.0001} |
| edmd | 1 | 0.156543 | 0.144779 | {'trig': False, 'ridge': 0.01} |
| edmd | 2 | 0.145271 | 0.158798 | {'trig': False, 'ridge': 0.01} |
| sindy | 0 | 0.0693276 | 0.0686159 | {'degree': 3, 'trig': True, 'threshold': 0.1} |
| sindy | 1 | 0.0740636 | 0.0710307 | {'degree': 3, 'trig': True, 'threshold': 0.1} |
| sindy | 2 | 0.0710923 | 0.0718478 | {'degree': 3, 'trig': True, 'threshold': 0.01} |

EDMD pilot completed 16 cases per arm and projects 0.038 core-hours for the minimum 177 runs per arm; its guarded matrix is running. SINDy pilot is running with matched CEM and bounded nonlinear optimization, native/+common-L arms. Native solve rejection applies explicitly recorded zero input. Scientific outcomes are not yet summarized. Public source revisions and license records are in `sources/manifest.json`. The host has an idle NVIDIA L4; separate neural environment setup is detached and recorded.

Independent review remains pending: delivery to existing `/root/overseer` again returned “live agent path not found.” No replacement was created. Corrected uncertainty/results packet: `reviews/classical_checkpoint.json`. Continue approved stages 3–8 after the EDMD/SINDy checkpoint; no per-stage approval is required.

## EDMD/SINDy checkpoint after supervised resumption

EDMD completed all 177 cases per arm (354 total). QP and QP + common L each achieved **0/59 per seed**, comprising 0/6 authored 14 s, 0/50 random 14 s and 0/3 authored 60 s. All simulated states remained finite. QP rejection counts remain in `edmd/summary.json`; zero-input rejection fallback is explicit. The common L branch did not rescue these cases. These finite-dictionary results do not establish an impossibility for Koopman control. Original and fresh 200-state expansions are running under the EDMD allocation.

SINDy completed 16 cases per arm at seed 0. The serial compiled predictor matched PySINDy to at most 5.4e-15. Exact sensitivities of the **learned polynomial/trigonometric map**, independently checked against objective finite differences (maximum error 6.2e-8), replaced costly finite differences in the primary nonlinear optimizer. No plant derivatives, coefficient changes, extra feedback or expanded solve limits were introduced. Finite-difference pilot cases remain a separate diagnostic and their cost remains charged.

| SINDy arm | Successes/n | Rejected solves |
|---|---:|---:|
| CEM | 0/16 | 2387 |
| CEM+L | 6/16 | 2387 |
| NMPC_exact | 0/16 | 6544 |
| NMPC_exact+L | 1/16 | 6256 |

The four-arm SINDy minimum projects **1.044 aggregate CPU core-hours**, within its remaining allocation. High rejection counts must be separated from prediction error in interpreting control outcomes. The bounded minimum matrix is authorized to continue.

The prescribed SMC 200-state extension is complete: **125/200, 132/200, 131/200**, versus matched L **122/200, 130/200, 128/200**, and H **48/200, 31/200, 78/200**. Paired counts are in `smc_C200/summary.json`; the small gains over L are descriptive across three training seeds. Saved C50 artifacts were reused only after law/state/endpoint checks, and new records remain under this campaign. SMC used the exact saved surface and calibration; a strict cross-SciPy reconstruction tolerance was corrected before any case.

PETS finished three-seed training, source variance/particle checks and its 16-case pilot. The largest declared configuration (20 persistent particles, 256 candidates, three iterations) projects **2.159 GPU-hours** for its minimum matrix, within the 8-hour stage allocation. All 16 pilot cases were unsuccessful, with zero rejected solves. Independent reviewer packets remain pending because the existing overseer is unreachable; no independent approval is claimed.

Deep Koopman frequency-block/loss/force checks and DFINE source-filter/loss/independent controlled-Kalman/causality checks passed. Passive zero-input collection and CPU fitting are now underway on predeclared near/broad ranges; no fallen trajectory is excluded. This is separate from the forced-control cohort.
