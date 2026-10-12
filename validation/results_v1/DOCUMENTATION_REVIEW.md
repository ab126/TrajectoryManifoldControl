# Completion review and documentation follow-up

October 12, 2026. Reviewed campaign commit `0de86a50` and the accompanying local artifacts. This is a bounded source/evidence review, not a new experiment campaign or an independent replication. Three Sol agents reviewed implementation correspondence, validation design and documentation; the main reviewer consolidated their findings. Scratch work was kept in `/tmp/trajectory-collaborator-review`.

The [collaborator briefing](COLLABORATOR_BRIEF.md) is the short entry point. The [full report](RESULTS.md) remains the detailed scientific record. E0–E4 and the approved E6 feedback extension are complete; optional experiments remain explicitly deferred. The original labmate source and historical plans were preserved.

## Findings that should guide interpretation

| Finding and status | Evidence and consequence |
|---|---|
| **Confirmed implementation concern: local identification is not necessarily local to upright.** | The [runner](../../scripts/run_inverted_pendulum_mc.py) selects 6000 samples by standardized state distance, without restricting inputs. [E2 diagnostics](E2/model_diagnostics.csv) measure broad neighborhoods; [true-plant local stability](E2/local_stability.csv) shows some fitted gains destabilize the upright equilibrium. This supports a concrete limitation, not a general impossibility of data-driven feedback. |
| **Confirmed numerical edge case; observed impact unresolved.** | The [planner](../../src/manifold_control.py) applies `topk`/`argmin` to costs without masking nonfinite values. A selected NaN can prevent accepting a finite alternative. The wrapper falls back to zero input when no finite incumbent is accepted. The campaign does not isolate how often this caused failures. |
| **Confirmed stale-plan behavior; causal effect unresolved.** | The runner retains its previous plan across local-feedback intervals and shifts it by five entries on the next solve regardless of elapsed time. Queued inputs also reduce feedback frequency. [Five failure diagnoses](E1/representative_failures.json) provide optimizer/model witnesses and switching associations, not a controlled estimate of this stale-plan effect. |
| **Real-time feasibility unestablished.** | [Authored timing](E0/authored_solve_timing.json) exceeds the 20 ms sample interval for every measured solve. Nominal simulation waits for computation without advancing plant time. The [separate delay test](E3/manifest.json) is a one-sample input delay, not a replay of actual variable computation latency. Feedback evaluation timings measure a different workload. |
| **Specification omissions, not a false stability theorem.** | The [27 annotations](reviews/implementation_tex_v3/annotations.json) identify notation, normalization, collection and evaluation omissions. No equation/code contradiction was found in the bounded review. The note explicitly disclaims stability guarantees. The [proof audit](reviews/paper_review_v1/proof_audit.md) accepts the deterministic graph result and qualifies output/noise extensions; geometry alone cannot certify learned prediction or control. |

## Weaknesses in our validation

**Endpoints and sampling.** The finite-time angle/no-fall endpoint is explicit and supported by tests; it does not require cart recentering or establish eventual full-state stability. The [saved 14/60-second witness](E4/long_horizon_prefix_check.json) confirms why that distinction matters. Noise is added to arbitrary state coordinates, including direct angle/position jumps, and scaled using broad collection statistics. [Fluctuation sizes and fall counts](E3/angular_tail_distributions.csv) must accompany strict endpoint failures.

**Statistical scope.** Fixed authored cases and grids are not random population samples. Within-seed matched success counts and seed variation are the clearest evidence. Wilson intervals and paired p values do not establish coverage over an untested operating region; numerous comparisons were made without a campaign-wide multiplicity correction. Nominal SMC differences remain unresolved rather than evidence of equivalence.

**Comparison scope.** The [E4 protocol](E4/manifest.json) uses linear trajectory spanning on a nonlinear plant, a simplified deterministic ensemble, and output-only initialization/velocity-cost approximations. Its failures concern those choices. E4 shares data internally but changes collection length from the original cohort; E6 reuses that original cohort. Cross-cohort paired claims are unsupported. No matched deep-reinforcement-learning sample-efficiency comparison or hardware trial was performed.

**Audit and replay coverage.** [Completion checks](completed_campaign_checks.json) verify final counts, reused references, ten selected E6 raw endpoints and matched noise arrays; they do not independently rescore every trajectory. The [campaign cache](../experiments/campaign.py) resumes by existing filenames without binding every result to hashes of all source/configuration/model dependencies. This is a provenance gap; no stale-result contamination was demonstrated. Raw arrays and weights are ignored by Git. In the reviewed checkout, the supervisor, execution logs and pinned-paper source directory are also not tracked, and the original implementation TeX resides outside the nested project repository. A fresh clone therefore cannot reproduce every linked review or run. Preserve or transfer that supporting material before external replay; its local existence is not evidence of Git publication.

**Resource accounting.** The recorded 91,181 seconds (25.33 hours) sum experiment elapsed times. Parallel workers mean this is not aggregate CPU worker-hours, and no comprehensive GPU utilization accounting is supplied. It is below a 40-hour elapsed-time interpretation, but these manifests alone do not certify every interpretation of the handoff's CPU/GPU cap. No cap breach is established by this review. E6 separately records its CPU and wall allowance.

## Evidence and reproducibility

The validation README indexes the current findings and historical experiments. The full report provides statistical/resource scope and replay limitations. The preserved implementation annotations describe an earlier source check: R05 predates completed baseline comparisons, and R13's “REPRODUCTION OPEN” category is stale although its body records reproduction. Use the completed experiment manifests and counts for present empirical status.

This review checked required deliverable presence, completion manifests, source correspondence, selected raw evidence and local links. It launched no scientific jobs. Usage was checked between work chunks using the existing supervisor's read-only `--check`, with unchanged thresholds and credit guard. `STATUS` remains `DONE`; no experiment or detached supervisor was restarted. This monitoring is not a hard per-request token budget, and balance polling is not a provider-side prohibition on credit spending.

## Operating-region evidence

The [briefing](COLLABORATOR_BRIEF.md) gives per-seed counts, defines data-fitted L and the two baseline cohorts, and links recorded gain evidence. The [H/L initial-state maps](briefing_figures/initial_state_regimes.png) and [baseline case comparison](briefing_figures/matched_baseline_regimes.png) use saved 14-second cases only. The [caption and reproduction guide](briefing_figures/README.md) explains their three-seed frequencies and limits. The [stability handoff](STABILITY_GUARANTEE_HANDOFF.md) separates a conditional local theorem from the unfinished certification of an explicit region.

L fits measured state/input transitions and does not receive the plant equations
or its exact linearization. It benefits from noiseless full-state feedback in the
nominal tests and from strong actuation. DeePC's failure has separate evidence:
all recorded optimization calls pass acceptance checks, but six saved-plan
[matched-input replays](E4/deepc_matched_plan_replays.json) produce physically
invalid forecasts. Linear combinations of nonlinear trajectories can satisfy the
optimization constraints without satisfying plant dynamics. These diagnostics
support a model-validity limitation of this baseline, not failure of the entire
DeePC family or an advantage due to hidden equations in L.
