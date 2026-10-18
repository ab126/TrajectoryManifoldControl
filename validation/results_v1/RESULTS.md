# Validation of the submitted trajectory-manifold controller

Start with the [collaborator briefing](COLLABORATOR_BRIEF.md) for a one-page overview. The [matched H/L notebook](../notebooks/collaborator_hybrid_linear_examples.ipynb) shows saved successes, failures and cases favoring either controller. Saved-result visualizations show [initial-state regimes](briefing_figures/initial_state_regimes.png) and [baseline case comparisons](briefing_figures/matched_baseline_regimes.png); their [captions](briefing_figures/README.md) define the empirical frequencies. A separate [stability handoff](STABILITY_GUARANTEE_HANDOFF.md) specifies a conditional local result and the unfinished certification work. The [October 12 completion review](DOCUMENTATION_REVIEW.md) consolidates implementation concerns, validation limitations and artifact-sharing caveats.

## Q1 — Does it work as claimed?

**The authored seed-234 run reproduces all six angle/no-fall successes, but the hybrid is sensitive to seed and its local-feedback ablation is stronger on the authored cases.** The fresh unchanged `--profile dev` run is saved in [E0/authored_summary.json](E0/authored_summary.json); expanded endpoints and ten-seed ablations are complete. Nine focused [metric and arm tests](E0/metric_arm_tests.txt) passed. The authored notebook already contains a development-profile demonstration: all six initial angles (±15°, ±30°, ±60° from rest, 14 s) have `balanced=True`; three also have `cart_rest=True`. Local feedback supplies 50–97% of steps, with 6–70 CEM solves per case. The notebook is the authored baseline; the fresh independent reproduction is reported below. Source: [notebook](../../manifold_control.ipynb), cells 10 and 12.

The primary endpoint is the runner's `balanced` metric (|θ|≤0.02 rad and physical |θ̇|≤0.05 rad/s in the last 3 s) **and no fall** (|θ|<π/2 throughout). Cart excursion, drift, settling, final position, input saturation and local feedback use are reported separately. There is no rail limit, recentering deadline or acceptance gate.

The fresh authored reproduction gives **6/6 primary successes** (Wilson 95% interval 0.610–1.000), with cart rest in 4/6 rather than the notebook's 3/6. Local feedback supplies 43.1–96.7% of steps and CEM solves range from 6 to 81. Maximum cart excursions span 0.413–7.858 m. These differences from the saved notebook should be retained as reproduction variability, not overwritten. The authored local-fit selection reaches |θ|=1.775 rad (101.7°), with standardized radius 0.809 and inputs spanning the full ±10 limit. Sources: [authored endpoints](E0/authored_cases.csv), [angle/cart/input figure](E0/authored_excursions.png), [model diagnostics](E0/authored_model_diagnostics.json).

The fresh authored run records 216 CEM solves: median 30.2 ms, 95th percentile 31.0 ms, maximum 130.2 ms. All exceed the 20 ms physical sample interval; one exceeds the nominal five-step queue interval of 100 ms. The plant simulator advances a fixed step after the controller returns, so this demonstration does not include computation delay. The measured timing matters for a hardware implementation and for comparing online shooting with direct policy evaluation. Sources: [timing summary](E0/authored_solve_timing.json), [original timings](E0/authored_summary.json), [submitted simulator](../../scripts/run_inverted_pendulum_mc.py).

On IC-A at 14 s, H succeeds in 4–5/6 cases per seed, while L succeeds in 6/6 for every seed. M and M1 each have 0/6 for every seed; H* has 4/6 for every seed. At 60 s, H improves to 4–6/6, L remains 6/6, and H* gives 4–5/6. This supports the local catch's importance; the broader E1 comparisons are below. The true-model failures show that learned-model error is not the only limitation. Sources: [per-seed rates](E0/rates.csv), [paired comparisons](E0/paired_comparisons.csv), [figures](E0/success_rates.png).

M and M1 can approach upright while missing the authored settling limits. In the illustrative seed-0 −15° case, M1's final-three-second maximum angle is 0.0388 rad and maximum physical angular speed is 1.047 rad/s; L gives 0.000275 rad and 0.000102 rad/s. The [terminal-window figure](E0/terminal_behavior.png) and [all 300 seeded 14 s terminal summaries](E0/terminal_behavior.csv) show the distinction between approaching upright and meeting the authored settling limits.

A [feasible-alternative diagnostic](E0/feasible_alternative_diagnostics.json) examines the last saved plans of three illustrative seed-0 misses. It found no lower internal-model cost among the local-law input sequences tried; that does not establish global optimality. Some older E0 arrays lack full planned inputs, so their chosen true-plan costs are left unavailable. E1 saves full plans for matched-input replays. Five [objective, missing-data, interval and matched-selection checks](failure_diagnostic_tests.txt) pass.

A completed E4 long-horizon witness also limits the eventual-stability claim. Seed-1 L from −15° succeeds at 14 seconds but misses the angular endpoint at 60 seconds without falling. Its final θ changes from −0.00115 to −0.0525 rad, final p from −1.63 to −74.64 m, and physical cart speed from −0.136 to −6.20 m/s. The two deterministic trajectories share an identical 14-second prefix. This gain stabilizes its fitted model (radius 0.99588) but destabilizes the true full-state upright Jacobian (radius 1.00166). This is an illustrative case under E4's disclosed 32-step collection, with no added recentering or excursion gate. Sources: [14-second case](E4/cases/seed1_L_A_0002_14s.json), [60-second case](E4/cases/seed1_L_A_0002_60s.json), [prefix check](E4/long_horizon_prefix_check.json), [recorded-gain stability diagnostic](E4/local_stability.csv), [figure](E4/local_stability.png), [independent Jacobian checks](E4/local_stability_checks.json). The completed E4 comparison is reported under Q3.

A post hoc horizon check scores saved 20-second prefixes only for H/H* runs that missed the 14-second angular endpoint without falling: H adds 2 successes among 11 selected cases and H* adds 0 among 20. This selected diagnostic does not change the authored horizon or estimate a new population success rate. Sources: [selection](E0/horizon_followup.json), [prefix endpoints](E0/horizon_limited_20s.csv).

## Q1.1 — Failure region, data needs, novel systems

**E1 and the nominal E2 data sweep are complete. At 3,500 and 10,000 episodes, local feedback alone (L) is stronger than H on the tested sets. Success first appears at 1,000 episodes and remains seed-dependent; additional data improves predictions without consistently improving control. Collection changes alter capture counts. For H and L, the tested measurement/process noise and one-sample delay give no primary successes, while ±25% mass/length mismatch preserves L's six authored catches. On the additional LTI, L and the known-model LQR catch all tested ICs, while H gives 30–31/50 random-box successes. The acrobot configuration gives no successes and every run falls, including its 60-second subset.**

The table gives the range of primary success counts across seeds 0–2, with each seed kept separate in the [rates and Wilson intervals](E1/rates.csv). These are tested ICs, not continuous-domain guarantees.

| IC set | H | L | M | M1 | H* |
|---|---|---|---|---|---|
| B: from rest | 72–107/171 | 137–141/171 | 0/171 | 0/171 | 85–97/171 |
| C: random box | 31–78/200 | 122–130/200 | 0/200 | 0/200 | 42–53/200 |
| D: angle/rate grid | 356–717/1681 | 1073–1089/1681 | 0/1681 | 0/1681 | 454–548/1681 |

On IC-B and IC-D, every H success also succeeds under L in all three seeds. L adds 54, 69 and 34 matched from-rest successes, and 661, 733 and 368 grid successes. In the random box, H adds 3, 0 and 1 catches beyond L, while L adds 77, 99 and 51 beyond H. Thus H occasionally catches an IC that L misses, but it does not expand the tested from-rest or angle/rate-grid reach over L. Together with the zero primary successes for M and M1, this supports the identified feedback law as the useful control component in this configuration. Sources: [matched comparisons](E1/paired_comparisons.csv), [rates and per-seed Wilson intervals](E1/rates.csv).

On the tested 1° from-rest grid, L succeeds from −68° to +68° in seed 0 and −70° to +70° in seeds 1–2. H’s largest uninterrupted successful spans are respectively [−36°, +34°], [−39°, +30°] and [−38°, +49°]. Additional isolated successful spans remain in the [exact interval table](E1/from_rest_intervals.csv); failed points are not bridged, and the grid establishes no guarantee between tested angles. H* is not a consistent upper bound: it improves seed 1, while seed 2 gives H 16 additional from-rest and 31 additional random-box successes over H*, against 6 and 4 in the opposite direction. These are matched per-seed comparisons on fixed ICs, not a pooled population estimate.

The [five selected failure diagnoses](E1/representative_failures.json) show both optimizer and model limitations. Every selected initial CEM plan admits a bounded local-law input sequence with lower internal-model cost. For the true-model −60° case, those costs are 7671.793 versus 6681.425, witnessing optimizer suboptimality despite an accurate transition. All four learned-model examples also have later model/true-cost preference reversals. For H at −43°, step 698 prefers its chosen inputs internally (95.216 versus 330.518), while true-plant replay favors the alternative (286.494 versus 122.846). The [−60° H trace](E1/failure_2.png) also shows repeated local-feedback exits where matched L succeeds. These are illustrative optimizer/model witnesses and switch associations, not proof of a sole closed-loop failure cause or global optimality. Sources: [matched replay for H](E1/failure_1_prediction_errors.json), [matched replay for H*](E1/failure_5_prediction_errors.json), [overlapping failure categories and toggle counts](E1/failure_categories.csv).

The 60 s subsets do not uniformly resolve misses. On the fixed −60°, 0°, +60° from-rest subset, L gives 3/3 primary successes in every seed, H gives 1–3/3 and H* gives 1/3. Angle success and full cart settling remain distinct: four of the six nonzero-angle L runs have no full-settle time by 60 s under the stated position-change criterion. Conversely, seed-2 H from −60° records a full-settle time of 29.1 s with final cart position −5.603 m; settling does not require recentering. Earlier falls remain disqualifying even when a run later balances. These small, prespecified subsets are descriptive; the [60 s per-seed counts](E1/rates.csv), [settling/final-position case metrics](E1/summary.csv) and [excursion, saturation and local-use distributions](E1/distributions.csv) retain all runs, including failed simulations with large excursions.

[Exact tested success intervals from rest](E1/from_rest_intervals.csv) preserve holes in the 1° grid. The [D map for seed 0](E1/map_D_seed0.png) and [cart-excursion map](E1/cart_excursion_D_seed0.png) have counterparts for seeds 1–2. [Paired comparisons](E1/paired_comparisons.csv), [overlapping failure categories](E1/failure_categories.csv) and [five selected failure diagnoses and evidence paths](E1/representative_failures.json) ([example plot](E1/failure_1.png)) separate the endpoints, switch associations and model/optimizer witnesses. Cart excursions remain descriptive; no rail criterion is used.

The previous true-model wide-box diagnostic has 147/200 angle-balanced runs, 67/200 also reaching cart rest, and 103/200 crossing π/2. The first two counts omit the newly required no-fall condition; they must not be presented as primary successes. Initial speeds reach 7.8 m/s and angles reach 74°, unlike the authored demonstration. Source: [independent audit](../overnight_runs/20261007T054549.695689Z/calibration/independent_audit.json); recomputation from saved trajectories gives **65/200** angle-balanced with no fall (Wilson 95% interval 0.264–0.393). Cart-excursion median is 3.829 m, 95th percentile 14.196 m and maximum 20.009 m. This historical run used N=60 and 20 s, so it is not a matched comparison with the new N=30, 14 s arms. Sources: [recomputed summary](historical_oracle/endpoint_summary.json), [all cases](historical_oracle/summary.csv), [manifest](historical_oracle/manifest.json).

`run_inverted_pendulum_mc.log`, `inverted_pendulum_mc_debug.ipynb`, and `linear_system_mc.ipynb` contain historical latent-MPC outputs. They are not evidence about the active CEM controller.

### E2 model diagnostics (all 30 models complete)

The held-out 30-step prediction errors decrease substantially as the data budget grows. The nearest-6000 local fit uses every training transition at 35 and 100 episodes. Those unwrapped training trajectories reach well beyond a single pendulum revolution; the small-data fit is not local to upright. Even at 3,500 episodes the selected angles reach 98–104°. At 10,000 episodes they reach 78–84°. These are collection-neighborhood diagnostics, not successful closed-loop angles. Sources: [per-model prediction and local-fit diagnostics](E2/model_diagnostics.csv), [saved models' diagnostics](models/episodes35_seed0/diagnostics.json).

The table gives min–max across the five whole-run seeds. RMSE columns use stored normalized state coordinates (physical velocities multiply by √9.81); no aggregate across unlike state units is used. Collected transitions are 30 × episodes; the fitted counts below follow the actual whole-episode split.

| Episodes | Fitted transitions | Local-fit samples | Selected max abs θ [degrees] | N-step p RMSE | N-step v RMSE | N-step θ RMSE | N-step ω RMSE |
|---|---|---|---|---|---|---|---|
| 35 | 750 | 750 | 420.6–506.0 | 0.5581–0.7051 | 0.3314–0.8815 | 1.48–2.858 | 1.671–3.086 |
| 100 | 2100 | 2100 | 445.0–517.9 | 0.4035–0.6097 | 0.4916–0.6615 | 1.561–3.422 | 1.396–3.26 |
| 350 | 7380 | 6000 | 343.6–372.9 | 0.4318–0.608 | 0.446–0.6766 | 0.5935–1.995 | 0.8578–2.311 |
| 1000 | 21000 | 6000 | 154.8–196.0 | 0.1224–0.134 | 0.1761–0.2471 | 0.1895–0.2952 | 0.2947–0.4151 |
| 3500 | 73500 | 6000 | 98.3–103.7 | 0.03114–0.04086 | 0.04457–0.07288 | 0.04959–0.06936 | 0.08783–0.1553 |
| 10000 | 210000 | 6000 | 78.5–83.5 | 0.01044–0.01186 | 0.01396–0.02541 | 0.0185–0.02189 | 0.02975–0.08188 |

### E2 data efficiency: all six budgets complete

**No primary successes occur at 35, 100 or 350 episodes. Success appears at 1,000 episodes, with strong seed dependence. At 3,500 and 10,000 episodes L succeeds on every authored IC in all five seeds, while H remains at 4–5/6. M1 has no primary successes at any tested budget.** These are observed results under the submitted collection, split, fitting and controller protocol, not a lower bound for a redesigned method.

The table gives success-count ranges across five whole-run seeds. Counts and Wilson intervals remain separate by seed in the [rates](E2/rates.csv); the [data-efficiency figure](E2/data_efficiency.png) shows descriptive means across those seeds against fitted transitions.

| Episodes | Collected / fitted transitions | IC-A H | IC-A L | IC-A M1 | IC-C50 H | IC-C50 L | IC-C50 M1 |
|---|---|---|---|---|---|---|---|
| 35 | 1050 / 750 | 0/6 | 0/6 | 0/6 | 0/50 | 0/50 | 0/50 |
| 100 | 3000 / 2100 | 0/6 | 0/6 | 0/6 | 0/50 | 0/50 | 0/50 |
| 350 | 10500 / 7380 | 0/6 | 0/6 | 0/6 | 0/50 | 0/50 | 0/50 |
| 1000 | 30000 / 21000 | 0–4/6 | 0–6/6 | 0/6 | 0–16/50 | 0–37/50 | 0/50 |
| 3500 | 105000 / 73500 | 4–5/6 | 6/6 | 0/6 | 6–19/50 | 30–36/50 | 0/50 |
| 10000 | 300000 / 210000 | 4–5/6 | 6/6 | 0/6 | 5–20/50 | 29–34/50 | 0/50 |

All 840 matched runs fall at each of the 35- and 100-episode budgets. At 350 episodes, 736/840 fall; the 104 runs without a fall still miss the angular settling limits at 14 seconds. These totals describe five seeds and three arms on fixed ICs, not pooled population estimates. Zero-count Wilson intervals are 0–0.390 for six ICs and 0–0.071 for fifty.

Component attribution depends on the fitted gain and budget. At 1,000 episodes, seed 2 gives H 11/50 random-subset successes and L 0/50: H adds 11 catches and L none (exact matched McNemar p=0.000977). This qualifies the nominal E1 conclusion; H can help when the small-data local law misses. At 3,500 episodes, H adds 0–2 random-subset catches beyond L per seed, while L adds 12–30. At 10,000, those counts are 0–1 and 11–29, with per-seed McNemar p between 3.73×10⁻⁹ and 0.00635. All H successes on IC-A at both larger budgets also succeed under L. Sources: [matched comparisons](E2/paired_comparisons.csv), [case metrics](E2/summary.csv).

| Seed at 10,000 episodes | IC-A H | IC-A L | IC-A M1 | IC-C50 H | IC-C50 L | IC-C50 M1 |
|---|---|---|---|---|---|---|
| 0 | 5/6 | 6/6 | 0/6 | 18/50 | 34/50 | 0/50 |
| 1 | 4/6 | 6/6 | 0/6 | 5/50 | 34/50 | 0/50 |
| 2 | 4/6 | 6/6 | 0/6 | 10/50 | 33/50 | 0/50 |
| 3 | 4/6 | 6/6 | 0/6 | 8/50 | 29/50 | 0/50 |
| 4 | 5/6 | 6/6 | 0/6 | 20/50 | 30/50 | 0/50 |

Prediction quality improves substantially at 10,000 episodes, but IC-C50 success declines relative to 3,500 episodes in three of five H seeds and four of five L seeds. The sweep changes the whole collection/training run, not only one model-error parameter. It establishes neither monotone control improvement nor a sufficient data budget for reliable H success on this random box. The [prediction/local-fit figure](E2/model_quality_vs_data.png) and stability diagnostic below show why prediction RMSE alone is insufficient evidence of closed-loop behavior.

Cart excursion remains descriptive. At 10,000 episodes, per-seed median maximum excursion spans 1.82–4.41 m for H and 1.05–2.95 m for L on IC-A; on IC-C50 it spans 7.36–12.01 m and 4.33–11.57 m. These include failed cases. Only 0–2 of six H runs and 0–4 of six L runs per seed record full settling by 14 seconds. Settling times can exist after an earlier fall and do not override the no-fall endpoint. [Distributions](E2/distributions.csv) retain each budget, seed, arm and IC set, with observed-time denominators, final position, saturation and local fraction. E2 finished all 5,040 cases in 3.60 wall hours, including final postprocessing ([manifest](E2/manifest.json)). The completed collection-distribution ablations are reported below.

### Local-fit stability diagnostic

Every one of the 30 identified DARE gains stabilizes its fitted linear model (spectral radius 0.9765–0.9984), but that does not ensure stability on the true plant. At 35, 100 and 350 episodes, all five gains give a true upright closed-loop Jacobian with spectral radius greater than one. This diagnoses a limitation of fitting a broadly selected neighborhood as a local linear model. Larger budgets reduce the discrepancy substantially, but the stability result is not monotone with data size.

The table uses the exact derivative of the submitted sample-and-hold RK4 map with input u=−Kx, before saturation. Independent central differences agree to at most 6.4×10⁻¹²; a gain designed on that true derivative gives radius 0.998382. These are post hoc diagnostics of the recorded gains, with no controller modification or gain selection. A radius above one concerns the full-state upright equilibrium; slowly growing cart modes can still pass a finite-horizon angle endpoint. A radius below one does not establish a capture region for the finite ICs. Sources: [per-model spectral radii](E2/local_stability.csv), [figure, including a zoom near one](E2/local_stability.png), [independent checks](E2/local_stability_checks.json), [calculation](../experiments/local_stability.py).

| Episodes | True local Jacobian radius, min–max across five seeds | Seeds above one |
|---|---|---|
| 35 | 1.291198–9.687442 | 5/5 |
| 100 | 1.123959–5.703622 | 5/5 |
| 350 | 1.420435–16.854472 | 5/5 |
| 1000 | 0.999522–1.003937 | 4/5 |
| 3500 | 0.986200–1.000032 | 1/5 |
| 10000 | 0.990098–1.001227 | 3/5 |


### Collection-distribution sensitivity

At the same 3,500-episode budget, the half-box collection scales all four initial-state bounds by 0.5 and retains the nominal input policy. The uniform-only collection retains the original box and uses the odd-episode uniform input policy on every episode, with the existing random input-hold schedule. All arms are evaluated on the same original IC-A and IC-C50, including ICs beyond the narrowed initial collection box; costs, endpoints and fitting/training settings are unchanged. The [adapter](../experiments/distribution_ablation.py) parameterizes the imported submitted collector in memory.

Ranges below span five whole-run seeds. M1 scores 0/6 and 0/50 in every seed under both ablations. The [comparison figure](E2_distribution/collection_comparison.png) shows separate seed traces, and [rates with Wilson intervals](E2_distribution/rates.csv) retain the denominators.

| Collection | IC-A H | IC-A L | IC-C50 H | IC-C50 L | Selected max abs θ [degrees] | Held-out final θ RMSE at 30 steps |
|---|---|---|---|---|---|---|
| Nominal | 4–5/6 | 6/6 | 6–19/50 | 30–36/50 | 98.3–103.7 | 0.0496–0.0694 |
| Half box | 4–5/6 | 6/6 | 16–22/50 | 28–33/50 | 55.6–60.2 | 0.0279–0.0341 |
| Uniform only | 3–5/6 | 6/6 | 5–20/50 | 29–35/50 | 98.3–103.6 | 0.0473–0.0582 |

The half-box collection improves H's IC-C50 count in three seeds, leaves one equal and reduces one; L's count declines in all five. Uniform-only effects are mixed: H improves two seeds and declines three relative to nominal, while L improves two and declines three. Thus the collection distribution matters, but neither tested change makes H consistently stronger than L or resolves M1's settling failure. The narrower model's smaller held-out prediction errors are measured on its own narrower test distribution and do not establish greater accuracy on a common wide-box test. Sources: [all 15 model diagnostics](E2_distribution/model_diagnostics.csv), [nominal counts](E2/rates.csv), [ablation counts](E2_distribution/rates.csv).

In the random subset, the half-box hybrid adds 0–2 catches beyond L per seed, while L adds 8–15; seed 0's exact paired p=0.109, so the count advantage is not uniformly statistically resolved with fifty ICs. Uniform-only gives 0–3 H-only versus 13–26 L-only catches (per-seed p from 2.98×10⁻⁸ to 0.0213). These are prespecified per-condition comparisons, not a pooled or multiplicity-adjusted global conclusion. Sources: [paired comparisons](E2_distribution/paired_comparisons.csv), [cart/settling/input distributions](E2_distribution/distributions.csv), [case metrics](E2_distribution/summary.csv). Both ablations completed all 1,680 runs in 0.81 wall hours ([manifest](E2_distribution/manifest.json)).

### E3 robustness

**All tested measurement-noise, process-noise and one-sample-delay conditions give H and L 0/6 on IC-A and 0/50 on IC-C50 in every seed. Under ±25% mass/length mismatch, L retains 6/6 authored successes and H gives 4–5/6.** The endpoint remains the same pointwise angular limits throughout the last three seconds and no fall. Noise levels are fractions of the nominal collection-state standard deviations, not fractions of the settling thresholds.

Mismatch models remain trained on nominal data. Length changes retain nominal physical measurement, time and force coordinates. Counts below are min–max across seeds 0–2; the [rates](E3/rates.csv) give per-seed counts and Wilson intervals, and the [matched comparisons](E3/paired_comparisons.csv) retain paired ICs.

| Plant mismatch | IC-A H | IC-A L | IC-C50 H | IC-C50 L |
|---|---|---|---|---|
| Mass ratio 1.5 (−25%) | 4–5/6 | 6/6 | 6–20/50 | 38/50 |
| Mass ratio 2.5 (+25%) | 4/6 | 6/6 | 6–19/50 | 29–32/50 |
| Length 0.75 (−25%) | 4–5/6 | 6/6 | 7–18/50 | 38/50 |
| Length 1.25 (+25%) | 4/6 | 6/6 | 5–14/50 | 26–29/50 |

The absence of primary successes under noise/delay includes runs that remain upright. The table separates those outcomes, with ranges across three seeds. Measurement noise perturbs controller observations; process noise is applied independently at every plant step and also during model collection. All arms share the same disturbance sequence for an IC/seed/condition. At process level 0.1, every run falls.

| Stress | IC-A runs without a fall, H / 6 | IC-A runs without a fall, L / 6 | IC-C50 falls, H / 50 | IC-C50 falls, L / 50 |
|---|---|---|---|---|
| Measurement 0.01 × std | 5–6/6 | 6/6 | 17–30/50 | 15–18/50 |
| Measurement 0.03 × std | 4–6/6 | 6/6 | 18–30/50 | 16–18/50 |
| Measurement 0.1 × std | 1–2/6 | 6/6 | 37–46/50 | 18–23/50 |
| Process 0.01 × std | 0–6/6 | 6/6 | 16–46/50 | 15–17/50 |
| Process 0.03 × std | 0–3/6 | 4–6/6 | 35–49/50 | 18–19/50 |
| Process 0.1 × std | 0/6 | 0/6 | 50/50 | 50/50 |
| One-sample delay (20 ms) | 5–6/6 | 6/6 | 21–30/50 | 21/50 |

The nominal unwrapped collection-angle standard deviation is 2.784–2.812 rad, so the smallest process-noise level adds θ noise with standard deviation 0.0278–0.0281 rad at each 20 ms step, exceeding the 0.02 rad endpoint bound. Continuous disturbances make this pointwise endpoint stringent. The [angular-fluctuation figure](E3/angular_fluctuations.png) and [per-seed tail distributions](E3/angular_tail_distributions.csv) report RMS and peak values for both all runs and runs without a fall; RMS is descriptive and never substitutes for primary success.

For L on IC-A with measurement level 0.01, per-seed median final-window angle RMS is 0.00814–0.00909 rad, while physical angular-velocity RMS is 0.256–0.260 rad/s. All six remain upright in every seed but miss the angular-rate limit. At level 0.1, all six L cases still remain upright, with median angle RMS 0.109–0.157 rad and angular-velocity RMS 1.24–1.42 rad/s; H falls in four or five of six. Thus strict endpoint failure and loss of the upright region are distinct measured outcomes.

The one-sample delay likewise leaves all six L cases upright but gives per-seed median tail angle RMS 0.0211–0.0212 rad and physical angular-velocity RMS 1.110–1.113 rad/s. This isolates sensitivity to a 20 ms input delay; it is separate from E0's measured planner computation times, since those nominal simulations apply commands without computation latency. E3 uses 14-second runs, so its endpoints do not establish eventual behavior beyond that horizon. Sources: [case diagnostics](E3/summary.csv), [angular distributions](E3/angular_tail_distributions.csv), [cart/input/local-use distributions](E3/distributions.csv), [condition figure](E3/robustness_conditions.png), [noise amplitudes and protocol](E3/manifest.json). All 3,696 runs completed in 0.45 wall hours.

### Novel linear system

**The method transfers to this four-state LTI through its identified local law: L and the known-model clipped LQR both succeed on every tested IC. H succeeds on six small diagonal ICs but only 30–31/50 random-box ICs per seed. M1 has no primary successes, including on the 60-second subset.** This is one additional system, not a guarantee over linear or nonlinear plants.

The plant has discrete A=diag(1.06, 1.02, 0.96, 0.90), B=(0.2, 0.2, 0.1, 0.1)^T, Q=I, R=0.03 and inputs limited to ±10, at 20 ms samples. Its primary endpoint requires every state component within ±0.02 throughout the last three seconds. IC-A consists of x0=v(1,1,1,1) for v=±0.125, ±0.25, ±0.5; IC-C50 is uniform in the four-dimensional ±1 box. The 60-second subset uses A indices 0, 2 and 5. These plant-specific ICs and endpoint are separate from the pendulum's angular endpoint. Sources: [protocol](E3_linear/manifest.json), [fixed ICs](E3_linear/initial_conditions.json).

| Arm | IC-A, 14 s | IC-C50, 14 s | IC-A subset, 60 s |
|---|---|---|---|
| H | 6/6 in all seeds | 30, 31, 31 / 50 | 3/3 in all seeds |
| L | 6/6 in all seeds | 50/50 in all seeds | 3/3 in all seeds |
| Known-model clipped LQR | 6/6 in all seeds | 50/50 in all seeds | 3/3 in all seeds |
| M1 | 0/6 in all seeds | 0/50 in all seeds | 0/3 in all seeds |

On the random box, L adds 20, 19 and 19 successes beyond H, with no H-only catches (per-seed exact McNemar p=1.91×10⁻⁶, 3.81×10⁻⁶, 3.81×10⁻⁶). H's per-seed median local fraction is 94.9–95.9% there and 100% on IC-A; the largest diagonal ICs do invoke CEM, so that median does not mean every H run uses only L. Sources: [counts and Wilson intervals](E3_linear/rates.csv), [matched comparisons](E3_linear/paired_comparisons.csv), [case metrics](E3_linear/summary.csv), [success figure](E3_linear/success_rates.png).

The local linear fit's per-state RMSE is 1.86×10⁻⁸–6.10×10⁻⁸ across seeds and coordinates. Held-out final 30-step NN errors range across seeds as x0: 0.0465–0.0630, x1: 0.0159–0.0232, x2: 0.00430–0.00601 and x3: 0.00526–0.00574. These are supplied state coordinates. A successful local regression and a usable NN forecast do not make the submitted planner a successful regulator on every tested IC. Sources: [model diagnostics](E3_linear/model_diagnostics.csv), [saved seed-0 diagnostics](E3_linear/models/seed0/diagnostics.json).

For the known LTI, each case records its accumulated stage cost plus the unconstrained optimal continuation x_T^T P x_T, divided by the unconstrained optimal initial cost x_0^T P x_0. This continuation comparison is distinct from a constrained-MPC optimum or a measured infinite future; the simulated LQR reference is clipped. On IC-C50, the per-seed median ratio is approximately 1 for both L and the known-model reference, 1.49–1.89 for H and 212–397 for M1. The largest L/reference ratio is 1.00359 because clipping can add cost. Failed H runs have much larger tails: the maximum recorded state norm reaches 7.22×10⁹. The [cost ratios](E3_linear/linear_cost_ratios.csv), [descriptive distributions](E3_linear/distributions.csv) and [figure](E3_linear/descriptive_comparison.png) retain those failures; the figure shows within-seed 25th–95th percentiles alongside medians. All 708 cases completed in 0.29 wall hours, including postprocessing ([manifest](E3_linear/manifest.json)).

### Novel nonlinear system: acrobot

**Under the disclosed acrobot collection, costs and input limit, H, L and M1 all fail every tested case in all three seeds.** Each arm gives 0/6 on IC-A, 0/50 on IC-C50 at 14 seconds, and 0/3 on the 60-second IC-A subset. Every run crosses a link's ±π/2 boundary; no nonfinite state is recorded. Longer simulation therefore does not rescue the no-fall endpoint on these ICs. Zero-count Wilson upper limits are 0.390, 0.071 and 0.561 respectively. Sources: [rates](E3_acrobot/rates.csv), [case metrics](E3_acrobot/summary.csv), [success figure](E3_acrobot/success_rates.png).

The plant has two unit uniform rods with COM at 0.5 and COM inertia 1/12, one elbow input limited to normalized torque ±1, and eight RK4 substeps per 20 ms sample. State coordinates are q1 from upright, relative q2, and their rates divided by √9.81; costs use 100 per joint angle, 10 per normalized joint rate and R=0.03, shared by all arms. The endpoint uses both absolute link angles and physical angular rates. IC-A starts from q1=±15°, ±30°, ±60° with the other coordinates zero. IC-C50 rejects second-link absolute initial angles beyond ±1.3 rad; collection retains the full joint-coordinate box. Sources: [plant and training protocol](E3_acrobot/manifest.json), [fixed ICs](E3_acrobot/initial_conditions.json), [dynamics and endpoint](../experiments/novel_plants.py).

A post hoc local diagnostic finds a substantial model/gain mismatch. The three fitted closed-loop Jacobians have spectral radii 0.94815–0.94823, while those same gains on the true upright sample-and-hold Jacobian give **1.19694, 1.17216 and 1.28404**. Thus none stabilizes the true local upright equilibrium, even before saturation. Independent analytic mass/gravity differentiation and central differences agree within 4.8×10⁻¹¹; DARE on the true derivative gives radius 0.94763. That last value is a local mathematical reference, not an extra simulated controller or a finite-IC capture guarantee. The diagnostic identifies a local-fit limitation; it does not assign a sole cause to H or M1's complete failures. Sources: [recorded-gain radii](E3_acrobot/local_stability.csv), [independent checks](E3_acrobot/local_stability_checks.json), [calculation](../experiments/acrobot_local_stability.py).

The nearest-6000 fit selects joint angles reaching |q1|=1.07–1.10 rad and |q2|=1.22–1.25 rad. Held-out final 30-step NN errors across seeds are q1: 0.132–0.171, relative q2: 0.310–0.454, normalized rate 1: 0.398–0.469 and normalized relative rate 2: 0.908–1.479. Sources: [per-model errors](E3_acrobot/model_diagnostics.csv), [seed-0 neighborhood and fit](E3_acrobot/models/seed0/diagnostics.json). The [descriptive distributions](E3_acrobot/distributions.csv) and [figure](E3_acrobot/descriptive_comparison.png) retain failed-run excursions and saturation rather than adding acceptance thresholds. All 531 cases completed in 0.40 wall hours, including diagnostics ([manifest](E3_acrobot/manifest.json)). Together with the successful LTI local law, this limits the demonstrated transfer: this campaign supports one additional linear system and records failure on its harder nonlinear test.


## Q2 — Errors in the implementation note

The reviewed transcription is [trajectory_manifold_implementation.tex](../../../trajectory_manifold_implementation.tex), stored in the parent directory.

**Review complete: no equation/code contradiction found, but the note needs 27 qualifications and clarifications.** The [v3 annotated copy](reviews/implementation_tex_v3/trajectory_manifold_implementation_reviewed_v3.tex) preserves the original text and flags notation overloads, implicit normalized costs, collection details, switch geometry, numerical fallback and evaluation omissions. The [comment ledger](reviews/implementation_tex_v3/annotations.json) records each issue. The selected local-fit neighborhood is substantially broader than the handoff estimate; see the measured E0 diagnostics above.

## Q3 — Comparison and contribution

**H is a full-state learned-dynamics shooting controller with an identified-feedback catch. The catch supplies the strongest empirical evidence; the full-state graph theorem establishes neither prediction accuracy nor control stability.** It combines one-step transition regression, CEM planning, five-step queues and a ridge/DARE local catch. E0–E1 support the catch's importance on the authored and broader tested sets. The graph theorem supplies explicit trajectory coordinates but does not constrain prediction error or guarantee closed-loop success; see the [proof audit](reviews/paper_review_v1/proof_audit.md).

| Comparison | What differs from H | What this campaign can establish |
|---|---|---|
| DeePC | Input/output trajectory spanning and a convex QP; exact LTI representation requires excitation and observability assumptions. | Matched nonlinear cart-pole performance and regularization sensitivity; the LTI theorem does not guarantee this nonlinear baseline. |
| Lusch Deep Koopman | Autonomous state embedding trained with multi-step and linearity losses; no implemented controller. | Prose/model comparison, not an invented Lusch control baseline. |
| DFINE | Unforced noisy partial observations, latent factors and linear dynamics with Kalman inference. | Observation and uncertainty differences; no DFINE controller is constructed. |
| Baek deep RL | Model-free SAC, symmetry augmentation, triple pendulum and hardware demonstrations. | Qualitative budget/task comparison; its 0.8–2 million steps cannot establish a matched sample-efficiency advantage for H. |
| Same-data SMC | Designed scalar sliding variable and a sampled boundary-layer reaching law, using the saved local fit. | E6 compares SMC with its zero-correction control, L and H; nonlinear-plant robustness remains empirical. |

The [verified primary-source matrix](reviews/paper_review_v1/literature_matrix.csv) supports these distinctions. Nagabandi neural shooting MPC and single-model variants of PETS are the closest algorithmic antecedents. H's training and online controller do not use plant equations. Its limitations include full-state measurement, deterministic one-step learning, no uncertainty estimate, open-loop queues, a hard switch and no stability guarantee. So far, the identified local law supplies the strongest control-performance evidence. Completed E2–E3 quantify data dependence, noise/delay sensitivity and limited transfer. E4 and E6 complete the matched baseline comparisons below.

### Matched NN-MPC, ensemble and DeePC comparison

E4 uses one shared new collection for every arm: 3281 × 32 steps, or 104992 transitions, with 73504 fitted transitions. The longer collection permits two observed past samples plus thirty future inputs without joining episodes; H keeps its submitted N=30 control/training algorithm. This is a different collection from E0/E1/E2's 3500 × 30, so E4 is assessed within its own cohort. Costs, input limit and primary endpoint are shared. Output-only DeePC receives (p, θ), starts with two actual zero-input samples and uses forward differences for velocity costs; that disclosed cost/initialization difference remains in the comparison. Three prespecified coefficient penalties, 0.001, 0.01 and 0.1 per column, are all reported. Sources: [protocol](E4/manifest.json), [code](../experiments/e4_deepc.py).

| Seed | IC-A H | IC-A L | IC-C50 H | IC-C50 L |
|---|---|---|---|---|
| 0 | 6/6 | 6/6 | 21/50 | 35/50 |
| 1 | 2/6 | 6/6 | 4/50 | 36/50 |
| 2 | 6/6 | 6/6 | 33/50 | 33/50 |

M1 and the five-member mean-cost ensemble E5 score 0/6 and 0/50 in every seed. Both also score 0/3 on every 60-second authored subset. E5 is PETS-lite: deterministic episode-bootstrap models, mean member costs, one-step feedback and no local catch; it lacks full PETS's aleatoric variance and particle propagation. All six state/output DeePC settings have zero successes and every run falls, including their long subsets. This evaluates the specified regularized linear-span baseline on nonlinear data, with the nonlinear limitation stated in the manifest. It does not establish a result for every nonlinear DeePC construction. Sources: [all per-seed counts and Wilson intervals](E4/rates.csv), [case metrics](E4/summary.csv), [success figure](E4/matched_success_rates.png).

On IC-C50, L adds 14 and 32 catches beyond H in seeds 0 and 1, with no H-only catches (exact paired p=0.000122 and 4.66×10⁻¹⁰). Seed 2 has two catches unique to each arm, p=1. H therefore has no success-count advantage in this cohort, although it sometimes catches an individual IC that L misses. At 60 seconds H and L each score 3/3, 0/3 and 3/3 across seeds. The seed-1 local-law miss and fitted/true stability mismatch are detailed under Q1. Sources: [paired comparisons](E4/paired_comparisons.csv), [cart excursion, settling, final position and input distributions](E4/distributions.csv).

An E4 diagnostic examines six saved first plans from seed 0, −15°, covering every full-state/output-only regularizer. Each predicts a final angle within 0.00063 rad of zero after 0.6 seconds. True-plant replay on the **same saved input sequence** ends at −1.58 to −1.74 rad and crosses the fall boundary in every case. Initial observations and applied first inputs match the saved data; independent first-step replays have zero residual. These six closed-loop cases record no QP failures. This is an illustrative saved-case diagnostic, not a population result or a claim about every DeePC variant. Sources: [matched-input figure](E4/deepc_matched_plan_replays.png), [numbers and wiring checks](E4/deepc_matched_plan_replays.json), [replay code](../experiments/deepc_rollout_diagnostics.py).

The retained nonlinear-data bases have full row rank in all three seeds: 154/154 for full state and 98/98 for output only, at the stated 1e−8 relative cutoff. The trajectories are stored in float32. For comparison, a noise-free fully observed four-state LTI rollout with thirty scalar inputs has linear behavior dimension 34. A nonlinear graph can have that intrinsic dimension while its linear span fills a much larger ambient space. Regularization supplies a coefficient prior here; it does not require the predicted sequence to satisfy the nonlinear plant. This helps explain the model-validity issue in the six replays and the relevance of learning a nonlinear rollout map. It does not establish new algorithmic novelty beyond existing learned-dynamics MPC. Sources: [basis dimensions and retained singular values](E4/models/seed0/state_basis.json), [all three basis dimensions](E4/deepc_matched_plan_replays.json), [Part I proof audit](reviews/paper_review_v1/proof_audit.md). The comparison retains all six DeePC settings.

All 866538 DeePC calls pass the recorded acceptance checks, with zero QP fallbacks; the largest initial-condition residual is 1.90×10⁻⁸. The six matched-input replays show a model-validity failure despite feasible QPs. Six [hand-computed DeePC checks](deepc_tests.txt), two [recorded-data wiring checks](deepc_data_wiring_tests.txt) and three [ensemble checks](ensemble_tests.txt) pass. E4 completed all 1770 cases in 2.02 wall hours ([manifest](E4/manifest.json)).

### Same-data sliding feedback and zero-correction comparison

Classical SMC designs desired motion on a state/error surface and a law that attracts and maintains that motion. Its ideal rejection result concerns uncertainty entering through the control channel; smoothing changes exact invariance to practical accuracy. H's quadratic ellipsoid selects the CEM or LQR branch, and its [source](../../scripts/run_inverted_pendulum_mc.py) supplies no sliding-variable reaching law. Mode switching alone does not establish a sliding-mode design. Source: [Spurgeon's tutorial](https://kar.kent.ac.uk/41730/1/sliding_mode.pdf).

A relevant planning/feedback connection is [Dynamic Tube MPC](https://arxiv.org/abs/1907.06553), which combines nominal trajectory optimization with boundary-layer feedback and an uncertainty-dependent tube. H's origin-centered catch does not implement that construction. The approved [E6 specification](smc_research.md) instead tests one same-data discrete feedback design. Its boundary width comes from validation residuals, not a certified uncertainty bound; its fitted linear-model guarantee does not extend automatically to the nonlinear plant, clipping, measurement noise or delay. Comparing with the zero-correction control separates the added nonlinear correction from the different underlying feedback gain. Fifteen [scientific wiring and numerical checks](E6_SMC/smc_tests.txt) passed during preparation; the completed empirical comparison follows.

E6 reuses the original 3500 × 30 nominal/process-trained fits for seeds 0–2, together with H/L cases from E0/E1/E3. It requires no NN training or GPU planning. Its results are matched within that original cohort; E4's different collection prevents a direct paired DeePC-versus-SMC conclusion. The prescribed discrete surface law uses ρ=0.9, β=0.45φ and input clipping at ±10; the zero-correction control uses β=0 with the same A, C, φ and ρ. The nominal validation selection contains 1205–1445 transitions per seed and gives φ=21.46–24.85 in the normalized surface coordinate. The width is an empirical residual scale, not a certified uncertainty bound. Sources: [approved request](E6_SMC/REQUEST.json), [design and assumptions](smc_research.md), [saved calibration](E6_SMC/models/nominal_seed0.json), [implementation](../experiments/smc_baseline.py).

Counts in each list retain seeds 0, 1 and 2.

| Arm | IC-A, 14 s | IC-C50, 14 s | IC-A subset, 60 s |
|---|---|---|---|
| H | 4/6, 4/6, 4/6 | 11/50, 6/50, 16/50 | 3/3, 1/3, 2/3 |
| L | 6/6, 6/6, 6/6 | 33/50, 36/50, 36/50 | 3/3, 3/3, 3/3 |
| SMC | 6/6, 6/6, 6/6 | 35/50, 36/50, 36/50 | 3/3, 3/3, 3/3 |
| Zero correction | 6/6, 6/6, 6/6 | 35/50, 37/50, 37/50 | 3/3, 3/3, 0/3 |

Nominal gains over L are small: SMC adds two random-subset catches in seed 0 and none in the others; zero correction adds two, one and one, with no L-only catches. Their exact paired p values are 0.5 or 1. This sample provides no statistically resolved nominal success advantage over L. The correction loses one random catch relative to zero correction in each of seeds 1 and 2, but gains three long-subset successes in seed 2 (n=3, paired p=0.25). Sources: [counts and Wilson intervals](E6_SMC/rates.csv), [feedback paired comparisons](E6_SMC/feedback_paired_comparisons.csv), [success figure with per-seed intervals](E6_SMC/feedback_success_comparison.png).

With a 20 ms input delay, zero correction scores **6/6 and 33–34/50 in every seed**, while H, L and SMC each score 0/6 and 0/50. Against either L or SMC, the zero-correction design adds 33, 34 and 34 random catches, with no opposing catches (paired p=2.33×10⁻¹⁰, 1.16×10⁻¹⁰, 1.16×10⁻¹⁰). All six new-feedback authored cases remain upright under delay; the difference is in the strict settling endpoint. Thus the added correction reduces delay tolerance for these fixed parameters. This is a controlled comparison of the correction term, rather than attribution of every feedback improvement to sliding-mode theory.

| Stress, IC-C50 at 14 s | L | SMC | Zero correction |
|---|---|---|---|
| Mass ratio −25% | 38/50 | 38/50 | 38/50 |
| Mass ratio +25% | 29–32/50 | 30–34/50 | 30–34/50 |
| Length −25% | 38/50 | 38/50 | 38/50 |
| Length +25% | 26–29/50 | 28–29/50 | 29–34/50 |
| One-sample delay | 0/50 | 0/50 | 33–34/50 |

All three feedback designs retain 6/6 under the four mismatch conditions. Both new designs have zero primary successes at every measurement/process-noise level, under the same stringent endpoint discussed in E3. Their fall counts differ: at process level 0.03 in seed 1, zero correction falls in 46/50 random cases, versus 19/50 for SMC and L. At process level 0.1 every arm falls in every case. Mismatch, arbitrary state noise and delay are distinct protocols and are not collectively certified matched disturbances. Sources: [all stress counts and falls](E6_SMC/rates.csv), [H-reference comparisons](E6_SMC/paired_comparisons.csv), [case records](E6_SMC/summary.csv).

The effort tradeoff is substantial. Across seeds, nominal IC-A median applied-input total variation is 13.31–13.62 for L, 12.58–12.86 for SMC and 7.28–7.59 for zero correction, in normalized-input units. Under delay these medians become 4288–4289, 4315–4328 and 15.04–15.34 respectively. Nominal median maximum cart excursion is 1.17–1.48 m for L, 1.21–1.68 m for SMC and 1.76–5.34 m for zero correction; weaker effort can accompany more cart travel. Full settling is separate: on the nominal 60-second subset, SMC records full settling in 3/3, 1/3 and 1/3 cases across seeds, while zero correction records 3/3, 0/3 and 0/3. These finite-time observations impose no recentering requirement or asymptotic guarantee. Sources: [effort distributions](E6_SMC/feedback_distributions.csv), [effort figure](E6_SMC/feedback_effort_comparison.png), [excursion/settling/final-position distributions](E6_SMC/distributions.csv).

The independently checked true upright radii are 0.99197, 0.99939 and 1.00017 for SMC, and 0.98733, 1.00039 and 1.00145 for zero correction; all six fitted-model radii are below one. Finite angle success therefore does not certify full-state eventual stability, even for these alternative feedback laws. Sources: [radii](E6_SMC/feedback_local_stability.csv), [analytic/finite-difference plant and law checks](E6_SMC/feedback_local_stability_checks.json). Per-seed medians of per-case mean CPU-law evaluation times on nominal IC-A are 14.4–14.7 μs ([timing](E6_SMC/feedback_timing.csv)); these time the feedback calculation only. E6 completed 4050 new and reused 4050 reference cases in 9.7 wall minutes and 4.3 CPU minutes, within its one-hour allowance ([manifest](E6_SMC/manifest.json)). Its fifteen [scientific checks](E6_SMC/smc_tests.txt) pass. The evidence supports a particular same-data feedback adaptation with accuracy, effort, delay and long-horizon tradeoffs.

A plausible research gap is nonlinear input/output data-enabled prediction with explicit identifiability, noise and approximation conditions. Part I flags the output injectivity issue and defers learning/control; the current full-state implementation does not exercise that extension. The [15-row claim ledger](reviews/paper_review_v1/claim_ledger.csv) separates correct graph/rank results from these unsupported extensions.


## Protocol and resume

All required stages and the approved feedback extension are complete. Recorded experiment wall times, including the separate authored reproduction, sum to **25.33 hours**, below 40 hours of summed experiment elapsed time. These records do not measure aggregate parallel CPU worker-hours or GPU utilization, so they alone cannot certify every interpretation of the handoff’s CPU/GPU cap. E6 also remains below its separate one-hour CPU/wall limits. The [completion checks](completed_campaign_checks.json) verify final counts, reused-reference identity, sampled raw-trajectory endpoints and matched saved noise arrays.

Completed-campaign operating record (not a pending queue or restart instruction): [superseding handoff](../codex_handoff_from_claude_v1.md). Jobs longer than two minutes run detached; each experiment's `JOB` records its PID and command. New evaluations save per-case outputs and skip completed cases. The usage supervisor handles account limits; this campaign does not track them.

Time-to-angle-settle is the earliest sample after which both angular limits hold through the end, requiring at least 3 s of remaining observation. Time-to-full-settle also requires physical cart speed ≤0.05 m/s and position change ≤0.01 m/s over rolling 1 s windows. These are descriptive times, not additional primary pass/fail criteria. Prediction RMSE is reported in stored normalized state units; physical velocities multiply by √9.81.

The runner's seed jointly controls data collection, episode splitting, network initialization and CEM draws. Seeds 0–9 therefore test whole-run reproducibility, not training randomness alone. Arms share data, fitted gain and initial conditions within each seed; every case restarts the controller RNG as the authored runner does. H* copies the learned state scaling into the existing true-RK4 transition so its exploration scaling stays matched.

Wilson 95% intervals are reported separately per seed/condition. Fixed authored cases and grids describe tested conditions; their intervals do not imply random population coverage over an untested operating domain. Exact McNemar comparisons use matched ICs within seed. Fixed-grid comparisons are descriptive rather than evidence of population-wide coverage; the many comparisons have no campaign-wide multiplicity correction. Emphasize matched catch counts and variation across seeds. Settling times are right-censored when the observation is too short; omitted settling times are counted in distribution tables.

The bounded [paper ledger and proof audit](reviews/paper_review_v1/proof_audit.md) and [primary-source literature matrix](reviews/paper_review_v1/literature_matrix.csv) are available. The graph and rank results are correct under their assumptions. The process-noise extension needs qualification if noise is not retained in the behavior vector. The [v3 annotated implementation copy](reviews/implementation_tex_v3/trajectory_manifold_implementation_reviewed_v3.tex) contains 27 comments and preserves source prose/equations. No equation/code contradiction was identified; normalization, notation, evaluation and collection omissions remain real specification issues.

## Deviations and remaining limits

2026-10-09: E0 was parallelized across eight workers after measured serial GPU utilization was 9%; saved models/cases were reused. Unused CVXPY imports were omitted when loading the submitted functions. The submitted source files remain unchanged. E1's three-IC long subsets were fixed before evaluation; their exact indices are recorded in the manifests and case tables.

E4 changed collection length to 32 steps to support a two-sample IO past without joining episodes, while retaining N=30 control and approximately equal transition count. Every E4 arm shares that new collection. IO initialization, forward-difference velocity costs, SVD truncation and all prespecified regularizers are disclosed above and in its manifest.

2026-10-11 (user addition): E6 automatically followed the core stages, using saved fits and matched references, CPU feedback only, and a one-hour CPU/wall allowance inside the original cap. All planned nominal and stress cases completed. No active core job was restarted to adopt it.

Optional EDMD/SAC and the second cart-velocity grid were not run. Comparisons with Lusch, DFINE and Baek are limited to verified primary sources; a matched deep-RL sample-efficiency advantage remains **unresolved**. Hardware behavior, certified nonlinear robustness and general eventual stabilization remain **unresolved**. The completed experiments delimit the observed claims and do not supply those guarantees.

Validation includes the linked metric/arm, collection/noise, novel-plant, ensemble, DeePC and feedback checks, plus independent saved-plan and local-Jacobian diagnostics. Raw arrays and weights remain local and ignored by git; their paths are recorded in case/model metadata.

2026-10-12 (documentation review): Required campaign deliverables and selected source/artifact evidence were reviewed without new experiments. Added the collaborator briefing and completion-review note, corrected stale README navigation, and clarified statistical and resource-accounting scope. Raw arrays/weights, runtime logs and some supporting review sources are local rather than tracked; a fresh clone is not a complete replay bundle. The completion checks are sampled checks, not an independent rescore of every trajectory. See [review details](DOCUMENTATION_REVIEW.md). Historical annotation R05’s evidence-needed wording and R13’s reproduction-open category are superseded by the completed E0–E4 findings; the preserved annotated text is unchanged.

2026-10-12 (reader clarification): The briefing now gives explicit success counts by seed, defines linear feedback and experiment cohorts, attributes the broad local-fit finding to validation, and links named unstable gains. New figures only postprocess saved 14-second cases, excluding 60-second follow-ups. Three-seed success frequencies are descriptive, not calibrated operating-domain probabilities. The stability handoff proposes proof/certification work; it does not report a certified region or authorize new experiments.
