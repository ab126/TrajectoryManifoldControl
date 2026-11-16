# Codex Progress Log

Append dated entries for meaningful project milestones.
Include tested results, important decisions, and commit IDs when available.

## 2026-10-17 — Integration protocol and source audit

- Started main at 325beae; untracked AGENTS.md belongs to user. Read STATE/LOG.
- `git submodule add https://github.com/aroyphillips/TrajectoryManifoldControl.git external/TrajectoryManifoldControl`
  pins e66a2b79f27f347286576f678f424625f60787b4. No commit/push.
- `git diff --no-index` confirmed identical original manifold_control.py and
  run_inverted_pendulum_mc.py. Read full implementation TeX in upstream handoff.
- No project-level LICENSE/COPYING, checkpoint .pt, or dataset/raw .npz tracked.
  Do not confuse third-party licenses or archived result tables with these.
- Added PLAN.md, isolated subprocess entry point, A/B/C scoring and metric tests.
  Test discovery explicitly excludes external to avoid upstream import collisions.
- System python3=3.12 lacks torch and ensurepip; existing fmri311 also lacks torch
  and could not bootstrap venv. Created /tmp/tmc-validation-env via micromamba;
  dependency installation underway. Sandbox bwrap network setup fails; commands
  run through approved escalation. No performance/success claim yet.

## 2026-10-17 — Baseline/metric verification and decoder diagnostics

- Isolated scientific Python at /tmp/tmc-validation-env (Python3.11.17,
  torch2.14.1+cpu, numpy2.4.6, scipy1.17.1, real cvxpy1.9.3).
- `python -m pytest -q`:40 passed in6.22s after correcting one malformed test
  helper. Discovery targets tests/, excludes external and root test_lqr_dd.py
  (an executable simulation/animation script, not a unit-test module).
- `scripts/validation_benchmark.py --verify`: saved C identities exactly match
  IC_SEED20261009; direct upstream imports, held-input RK4 timing, normalized dt
  and episode splits pass. No CVXPY workaround.
- Archived-gain L, six A + first3 C: binary and compared scalar trajectory values
  exactly agree with archive. Figures saved under saves/validation/initial-replay.
- Reconstructed seed0 original dev:3500episodes,100epochs,76036parameters;
  train2450episodes/73500transitions; validation525; test525. H4/6,L6/6 on A;
  archived binary agreement12/12. CPU reconstruction is not original CUDA replay.
- `scripts/diagnose_validation_decoder.py`: test NRMSE0.01018/0.01695/0.00910/
  0.01839; initial,input,predicted-prefix residual0; numerical rank34. Candidate
  input replay on4 validation states: terminal angle errors0.029/0.071/0.124/0.022rad.
  One CEM cost762.67 exceeds L diagnostic candidate702.23 under same learned cost.
  No L candidate used as manifold action. Four CEM solves~83ms, all miss20ms.
- Full L launches named seed1/2 initially omitted --seed: summary duplicate guard
  caught this. Preserve and exclude these seed0 duplicates. Relaunched explicit
  seed1/2 into *-corrected paths; final summary pending.

## 2026-10-17 — Matched baselines and native implementation milestone

- Corrected full L reproduction:122/200,130/200,128/200, all three endpoints;
  all600 A/B records match upstream, compared final_theta/final_p/max_abs_p
  differences0. Reconstructed H also agrees on C3 (H1/3,L2/3).
- Original checkpoint SHA256 c3f6032f5e35b6bb36395a4eec23813ac41e44a4c72a04037b03629f5cbf4b4f;
  measured data SHA2560e825cda7821de4afbe839f5da022b5942c6f8b14dd5208a040f05278cbde119.
- Added opt-in native safe CEM/projected Adam, actual-age warm starts, one-action
  feedback, optional periodic and terminal-window objectives. Original sources
  untouched.46tests passed. Native controller never invokes L.
- Static matched validation-state solver screen: safe/original CEM same costs;
  Adam20iter beats CEM on1/4, worse3/4;~0.21s after startup versus~0.08sCEM.
- Matched10epoch fine-tuning5step loss: common validation criterion0.00398455
  versus0.00397709 one-step control. Original M on tuning starts0/2 for both;
  L2/2. No multistep improvement claimed; no expensive direct decoder/geometric
  penalty introduced without evidence.
- Fixed-model tuning screen in progress, independent seed20261017. First state:
  M,H,cem5,cem1 fail; L and adam1 recover. Adam terminal angle max0.00321rad,
  rate max0.00983rad/s, cart final4.304m, no fallback,~0.200s median solve.
  This is tuning evidence only, not final benchmark superiority.
- Added Markdown report/reproduction commands and a frozen-protocol evaluator
  (not yet executed). Full improved C200/fresh200 cost remains high on CPU.

## 2026-10-17 — Resumability and preliminary objective ablation

-47tests pass (6.17s), including freeze rejection for incomplete evidence and
  changed source. Syntax/whitespace checks pass; upstream submodule stays clean.
- Tuning invocation1 stopped at a between-case budget boundary after379.96s
  (a complete periodic case ran across the300s boundary). Seven of16cases saved.
  Both quadratic Adam and periodic Adam recover the first start, no advantage
  of periodic cost demonstrated. Same command resumed; prior cases skipped by
  verified unchanged protocol/model/source hashes.
- Full native C200 estimated7.8CPU hours per fitted seed at0.2s×700×200;
  final three-seed + fresh-holdout evaluation is beyond this bounded screen.
  `scripts/frozen_manifold_validation.py` provides hash-checked resumable final
  evaluation and within-seed paired endpoints; not a claim it has run.

## 2026-10-18 — Second tuning start and final-screen boundary

- Invocation2 finished through adam1 on second tuning state;14/16cases complete.
  Native quadratic Adam2/2 versus original M0/2,H1/2,L2/2, safe CEM5/1 both0/2.
  Second Adam terminal-angle maximum0.00622rad. This is tuning evidence only.
- Resumed invocation3 for the remaining periodic/window second-start cases.
  Both model-training ablations are completed negative results (M0/2,L2/2).
- Verified eight tuning states distinct from pinned fresh200, in addition toC200.
  Success/failure five-panel figures generated and Adam success figure inspected.
- Frozen evaluator now pins original H/L artifacts separately from candidate
  artifacts and refuses a fine-tuned checkpoint as the original H baseline.

### Selection rule recorded before remaining objective results

Maximize terminal recovery on the declared independent tuning screen; on tied
binary outcomes retain the original quadratic objective/terminal weight instead
of adding an unproven objective change. Plain Adam already2/2; periodic/window
second-start outcomes not yet available. No final-score selection is allowed.
Upstream robustness helper nominal-mismatch identity and delayed-input timing
checked directly; frozen evaluator supports those interventions without passing
parameters to controllers.

### Completed screen and frozen release

- All16 tuning cases complete. M0/2,H1/2,L2/2,CEM5/CEM1 both0/2, Adam2/2,
  periodic1/2, periodic+terminalwindow2/2 for each A/B/C endpoint.
- Selected plain Adam by the recorded tie rule; original model/objective/terminal
  weight, one-action feedback,20iterations,3starts,lr0.15. Freeze SHA256
  97a3d77c12d69a30a8ef799f8703c77f425f0dc94f762c88a0403c7a01ad9f41.
- `frozen_manifold_validation.py evaluate --verify-only`: all hashes/config pass.
  Began bounded C200 evaluation, arms L H native,180s budget checked between
  cases, preserving partial results. Also evaluating predeclared authored index2
  under input_delay and mass_plus25. No retuning allowed from these outcomes.
- Completed inexpensive baseline60s Aindices2,3: originalH/L both2/2 under all
  endpoints. Native60s remains pending; do not imply these were native tests.
- Compact EVIDENCE.json exported; original-arm forward counts derived from
  raw CEM records (ablation's original-arm zero counter means uninstrumented).

### Frozen robustness and first nominal result (no retuning)

- Delay1 sample, fixed authored−15deg: native1/1,H0/1,L0/1 under all endpoints.
  One native-only paired win; exact paired p=1 with n=1, no broad inference.
- Cartmass+25%, same start: native/H/L all1/1. Upstream interventions reused.
- First nominal C state: native fails all endpoints (angle max0.01271rad passes,
  physical speed max0.06899rad/s exceeds0.05), L succeeds,H fails. No numerical
  failure/fallback; native median controller call0.2115s,100%20ms misses.
  Do not change tolerances or tune from this result. Second nominal native case
  still running; C200 partial and resumable.

### Final bounded comparison and session handoff

- Bounded frozen C200 invocation completed2paired starts in306.20s, then stopped
  at its own between-case boundary. A/B/C: native1/2,H1/2,L2/2. Native versusL
  0wins/1loss, exact paired p=1.0 (n=2); target not achieved. No retuning.
- Native case1 settles angularly at2.40s (L0.98,H1.26); terminal angle/speed maxima
  0.00715rad/0.02587rad/s. Native C0/1 finalcart−6.213/4.259m, energy22.817/1.794,
  median call0.2115/0.2035s,100%20ms deadline misses, no fallback/numerical failure.
-47tests pass; syntax/whitespace checks pass, submodule clean. Figures generated
  for successful and failed tuning/frozen/robust/long-baseline trajectories.
  `EVIDENCE.json` preserves compact machine-readable data and hashes; raw/model
  artifacts remain ignored under saves/validation. No commit/push or deletion.
- Full native/H C200, remaining fitted seeds1/2, independent fresh200 and native
  60s not completed. Frozen protocol and exact resume commands in REPORT/STATE.
  Estimated native C200~7.8CPU hours/seed before other arms/holdouts. No active
  controller experiment remains after the bounded run. Future configuration
  selection must not use the observed final C scores.

## 2026-10-18 — Minimal root validation integration (user-requested replacement)

- Started clean on main at f41056d670cdb60a03ca73e118be9517fff987cd; read STATE/LOG
  and verified upstream remained at e66a2b79. Reviewed Roy's checked-in reader
  brief and plans (harness under root validation/, submitted source unchanged).
  Public GitHub commit/issue comment endpoints returned empty lists.
- Replaced the full submodule with root validation/:19files,2.2MB. Selected six
  runtime helpers plus saved ICs, seed diagnostics, four reports and method note.
 16source/data/report files byte-identical;1236 H/L reference JSON objects packed
  losslessly and individually compared with originals before removal.
- SOURCE.json pins upstream URLs/paths/commit and hashes, including local reused
  SUT. SHA2567d97e6bd3a8f082b5b87c0c9c6372545e203e6cde06643c695ee347cbb913dea.
  No duplicate controller, plant or root notebooks; no runtime submodule needed.
- `git submodule deinit -- external/TrajectoryManifoldControl` and `git rm --
  external/TrajectoryManifoldControl .gitmodules`:clean checkout removed, Git
  object store retained locally. Saved local models/results unchanged; no commit.
- Initial patch request timed out in permission review without execution; retry
  succeeded. This was a timeout, not a safety rejection. Sandbox bwrap remains
  broken, so approved escalated commands were used.
- Root adapter --verify passes. L six authored + first3random replays:9archived
  comparisons,0endpoint mismatches and0differences in compared trajectory scalars.
- New integration-only bundle01e556c2f7100e95bb121482bbb76ff596b2bd04d854e340df6ef16e80e04767;
  scientific settings/models/scoring/protocol identical to prior release97a3d77c.
  Authenticated old integration hashes against f41056d. Copied6caseJSON+6rawNPZ+
  summary into final-root-layout-seed0-C;13files byte/hash matched, old artifacts
  untouched. LAYOUT_MIGRATION.json records ancestry; --verify-only succeeds.
- Importer and guarded migration commands in docs/validation/LAYOUT.md. Historical
  report commands explicitly labeled for the old source version. No new native
  simulations, training, parameter changes or validation-score selection.
-49tests passed before adding the final migration guard test; final suite pending.

### Final layout verification

- `env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
  /tmp/tmc-validation-env/bin/python -m pytest -q`:50passed in5.83s.
- Syntax compilation and git diff --check pass. git submodule status is empty.
  Root snapshot has19tracked-candidate files (2.2MB); no duplicate src/notebooks.
- Historical freeze still97a3d77c12d69a30a8ef799f8703c77f425f0dc94f762c88a0403c7a01ad9f41;
  historical EVIDENCE.json untouched. All13 copied historical files verified
  byte-identical in both old and migrated outputs. Root-layout freeze verification
  passes; no new native control, tuning or training performed.
- Handoff now names the new bundle/output and exact continuation command.
  Uncommitted cleanup/import changes ready for review; no commit or push.

## 2026-10-18 — Authorized publication and upstream history preservation

- User requested: first commit/push the local cleanup, then preserve Roy's
  current fork as a branch and merge history while keeping our local version.
- Pre-publication verification: 50 tests passed in5.75s; git diff --check passed.
- Cleanup commit/push and history-only merge are next; no experiment reruns.

### Published cleanup and Roy's original history

- Cleanup committed as 7d495857bf0f5429f288f0d80b11218296081805 and pushed to
  origin/main. Queried Roy's current default main: e66a2b79f27f347286576f678f424625f60787b4.
- Added remote roy (https://github.com/aroyphillips/TrajectoryManifoldControl.git),
  fetched main, created roy-validation, and pushed that branch to origin.
  Original author metadata and commit IDs are retained, not filtered or squashed.
- `git merge --no-ff -s ours roy-validation` created and published merge
  75f0671222788b70a77f7074a3f5b787b175ac39. Parents are cleanup 7d495857 and
  Roy's e66a2b79. Shared ancestor was 325beae718ab093d3ca48b18a59ae61fd93798a3.
- `git diff --exit-code HEAD^1 HEAD` and ancestry verification passed. Both
  cleanup and merge trees equal a379977e5319f406b4d30bd6fb27ecc4132b8777:
  no upstream redundant files or source changes entered the working tree.
- Handoff documentation updated separately; no research results changed.

## 2026-10-21 — Trajectory Manifold Control Visualization

- Started clean main at dd6de61a; read STATE/LOG, source and all three pendulum
  notebooks. Canonical decoder packs input-first, legacy decoder state-first;
  visualization explicitly repacks complete horizons state-first for projection.
- New observational module src/trajectory_visualization.py, example/export driver
  scripts/visualize_trajectory_manifold.py, notebook-only execution helper
  scripts/execute_visualization_notebook.py and docs/validation/VISUALIZATION.md.
- Canonical conditioning fixes x0 and samples bounded future inputs. Legacy search
  operates only on temporary encoder-derived latent coordinates, with initial-state
  and bound residual masks. No plant access in sampling, projection or rendering.
- Normalization/projection/background/camera/color scale fixed across run. All
  sampling, costs, selected plans and projections cached before rendering; pure
  rendering verified by tests replacing decoder/sample/transform with failures.
- New sections appended to primary manifold_control.ipynb and historical debug
  notebook. Historical preflight references missing runner interfaces; those cells
  are untouched. Explicit canonical-checkpoint path permits isolated execution.
- System sandbox still fails before commands (bwrap loopback); approved escalated
  commands used. Installed nbclient/nbformat/ipykernel only into /tmp environment,
  registered tmc-viz kernel within that environment. No mandatory runtime dependency.
- 59tests passed in10.40s, including GIF frames, latent search, determinism,
  conditioning, fixed projection and preserving live notebook model flags.
- CLI .6s seed234 visualization using unchanged reconstructed seed0 checkpoint:
  256background,24candidates/frame,31frames including final measurement. PCA variance
  fraction .729988, neighbor overlap .563965, temporal distortion .883913. Optional
  UMAP absent, explicitly fell back to PCA. No ffmpeg; GIF works, MP4 guard tested.
- Debug notebook's three new cells executed successfully using nbclient; static
  figure visually inspected. Original62cells/metadata verified unchanged. Primary
  notebook's appended cells and final full-suite rerun pending at this checkpoint.
- No training/controller code altered, no benchmark retuning, commit or push.

### Final visualization verification

- Final full suite:60passed in10.46s; compileall and git diff --check passed.
  Added primary notebook summary-config reuse and model-isolation tests, plus
  an exact rank-three embedding regression for continuity normalization.
- Corrected temporal distortion to use RMS Euclidean background radius in each
  space, avoiding dimension-dependent scaling. Final value .168267 replaces the
  earlier .883913. PCA variance .729988 and neighbor overlap .563965 unchanged.
- `python -m scripts.execute_visualization_notebook --kernel tmc-viz` executed
  all3 primary visualization code cells successfully, after the debug section was
  independently executed. Final source notebooks preserve all16primary/62debug
  original cells and metadata. Existing primary missing-ID metadata warning is
  harmless; no historical cells were normalized/re-written.
- Refreshed CLI with --projection compare: optional UMAP absent, clear PCA fallback.
  Model checkpoint SHA c3f6032f5e35b6bb36395a4eec23813ac41e44a4c72a04037b03629f5cbf4b4f.
- Validated both final GIFs:31frames,696949bytes; static PNG276226bytes. Full cached
  backgrounds(256,154), candidates(31,24,154); all744 current-state candidates
  decoder-feasible, max x0 residual1.41472e-8. Terminal selected horizon absent
  as intended. Executed notebook has no error outputs. No model fitting/evaluation
  claims follow from this .6s visualization demonstration.
- Changed files: new visualization module, two scripts, test module and docs;
  README and append-only notebook sections; STATE/LOG. No commit/push requested.

## 2026-10-21 — Named micromamba project environment and publication

- Started main at03733c2e (visualization already committed); only AGENTS.md was
  dirty. User requested control311, complete dependencies, README and
  docs/visualization.md updates, then commit/push including other changes.
- README recommends Python3.11. micromamba2.9.0 selected3.11.17 under
  /home/ab126/micromamba/envs/control311. Updated existing environment.yml,
  retaining3.11 line rather than switching to3.12+. CPU PyTorch is portable default.
- Added notebook execution, UMAP, ffmpeg, CasADi, optional do-mpc/PyDeePC. First
  pip step failed (no pydeepc index release); inspected official upstream setup,
  pinned Git revisionab05d231e39c5dfe616f3edd60a317b4f9419665 and retried using
  micromamba install -n control311 -f environment.yml -y. Installation succeeded.
- pip check clean; scientific/notebook/optional-control imports pass. do-mpc
  warns about unused ONNX/OPC-UA extras, not required by repository workflows.
- Registered environment-local control311 kernel. Full suite60passed in10.50s.
  Primary appended notebook3code cells executed under control311 successfully.
- Real UMAP comparison and MP4 export completed for .6s run,31frames. UMAP
  overlap .59375 > PCA .563965 but temporal distortion7.0891 > PCA .16825;
  reliability criterion correctly falls back to PCA. Artifacts under
  saves/visualizations/control311-compare and control311-notebook (execution).
- README/environment setup updated; new docs/visualization.md tracked via an
  explicit .gitignore exception. Historical detailed docs retain old validation
  notes with pointer to maintained instructions. No source/model changes.
- User's AGENTS.md edit preserved verbatim and included in authorized publication.
- Final ffprobe verification: H.264,31frames. git diff --check clean. Publishing
  all8setup/docs/instruction files with git commit and git push origin main;
  resolve exact commit hash from Git history (not embedded recursively here).

## 2026-10-27 — Projection mode debugging and comparison output

- Started main with user's manifold_control.ipynb modified. Verified argument
  forwarding was correct: FixedProjection treated explicit umap and compare
  identically, rejecting UMAP for poorer diagnostics; compare rendered only winner.
- Changed explicit UMAP to retain a finite fitted embedding with quality warning.
  compare retains automatic top-level selection while exposing pca/umap views from
  identical cached trajectories. Notebook static and animation cells show both;
  export writes both complete artifact sets. Metadata records requested method.
- src/trajectory_visualization.py, scripts/visualize_trajectory_manifold.py,
  both visualization notebook sections, tests and docs/visualization.md updated.
  User's prior notebook output/source edits preserved and excluded from commit.
- NUMBA_CACHE_DIR=/tmp/tmc-numba-cache MPLBACKEND=Agg python -m pytest -q:
  61 passed, 1 failed (unrelated archived validation set exact comparison,
  max difference 4.4408921e-16). Focused visualization suite:12 passed in16.91s.
- Fresh-kernel nbclient execution of actual primary notebook visualization cells
  passed for pca, umap and compare using synthetic canonical live decoder.
  24 background samples,3 candidates,.04s; artifacts under
  saves/visualizations/projection-notebook-debug/<method>/executed.ipynb.
- Real UMAP0.5.12 integration and GIF exports passed; UMAP static visually checked.
  CLI compare with saved canonical_decoder_dev.pt also passed and generated
  both labeled views at saves/visualizations/projection-checkpoint-debug/.
  Syntax and git diff --check pass. Jupyter socket access required escalation;
  Numba required writable /tmp cache under sandbox. No scientific models changed.

## 2026-10-27 — Right-panel horizon selection clarification

- Inspected DecoderSampler.sample, visualization frame construction/rendering,
  notebook configuration and notebook_visualization driver. Canonical colored
  candidates use measured current state and independently sampled bounded future
  input sequences, rolled forward by the learned decoder. Full H+1 state/H input
  behaviors are projected to single points. Gray background varies initial state;
  red star is the separate actual controller solve, not best-of-plotted samples.
- Legacy path samples encoded training latents and optimizes initial-state/input
  feasibility. Neither path constructs current-state candidates from H past samples.
- No code/notebook edits or runtime tests; only continuity docs updated. Main at
  e2908c95; pre-existing modified manifold_control.ipynb preserved. No pending task.

## 2026-10-27 — Fix visualization imports in running notebook kernels

- User notebook saved traceback fails at viz.comparisons after displaying an old
  metadata schema, confirming stale imported class; current disk constructor
  initializes comparisons for every method. Both notebooks now reload dependency
  then driver in configuration cell and rebind helpers. Model/training untouched.
- Added two regression cases executing actual notebook cells in subprocesses with
  deliberately injected old visualization classes missing comparisons. Both pass,
  including static/GIF/data export. Updated docs/visualization.md with rerun steps.
- NUMBA_CACHE_DIR=/tmp/tmc-numba-cache MPLBACKEND=Agg python -m pytest -q:
  63 passed, 1 pre-existing failure. Direct validation_benchmark.py --verify
  reproduces exact-array mismatch, max absolute difference 4.4408921e-16.
- /tmp/tmc_verify_viz_cells.py executed all three actual visualization cells in
  fresh control311 Jupyter kernels with simulated stale imports: both notebooks,
  pca/umap/compare (six passes), real canonical_decoder_dev.pt, .04s, 24 background,
  3 candidates. Verified requested/actual methods, both comparison directories,
  nonempty exports, 3-frame GIFs, and no notebook error outputs. Artifacts under
  saves/visualizations/stale-import-check/<notebook>/<mode>/executed.ipynb.
  Local kernel sockets required sandbox escalation; execution approved/succeeded.
- git diff --check clean. Preserve user's existing primary notebook changes and
  stage only configuration import source changes; publish fix per AGENTS.md.
  User action: rerun all three viz code cells starting with configuration.
- Publication blocked: automatic approval review rejected combined git commit
  and push to origin/main because default branch/external remote publication
  needs explicit approval of this payload/target. Neither action executed.
  Verified fix remains staged; this final blocker note and STATE update unstaged.

## 2026-11-13 — Audit fallback LQR and nonzero target support

- Main e61ba0ae; pre-existing manifold_control.ipynb edit preserved.
- Traced scripts/run_linear_mc.py:510–536: same u_lqr callable injected into
  failure fallback and simulated baseline; x_ref forwarded; CLI u_ref zero.
  src/linear_system.py:485–561 supports u_ref-K@(x-x_ref), clipping. Constant
  reference regulation assumes equilibrium A*x_ref+B*u_ref=x_ref.
- Validation campaign.controller L uses clipped -gain@state; H delegates to
  scripts/run_inverted_pendulum_mc.py:199–249 with same gain. H switches only
  when state@Q@state < .75; failed solves elsewhere return zero. No reference
  parameters. Frozen native uses src/manifold_experiments.py:native_controller,
  deliberately no LQR, failure uses shifted seed or zero. Fallback event counts
  in reports must not be interpreted as LQR usage.
- Runtime audit via validation adapter and mocked failing solver confirms H/L
  exact action equality locally including saturation, failed H returns zero
  unlike L outside region, and nonzero finite-horizon LQR x_ref/u_ref honored.
  Initial clipping probe lay outside H switch region; corrected synthetic gain
  tests clipping within region. No scientific source changes or benchmark rerun.
- MPLCONFIGDIR=/tmp/tmc-lqr-mpl control311/bin/python -m pytest -q
  tests/test_manifold_experiments.py: 6 passed in 2.33s.
- Audit only; checkpoint documents updated. Any target-aware LQR failure policy
  would require a separately identified experiment, preserving frozen contracts.

- Audit documentation committed as 5ab64b0b. Push to origin/main rejected by
  automatic approval review: external default-branch destination not verified
  trusted; requires explicit payload/destination approval. No push performed.
  This publication note remains uncommitted; user notebook edit preserved.

## 2026-11-13 — Shared target-aware LQR/hybrid and strict TMX launch

- User requested validation L as target, arbitrary x_ref/u_ref, prepared hybrid,
  new standalone TMX-versus-linear validation and parameter/CLI guidance.
- Added src/validation_controllers.py with shared clipped validation feedback,
  reference-centered hybrid switch, reference-aware manifold objective and
  strict TMX failure exception. Hybrid L outputs preserve float64 baseline
  precision; no cast through torch before returning linear actions.
- Added scripts/validate_tmx.py with paired arms, artifact/source/config guards,
  bounded between-case budget/resume, raw trajectories, reference-relative
  metrics, latency, explicit failures and paired summary. TMX never applies a
  substitute action after failed solve. New experiment; frozen sources unchanged.
- Added tests/test_validation_controllers.py (6 tests), docs/validation/TMX.md,
  README pointer. Local seed234 canonical checkpoint lacks training data; CLI
  requires either --data for matched fit or explicit --archived-gain-seed for an
  accurately labeled unmatched screen. Reconstruction command documented.
- Recommendations grounded in docs/validation/REPORT.md small screen: Adam/plain
  2/2 vs CEM0/2 and L2/2; no superiority claim. Start20iterations/.15/3starts,
  replan1, terminal window1. Screen runtime budgets and rates on tuning only.
- control311: six new tests pass (7.30s), initial focused tests11pass. Full suite
  68passed,2failed in48.67s: unchanged notebook compare configuration vs test
  expecting comparisons=={}, and pre-existing adapter exact-array floating-point
  mismatch. Verified notebook source equals HEAD; user output edits preserved.
- Real checkpoint CLI .04s zero and nonzero references, TMX/L/hybrid, resume and
  exact archived-L-law action replay passed. Outputs under ignored saves/validation/
  tmx-smoke and tmx-reference-smoke. verify_snapshot and py_compile pass.
- No long scientific run launched or new recovery improvement claimed. Prior
  origin/main push rejection still requires explicit publication approval.

## 2026-11-13 — Correct controller abbreviation to TMC; authorized publication

- User clarified TMX was a typo for trajectory manifold control (TMC) and
  explicitly requested all necessary renaming and push.
- Renamed scripts/validate_tmx.py to scripts/validate_tmc.py and
  docs/validation/TMX.md to docs/validation/TMC.md. Current CLI arm identifiers,
  filenames, manifest source path, tests, README, guide and STATE now use TMC.
- Historical log entries and ignored old smoke artifacts retain their actual
  original names; no saved experimental evidence rewritten. Controller logic
  unchanged. User's existing manifold_control.ipynb edits preserved.
- control311 pytest -q tests/test_validation_controllers.py
  tests/test_manifold_experiments.py:12 passed in9.08s. Includes real subprocess
  CLI strict-failure/resume/protocol-guard test through renamed entrypoint.
  CLI --help and git diff --check pass. No long validation launched.
- User authorized publishing the pending audit/controller commits and rename
  to origin/main; publication outcome recorded in current handoff/final response.
- Publication verified: rename commit65c82e4f pushed successfully to origin/main
  (remote advanced62e9eb84..65c82e4f); prior audit/implementation included in ancestry.

## 2026-11-14 — Clean-checkout validation instructions

- Confirmed main starts clean; saves/saved_models is empty and saves/ ignored.
  Earlier guide incorrectly assumed prior session's local dev checkpoint.
- Documented seed0 reconstruction, artifact presence check, separate matched
  smoke and tuning commands, existing-dev alternative and archived-gain limits.
- Clarified training reuse, cost, per-invocation between-case budget and fresh
  output paths versus resume. No controller changes or training launched.
- .venv/bin/python launcher --help checks passed for both documented launchers.
  First edit attempt used unavailable bare python; reran with .venv/bin/python.
- git diff --check passed; documentation commit 31f91ef6 pushed to origin/main.

## 2026-11-14 — Saved TMC report and convergence comparisons

- Found user screenshot in saves/validation/temp; legacy phase-map code still in
  validation/experiments/figures.py, uses legacy schema. Screenshot hybrid/3-seed
  provenance not verified; explicitly historical context in new report.
- Current tmc-c200-seed0-v1 contains7/400 cases: TMC3/4,L2/3 recovery; shared3,
  wins1/losses1. Median case median latency232.680ms vs0.014ms. Partial evidence.
- Implemented src/validation_report.py + scripts/report_tmc.py, report/evidence,
  saved-outcome map and all shared-start convergence pages. Frozen sources and
  launcher unchanged, allowing original run resume. Updated usage figure commands.
- Generated docs/validation/tmc-seed0-results with historical screenshot included.
  Inspected convergence figure. No simulation rerun or expensive training.
- control311 pytest -q tests/test_validation_report.py:1passed in1.48s;
  git diff --check passed. .venv has no numpy; control311 rendering succeeded.
- Publication verified: cf4c80d3 pushed to origin/main.

## 2026-11-14 — Exact original convergence map and new sweep launcher

- Cloned requested upstream into /tmp/tmc-upstream-figure at053b14890a107063db9b58859884f655173528b8.
  campaign.py SHA matches pinned local887f6fffe96da2b03d67d674613763b6a04f99f3c5b50babce7b99833640dd4f.
- Initially identified primary no-crossing endpoint, then compared both archived
  maps: screenshot pattern matches terminal recovery, including earlier falls.
  Corrected default, documented distinction, export both endpoints.
- Reproduced10086 archived D cases: terminal H2864,L3247; primary H1485,L3247
  successes out of5043each. Stored provenance/hashes, maps and NPZ counts in
  docs/validation/grid-reproduction. No new controller runs for archived maps.
- Added src/validation_sweep.py, scripts/sweep_tmc.py, scripts/plot_legacy_grid.py,
  test_validation_sweep.py; updated guide/report. Original CEM hybrid default,
  optional standalone Adam; matched per-seed model/data; resumable raw hash checks.
- control311 focused sweep/report/controller tests:9passed in16.21s. Actual
  seed0 first D H/L14s pair verified legacy metric parity;H15.64s,140solves.
  Exact resume and plot-only passed; actual hybrid/standalone .04s smoke passed.
- No new full grid/training1,2 launched. User receives full launch commands;
  original artifacts absent, reconstruction cannot promise identical weights.
- git diff --check passed. Source and docs prepared for publication.
- Publication verified:24fef25c pushed to origin/main.

## 2026-11-16 — Completed-grid investigation and candidate pipeline

- Audited completed10086-case grid: H2880/5043,L3247. Seed2H1225vsL1085
  already wins; seed0/1 lose. H-only1275,L-only1642. No recorded solve failures.
- Added investigate_tmc.py: matched artifact audits, selected raw replay checks,
  witness, coverage and disagreement plots, multistep-error and optimizer probes.
  24 independent tuning seed/start pairs per solver: CEM10 true replay median
  cost ratio.869,Adam40ratio1.233 againstCEM3; short-horizon costs not coverage.
- Launched three real two-start14s seed0 tuning screens. CEM3/queue5 and
  CEM10/queue5 eachH1/2,L2/2;Adam20/replan1 same. All jobs finished. Small
  screens reject claiming measured improvement, not a full tuning conclusion.
- Added configurable candidate train/data mixture and candidate hybrid/L screen
  (tuning orDgrid); preserved frozen sources/original sweep protocol. Tiny
  40episode/1epoch/16x16 train plus one-step/multistep fine-tuning and eval passed.
- Nine candidate/sweep/controller tests pass30.73s; new candidate workflow test
  checks CLI data/train/dimensions/short endpoint/resume/protocol change guard.
- Visual report and LAUNCH.md propose14000episodes/256x3/200epochs +20epochs
  multistepwindow10 with broad+25%upright data; explicitly unverified, no promised
  near-certain win. Seed2 repeat is posthoc existing-grid evidence, not new test.
- Large candidate training and new dense grids not launched. Final audit rerun
  under saves/validation/investigation-final-v1; report evidence updated on completion.
- Final raw-hash audit and four-figure regeneration succeeded; py_compile and
  diff check passed. c75c9f39 pushed to origin/main; all three tuning jobs finished.

## 2026-11-16 — Launch pure TMC versus LQR

- User clarified pure TMC, no hybrid, versus LQR using previous setup.
- Read controller path: standalone has no linear switch or fallback.
- Launched scripts/sweep_tmc.py --seeds 0 1 2 --controller standalone
  --models-root saves/validation/models --seconds 14
  --output saves/validation/grid-standalone-3seed-v1 --max-wall-seconds 172800.
- Detached PID1499380 verified live; manifest verified arms TMC/L,3seeds,
  10086expected cases. No new training; original matched checkpoints reused.
- Adam20/lr.15/3starts/replan1 differs from prior hybrid CEM; disclosed to user.
- Simulation in progress; no recovery results or completion claimed.

## 2026-11-16 — Core TMC launch guidance and tuning defaults

- User requested concise existing docs, core TMC for new launches/tuning, and push.
- Added docs/launch.md entry; compressed investigation LAUNCH.md and adjusted
  README, TMC guide and grid launch commands to explicit core TMC versus LQR.
- Candidate screen previously hard-coded hybrid; now defaults to --arms TMC L,
  with hybrid opt-in. Invalid standalone plans become recorded failed cases.
  Existing hybrid candidate outputs cannot resume under changed source/protocol.
- Preserved historical hybrid evidence and sweep source/default to avoid breaking
  running grid source hashes; docs always pass --controller standalone.
- PID1499380 still live; first TMC/L grid pair saved. No restart or new long job.
- control311 pytest -q tests/test_candidate_workflow.py
  tests/test_validation_controllers.py:7passed in29.34s; git diff --check passed.
- Tuning/model improvements remain hypotheses pending active-grid completion.

## 2026-11-16 — Longer convergence visualization

- User requested another GIF spanning near-target stabilization, with left axes
  angle/angular velocity/cart velocity. Updated rendering coordinates/labels/
  fixed bounds consistently; preserved original short output.
- Added --until-converged (canonical), --frame-stride and metadata for actual
  duration/cap/final SI state. Stop: all4 SI errors within [.05,.05,.02,.05]
  for50successive samples; controller/plant/weights/checkpoint unchanged.
- Generated saves/visualizations/trajectory_manifold_convergence with command:
  control311 python -m scripts.visualize_trajectory_manifold
  --checkpoint saves/validation/models/reconstructed-seed0/transition.pt
  --output saves/visualizations/trajectory_manifold_convergence
  --seconds 30 --until-converged --frame-stride 5 --projection pca.
- 30s cap reached:1500actions,301GIFframes,15FPS. Final SI state
  [4.402092507,.000104232,-.000296787,-.004603840]. Final second meets all
  three displayed-coordinate tolerances; full-state convergence false because
  position drift4.40m. Stabilization cannot be presented as return to origin.
- Pillow verified301frames and decoded final frame; NPZ shapes and final-second
  tolerance verified. Static preview inspected. Outputs ignored by Git as usual.
- Initial focused suite13passed/1failed: old notebook refresh test assumed empty
  comparisons despite notebook compare default. Test explicitly sets PCA now.
  Final tests/test_trajectory_visualization.py:17passed in34.35s.
  Three new cases verify sustained full-state stopping/time cap/axis labels.
- Active standalone grid PID1499380 verified alive; no changes to its sources,
  artifacts, training, settings or process. git diff --check passed.

## 2026-11-16 — Distinguish selected horizon from target reference

- User requested a distinct selected marker/color and red target star (or plane),
  retaining source functions for both visualization versions.
- Right selected horizon now blue diamond, infeasible selection blue X, history
  blue; red star projects ideal constant target state/zero future inputs. It is
  labelled as a reference, not guaranteed decoder/plant feasible. No plane:
  arbitrary future inputs generally move states away from target, and projected
  fixed-x0 decoded slices are generally curved, not guaranteed planes.
- Preserved all existing source function names (verified via AST against HEAD),
  notebook/control/export APIs, existing GIF directories; added optional legacy
  selected-red-star style and PCA cache replay with from_export / --replay.
  Replay restores full behavior arrays and original PCA mean/scales/components;
  does not fit, sample, or solve. Old UMAP caches explicitly rejected because
  fitted reducer was never serialized. Replay requires separate output path.
- Updated source:src/trajectory_visualization.py and
  scripts/visualize_trajectory_manifold.py; commands/semantics in
  docs/visualization.md. New sibling output folders end in _target.
- Focused visualization tests:18passed in71.44s. Roundtrip test checks raw data,
  projected coordinates, target semantics, legacy style and forbidden fit/sample.
- Both updated GIFs completed: trajectory_manifold_cli_target/animation.gif
  (31frames) and trajectory_manifold_convergence_target/animation.gif (301frames).
  Pillow decoded final frames,1170x540. Both closed_loop.npz and trajectories.npz
  match originals to1e-12 across every field (NaNs equal). No controller rerun.
- Original function-name comparison used ef522a23 source. Both prior GIFs and
  all source functions retained. Source/docs/tests/handoff ready for publication;
  git diff --check passed.

## 2026-11-16 — Right-plot audit, training provenance and exact target plane

- User flagged distant target/star, asked for a plane matching left target,
  explanation with figures, and whether gray/nearby points were training data.
- Verified original points:256random decoded background horizons,24random decoded
  current-state candidates; neither is measured training data/optimizer population.
- Original full-zero-state/zero-input reference constrains cart position, unlike
  left3coordinate target. PCA star distance1.32094->48.20309; background-center
  distance1.79787->48.33116; gray cloud ranges37.30/33.45/15.49, not a point.
  x0 background position±.5 versus actual final4.402m explains cloud placement.
- Corrected geometry explicitly changes embedding: first2standardized full-behavior
  PCs fitted on2450actual training episodes, third coordinate log1p(norm of SI
  x0 angle/angular-speed/cart-speed errors divided by [.02,.05,.05]). Planezero
  contains exactly left target, futureinputs and position free; not a feasible
  decoder manifold, terminal condition or guaranteed future tracking. Generic
  arbitrary full-behavior PCA cannot guarantee such a2Dplane. No point moved by hand.
- Added256actual measured train episodes with original dataset IDs as separate
  filled squares; old synthetic background hollow circles; decoded candidates
  keep cost colors. Original split seed0 excludes525validation/525test episodes.
  Verified checkpoint SHA and companion data SHA0e825cda7821de4afbe839f5da022b5942c6f8b14dd5208a040f05278cbde119.
- New src/trajectory_target_visualization.py; scripts/explain_trajectory_visualization.py.
  Command: control311 python -m scripts.explain_trajectory_visualization
  --source saves/visualizations/trajectory_manifold_convergence
  --dataset saves/validation/models/reconstructed-seed0/data.npz
  --output saves/visualizations/trajectory_manifold_convergence_plane.
  --skip-animation refreshes report/cache without control or GIF encoding.
- Added docs/visualization/RIGHT_PLOT.md, target-plane/{elements,progress,snapshots}.png
  and audit.json. Added tracked-doc ignore exceptions; docs/visualization.md links
  corrected explanation and identifies historical constant-reference view.
- Normalized displayed error7.5->.0932882; planeheight2.140066->.0891899.
  Final cart position4.402m remains off full target; no false convergence claim.
- Focused visualization suite20passed in71.40s; py_compile/diff check passed.
  Tests prove exact plane with nonzero target/freefuturestates/inputs/position,
  fixed SI scaling and cached rendering; existing18tests retained.
- GIF written301frames1170x540; renderer session returned143 after file write,
  so artifact was independently decoded through EVERY frame. Verified raw closed
  loop arrays identical, every plotted training row exactly equals its dataset
  row and belongs to train split, selected height equals state-error height to2e-6.
  Report regeneration/provenance verification completed with exit0.
- All earlier source functions/GIF directories retained unchanged. Active grid
  not touched or queried; no controller/training experiments rerun.
