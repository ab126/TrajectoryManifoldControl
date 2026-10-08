# Saved Gate A animations

**October 8:** the GPU diagnosis has now completed; its aggregate review is in
[cem_findings.md](../diagnostics/cem_findings.md). This notebook/gallery show
the original Gate A and local-feedback traces; CEM comparison plots have not
been added. Prepopulated narrative was generated before that diagnosis; current
status and next actions come from the updated handoff/findings, not older output.

For Git, use `gate_a_examples_source.ipynb`, the same notebook cells with outputs
and execution counts cleared and a portable Python 3 kernel selection. The
populated `gate_a_examples.ipynb`, generated kernel specification, and Jupyter
runtime state are ignored. Refresh the source companion after editing or
regenerating the populated notebook:

```bash
python3 TrajectoryManifoldControl/validation/notebooks/export_source.py
```

Running the source notebook requires the separately preserved run data/figure
archive and the display dependencies below; Git retains reports and manifests,
rather than the raw trajectories and rendered media.

Open `gate_a_examples.ipynb`: prepopulated figures and eleven animations, including
all three original strict successes, five original failure examples, and three
new angle-nonsettling examples. The **Cases where the angle does not settle near
zero** section includes all 52 angle failures, zoom plots for 17–20 s, explicit
±0.02 rad/±0.05 rad/s bands, and cases 063/139/028. Case 139 ends near zero but
fails to stay there; case 051 is identified as angular-speed-only. New animations
open at 17 s, with the full recovery available by scrubbing/resetting. Play,
pause, scrub through all 20 physical seconds, change speed, or set CASE_ID to
animate any of the 200 cases. The additional local-feedback diagnostic has a
separate comparison section; it is not mixed into the original success count.

Launch on the lab server from the workspace root:

```bash
bash TrajectoryManifoldControl/validation/notebooks/open_notebook.sh --no-browser
```

This launcher trusts the generated notebook for its embedded scripts, uses a
workspace-local Jupyter data/runtime directory, and exposes the bundled kernel
specification. The display kernel uses the existing `overseeg` Python 3.12
environment (NumPy 2.4.6, Matplotlib 3.10.8); its freeze is saved here. It only
reads saved trajectories and displays them. The isolated scientific CPU and
CUDA environments remain unchanged. All 27 Python cells have been run offline
and the notebook schema validated. JavaScript playback/scrubbing/reset were
checked with a mocked DOM/canvas; an actual browser rendering test was unavailable.

Without Jupyter, open `../figures/20261007T054549.695689Z/index.html` in a browser.
Its self-contained animations need no Python, GPU, network or video encoder.
The figure archive has PNG/PDF dashboards and derived-data NPZs for all cases,
an entire-cohort overview, and seven paired local-diagnostic comparisons.

To regenerate, run `trajectory_media.py`, then `finalize_trajectory_media.py`
with the private validation CPU environment/native-library settings. The
finalizer also preserves the angle-nonsettling section and refreshes the output-
free source notebook. To update just the angle section, run `angle_failure_media.py`
with the same CPU/native-library settings; the other dashboards are not rendered
again. These are saved-data export tools; they never launch controllers or acquire
new trajectories.
