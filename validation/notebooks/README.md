# Saved Gate A animations

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

Open `gate_a_examples.ipynb`: prepopulated figures and eight animations, including
all three original strict successes and five distinct failure examples. Play,
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
CUDA environments remain unchanged. All 20 Python cells have been run offline
and the notebook schema validated. JavaScript playback/scrubbing/reset were
checked with a mocked DOM/canvas; an actual browser rendering test was unavailable.

Without Jupyter, open `../figures/20261007T054549.695689Z/index.html` in a browser.
Its self-contained animations need no Python, GPU, network or video encoder.
The figure archive has PNG/PDF dashboards and derived-data NPZs for all cases,
an entire-cohort overview, and seven paired local-diagnostic comparisons.

To regenerate, run `trajectory_media.py`, then `finalize_trajectory_media.py`
with the private validation CPU environment/native-library settings. These are
saved-data export tools; they never launch controllers or acquire new trajectories.
