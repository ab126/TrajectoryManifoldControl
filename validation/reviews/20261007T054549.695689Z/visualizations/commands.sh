# Commands executed for this task; all paths relative to workspace root.
# Export figures (private CPU Python with the native settings in validation/run.sh).
# trajectory_media.py
# trajectory_media.py --refresh-examples-only (example selection/layout only; no dashboard rerender)
# finalize_trajectory_media.py (read-only diagnostic plots and existing display kernel)
bash TrajectoryManifoldControl/validation/run_diagnostics.sh local --execute
# Prepared/preflighted only; NOT run in the GPU-inaccessible agent session:
# bash TrajectoryManifoldControl/validation/run_diagnostics.sh cem --execute
# Notebook app launcher, not launched by agent:
# bash TrajectoryManifoldControl/validation/notebooks/open_notebook.sh --no-browser
