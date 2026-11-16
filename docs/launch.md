# Simulation launch plan

Use [core TMC versus LQR launch and tuning](validation/investigation-2026-11-16/LAUNCH.md).
It includes the active three-seed grid, resume command, solver tuning, candidate
training and final grid evaluation. Wait for the active grid before tuning.

New launches explicitly select core TMC with `--controller standalone` (sweep)
or `--arms TMC L` (candidate/reference-aware launchers). Core TMC never switches
to LQR or uses it as fallback. Historical hybrid reports retain their labels.
