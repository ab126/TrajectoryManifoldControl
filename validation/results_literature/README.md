# Launch and monitor the approved literature comparisons

Run in your host terminal:

```bash
bash /home/ap60/github/data-driven-control/TrajectoryManifoldControl/validation/launch_literature.sh
```

The launcher creates or reuses the dedicated supervisor venv under
`/tmp/trajectory-literature-envs/supervisor`, validates package isolation, checks
usage and starts a detached supervisor on CPUs80–127. No environment activation
is needed. Scientific methods must use their own recorded project environments;
`overseeg` and other projects' environments must not be borrowed. The old project
venvs are preserved. Scratch environments must be reconstructible after cleanup.

Inspect progress with:

```bash
cd /home/ap60/github/data-driven-control/TrajectoryManifoldControl/validation
cat results_literature/STATUS
tail -n 30 results_literature/launcher.log results_literature/supervisor.log results_literature/codex_exec.log
```

The approved stages1–8 run sequentially under the unchanged usage guard, with
automatic usage pauses and resumption. This interactive container cannot launch
the model because it cannot write `~/.codex`; READY means prepared, not active.
Review packets target the existing overseer only. Until that reviewer is reachable,
record independent review as pending rather than claiming approval.
