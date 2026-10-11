"""Replay six completed DeePC first plans on the true plant, on identical inputs."""
from campaign import *
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    start = time.perf_counter()
    out = OUT / "E4"
    config = runner.make_config("dev")
    rows = []
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, kind in zip(axes, ("state", "output")):
        for color, penalty in enumerate(("0.001", "0.01", "0.1")):
            key = f"seed0_DeePC_{kind}_{penalty}_A_0002_14s"
            case = json.loads((out / "cases" / (key + ".json")).read_text())
            with np.load(ROOT / case["raw_path"]) as data:
                k = int(data["solve_steps"][0])
                plan = data["plan_inputs"][0].copy()
                predicted = data["predictions"][0].copy()
                initial = data["X"][k].copy()
                first_actual_step = data["X"][k + 1].copy()
                first_applied_input = data["U"][k].copy()
            observed = initial if kind == "state" else initial[[0, 2]]
            assert np.allclose(predicted[0], observed, rtol=0, atol=1e-7)
            assert np.allclose(plan[0], first_applied_input, rtol=0, atol=1e-10)
            assert np.max(np.abs(plan)) <= 10 + 1e-9
            true = [initial]
            for value in plan:
                true.append(runner._plant_step(true[-1], value, config.dt))
            true = np.asarray(true)
            first_error = float(np.max(np.abs(true[1] - first_actual_step)))
            assert first_error < 1e-10
            angle = predicted[:, 2 if kind == "state" else 1]
            horizon = np.arange(len(true)) * .02
            falls = np.flatnonzero(np.abs(true[:, 2]) >= np.pi/2)
            rows.append(dict(case=key, arm=case["arm"], planning_step=k,
                initial_observation_max_error=float(np.max(np.abs(predicted[0] - observed))),
                first_applied_input_max_error=float(np.max(np.abs(plan[0] - first_applied_input))),
                independent_first_step_max_error=first_error,
                predicted_final_theta=float(angle[-1]), true_same_plan_final_theta=float(true[-1, 2]),
                theta_RMSE_same_inputs=float(np.sqrt(np.mean((angle - true[:, 2])**2))),
                true_same_plan_first_fall_seconds=float(horizon[falls[0]]) if len(falls) else None,
                qp_failures_in_closed_loop=case["qp_failures"]))
            ax.plot(horizon, angle, "--", color=f"C{color}", label=f"Predicted, λ={penalty}")
            ax.plot(horizon, true[:, 2], color=f"C{color}", label=f"True replay, λ={penalty}")
        ax.axhline(-np.pi/2, color="black", ls=":", label="Fall boundary")
        ax.set(xlabel="Physical time from planning instant [s]",
               title="Full state" if kind == "state" else "Output only (after two zero-input samples)")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Pendulum angle [rad]")
    fig.suptitle("DeePC first plans: seed 0, −15°; identical planned inputs in prediction and true replay")
    fig.tight_layout()
    fig.savefig(out / "deepc_matched_plan_replays.png", dpi=150)
    plt.close(fig)
    basis = []
    for path in sorted((out / "models").glob("*/*_basis.json")):
        d = json.loads(path.read_text())
        basis.append(dict(seed=path.parent.name, kind=path.stem, rows=d["rows"],
                          rank=d["rank"], columns=d["columns"]))
    dump(out / "deepc_matched_plan_replays.json", dict(
        selection="Illustrative saved-case selection: seed 0, IC-A index 2 (−15°), first plan, all six DeePC settings",
        interpretation="True open-loop replay of the saved planned input sequence; subsequent closed-loop inputs may differ",
        cases=rows, basis_dimensions=basis, diagnostic_wall_seconds=time.perf_counter()-start))
    print("Matched-input and independent first-step checks passed for six saved plans.", flush=True)


if __name__ == "__main__":
    main()
