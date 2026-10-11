"""Compare completed collection ablations on their shared evaluation ICs."""
import csv
import json
from pathlib import Path
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "validation/results_v1"


def main():
    start = time.perf_counter()
    out = OUT / "E2_distribution"
    assert json.loads((out / "manifest.json").read_text())["status"] == "DONE"
    sources = [
        ("nominal", OUT / "models", OUT / "E2/rates.csv"),
        ("half_box", out / "models/half_box", out / "rates.csv"),
        ("uniform_only", out / "models/uniform_only", out / "rates.csv"),
    ]
    quality, rates = [], []
    for name, models, source in sources:
        for seed in range(5):
            d = json.loads((models / f"episodes3500_seed{seed}/diagnostics.json").read_text())
            local = d["local"]
            row = dict(collection=name, seed=seed, training_transitions=local["training_transitions"],
                       selected_samples=local["selected_samples"], uses_all_data=local["uses_all_data"],
                       max_abs_theta_degrees=np.rad2deg(local["max_abs_theta"]),
                       max_normalized_norm=local["max_normalized_norm"])
            for prefix, values in (("N_step_rmse", d["prediction"]["N_step_rmse_by_state"]),
                                   ("local_fit_rmse", local["fit_rmse"])):
                row.update({prefix + "_" + state: value for state, value in zip(("p", "v", "theta", "omega"), values)})
            quality.append(row)
        with source.open() as handle:
            rates.extend(dict(r, collection=name, evaluation_set=r["ic_set"].removeprefix(name + "_"))
                         for r in csv.DictReader(handle)
                         if r["episodes"] == "3500" and float(r["seconds"]) == 14
                         and r["ic_set"] in (["A", "C50"] if name == "nominal" else [name + "_A", name + "_C50"]))
    with (out / "model_diagnostics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(quality[0]))
        writer.writeheader()
        writer.writerows(quality)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, ic in zip(axes, ("A", "C50")):
        for a, arm in enumerate(("H", "L", "M1")):
            for seed in range(5):
                items = [next(r for r in rates if r["collection"] == name and r["seed"] == str(seed)
                              and r["arm"] == arm and r["evaluation_set"] == ic)
                         for name, _, _ in sources]
                xs = np.arange(3) + .12 * (a - 1) + .02 * (seed - 2)
                ax.plot(xs, [float(r["rate"]) for r in items], "o-", color=f"C{a}", alpha=.55,
                        markersize=4, linewidth=.6, label=arm if seed == 0 else None)
        ax.set(xticks=range(3), xticklabels=["Nominal", "Half box", "Uniform inputs"],
               ylim=(-.04, 1.04), title="IC-" + ic, xlabel="Collection distribution")
        ax.legend()
    axes[0].set_ylabel("Angle stabilization and no fall at 14 s")
    fig.suptitle("3,500 episodes; five whole-run seeds shown separately")
    fig.tight_layout()
    fig.savefig(out / "collection_comparison.png", dpi=150)
    plt.close(fig)
    path = out / "manifest.json"
    manifest = json.loads(path.read_text())
    elapsed = time.perf_counter() - start
    manifest["comparison_postprocess_wall_seconds"] = manifest.get("comparison_postprocess_wall_seconds", 0.) + elapsed
    manifest["wall_seconds"] += elapsed
    path.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
