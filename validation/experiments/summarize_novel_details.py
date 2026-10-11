"""Describe completed novel-system runs using their saved metrics and models."""
import csv
import json
from collections import defaultdict
from pathlib import Path
import sys
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(experiment):
    assert experiment in ("E3_linear", "E3_acrobot")
    start = time.perf_counter()
    out = ROOT / "validation/results_v1" / experiment
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["status"] == "DONE"
    cases = [json.loads(p.read_text()) for p in (out / "cases").glob("*.json")]
    groups = defaultdict(list)
    fields = ["max_state_norm", "settle_seconds", "max_abs_u", "input_limit_seconds",
              "local_fraction", "accumulated_stage_cost"]
    if experiment == "E3_linear":
        fields.append("cost_ratio")
        ratios = []
        for r in cases:
            reference = r["unconstrained_optimal_infinite_cost"]
            assert reference > 0
            cost = r["cost_plus_optimal_tail"]
            assert cost is None or cost >= reference * (1 - 1e-9)
            r["cost_ratio"] = cost / reference if cost is not None else None
            ratios.append({k: r[k] for k in ("case", "seed", "arm", "ic_set", "seconds",
                                            "unconstrained_optimal_infinite_cost",
                                            "cost_plus_optimal_tail", "cost_ratio")})
        write_csv(out / "linear_cost_ratios.csv", sorted(ratios, key=lambda r: r["case"]))
    for r in cases:
        groups[r["seed"], r["arm"], r["ic_set"], r["seconds"]].append(r)
    distributions = []
    for key, items in sorted(groups.items()):
        for field in fields:
            values = [r[field] for r in items if r[field] is not None]
            distributions.append(dict(zip(("seed", "arm", "ic_set", "seconds"), key),
                field=field, n=len(items), n_observed=len(values),
                **{f"q{q:02d}": float(np.quantile(values, q / 100)) if values else None
                   for q in (0, 25, 50, 75, 95, 100)}))
    write_csv(out / "distributions.csv", distributions)
    quality = []
    for seed in range(3):
        d = json.loads((out / "models" / f"seed{seed}/diagnostics.json").read_text())
        row = dict(seed=seed, plant=d["plant"], n=d["n"], m=d["m"],
                   training_transitions=d["local"]["training_transitions"],
                   selected_samples=d["local"]["selected_samples"])
        for prefix, values in (("one_step_rmse", d["prediction"]["one_step_rmse_by_state"]),
                               ("N_step_rmse", d["prediction"]["N_step_rmse_by_state"]),
                               ("local_fit_rmse", d["local"]["fit_rmse"])):
            row.update({f"{prefix}_x{i}": value for i, value in enumerate(values)})
        quality.append(row)
    write_csv(out / "model_diagnostics.csv", quality)
    plotted = [("max_state_norm", "Maximum state norm in supplied coordinates"),
               ("input_limit_seconds", "Time at the input limit [physical s]")]
    if experiment == "E3_linear":
        plotted.append(("cost_ratio", "Prefix cost + optimal tail / optimal initial cost"))
    fig, axes = plt.subplots(1, len(plotted), figsize=(5 * len(plotted), 4))
    arms = manifest["config"]["arms"]
    for ax, (field, label) in zip(axes, plotted):
        for a, arm in enumerate(arms):
            for j, ic in enumerate(("A", "C50")):
                for seed in range(3):
                    distribution = next(r for r in distributions if r["seed"] == seed
                                 and r["arm"] == arm and r["ic_set"] == ic
                                 and r["seconds"] == 14 and r["field"] == field)
                    value = distribution["q50"]
                    if value is not None:
                        ax.vlines(a + .15 * (j - .5) + .03 * (seed - 1), distribution["q25"], distribution["q95"], color=f"C{j}", alpha=.3)
                        ax.scatter(a + .15 * (j - .5) + .03 * (seed - 1), value,
                                   marker="o" if j == 0 else "x", color=f"C{j}",
                                   label=ic if a == 0 and seed == 0 else None)
        ax.set(xticks=range(len(arms)), xticklabels=arms, ylabel=label)
        if field != "input_limit_seconds":
            ax.set_yscale("log")
        ax.legend()
    fig.suptitle(experiment + ": 14 s, per-seed medians and 25th–95th percentiles within each IC set")
    fig.tight_layout()
    fig.savefig(out / "descriptive_comparison.png", dpi=150)
    plt.close(fig)
    elapsed = time.perf_counter() - start
    manifest["details_postprocess_wall_seconds"] = manifest.get("details_postprocess_wall_seconds", 0.) + elapsed
    manifest["wall_seconds"] += elapsed
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main(sys.argv[1])
