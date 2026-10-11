"""Describe saved angular fluctuations separately from the unchanged endpoint."""
import csv
import json
from collections import defaultdict
from pathlib import Path
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "validation/results_v1/E3"
FIELDS = ("tail_angle_rms", "tail_angular_velocity_rms_physical",
          "tail_max_abs_angle", "tail_max_abs_angular_velocity_physical")


def main():
    start = time.perf_counter()
    manifest = json.loads((OUT / "manifest.json").read_text())
    assert manifest["status"] == "DONE"
    cases = [json.loads(p.read_text()) for p in (OUT / "cases").glob("*.json")]
    groups = defaultdict(list)
    for case in cases:
        groups[case["seed"], case["arm"], case["ic_set"]].append(case)
    assert len(cases) == 3696 and len(groups) == 132
    rows = []
    for (seed, arm, ic), items in sorted(groups.items()):
        assert len(items) == (6 if ic.endswith("_A") else 50)
        for population in ("all", "no_fall"):
            selected = items if population == "all" else [r for r in items if r["no_fall"]]
            for field in FIELDS:
                values = [r[field] for r in selected if r[field] is not None]
                rows.append(dict(seed=seed, arm=arm, ic_set=ic, population=population,
                                 field=field, n_total=len(items), n=len(selected), n_observed=len(values),
                                 **{f"q{q:02d}": float(np.quantile(values, q / 100)) if values else None
                                    for q in (5, 50, 95)}))
    with (OUT / "angular_tail_distributions.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for col, mode in enumerate(("measurement", "process")):
        for row, (field, label) in enumerate(zip(FIELDS[:2], ("Angle RMS [rad]", "Angular velocity RMS [physical rad/s]"))):
            ax = axes[row, col]
            for a, arm in enumerate(("H", "L")):
                for seed in range(3):
                    values = [next(r["q50"] for r in rows if r["seed"] == seed and r["arm"] == arm
                                   and r["ic_set"] == f"{mode}_{level}_A"
                                   and r["population"] == "no_fall" and r["field"] == field)
                              for level in (.01, .03, .1)]
                    ax.plot([.01, .03, .1], [np.nan if v is None else v for v in values], "o-",
                            color=f"C{a}", alpha=.6, label=arm if seed == 0 else None)
            ax.set(xscale="log", yscale="log", ylabel=label, title=mode if row == 0 else "")
            ax.set_xticks([.01, .03, .1], ["0.01", "0.03", "0.1"])
            if row == 1: ax.set_xlabel("Noise multiplier × nominal training state std")
            ax.legend()
    fig.suptitle("No-fall IC-A runs: final-three-second RMS, medians per seed")
    fig.text(.5, .015, "Missing points: no runs without a fall. RMS is descriptive; primary success uses pointwise limits.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .04, 1, .95))
    fig.savefig(OUT / "angular_fluctuations.png", dpi=150)
    plt.close(fig)
    elapsed = time.perf_counter() - start
    manifest["angular_postprocess_wall_seconds"] = manifest.get("angular_postprocess_wall_seconds", 0.) + elapsed
    manifest["wall_seconds"] += elapsed
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
