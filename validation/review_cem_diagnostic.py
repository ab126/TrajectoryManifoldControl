"""Review saved CUDA diagnostics without executing controllers or transitions.

Requires NumPy only. Writes aggregate findings outside the original run folder.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def safe(attempt):
    return attempt["predicted_rail_safe"] and attempt["predicted_angle_safe"]


def review(directory):
    job = json.loads((directory / "job.json").read_text())
    config = json.loads((directory / "config.json").read_text())
    runtime = json.loads((directory / "cuda_runtime_check.json").read_text())
    results = json.loads((directory / "results.json").read_text())
    assert job["mode"] == "cem" and job["status"] == "finished" and job["returncode"] == 0
    assert runtime["passed"] and not results["incomplete"]
    assert results["new_closed_loop_diagnostic_plant_samples"] == 0
    assert config["CEM"] == {"population": 256, "elites": 32, "iterations": 3,
                            "horizon": 60, "starts": [1, 5]}
    assert len(config["states"]) == 20 and len(results["cem"]) == 40
    states = {x["label"]: x for x in config["states"]}
    assert len(states) == 20
    for name, expected in job["source_SHA256"].items():
        assert digest(directory / "source_snapshots" / name) == expected, name

    inputs = [directory / name for name in ("job.json", "config.json", "results.json", "cuda_runtime_check.json")]
    origin_name = Path(job["command"][job["command"].index("--run-dir") + 1]).name
    origin = HERE / "overnight_runs" / origin_name / "calibration"
    for state in states.values():
        if isinstance(state["case"], int):
            path = origin / f'case_{state["case"]:03d}.npz'
            with np.load(path, allow_pickle=False) as trace:
                np.testing.assert_array_equal(trace["X"][state["sample"]], state["x"])
            inputs.append(path)
        else:
            sign = int(state["label"].split("_")[-1])
            np.testing.assert_array_equal(state["x"], [sign, 0, 0, 0])
    pairs = results["cem"]
    comparisons = {}
    max_cost_error = 0.0
    durations = []
    Q = np.diag([0.02, 30.0, 100.0, 10.0])
    for pair in pairs:
        label = pair["state"]["label"]
        assert pair["state"] == states[label]
        assert pair["warm_kind"] in ("zeros", "oracle_local_rollout")
        assert pair["warm_kind"] not in comparisons.setdefault(label, {})
        comparisons[label][pair["warm_kind"]] = pair
        assert [a["attempt"] for a in pair["attempts"]] == list(range(5))
        for attempt in pair["attempts"]:
            path = directory / f'cem_{label}_{pair["warm_kind"]}_{attempt["attempt"]}.npz'
            inputs.append(path)
            with np.load(path, allow_pickle=False) as saved:
                X, U = saved["X"], saved["U"]
                assert X.shape == (61, 4) and U.shape == (60, 1)
                assert np.isfinite(X).all() and np.isfinite(U).all() and attempt["finite"]
                np.testing.assert_array_equal(X[0], states[label]["x"])
                assert np.max(np.abs(U)) <= 10.0 + 1e-12
                stage = np.einsum("ki,ij,kj->", X[:-1], Q, X[:-1])
                terminal = float(X[-1] @ (12 * Q) @ X[-1])
                total = stage + terminal + 0.03 * np.sum(U * U) + 0.002 * np.sum(np.diff(U, axis=0) ** 2)
                expected = attempt["objective_parts"]
                np.testing.assert_allclose(total, attempt["loss"], rtol=1e-9, atol=1e-8)
                np.testing.assert_allclose(total, expected["total"], rtol=1e-9, atol=1e-8)
                components = sum(value for key, value in expected.items() if key != "total")
                np.testing.assert_allclose(total, components, rtol=1e-9, atol=1e-8)
                for index, name in enumerate(("position", "speed", "angle", "angular_speed")):
                    np.testing.assert_allclose(np.sum(X[:-1, index] ** 2) * Q[index, index],
                                               expected[name], rtol=1e-9, atol=1e-8)
                np.testing.assert_allclose(terminal, expected["terminal"], rtol=1e-9, atol=1e-8)
                np.testing.assert_allclose(.03 * np.sum(U ** 2), expected["control"], rtol=1e-9, atol=1e-8)
                np.testing.assert_allclose(.002 * np.sum(np.diff(U, axis=0) ** 2),
                                           expected["rate"], rtol=1e-9, atol=1e-8)
                assert bool(np.all(np.abs(X[:, 0]) <= 2.4)) == attempt["predicted_rail_safe"]
                assert bool(np.all(np.abs(X[:, 2]) <= np.pi / 2)) == attempt["predicted_angle_safe"]
                np.testing.assert_allclose(U[0, 0] * 9.81, attempt["first_force_N"], atol=1e-10)
                max_cost_error = max(max_cost_error, abs(float(total) - attempt["loss"]))
            assert attempt["loss"] <= attempt["warm_candidate_cost"] + 1e-7 * max(1.0, abs(attempt["warm_candidate_cost"]))
            durations.append(attempt["solve_seconds"])
        best = min(range(5), key=lambda index: pair["attempts"][index]["loss"])
        assert best == pair["best_attempt"]
        assert pair["one_start_loss"] == pair["attempts"][0]["loss"]
        assert pair["five_start_loss"] == pair["attempts"][best]["loss"]
    assert len(comparisons) == 20 and len(durations) == 200

    aggregates = {}
    for warm in ("zeros", "oracle_local_rollout"):
        group = [pair for pair in pairs if pair["warm_kind"] == warm]
        reductions = np.array([1 - pair["five_start_loss"] / pair["one_start_loss"] for pair in group])
        one = [pair["attempts"][0] for pair in group]
        best = [pair["attempts"][pair["best_attempt"]] for pair in group]
        aggregates[warm] = {
            "pairs": len(group), "five_start_cost_improvements": int(np.sum(reductions > 1e-12)),
            "relative_cost_reduction_median": float(np.median(reductions)),
            "relative_cost_reduction_max": float(np.max(reductions)),
            "one_start_safe": sum(map(safe, one)), "five_start_safe": sum(map(safe, best)),
            "one_start_rail_failures": sum(not a["predicted_rail_safe"] for a in one),
            "five_start_rail_failures": sum(not a["predicted_rail_safe"] for a in best),
            "one_start_angle_failures": sum(not a["predicted_angle_safe"] for a in one),
            "five_start_angle_failures": sum(not a["predicted_angle_safe"] for a in best),
            "safe_to_unsafe": sum(safe(a) and not safe(b) for a, b in zip(one, best)),
            "unsafe_to_safe": sum(not safe(a) and safe(b) for a, b in zip(one, best)),
        }
    state_rows = []
    for label, methods in comparisons.items():
        assert len(methods) == 2
        z, local = methods["zeros"], methods["oracle_local_rollout"]
        state_rows.append({"label": label,
                           "zero_one_cost": z["one_start_loss"], "zero_five_cost": z["five_start_loss"],
                           "local_one_cost": local["one_start_loss"], "local_five_cost": local["five_start_loss"],
                           "zero_five_safe": safe(z["attempts"][z["best_attempt"]]),
                           "local_five_safe": safe(local["attempts"][local["best_attempt"]])})
    seconds = np.asarray(durations)
    output = {
        "run_directory": str(directory.relative_to(HERE)),
        "scope": "saved-data review only; no new solves, training, or plant transitions",
        "checks": {"complete_finite_solves": 200, "saved_predictions_verified": 200,
                   "fixed_states_match_original_traces_or_pure_probes": True,
                   "objective_and_safety_recomputed": True, "warm_candidate_retention_passed": True,
                   "execution_source_hashes_verified": True, "runtime_check_passed": True,
                   "max_objective_absolute_discrepancy": max_cost_error},
        "aggregates": aggregates, "states": state_rows,
        "local_warm_lower_best_cost_states": sum(x["local_five_cost"] < x["zero_five_cost"] for x in state_rows),
        "zero_warm_lower_best_cost_states": sum(x["zero_five_cost"] < x["local_five_cost"] for x in state_rows),
        "solve_seconds": {"mean": float(seconds.mean()), "median": float(np.median(seconds)),
                          "p95": float(np.quantile(seconds, .95)), "p99": float(np.quantile(seconds, .99)),
                          "max": float(seconds.max()), "sum": float(seconds.sum())},
        "worker_seconds": job["worker_seconds"], "new_closed_loop_plant_transitions": 0,
        "input_sha256": {str(path.relative_to(HERE)): digest(path) for path in inputs},
        "limitations": ["selected development states, not IID calibration",
                        "open-loop 1.2-second predictions, not 20-second closed-loop success",
                        "synthetic explicit warm starts; historical full plans unavailable",
                        "global optimum and physical viability not certified",
                        "solve timings exclude full controller-step overhead"],
    }
    (HERE / "diagnostics/cem_review.json").write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: output[k] for k in ("checks", "aggregates", "local_warm_lower_best_cost_states",
                                            "zero_warm_lower_best_cost_states", "solve_seconds", "worker_seconds")}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", nargs="?", default=str(HERE / "diagnostics/20261008T190703.853434Z"))
    review(Path(parser.parse_args().directory).resolve())
