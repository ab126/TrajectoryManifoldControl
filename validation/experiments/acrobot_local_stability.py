"""Post hoc upright Jacobians of the saved acrobot gains; no controller changes."""
from novel_plants import *
from scipy.linalg import solve_discrete_are


def analytic_linearization(dt):
    mass = np.array([[8/3, 5/6], [5/6, 1/3]])
    gravity_derivative = np.array([[2., .5], [.5, .5]])
    A = np.zeros((4, 4)); A[:2, 2:] = np.eye(2)
    A[2:, :2] = np.linalg.solve(mass, gravity_derivative)
    B = np.r_[np.zeros(2), np.linalg.solve(mass, [0., 1.])].reshape(4, 1)
    substep = dt / 8
    As = np.eye(4); Bs = np.zeros((4, 1)); power = np.eye(4)
    for order in range(1, 5):
        Bs += substep**order / math.factorial(order) * power @ B
        power = power @ A
        As += substep**order / math.factorial(order) * power
    Ad = np.eye(4); Bd = np.zeros((4, 1))
    for _ in range(8):
        Ad = As @ Ad
        Bd = As @ Bd + Bs
    return Ad, Bd


def radius(matrix):
    return float(np.max(np.abs(np.linalg.eigvals(matrix))))


def main():
    start = time.perf_counter()
    plant = make_plant("acrobot")
    dt = runner.make_config("dev").dt
    Ad, Bd = analytic_linearization(dt)
    errors = []
    for epsilon in (1e-5, 1e-6):
        An = np.column_stack([(plant.step(epsilon * axis, [0.], dt) - plant.step(-epsilon * axis, [0.], dt)) / (2 * epsilon)
                              for axis in np.eye(4)])
        Bn = ((plant.step(np.zeros(4), [epsilon], dt) - plant.step(np.zeros(4), [-epsilon], dt)) / (2 * epsilon)).reshape(4, 1)
        errors.append(max(float(np.max(np.abs(An - Ad))), float(np.max(np.abs(Bn - Bd)))))
    assert max(errors) < 1e-8
    P = solve_discrete_are(Ad, Bd, plant.Q, plant.R)
    K = np.linalg.solve(plant.R + Bd.T @ P @ Bd, Bd.T @ P @ Ad)
    assert radius(Ad) > 1 and radius(Ad - Bd @ K) < 1
    out = OUT / "E3_acrobot"
    rows = []
    for seed in range(3):
        d = json.loads((out / "models" / f"seed{seed}/diagnostics.json").read_text())["local"]
        gain = np.asarray(d["gain"])
        rows.append(dict(seed=seed,
                         fitted_closed_loop_radius=radius(np.asarray(d["A"]) - np.asarray(d["B"]) @ gain),
                         true_closed_loop_radius=radius(Ad - Bd @ gain)))
    csv_write(out / "local_stability.csv", rows)
    dump(out / "local_stability_checks.json", dict(
        epsilons=[1e-5, 1e-6], analytic_vs_central_difference_max_abs_errors=errors,
        uncontrolled_radius=radius(Ad), true_DARE_radius=radius(Ad - Bd @ K),
        interpretation="True sample-and-hold upright Jacobian before input saturation; local diagnostic, no finite-IC capture guarantee"))
    manifest = json.loads((out / "manifest.json").read_text())
    elapsed = time.perf_counter() - start
    manifest["stability_postprocess_wall_seconds"] = manifest.get("stability_postprocess_wall_seconds", 0.) + elapsed
    manifest["wall_seconds"] += elapsed
    dump(out / "manifest.json", manifest)
    print(rows, flush=True)
    print("Independent analytic/finite-difference checks:", errors, flush=True)


if __name__ == "__main__":
    main()
