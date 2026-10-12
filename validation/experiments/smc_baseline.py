"""Same-data discrete boundary-layer sliding feedback; no plant equations."""
import numpy as np
from scipy.linalg import solve_discrete_are


def surface(A, B, Q, R):
    A, B, Q, R = (np.asarray(v, dtype=float) for v in (A, B, Q, R))
    if B.shape != (len(A), 1):
        raise ValueError('This baseline requires one input')
    P = solve_discrete_are(A, B, Q, R)
    C = np.linalg.solve(B.T @ P @ B, B.T @ P)
    equivalent = (np.eye(len(A)) - B @ C) @ A
    if not np.allclose(C @ B, 1., atol=1e-10):
        raise ValueError('Ill-conditioned sliding input gain')
    if np.min(np.linalg.eigvalsh(P)) <= 0 or max(abs(np.linalg.eigvals(equivalent))) >= 1:
        raise ValueError('Fitted equivalent dynamics are not stable')
    return C, P, equivalent


def calibrate(A, B, C, train, validation, samples=6000):
    """Use validation residuals inside the training fit's selected state radius."""
    X, _ = train
    VX, VU = validation
    n = X.shape[-1]
    x = X[:, :-1].reshape(-1, n).astype(float)
    scale = x.std(axis=0).clip(1e-6)
    radius = np.sort(np.linalg.norm(x / scale, axis=1))[min(samples, len(x))-1]
    vx = VX[:, :-1].reshape(-1, n).astype(float)
    vn = VX[:, 1:].reshape(-1, n).astype(float)
    vu = VU.reshape(-1, 1).astype(float)
    chosen = np.linalg.norm(vx / scale, axis=1) <= radius
    residual = ((vn - vx @ A.T - vu @ B.T) @ C.T)[chosen, 0]
    if len(residual) < 10 or not np.isfinite(residual).all():
        raise ValueError('Insufficient finite validation residuals in the training fit radius')
    delta95 = float(np.quantile(np.abs(residual), .95))
    return dict(phi=max(1e-3, 4*delta95), validation_samples=len(residual),
                residual_abs_q95=delta95, training_scale=scale.tolist(),
                training_fit_radius=float(radius), training_samples=min(samples, len(x)),
                uncertainty_bound_certified=False)


class SlidingFeedback:
    def __init__(self, A, C, phi, correction=True, rho=.9, bound=10.):
        self.A, self.C = np.asarray(A), np.asarray(C)
        self.phi, self.rho, self.bound = float(phi), float(rho), float(bound)
        if self.phi <= 0 or self.bound <= 0 or not 0 < rho < 1:
            raise ValueError('Invalid reaching-law parameters')
        self.beta = .45*self.phi if correction else 0.

    def command(self, x):
        x = np.asarray(x, dtype=float)
        s = float((self.C @ x).item())
        raw = float((-self.C @ self.A @ x).item()) + self.rho*s
        raw -= self.beta*float(np.clip(s/self.phi, -1, 1))
        # A fallen nonfinite simulated state is already a failed endpoint. Keep
        # the actuation finite rather than allowing NaN commands to spread.
        applied = float(np.clip(raw, -self.bound, self.bound)) if np.isfinite(raw) else 0.
        return np.array([applied]), s, raw


def effort(U, commands=None):
    U = np.asarray(U, dtype=float).reshape(-1)
    def number(v):
        return float(v) if np.isfinite(v) else None
    result = dict(input_total_variation=number(np.abs(np.diff(U)).sum()),
                  input_rms=number(np.sqrt(np.mean(U**2))))
    if commands is not None:
        values = np.asarray(commands).reshape(-1)
        result['command_total_variation'] = number(np.abs(np.diff(values)).sum())
    return result
