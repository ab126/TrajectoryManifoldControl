"""Explicit angular endpoints for normalized, continuous-angle plant traces.

No controller or simulator dependency. Upstream metrics are supplied by the
adapter to retain its exact legacy endpoint and secondary settling convention.
"""
from __future__ import annotations

import numpy as np


def periodic_error(theta):
    return np.arctan2(np.sin(theta), np.cos(theta))


def settling_time(t, good, window=3.0):
    sustained = np.logical_and.accumulate(np.asarray(good, bool)[::-1])[::-1]
    eligible = np.flatnonzero(sustained & (t <= t[-1] - window + 1e-9))
    return float(t[eligible[0]]) if len(eligible) else None


def score_trace(t, X, U, *, upstream, fallback_events=0):
    """A delegates to upstream; B/C use a complete final three seconds.

    X columns: p/l, p_dot/sqrt(l*g), theta, theta_dot/sqrt(g/l).
    U is normalized force F/(m_p*g); physical units here use benchmark l=1,
    g=9.81 solely for reporting. Input is held over [t[k],t[k+1]).
    """
    t, X, U = np.asarray(t), np.asarray(X), np.asarray(U)
    if (t.ndim != 1 or len(t) < 2 or X.shape != (len(t), 4)
            or U.shape != (len(t)-1, 1) or not np.isfinite(t).all()
            or np.any(np.diff(t) <= 0)):
        raise ValueError('expected increasing time, X=(T,4), U=(T-1,1)')
    finite = bool(np.isfinite(X).all() and np.isfinite(U).all())
    theta = X[:, 2]
    error = periodic_error(theta)
    speed = X[:, 3] * np.sqrt(9.81)
    tail = t >= t[-1] - 3.0 - 1e-9
    complete = t[-1] - t[0] >= 3.0 - 1e-9
    unwrapped_good = (np.abs(theta) <= .02) & (np.abs(speed) <= .05)
    periodic_good = (np.abs(error) <= .02) & (np.abs(speed) <= .05)
    dt = np.diff(t)
    # Count crossings of all horizontal configurations, including after turns.
    # Counts are sampled lower bounds if multiple crossings reverse within a step.
    if finite:
        bands = np.floor((theta + np.pi/2) / np.pi)
        crossings = int(np.abs(np.diff(bands)).sum())
        rotations = int(np.floor(np.ptp(theta) / (2*np.pi)))
        net_rotations = int(np.trunc((theta[-1]-theta[0]) / (2*np.pi)))
    else:
        crossings = rotations = net_rotations = None
    def number(value):
        return float(value) if np.isfinite(value) else None
    return {
        'legacy_uninterrupted': bool(upstream['primary_success']),
        'terminal_recovery': bool(finite and complete and unwrapped_good[tail].all()),
        'periodic_terminal_recovery': bool(finite and complete and periodic_good[tail].all()),
        'complete_terminal_window': bool(complete),
        'angle_settle_seconds': settling_time(t, unwrapped_good) if finite else None,
        'periodic_angle_settle_seconds': settling_time(t, periodic_good) if finite else None,
        'full_settle_seconds': upstream.get('full_settle_seconds'),
        'final_theta_unwrapped': number(theta[-1]),
        'final_theta_periodic': number(error[-1]),
        'final_angular_speed_physical': number(speed[-1]),
        'terminal_max_angle_unwrapped': number(np.abs(theta[tail]).max()),
        'terminal_max_angle_periodic': number(np.abs(error[tail]).max()),
        'terminal_max_angular_speed_physical': number(np.abs(speed[tail]).max()),
        'final_cart_position': number(X[-1, 0]),
        'max_abs_cart_position': number(np.abs(X[:, 0]).max()),
        'final_cart_speed_physical': number(X[-1, 1]*np.sqrt(9.81)),
        'control_energy_normalized': number(np.sum(U[:, 0]**2 * dt)),
        'control_total_variation_normalized': number(np.abs(np.diff(U[:, 0])).sum()),
        'saturation_seconds': number(np.sum((np.abs(U[:, 0]) >= 10.-1e-6)*dt)),
        'horizontal_crossings_sampled': crossings,
        'completed_rotations_range': rotations,
        'net_completed_rotations': net_rotations,
        'numerical_failure': not finite,
        'fallback_events': int(fallback_events),
    }
