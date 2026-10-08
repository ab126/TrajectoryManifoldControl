Uniform-rod plant derivation (local orchestrator review; no independent agent available).

Take physical generalized coordinates `(p, theta)` with positive theta tipping
the rod's center of mass toward positive p from upright. Its COM is at
`(p + l sin(theta)/2, l cos(theta)/2)`. For rod mass m and full length l,
`I_COM = m l²/12`, `I_pivot = m l²/3`. The kinetic and potential energies are

```
T = (M_cart+m) p_dot²/2 + m l cos(theta) p_dot theta_dot/2
    + m l² theta_dot²/6
V = m g l cos(theta)/2
```

Euler–Lagrange with applied cart force F gives

```
[(M_cart+m)       m l cos(theta)/2] [p_ddot    ]
[m l cos(theta)/2 m l²/3         ] [theta_ddot]
 = [F + m l sin(theta) theta_dot²/2, m g l sin(theta)/2].
```

Define `t0=sqrt(l/g)`, normalized time `tau=t/t0`, normalized state
`[p/l, p_dot/sqrt(l*g), theta, theta_dot/sqrt(g/l)]`, input `u=F/(m*g)`,
and mass ratio `rho=M_cart/m`. Then

```
D = [[rho+1, cos(theta)/2], [cos(theta)/2, 1/3]]
D [p_norm'', theta''] = [u + sin(theta) theta_norm_dot²/2, sin(theta)/2].
```

For rho=2, `det(D)=1-cos²(theta)/4 >= 3/4`: no angle singularity.
Any cart position with zero velocities and theta=0 or pi is a zero-force
equilibrium. Upright is unstable: a small positive theta causes positive angular
acceleration and negative cart acceleration. Positive force at upright causes
positive cart acceleration and negative angular acceleration. Normalized energy
is `3 v²/2 + cos(theta) v omega/2 + omega²/6 + cos(theta)/2`.

Evidence: `src/inverted_pendulum.py:24` defines scales, `:28` and `:40` wrap
callbacks, `:52` converts physical derivatives, `:66` implements D/RHS,
`:210` implements held-input RK4, and `:273` defines upright linearization.
The active runner `scripts/run_inverted_pendulum_mc.py:34` uses normalized
dt=0.0626418390534633, `:64` uses rho=2, `:199` defines the controller, and
`:249` labels output time in physical 0.02-second steps.

No governing-equation discrepancy was found. The plan's dt=0.02 is physical,
not the argument passed to normalized RK4. Force 10 is 98.1 N for m=1 kg,
g=9.81 m/s²; length is 1 m and cart mass is 2 kg in this physical interpretation.
Numerical checks are in `artifacts/plant_checks.json`. The rod assumptions follow
from the implemented inertia/coupling, not from the repository's name. No other
pendulum repository or baseline has been audited or reused. Smooth equations
and tested numerical accuracy do not establish stability or attainability.
