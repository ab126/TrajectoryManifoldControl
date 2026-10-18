# Stability guarantee research handoff

**We can formulate a conditional local stability guarantee for a specific recorded feedback gain on the nominal sampled nonlinear simulator. We have not completed a verified numerical certification, computed a certified capture region, or proved global stability of H.** This handoff proposes bounded proof work; it authorizes no simulations or controller changes.

## A precise claim to pursue

Let x=(p,v,θ,ω) denote the four normalized plant coordinates: cart position, cart velocity, upright angle, and angular velocity. Let F(x,u) be exactly the submitted fixed-step fourth-order Runge–Kutta (RK4) update with held normalized force u, mass ratio M=2, and recorded step h=0.0626418390534633. This is a discrete nonlinear plant, not the exact continuous-time physical flow. Define saturation sat₁₀(a)=min(10,max(−10,a)) and the local closed-loop map g(x)=F(x,sat₁₀(−Kx)).

Use the already recorded E4 seed-0 gain

K=(−1.20185169823597, −19.365432453692755, −57.58049888047022, −41.64140651197739).

Its provenance is [diagnostics](E4/models/seed0/diagnostics.json); the [recorded-gain table](E4/local_stability.csv) gives the true-plant derivative's spectral radius 0.9950987064886747. Spectral radius means the largest absolute eigenvalue. “True” here means the submitted nominal RK4 plant, not hardware. The [independent checks](E4/local_stability_checks.json) compare analytic derivatives with central differences; errors are at most 6.4×10⁻¹². These floating-point diagnostics support the hypothesis; they are not interval enclosures proving a strict eigenvalue margin.

The conditional statement is: **if the recorded gain, plant, step, and derivative are fixed as above and the closed-loop derivative has all eigenvalues strictly inside the unit circle, then zero is locally exponentially stable for g.** Thus sufficiently small full-state initial errors converge geometrically while remaining small. The size of “sufficiently small” is unspecified until a region is certified. See Bof, Carli and Schenato's [discrete-time Lyapunov theory, Section 3](https://arxiv.org/html/1809.05289), for the linearization and Lyapunov results.

The required regularity is checkable from [plant code](../../src/inverted_pendulum.py). Its mass matrix has determinant 1−cos²(θ)/4≥3/4 when M=2. Its inverse and trigonometric terms are smooth; RK4 is a finite smooth composition. F(0,0)=0. Because K is finite, |Kx|<10 on an open neighborhood of zero, where saturation is inactive and differentiable. Consequently g(0)=0 and Dg(0)=Aₕ−BₕK, with Aₕ and Bₕ the derivatives of F at zero, exactly the objects in [local_stability.py](../experiments/local_stability.py). This argument does not require the learned network to reproduce the equilibrium.

A recorded gain with radius above one cannot receive this local guarantee: under the same smoothness hypotheses its upright equilibrium is unstable. E4 seed 1 gives 1.001662860276656 despite stabilizing its fitted model. Its saved 14/60-second witness in [RESULTS](RESULTS.md) illustrates why finite angle success does not prove eventual full-state stability. At radius exactly one the linearization test is inconclusive.

## From existence to a computable catch region

For a verified stable matrix A꜀=Aₕ−BₕK, choose W positive definite and solve A꜀ᵀPA꜀−P=−W. Define V(x)=xᵀPx. Here P is a new proof certificate, distinct from the controller's state-cost matrix Q and its fitted-model Riccati solution. Write g(x)=A꜀x+r(x). Establish a uniform Taylor remainder bound ||r(x)||≤c||x||² on a stated ball ||x||≤b, using verified derivative bounds, interval arithmetic, or another sound enclosure of every RK4 stage. Sampled residuals or a dense grid cannot establish that bound.

Expansion gives

V(g(x))−V(x) ≤ −λmin(W)||x||² + 2||A꜀ᵀP||c||x||³ + ||P||c²||x||⁴.

Choose a positive radius where the right-hand side is strictly negative for x≠0. A sublevel set Ω={x:V(x)≤α} contained in that ball is then invariant and convergent, provided the bound holds everywhere in it. Also verify |Kx|<10 throughout Ω and all declared plant/state restrictions. The scalar α is a Lyapunov level, not an experimentally observed success rate. Sound numerical enclosures must cover the Lyapunov solve and inequality margins. No α, c, or certified Ω has been produced here.

## What this would establish for H

The [controller](../../scripts/run_inverted_pendulum_mc.py) checks xᵀQx<0.75 at every call, with Q=diag(0.02,30,100,10), before using queued inputs. Inside this strict switch it clears the pending queue and applies the saturated local gain. Require Ω to lie strictly inside that switch, with a verified margin. Invariance then keeps H in its feedback branch and gives the same local plant-state convergence, regardless of an initially pending plan.

Outside this region H carries a pending five-input queue, the previous plan used for warm starts, and optimizer random-generator state. Any guarantee involving planning, delay, or return to the catch must include the relevant controller memory in an augmented state and quantify over admitted randomness. A plant-state derivative alone proves no such result. Local convergence also does not prove convergence of all stored controller memory. A larger claim needs a certified path into Ω while respecting constraints, and a switch argument covering exits and boundaries.

The terminal cost is 12Q, not a certified terminal Lyapunov function with an invariant terminal constraint. Finite CEM search supplies no demonstrated optimality/decrease condition. The learned transition is not forced to satisfy F̂(0,0)=0; prediction error and equilibrium bias need separate bounds if used in a planning proof. The [graph proof audit](reviews/paper_review_v1/proof_audit.md) establishes trajectory geometry, not these control properties.

## Scope and staged acceptance

1. **Freeze and verify:** archive the stated gain and configuration; enclose the derivative, equilibrium, and eigenvalue/Lyapunov margins. Accept only a strict verified stability margin. Stop on mismatch, nonfinite bounds, or an undecidable margin; do not silently redesign the gain.
2. **Certify a local set:** bound the nonlinear remainder and produce a reproducible positive α with saturation, switch, and domain margins. Report a tiny certified set honestly; stop if bounds cannot certify a nonempty useful set within an agreed budget.
3. **Extend only by separate approval:** pursue larger attraction sets, queue-aware reachability, or robustness with explicit hypotheses. No automatic simulation sweep, new source modification, or global claim follows from this handoff.

Exponential/asymptotic convergence is not exact finite-time arrival at zero. A certified decay rate can bound time to a nonzero tolerance; the authored finite-window angle endpoint is a different property. Bounded disturbances and measurement errors may permit a local robust invariant set and eventual error bound after proving a disturbance-dependent decrease inequality and switch margin. Persistent noise generally prevents exact convergence. Unbounded Gaussian noise cannot support an unconditional bounded-set guarantee over all time; a probabilistic finite-horizon statement needs an explicit probability model and failure budget.

Finally, certification using analytic plant derivatives is a privileged nominal-model guarantee even though K was learned from data. A genuinely data-only guarantee requires justified uniform model/derivative error bounds, coverage and identifiability assumptions, or stated statistical confidence conditions. The E6 SMC validation residual's 95th percentile explicitly is not a certified uncertainty bound in [smc_baseline.py](../experiments/smc_baseline.py); neither that calibration nor switching alone supplies a robust sliding-mode theorem. Hardware, exact physical sampled flow, delay, and model mismatch remain separate obligations.
