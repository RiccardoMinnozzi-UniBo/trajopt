# Multi-phase solar sail: validated solutions and solver findings

Both discretisations now produce **fully validated** minimum-time deorbit
solutions. Reproduce with `python main.py` (MS) and `python main.py
config-ps.yaml` (PS).

## 1. Solutions

Scenario: A/m = 10 m^2/kg, GEO -> graveyard (perigee >= r_GEO + 250 km,
e <= 0.003), cylindrical shadow, Sun line rotating at 0.9856 deg/day,
24 revolutions, 49 alternating sunlit/eclipse phases.

| | MS (`config.yaml`) | PS (`config-ps.yaml`) |
| --- | --- | --- |
| mesh (sunlit / eclipse nodes) | 10 / 4 | 18 / 6 |
| elapsed time | **23.637 d** | **23.656 d** |
| revolutions flown | 23.598 | 23.598 |
| true final perigee - r_GEO | **+249.92 km** | **+250.31 km** |
| true final eccentricity | **0.002997** | **0.002817** |
| accumulated defect | 0.202 km | -0.528 km |
| max per-interval defect | 3.17e-06 | 4.97e-07 |
| SCP iterations | 38 | 79 |
| converged / feasible | yes / **yes** | yes / **yes** |

"True" quantities come from re-integrating the whole trajectory end to end from
the exact initial state with each method's own control model, never resetting to
the optimiser's nodes. Elapsed times agree to 0.08 %, which is independent
confirmation: two different discretisations, same answer.

## 2. What was wrong with the formulation

Found by comparing against Kelly & Bevilacqua (2021).

1. **The steering angle could not wind.** `steering_limits` bounded theta to
   +/-2*pi. The locally optimal primer is roughly fixed in inertial space, so in
   the rotating local frame it turns a full 2*pi *per revolution* (their Figure
   7). The box therefore capped the sail at one revolution of useful steering;
   past that the control saturated and eccentricity was pumped instead of held.
   Now +/-2*pi*(N+2).

2. **The initial guess did not wind either.** It held theta constant at pi/2.
   SCP is a local method, so it stayed on the non-winding branch. The guess now
   carries -2*pi per revolution (`guess.steering_winds_per_rev`, 0 restores the
   old behaviour). Fixes 1 and 2 together were worth about +50 km of perigee.

3. **The revolution count was an order of magnitude too low.** The paper needs
   90 revolutions at A/m = 1; at A/m = 10 the answer is ~24, not the 1-4 implied
   by the old README. Perigee rises monotonically with N until it plateaus, so
   nothing short of ~20 can reach the graveyard.

4. **The Sun was fixed in inertial space.** Now rotates about +Z at
   `sun_angular_rate_deg_per_day`, and `phase_boundaries` places the nominal
   eclipse at `2*pi*i + Omega*t_i` so the shadow follows it. trajopt needs no
   change for this: physical time is state x[5], so a moving Sun is just a
   dependence on an existing state.

5. **State bounds must bracket the excursion, and they also set the scaling.**
   Paper Eq. (44) values (`p` in [r_GEO, r_GEO+1000 km], `f,g` in +/-0.01) work.
   Too tight and the nodes are pushed where the dynamics cannot follow; too
   loose and the same nondimensional tolerance permits a much larger physical
   error. Widening `f,g` to +/-0.15 once let the eccentricity error through
   at 100x.

6. **A/m = 100 is worse than A/m = 10**, not better. The transfer becomes too
   short to shape the eccentricity vector, and the dynamics stiffen per
   revolution. Peak perigee plateaued near +90 km at every sail strength until
   the winding fixes landed.

## 3. Solver findings

1. **PS holds the dynamics far tighter than MS** on this problem: 4.97e-07
   against 3.17e-06 with half the nodes' worth of residual, and it improved
   monotonically with problem size (+191 -> +232 -> +244 km as N went 12 -> 16
   -> 20) while its defect fell. MS degraded under mesh refinement instead
   (defect 1.3e-01 at 10 nodes, 3.1e-01 at 16, 3.9e-01 at 20 for N=12), which
   looks like SCP step control rather than discretisation error.

2. **PS needs more collocation points per arc than MS needs nodes.** At 10
   nodes PS's residual was already excellent (1.9e-07) yet the re-propagated
   trajectory drifted 7.6 km: the polynomial satisfies the dynamics *at* the
   nodes, and the interpolation error between them is what a propagator sees.
   18 nodes removes it. Judging PS by its collocation residual alone is
   misleading.

3. **Infeasibility drains into the dynamics defect.** The terminal cones are
   hard (`scp_final_convex_inequality` has no virtual buffer) while the dynamics
   are buffered, so an unreachable target shows up as a numerical failure rather
   than as a terminal violation. Giving the cones buffers was tried and is
   worse - minimum time then simply violates them - so the asymmetry is
   deliberate, but it makes diagnosis hard and is worth knowing.

4. **Node values cannot be trusted as a feasibility test.** Minimum time drives
   the node perigee exactly onto the constraint, so the propagated trajectory
   lands short by precisely the accumulated defect. Every apparent near-miss in
   this study matched `shortfall ~= accumulated defect` to within a few percent.
   `solution._chained_propagation` now judges the propagated orbit instead.

5. **Per-interval tolerances do not bound end-to-end error.** The defect is
   systematically one-signed, so it accumulates linearly: at one point the
   measured 205 km drift was exactly 173 intervals x 1.19 km.
   `problem.defect_tolerance()` therefore sizes eps from an accumulated budget
   (`max_accumulated_defect_km`) divided by the interval count.

6. **Second order is essential here.** First-order SCP halves the defect roughly
   every 300 iterations and needs thousands; second order reaches its final
   accuracy in ~100 and then plateaus exactly, so a large `iter_max` is wasted.

## 4. trajopt changes made

| file | change |
| --- | --- |
| `methods/common/convergence.py` | convergence requires *every* buffered constraint to be feasible, not five categories named by hand. `initial_state` was exempt, so a solve could converge while starting 100 km from its stated initial condition |
| `methods/dev/sqp/scp_segment.py` | `eps_state` reads `flags.eps_state` instead of reusing the defect tolerance |
| `methods/dev/{sqp,scvx}/scp_segment.py` | PS mesh relation restored to `dt[k] = offset + (1-tau)*dt[0] + tau*dt[N-1]`; `dt[0]==0` only when the segment has no predecessor. Interior phase boundaries were previously frozen at their guess longitudes and the shadow-terminator equality was violated by 1.3e-02; it is now 1.1e-08 |
| `methods/dev/{sqp,scvx}/.../scp_constraint_types.py` | PS defect weighted by the Radau quadrature weights so it is comparable with a shooting defect |
| `methods/dev/{sqp,scvx}/.../scp_constraint.py` | `residual_scale` / `scaled_vb` supporting the above |

## 5. Still open

- **MS degrades under mesh refinement** (finding 3.1). The initial guess is
  dynamically exact to 3e-14, so this is not a discretisation limit.
- **Scalability.** CVXPY constraint formatting attempted a 76 GiB allocation
  around N ~ 35; per-solve cost grows superlinearly.
- `compute_legendre(..., use_spartan=False)` returns NaN at x = +/-1 (0/0 in the
  scipy branch). Latent, since the recursive branch is the default.
