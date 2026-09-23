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
| mesh (sunlit / eclipse nodes) | 12 / 3 | 24 / 4 |
| elapsed time | **23.654 d** | **23.676 d** |
| true final perigee - r_GEO | **+249.94 km** | **+250.04 km** |
| true final eccentricity | **0.002999** | **0.002997** |
| accumulated defect | 0.090 km | 0.027 km |
| max per-interval defect | 1.78e-06 | 1.41e-07 |
| SCP iterations | 33 | 28 |
| runtime | 584 s | 392 s |
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

## 6. Performance of the two methods

Final configurations, A/m = 10, 24 revolutions, 49 phases, moving Sun. Timings
are whole-solve wall clock on one machine and include JAX compilation and the
CVXPY build.

| | MS (`config.yaml`) | PS (`config-ps.yaml`) |
| --- | --- | --- |
| nodes per sunlit / eclipse arc | 12 / 3 | 24 / 4 |
| phases | 49 | 49 |
| total nodes | 324 | 648 |
| SCP iterations | 33 | 28 |
| runtime | 584 s | 392 s |
| elapsed time (solution) | 23.654 d | 23.676 d |
| true perigee - r_GEO | +249.94 km | +250.04 km |
| true eccentricity | 0.002999 | 0.002997 |
| accumulated defect | 0.090 km | 0.027 km |
| max per-interval defect | 1.78e-06 | 1.41e-07 |
| validated | yes | yes |

**PS is the cheaper method here despite twice the nodes** -- 392 s against 584 s,
in fewer iterations, with an order of magnitude smaller defect. Per node per
iteration it is roughly four times faster than multiple shooting. Earlier
comparisons in this study suggested the opposite; they were comparing the two at
meshes chosen for different reasons rather than at each method's own working
point.

The two need different meshes and it is not a free choice. Sunlit arcs are
5.997 rad and eclipse arcs 0.304 -- a 19.7:1 ratio -- so node counts have to be
split roughly in that proportion; equal counts over-resolve the eclipse, which
is also the easy part (illumination is zero, the motion is Keplerian and carries
no control).

Mesh studies behind the choices, at 24 revolutions:

| mesh | MS acc / runtime | PS acc / runtime |
| --- | --- | --- |
| 12/3 | 0.090 km / 605 s | 5.240 km / 180 s |
| 16/3 | 0.057 km / 1563 s | 0.592 km / 209 s |
| 20/3-20/4 | -0.025 km / 2743 s | -0.322 km / 910 s |
| 24/4 | (not reached) | 0.027 km / 384 s |

MS refines expensively: 1.6x the nodes costs 4.5x the runtime for ~0.1 km of
accumulated defect against a 3 km budget, so 12/3 is the sensible point. PS
refines cheaply and needs to: at 12/3 its collocation residual is already tiny
(1.7e-06) while the re-propagated trajectory drifts 5.2 km, because the
polynomial satisfies the dynamics *at* the nodes and the interpolation error
between them is what a propagator sees. 24/4 removes that and converges in the
fewest iterations of any mesh tried, so it is also not the slowest.

Phase counts are identical (49 = 25 sunlit + 24 eclipse) for both methods; only
the node counts differ.

## 7. Further bugs found in the example

Found while reworking the figures, all fixed.

1. **Eclipse arcs were mislabelled once the Sun moved.** `phase_boundaries`
   decided sunlit-versus-eclipsed by asking whether an arc's midpoint lay within
   `half_shadow` of a multiple of 2*pi -- true only for a fixed Sun. With the Sun
   drifting, the crossings move by `Omega*t`, and after about nine days that
   exceeds `half_shadow`, so 16 of the 24 eclipse arcs were labelled sunlit and
   given `illumination = 1.0`. The sail was thrusting through eclipse for two
   thirds of the transfer. Arcs are now labelled by construction from the
   crossing sequence.

2. **`max_state_jump` compared incompatible units.** It took the dimensional
   state difference across a phase boundary and compared the largest component
   against a single `1e-4`. That vector mixes km, dimensionless elements and
   seconds, and elapsed time runs to ~2e6 s, so the test was dominated by
   whichever state had the biggest numbers: a pass or fail at 1e-4 meant 0.1 mm
   on the semi-latus rectum or 0.1 ms on the clock, depending. It rejected an
   otherwise perfect PS solution. Now measured relative to each state's own
   scale, where both methods sit at 1e-12 to 1e-10.

