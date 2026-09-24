# Multi-phase solar sail: validated solutions and solver findings

Both discretisations now produce **fully validated** minimum-time deorbit
solutions. Reproduce with `python main.py` (MS) and `python main.py
config-ps.yaml` (PS).

## 1. Solutions

Scenario: A/m = 10 m^2/kg, GEO -> graveyard (perigee >= r_GEO + 250 km,
e <= 0.003), cylindrical shadow, Sun line rotating at 0.9856 deg/day,
23 revolutions, 47 alternating sunlit/eclipse phases.

| | MS (`config.yaml`) | PS (`config-ps.yaml`) |
| --- | --- | --- |
| mesh (sunlit / eclipse nodes) | 12 / 3 | 24 / 4 |
| phases / nodes | 47 / 311 | 47 / 622 |
| elapsed time | **22.669 d** | **22.671 d** |
| true final perigee - r_GEO | **+250.01 km** | **+250.26 km** |
| true final eccentricity | **0.003000** | **0.002994** |
| accumulated defect | -0.004 km | -0.066 km |
| max per-interval defect | 8.56e-06 (tol 9.68e-06) | 4.35e-06 (tol 4.83e-06) |
| SCP iterations | 23 | 29 |
| converged / feasible | yes / **yes** | yes / **yes** |

"True" quantities come from re-integrating the whole trajectory end to end from
the exact initial state with each method's own control model, never resetting to
the optimiser's nodes. Elapsed times agree to 0.007 %, which is independent
confirmation: two different discretisations, same answer.

### Why 23 revolutions, and why not 22

23 is the minimum, and both methods reach it. The threshold is sharp rather than
gradual, because the reachable endpoint is quantised: a transfer can only end at
the end of a sunlit arc, and the final arc's window caps the end at
`L = 2*pi*(revolutions + 1)`. Dropping one revolution removes a whole `2*pi` of
thrusting.

| revolutions | method | converged | propagated graveyard | max defect | iterations | runtime |
| --- | --- | --- | --- | --- | --- | --- |
| 23 | MS | yes | +250.01 km, e 0.00300 | 8.6e-06 | 23 | 622 s |
| 23 | PS | yes | +250.26 km, e 0.00299 | 4.3e-06 | 29 | 400 s |
| 22 | MS | yes | +246.87 km, e 0.00306 | 5.8e-06 | 558 | 2436 s |
| 22 | PS | **no** | +251.81 km, e 0.00298 | 2.9e-05 | 600 (cap) | 3865 s |

At 22 the two methods fail on opposite criteria, which is worth reading
carefully because either one alone would be misleading:

- **MS converges and is still wrong.** The terminal cones are hard, so the
  *nodal* final state satisfies the graveyard exactly; the propagated one does
  not. The miss is 6e-5 of eccentricity accumulated in `f` and `g`, and since
  `dr_p/de ~ -42150 km` that alone accounts for 2.5 of the 3.1 km perigee
  shortfall. The accumulated-defect budget is stated in km of semi-latus rectum
  and so cannot see it.
- **PS reaches the graveyard and never converges.** It runs to the 600-iteration
  cap carrying a defect of 2.9e-05 against a per-interval budget of 1.3e-07, so
  the trajectory it produces is not a solution of the stated problem however
  good its endpoint looks.

The iteration count goes 23/29 at 23 revolutions to 558/600 at 22. Treat a
near-miss at `N-1` as structural, not as something more solver effort will
close.

Dropping to 23 is also physically honest, which is not automatic. The 23-rev cap
of `L = 150.796` falls just below the true 24th shadow entry at `L ~ 151.07`, so
the final arc is genuinely sunlit. Had it landed inside the shadow the model
would have granted the sail thrust it does not have, and nothing in the existing
validation would have noticed: `_chained_propagation` takes illumination from
`segment.params.illumination`, so a mislabelled arc propagates consistently with
the wrong physics. Checked by recomputing the cylindrical shadow from the
propagated positions and comparing against the modelled flags -- 0.15 % of
samples for MS and 0.04 % for PS, both at arc boundaries.

At 24 revolutions the last arc was a 0.056 rad stub *after* the 24th eclipse and
the graveyard was first met at `L = 151.06`, the eclipse entry. Orbital elements
are frozen through eclipse, so that final eclipse and stub did no work; the
phase structure simply forced the trajectory through them.

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
   90 revolutions at A/m = 1; at A/m = 10 the answer is 23, not the 1-4 implied
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

1. **PS holds the dynamics tighter than MS** on this problem: 4.35e-06 against
   8.56e-06 with each method at its own working point, and it improved
   monotonically with problem size (+191 -> +232 -> +244 km as N went 12 -> 16
   -> 20) while its defect fell. MS degraded under mesh refinement instead
   (defect 1.3e-01 at 10 nodes, 3.1e-01 at 16, 3.9e-01 at 20 for N=12), which
   looks like SCP step control rather than discretisation error. That
   degradation does *not* reproduce at the shipped 23-revolution configuration,
   where the MS defect goes 8.6e-06 -> 1.4e-06 -> 4.0e-06 over 12/3, 16/3, 20/3
   at flat runtime; see section 5.

2. **PS needs more collocation points per arc than MS needs nodes.** At 12
   nodes per sunlit arc -- the mesh MS is happy with -- PS's residual is 1.1e-06
   yet the re-propagated trajectory accumulates 6.4 km of defect and misses
   perigee by 3.7 km: the polynomial satisfies the dynamics *at* the nodes, and
   the interpolation error between them is what a propagator sees. 24 nodes
   removes it. Judging PS by its collocation residual alone is misleading, which
   is why `max_propagation_error_nd` is a feasibility criterion rather than a
   diagnostic.

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

- **MS degrades under mesh refinement** (finding 3.1), but not always: the
  effect is absent at the shipped 23-revolution configuration, where runtime and
  defect are both flat over 12/3 .. 20/3 (section 6). The initial guess is
  dynamically exact to 3e-14, so it is not a discretisation limit either way.
  Open because it is unexplained, not because it always happens.
- **Scalability.** CVXPY constraint formatting attempted a 76 GiB allocation
  around N ~ 35; per-solve cost grows superlinearly.
- `compute_legendre(..., use_spartan=False)` returns NaN at x = +/-1 (0/0 in the
  scipy branch). Latent, since the recursive branch is the default.

## 6. Performance of the two methods

Final configurations, A/m = 10, 23 revolutions, 47 phases, moving Sun. Timings
are whole-solve wall clock on one otherwise idle machine and include JAX
compilation and the CVXPY build.

| | MS (`config.yaml`) | PS (`config-ps.yaml`) |
| --- | --- | --- |
| nodes per sunlit / eclipse arc | 12 / 3 | 24 / 4 |
| phases | 47 | 47 |
| total nodes | 311 | 622 |
| SCP iterations | 23 | 29 |
| runtime | 622 s | **370 s** |
| elapsed time (solution) | 22.669 d | 22.671 d |
| true perigee - r_GEO | +250.01 km | +250.26 km |
| true eccentricity | 0.003000 | 0.002994 |
| accumulated defect | -0.004 km | -0.066 km |
| max per-interval defect | 8.56e-06 (tol 9.68e-06) | 4.35e-06 (tol 4.83e-06) |
| max re-propagation error | 8.56e-06 | 3.24e-04 |
| validated | yes | yes |

**PS is the cheaper method here despite twice the nodes** -- 370 s against 622 s,
with half the defect. Per node per iteration it is roughly three times faster
than multiple shooting. Earlier comparisons in this study suggested the
opposite; they were comparing the two at meshes chosen for different reasons
rather than at each method's own working point.

The two need different meshes and it is not a free choice. Sunlit arcs are
5.997 rad and eclipse arcs 0.304 -- a 19.7:1 ratio -- so node counts have to be
split roughly in that proportion; equal counts over-resolve the eclipse, which
is also the easy part (illumination is zero, the motion is Keplerian and carries
no control).

### Mesh study at 23 revolutions

| method | mesh | nodes | iterations | accumulated defect | true r_p - r_GEO | max defect | re-propagation | runtime | feasible |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MS | **12/3** | 311 | 23 | -0.004 km | **+250.01 km** | 8.56e-06 | 8.56e-06 | 622 s | **yes** |
| MS | 16/3 | 407 | 29 | -0.108 km | +250.07 km | 1.36e-06 | 1.30e-06 | 702 s | yes |
| MS | 20/3 | 503 | 34 | -0.048 km | +249.99 km | 3.98e-06 | 3.98e-06 | 695 s | no |
| PS | 12/3 | 311 | 50 | +6.367 km | +246.26 km | 1.13e-06 | 1.29e-03 | 196 s | no |
| PS | 16/3 | 407 | 41 | +0.474 km | +251.83 km | 2.18e-05 | 1.71e-03 | 211 s | no |
| PS | 20/4 | 526 | 55 | -0.134 km | +249.97 km | 6.90e-08 | 5.29e-04 | 819 s | no |
| PS | **24/4** | 622 | 29 | -0.066 km | **+250.26 km** | 4.35e-06 | 3.24e-04 | 370 s | **yes** |

Three things worth taking from this table.

**Only the shipped meshes validate.** Every other mesh fails, and they fail for
different reasons: PS at 12/3 and 16/3 on end-to-end drift (re-propagation error
above the 1e-03 limit) even though its collocation residual is excellent, and MS
at 20/3 and PS at 20/4 on the graveyard itself -- by 10 m and 30 m of perigee.
That is not a coincidence of tuning so much as a consequence of how tight 23
revolutions is; see the caveat below.

**PS converges faster as it is refined.** 50, 41 and 55 iterations at 12/3, 16/3
and 20/4, against 29 at 24/4. The finest mesh is also the cheapest in wall clock
(370 s against 819 s at 20/4), so for PS there is no accuracy-for-time trade to
make here -- refine until the re-propagation error comes down.

**MS runtime is flat in the mesh.** 622, 702 and 695 s over 12/3, 16/3 and 20/3,
and the defect does not degrade either (8.6e-06, 1.4e-06, 4.0e-06). This
contradicts the behaviour recorded in section 3, finding 1, which was measured at
12 revolutions on the pre-winding formulation; at the shipped configuration the
effect is absent. 12/3 is chosen because it is the cheapest mesh that validates,
not because refining it is punitive.

### Caveat: 23 revolutions has no margin

The minimum-time solution sits exactly on both terminal constraints -- MS lands
with 7 m of perigee to spare and the eccentricity on the cap to six decimals.
A change that shifts the propagated perigee by tens of metres therefore flips
feasibility, which is what MS 20/3 and PS 20/4 demonstrate. The shipped
configuration is validated, but it is brittle in a way the 24-revolution one was
not: anyone changing the mesh, the defect budget or the solver tolerances should
re-run the validation rather than assume it carries over.

Phase counts are identical (47 = 24 sunlit + 23 eclipse) for both methods; only
the node counts differ.

## 7. Further bugs found in the example

Found while reworking the figures, all fixed.

1. **Eclipse arcs were mislabelled once the Sun moved.** `phase_boundaries`
   decided sunlit-versus-eclipsed by asking whether an arc's midpoint lay within
   `half_shadow` of a multiple of 2*pi -- true only for a fixed Sun. With the Sun
   drifting, the crossings move by `Omega*t`, and after about nine days that
   exceeds `half_shadow`, so two thirds of the eclipse arcs were labelled sunlit
   and given `illumination = 1.0`. The sail was thrusting through eclipse for
   most of the transfer. Arcs are now labelled by construction from the
   crossing sequence.

2. **`max_state_jump` compared incompatible units.** It took the dimensional
   state difference across a phase boundary and compared the largest component
   against a single `1e-4`. That vector mixes km, dimensionless elements and
   seconds, and elapsed time runs to ~2e6 s, so the test was dominated by
   whichever state had the biggest numbers: a pass or fail at 1e-4 meant 0.1 mm
   on the semi-latus rectum or 0.1 ms on the clock, depending. It rejected an
   otherwise perfect PS solution. Now measured relative to each state's own
   scale, where both methods sit at 1e-12 to 1e-10.

