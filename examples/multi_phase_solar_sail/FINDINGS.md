# Multi-phase solar sail: validated solutions and solver findings

Both discretisations now produce **fully validated** minimum-time deorbit
solutions. Reproduce with `python main.py` (MS) and `python main.py
config-ps.yaml` (PS).

## 1. Solutions

Scenario: A/m = 10 m^2/kg, GEO -> graveyard (perigee >= r_GEO + 250 km,
e <= 0.003), cylindrical shadow, Sun line rotating at 0.9856 deg/day.

Two configurations ship, solving the **same 21-revolution problem** with the two
discretisations. 21 is the minimum for this sail and both methods reach it;
20 fails for both. Each config carries the settings its own discretisation needs
-- the mesh, the terminal margin and the dynamics penalty initialisation -- and
nothing else differs. Section 8 records what it took to get there.

| | MS (`config.yaml`) | PS (`config-ps.yaml`) |
| --- | --- | --- |
| revolutions / phases | 21 / 41 | 21 / 41 |
| mesh (sunlit / eclipse) | 12 / 3 | 24 / 4 |
| total nodes | 272 | 544 |
| dynamics penalty `W.init` | 100 | 1.0 |
| terminal margin | 0.3 km | 0 |
| SCP iterations | 39 | **15** |
| runtime | 569 s | **234 s** |
| elapsed time | 20.101 d | 20.114 d |
| true perigee - r_GEO | +250.34 km | +250.10 km |
| true eccentricity | 0.003000 | 0.002995 |
| accumulated defect | -0.033 km | -0.022 km |
| max per-interval defect | 1.59e-06 (tol 1.11e-05) | 5.00e-07 (tol 5.52e-06) |
| validated | yes | yes |

The two elapsed times agree to **0.06 %**, which is independent confirmation
that the answer is real: two different discretisations, two different meshes,
two different penalty balances, same transfer.

"True" quantities come from re-integrating the whole trajectory end to end from
the exact initial state with each method's own control model, never resetting to
the optimiser's nodes. They were additionally checked against an adaptive DOP853
integration at rtol = atol = 1e-12, independent of trajopt's fixed-step RK4,
which agrees to 12 m on the shooting solution.

Collocation is the cheaper method here by a wide margin -- 15 iterations against
39, and 234 s against 569 s, despite twice the nodes -- but only once its
dynamics penalty is set correctly; see section 8.

### The transfer ends at the last shadow entry

A sail gains nothing inside an eclipse -- illumination is zero, so the orbital
elements are frozen across it -- and the phase chain therefore stops at the last
shadow entry rather than at a whole number of revolutions past the start.

Running to `start + 2*pi*n_rev` instead, as this example did until recently,
places that entry *inside* the span: the chain picks up one more eclipse and
then a final sunlit stub behind it, and the minimum-time solution ends in that
stub having gained nothing from the eclipse it was forced to fly through.  At
23 revolutions the stub was 0.056 rad long and was the shortest phase in the
problem (0.219 h, against 1.16 h for a real eclipse arc).  Ending at the shadow
entry removes both, which is worth two phases and ~0.06 d directly, and more
importantly stops the final-arc window from being positioned by an arbitrary
count rather than by the geometry.

### The terminal cone needs a margin

`graveyard.perigee_margin_km` is added to the *nodal* terminal cone and to
nothing else: `validation_report` judges the propagated orbit against
`perigee_altitude_km`, so the margin buys no credit in the verdict.

It is needed because minimum time drives the nodal perigee exactly onto whatever
the cone asks for, and the propagated trajectory then lands short by the
discretisation error.  Aiming exactly at the requirement produces a solution
that misses it: at 23 revolutions multiple shooting came out 75 m low and passed
only on the validator's 100 m slack, and at the 20/3 mesh it failed outright.
With 300 m of margin the same case lands 631 m *above* the requirement.

The gap belongs to the discretisation, so the margin does too, and it is set
alongside the node counts.  Collocation at 24 nodes per arc does not need it:
asking for the extra 300 m moves the propagated perigee from +250.11 to
+250.41 km but pushes the per-interval defect from 4.82e-06 to 6.91e-06, across
the 5.04e-06 tolerance, so the solve stops validating.  `config-ps.yaml` sets
the margin to zero for that reason.

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

1. **Neither method's collocation residual predicts its accuracy.** At the
   shipped working points MS carries 2.75e-06 against PS's 4.82e-06, but the
   end-to-end re-propagation errors are 2.75e-06 and 4.61e-04 -- two orders of
   magnitude apart in PS's disfavour, because a shooting defect *is* an
   end-to-end error over its interval while a collocation residual is not.
   Compare at the same mesh and the point is sharper still: at 12/3 PS's
   residual is 1.16e-07, the best in the study, and its re-propagation error is
   2.61e-03, the worst (section 6).

   MS was previously recorded as degrading under mesh refinement (defect 1.3e-01
   at 10 nodes, 3.1e-01 at 16, 3.9e-01 at 20, measured at 12 revolutions on the
   pre-winding formulation). That does *not* reproduce at the shipped
   configuration: the MS defect goes 2.75e-06 -> 1.49e-06 -> 1.61e-06 over 12/3,
   16/3, 20/3, all three validate, and runtime falls slightly as the mesh is
   refined.

2. **PS needs more collocation points per arc than MS needs nodes.** At 12
   nodes per sunlit arc -- the mesh MS is happy with -- PS's residual is 1.16e-07
   yet the re-propagated trajectory accumulates 5.3 km of defect and misses
   perigee by 3.4 km: the polynomial satisfies the dynamics *at* the nodes, and
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

- **The dynamics penalty initialisation has to be tuned per discretisation**
  and there is no guidance for choosing it. The right value differs by two
  orders of magnitude between shooting and collocation here, and the wrong one
  costs two revolutions of performance while looking like a structural limit
  (section 8). A weight normalised by node count would remove the trap.
- **MS degrades under mesh refinement** (finding 3.1), but not always: the
  effect is absent at the shipped configuration, where all three meshes validate
  and runtime falls slightly as the mesh is refined (section 6). The initial
  guess is dynamically exact to 3e-14, so it is not a discretisation limit
  either way. Open because it is unexplained, not because it always happens.
- **Scalability.** CVXPY constraint formatting attempted a 76 GiB allocation
  around N ~ 35; per-solve cost grows superlinearly.
- `compute_legendre(..., use_spartan=False)` returns NaN at x = +/-1 (0/0 in the
  scipy branch). Latent, since the recursive branch is the default.

## 6. Performance of the two methods

A/m = 10, moving Sun, 21 revolutions, 41 phases. Timings are whole-solve wall
clock on one otherwise idle machine and include JAX compilation and the CVXPY
build.

| | MS (`config.yaml`) | PS (`config-ps.yaml`) |
| --- | --- | --- |
| nodes per sunlit / eclipse arc | 12 / 3 | 24 / 4 |
| total nodes | 272 | 544 |
| SCP iterations | 39 | **15** |
| runtime | 569 s | **234 s** |
| elapsed time (solution) | 20.101 d | 20.114 d |
| true perigee - r_GEO | +250.34 km | +250.10 km |
| true eccentricity | 0.003000 | 0.002995 |
| accumulated defect | -0.033 km | -0.022 km |
| max per-interval defect | 1.59e-06 (tol 1.11e-05) | 5.00e-07 (tol 5.52e-06) |
| validated | yes | yes |

**Collocation is far the cheaper method** -- 234 s against 569 s, in 15
iterations against 39, despite twice the nodes. Per node per iteration it is
roughly six times faster than shooting, and it holds the dynamics three times
tighter. That advantage only appears once its dynamics penalty is set
appropriately for its node count; with the shooting value it cannot solve this
revolution count at all (section 8).

The two need different meshes and it is not a free choice. Sunlit arcs are
5.997 rad and eclipse arcs 0.304 -- a 19.7:1 ratio -- so node counts have to be
split roughly in that proportion; equal counts over-resolve the eclipse, which
is also the easy part (illumination is zero, the motion is Keplerian and carries
no control).

### Mesh study (measured at 23 revolutions)

Measured on the previous 23-revolution setting, before the dynamics-penalty
finding in section 8 moved both methods to 21.  The mesh conclusions carry over
unchanged -- they are about resolution, not revolution count -- but the absolute
numbers are one setting behind the shipped ones.

| method | mesh | nodes | iterations | accumulated defect | true r_p - r_GEO | max defect | re-propagation | runtime | feasible |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MS | **12/3** | 298 | 47 | -0.188 km | **+250.63 km** | 2.75e-06 | 2.75e-06 | 802 s | **yes** |
| MS | 16/3 | 390 | 59 | -0.009 km | +250.39 km | 1.49e-06 | 1.49e-06 | 694 s | yes |
| MS | 20/3 | 482 | 35 | -0.073 km | +250.38 km | 1.61e-06 | 1.61e-06 | 648 s | yes |
| PS | 12/3 | 298 | 73 | +5.346 km | +246.65 km | 1.16e-07 | 2.61e-03 | 220 s | no |
| PS | 16/3 | 390 | 45 | +0.547 km | +249.50 km | 2.23e-06 | 7.98e-04 | 210 s | no |
| PS | 20/4 | 504 | 51 | -0.619 km | +252.37 km | 1.10e-05 | 7.66e-04 | 648 s | no |
| PS | **24/4** | 596 | 24 | -0.057 km | **+250.11 km** | 4.82e-06 | 4.61e-04 | 315 s | **yes** |

**Shooting is now insensitive to the mesh, and the margin is why.** All three
counts validate and land within 250 m of each other, where before the margin was
introduced 20/3 missed the requirement by 10 m and failed.  Runtime falls
slightly as the mesh is refined (802, 694, 648 s) because the iteration count
does not grow with it.  12/3 is shipped because it is the cheapest that
validates, not because refining it is punitive.

**Collocation is not insensitive, and its residual says nothing useful.** At
12/3 it has the *smallest* collocation residual in the whole table -- 1.16e-07,
an order of magnitude better than any shooting run -- while accumulating 5.3 km
of defect and missing perigee by 3.4 km.  The polynomial satisfies the dynamics
at the nodes; the interpolation error between them is what a propagator sees, and
that is what the terminal condition feels.  This is why
`max_propagation_error_nd` is a feasibility criterion rather than a diagnostic:
it is the only quantity in the table that separates PS 12/3 from PS 24/4.

**Only the finest collocation mesh validates, and it is also the fastest.**
24/4 converges in 24 iterations against 73, 45 and 51 for the coarser meshes, so
it costs 315 s against 648 s at 20/4.  For collocation on this problem there is
no accuracy-for-time trade: refine until the re-propagation error comes down.

### Caveat: the eccentricity cap has no margin

`graveyard.perigee_margin_km` buys room on the perigee requirement, and the
shipped solutions land 110-630 m above it.  The eccentricity cap has no
equivalent and the minimum-time solution sits on it to six decimals in every
case (0.002997 to 0.003000 against 0.003).  A change that shifts the propagated
eccentricity by 1e-6 therefore still flips feasibility, and through
`dr_p/de ~ -42150 km` a few parts in 1e-6 of eccentricity is a few hundred
metres of perigee.  Anyone changing the mesh, the defect budget or the solver
tolerances should re-run the validation rather than assume it carries over.

Phase counts are identical (41 = 21 sunlit + 20 eclipse) for both methods; only
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

## 8. What it took to get both methods to 21 revolutions

Both discretisations solve the same 21-revolution problem, but neither does so
with trajopt's defaults or with the same settings as the other.  This is the
record of what actually mattered, because most of it was not obvious and two of
the settings pull in opposite directions.

### The two that decide feasibility

**1. The phase chain must end at the last shadow entry.**  Running to
`start + 2*pi*n_rev` instead puts the next shadow entry inside the span, so the
chain picks up one more eclipse and a sunlit stub behind it.  Elements are
frozen across an eclipse, so the minimum-time solution ends in that stub having
gained nothing from the eclipse it was forced to fly through -- the stub was
0.056 rad and was the shortest phase in the problem.  Fixing this took the
transfer from 23 structural revolutions to 21 on its own.

**2. `penalty.dynamics.W.init` must differ between the methods, and this is
worth two revolutions.**  `config.yaml` starts it at 100 because trajopt's
`min_time` cost sums the longitude over every node of the final segment, so its
magnitude grows with the node count and can outweigh the dynamics penalty, at
which point the optimiser buys a shorter trajectory by leaving a dynamics
defect.  That reasoning is correct for shooting.  It is wrong for collocation,
which has roughly twice the nodes, so the same weight sits very differently
against the objective:

| `W.init` | shooting at 20 rev | collocation at 22 rev | collocation at 21 rev |
| --- | --- | --- | --- |
| 1.0 (trajopt default) | +215.70 km, no convergence | **+250.03 km, converged, 59 it** | **+250.10 km, converged, 15 it** |
| 100 (config.yaml) | +245.00 km, converged | +246.44 km, no convergence | not reached |
| 10 000 | +217.07 km, no convergence | +246.34 km, no convergence | not reached |

Collocation at 100 cannot solve 22 revolutions *at all*, and at 1.0 it solves 21
in 15 iterations -- the fastest solve in the example.  Shooting wants the
opposite.  Each configuration therefore carries its own value, next to the mesh,
because it is the same kind of setting: a property of the discretisation, not of
the problem.

**3. The terminal cone needs a margin, and only for shooting.**  Minimum time
drives the *nodal* perigee exactly onto whatever the cone asks for, and the
propagated trajectory then lands short by the discretisation error.  Shooting at
12 nodes per arc came out 75 m low and passed only on the validator's 100 m
slack; with `graveyard.perigee_margin_km: 0.3` it lands 631 m above.  Collocation
at 24 nodes does not need it and is hurt by it -- the extra 300 m pushes its
per-interval defect from 4.82e-06 to 6.91e-06, across tolerance, so the solve
stops validating.  `validation_report` judges the propagated orbit against the
unmargined requirement, so the margin earns no credit in the verdict.

### What was tried and did not matter

Before the penalty was identified, collocation's floor was assumed to be
structural.  Everything in this list was tried against it and none of it helped;
recording it is the point, because each one is a plausible first guess.

| lever | setting | collocation at 22 rev |
| --- | --- | --- |
| mesh | 24/4 (576 nodes) | +246.44 km |
| mesh | 32/6 (788 nodes) | +246.38 km |
| mesh | 25/5 (613 nodes) | +246.45 km |
| hp refinement | `hp_segments: 4` | +246.10 km |
| conic solver | SCS | +246.03 km |
| conic solver | QOCO | +245.39 km |
| conic solver | ECOS | MemoryError, 8 GiB allocation |
| conic solver | PIQP | fails at iteration 0 |
| trust region | `alpha_z = alpha_nu = 10` | +246.019 km (bit-identical to baseline) |
| trust region | `alpha_z = alpha_nu = 0.1` | +246.019 km (bit-identical) |
| initial guess | seeded from a verified shooting solution | +202.84 km (worse) |
| initial guess | continuation from its own converged 23-rev answer | -305.81 km (much worse) |

Four meshes spanning 576-788 nodes and two polynomial structures agree within
0.35 km, so it was never resolution.  The trust region is provably inert: its
term vanishes as the step goes to zero, so it changes the path to a local
optimum and not the optimum itself, which is why both directions reproduce the
baseline bit for bit.  And seeding *on* a known-good trajectory made things
worse, which was the clue that a penalty was pulling the iteration away from it
rather than the mesh being unable to represent it.

The same sweep on shooting at 20 revolutions found nothing at all -- `equal_dt:
0` (+245.06), `nsub: 16` (+245.02), both trust-region directions (bit-identical)
and both penalty directions (worse).  20 revolutions is genuinely out of reach
for this sail; the propagated orbit stalls near +245 km against a +250 km
requirement.

### Node placement

`equal_dt: 0` frees each interior node time as its own variable, which shooting
supports and collocation structurally cannot -- the Radau abscissae are fixed by
the differentiation matrix, and the mesh relation makes interior node times
affine in the two arc endpoints.  It is worth 53 m at 20 revolutions and reduces
the 21-revolution iteration count from 39 to 26, so it is a mild convergence
aid, not a feasibility lever.  Left at `1` in the shipped configuration because
uniform spacing is easier to reason about and the solution is identical.
