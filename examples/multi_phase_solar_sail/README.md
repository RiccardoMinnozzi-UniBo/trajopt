# Multi-phase solar-sail GEO disposal

This example implements the first problem of Kelly and Bevilacqua, *"Geostationary
debris mitigation using minimum time solar sail trajectories with eclipse
constraints"* (2021): move a sail-equipped spacecraft from circular GEO to a
graveyard orbit with

- eccentricity no greater than `0.003`, and
- perigee radius at least `r_GEO + 250 km`,

in minimum time, using solar radiation pressure alone.

Only central Earth gravity and SRP are retained, and the Earth casts a
cylindrical shadow, so there are no ephemerides, third bodies or geopotential
terms. The Sun line does rotate, at `sun_angular_rate_deg_per_day`, which makes
the dynamics non-autonomous; the shadow follows it. Each sunlit and eclipsed arc
is a separate trajopt segment, tied to its neighbours by state, control and
longitude continuity.

The sail is sized at `A/m = 10 m^2/kg`, ten times the paper's, so the transfer
takes ~24 revolutions instead of ~90 and the example runs in minutes. Everything
else follows the paper.

## Running it

```bash
python main.py
```

Edit `config.yaml`, run this, get the minimum-time solution. The revolution count
cannot be optimised inside a single SCP solve — it sets the number of phases — so
`main.py` searches over it, bracketing the threshold from the guess in
`problem.revolutions` and then bisecting, which costs `O(log)` solves rather than
one per revolution. Change the sail, the terminal requirement or anything else
and it still lands on the minimum-time solution; a poor guess only costs extra
solves, never the right answer.

`problem.revolutions: 24` is already the answer for the shipped sail, so set
`problem.revolution_search.enabled: false` to solve once at the guess — that is
what produced the numbers below, and it is much quicker while iterating on
something else. A solve at this size takes a few minutes; the search multiplies
that by the number of attempts.

Both discretisations validate against **one shared problem definition**; only
the mesh differs between them.

| | multiple shooting | pseudospectral |
| --- | --- | --- |
| config | `config.yaml` | `config-ps.yaml` |
| nodes per sunlit / eclipse arc | 10 / 4 | 18 / 6 |
| revolutions | 24 | 24 |
| phases | 49 | 49 |
| elapsed time | **23.637 days** | **23.656 days** |
| revolutions flown | 23.598 | 23.598 |
| true final perigee radius | r_GEO + 249.92 km | r_GEO + 250.31 km |
| true final eccentricity | 0.002997 (cap 0.003) | 0.002817 |
| accumulated defect | 0.202 km (budget 3.0) | -0.528 km |
| max dynamics defect | 3.17e-06 | 4.97e-07 |
| shortest phase | 0.218 h | 0.219 h |
| SCP iterations | 38 | 79 |
| validated | **yes** | **yes** |

Both requirements are active: the sail arrives at the graveyard orbit with
essentially nothing to spare, which is what a minimum-time solution should look
like. The two elapsed times agree to 0.08 % — two independent discretisations
reaching the same answer is the strongest evidence available that it is right.

Every "true" quantity above comes from re-integrating the whole trajectory end to
end from the exact initial state, using each method's own control model and never
resetting to the optimiser's nodes. See [Verification](#verification): the node
values on their own cannot be trusted, because minimum time drives them exactly
onto the constraint.

### Discretisation: multiple shooting or pseudospectral

```bash
python main.py                  # RK4 multiple shooting (config.yaml)
python main.py config-ps.yaml   # flipped Legendre-Radau collocation
```

`config-ps.yaml` inherits `config.yaml` and overrides only the discretisation and
the mesh, so the two cannot drift apart on the physics. Both reach a validated
solution and agree on the answer to 0.08 %; the table at the top compares them.

**Collocation needs more points per arc than shooting needs nodes.** This is the
one place the two configurations genuinely differ, and it is not arbitrary. At 10
nodes per sunlit arc the Radau residual is already excellent — 1.9e-07, far
better than shooting manages — and yet the re-propagated trajectory drifts 7.6 km
over the transfer and misses the perigee requirement by 3 km. The polynomial
satisfies the dynamics *at* the collocation nodes; between them the interpolation
error is what a propagator sees, and that error is what the terminal condition
feels. 18 nodes per arc removes it.

The practical lesson: **do not judge a pseudospectral solution by its collocation
residual.** A residual six times smaller than the shooting run's accompanied a
drift fifteen times larger. Only end-to-end re-propagation separates the two.

With the mesh right, PS is the better-behaved method here. It holds the dynamics
tighter (4.97e-07 against 3.17e-06) and improves monotonically as the problem
grows — at 10 nodes per arc, perigee went +191 km at 12 revolutions, +232 at 16,
+244 at 20, with the defect falling from 1.4e-03 to 9.3e-06 along the way.
Multiple shooting does the opposite under mesh refinement; see the observations
below.

### Figures

All are white-background, 300 dpi, colourblind-safe (Okabe-Ito), and sized for a
one- or two-column figure. Each is also callable on its own from
`visualization.py` and takes `annotated=True/False`, so you can restyle or drop
any of them.

Every figure is written **twice**:

- `results/` — annotated. Each figure carries the settings it came from (A/m,
  revolution count, elapsed time, validation status), so a figure found on disk
  months later is self-describing.
- `results/paper/` — unannotated, same axes. The annotation sits outside the
  axes, so with a tight bounding box it changes the saved figure's extent; the
  clean version drops into a LaTeX float without disturbing the layout around
  it. Point your paper at this directory.

| file | what it shows |
| --- | --- |
| `elements.png` | apsides, eccentricity and semi-latus rectum against the requirements |
| `control.png` | steering angle, sail cone angle, and the resulting thrust magnitude |
| `transfer_plane.png` | polar plan view of the spiral with the shadow sector — the radial axis is `r − r_GEO`, which is the only way the transfer is visible at all |
| `trajectory_3d.png` | 3D view with the Earth and the cylindrical shadow |
| `phases.png` | phase timeline and per-phase durations |
| `convergence.png` | SCP convergence history against the tolerance |
| `summary.png` | all of the above on one sheet, for a talk |
| `trajectory_3d.gif` | rotating 3D view (validated solutions only) |

Edit the palette and `_RC` block at the top of `visualization.py` to restyle the
whole set at once.

## Files

| file | role |
| --- | --- |
| `config.yaml` | every physical, structural and solver setting worth editing |
| `solar_sail.py` | dynamics, sail steering law, shadow geometry, terminal cones |
| `problem.py` | expands the settings into the alternating trajopt segments |
| `search.py` | outer loop over the revolution count (the paper's Algorithm 1) |
| `solution.py` | extraction, independent verification, save/load |
| `visualization.py` | summary dashboard, 3D still, rotating GIF |
| `main.py` | solve (or search), validate, save, draw |

## Formulation

True longitude `L` is the independent variable, as in Section 2.3 of the paper.
The state is

```text
x = [p, f, g, h, k, elapsed_time]
```

and the control is a single **steering angle** `theta`, the direction of the
paper's primer vector in the local orbital frame:

```text
primer = cos(theta) * radial + sin(theta) * transverse
```

Only the primer's *direction* enters the locally optimal steering law of
Equation (24), so one angle carries exactly the same information as two
components plus a unit-norm equality — and removes that nonconvex constraint
entirely. The sail cone angle follows in closed form; when the primer points
into the Sun the law returns `alpha = pi/2`, so the sail feathers and produces no
thrust, which is the correct behaviour.

**`theta` must be free to wind, and the guess must wind with it.** The optimal
primer is roughly fixed in inertial space, so in the rotating local frame it
turns a full `2*pi` *per revolution* — the paper's Figure 7 shows exactly that.
Two consequences, both of which had to be got right before anything converged:

- `steering_limits` is `+/-2*pi*(revolutions + 2)`, not `+/-2*pi`. Bounding the
  angle to one turn caps the sail at one revolution of useful steering; past
  that the control saturates and the eccentricity is pumped instead of held at
  the cap.
- the initial guess carries `-2*pi` per revolution
  (`guess.steering_winds_per_rev`; `0` restores a constant angle). SCP is a
  local method, so a non-winding guess stays on the non-winding branch however
  wide the bounds are.

Together these were worth about +50 km of final perigee.

During eclipse the illumination factor is zero, so SRP vanishes regardless of
`theta`. Control continuity across the phase boundaries pins the steering angle
there, where it would otherwise be a free null direction.

### The revolution count is a guess, not a boundary condition

`problem.revolutions` sets how many phases the problem has and seeds the initial
guess. It does **not** fix where the trajectory ends:

- every interior arc ends where the spacecraft actually crosses the shadow
  terminator, imposed as the nonconvex equality `shadow_boundary = 0`, so the
  boundaries move with the orbit;
- the final arc ends wherever the minimum-time solution wants it to, within
  `final_longitude_freedom_rad` of the nominal endpoint.

A window of `terminator_window_rad` around each nominal boundary stops an arc
collapsing or inverting. The nominal eclipse arc is about 0.304 rad, so the
default window of 0.05 rad still leaves every arc at least 0.2 rad long.

The right revolution count is therefore searched over, by `search.py`, which
brackets the threshold and then bisects it:

- expand outward from the guess, doubling the stride, until one feasible and one
  infeasible count are known;
- bisect until they are adjacent. The infeasible side is what proves the feasible
  side is minimal.

This rests on monotonicity: more revolutions means more time under thrust, so
feasibility, once gained, is kept. Bracketing costs `O(log(distance))` solves
where stepping by one costs `O(distance)` — which matters, because the answer
here is 24 and each solve at that size takes minutes.

Feasibility is judged on the **propagated** trajectory, not the optimiser's final
node; see [Verification](#verification). Judged on the nodes, every count from 1
upward looks feasible, because minimum time drives the final node onto the
constraint by construction and the shortfall hides in the accumulated defect.

If the walk runs off `revolution_search.minimum` without a failure, or runs out
of `max_solves` before the bracket closes, the search says so.

Each attempt owns a compiled CVXPY subproblem that can run to gigabytes, so the
search keeps only the best one alive — retaining them all exhausts memory after a
handful of solves.

### Revolution count versus sail size

The transfer needs roughly one revolution per `90 / (A/m)` days, because the
perigee raise is set by the total impulse and the eccentricity has to be shaped
over many orbits rather than reacted to locally.

| `area_to_mass_m2_kg` | revolutions | phases | transfer |
| --- | --- | --- | --- |
| 10 (shipped) | 24 | 49 | 23.64 days |
| 1 (paper sizing) | ~90 | ~181 | ~90 days |

The paper's own sizing is reachable in principle but not in this example's
runtime: CVXPY's constraint formatting attempts a 76 GiB allocation somewhere
around 35 revolutions, and per-solve cost grows superlinearly. `A/m = 10` keeps
the structure of the problem — multirevolution, eclipsed, eccentricity-limited —
at a size that solves in minutes.

Counter-intuitively, **raising `A/m` further makes the problem harder, not
easier.** At `A/m = 100` the transfer is over in a couple of revolutions, which
leaves too little of the orbit to shape the eccentricity vector, and the dynamics
stiffen per revolution so the mesh has to be finer. Peak perigee plateaued near
`r_GEO + 90 km` at every sail strength tried until the winding fixes above
landed.

### The minimum-time objective

`problem.minimum_time_objective` picks between two forms, which fail in opposite
ways:

- **`longitude`** (default) — trajopt's `min_time`, acting on the independent
  variable. It cannot be gamed, because the longitude grid carries no
  virtual-buffer slack. But it is *indifferent* to how far the orbit is
  over-raised: going past the perigee requirement costs no extra longitude, so a
  flat set of near-optimal solutions is left for the solver to drift within.
- **`elapsed_time`** — a cost on the elapsed-time state. Not indifferent, since a
  higher orbit has a longer period and therefore costs time directly. This form
  was historically reducible by pushing elapsed time *backwards* through the
  dynamics' virtual buffer, which is what produced negative-duration phases; that
  turned out to be a symptom of the state scaling rather than of the objective,
  but it remains the riskier of the two.
- **`both`** — the longitude cost plus the elapsed-time cost at
  `elapsed_time_cost_weight`, purely to break ties inside the flat set.

`dt/dL > 0` throughout, so all three have the same minimiser; they differ only in
how well-determined the solution is and in what the solver can exploit.

### Scaling

The longitude is scaled by the nominal swept longitude (Eq. 45). For the states,
see `_state_scales` in `problem.py`.

trajopt nondimensionalises by **division only, with no offset**
(`src/trajopt/nondim.py`), so an interval's amplitude is the right scale only for
a state centred on zero. The semi-latus rectum is not: it sits at ~42164 km
inside an interval ~1000 km wide. Rather than work around that with a
magnitude-based scale — which leaves the motion at the 1e-2 level of the
variable, so a dynamics defect that is numerically negligible is physically large
— **the state carries `p - r_GEO`**, which is centred on zero and takes the
amplitude scale like everything else. `solar_sail.semi_latus_rectum()`
reconstitutes the absolute value wherever the dynamics need it.

The bounds in `state_bounds` must *bracket the whole trajectory*, not just its
endpoints, and they also set the scaling — so they are a coupled knob and both
directions hurt:

- too tight and the nodes are pushed where the dynamics cannot follow, and the
  defect absorbs the difference. Bounds of `+/-0.004` on `f, g` were worth ~10x
  on every metric.
- too loose and the same nondimensional tolerance permits a much larger physical
  error. Widening `f, g` to `+/-0.15` let the eccentricity error back in at
  roughly 100x.

Paper Eq. (44) values work: `p` in `[r_GEO, r_GEO + 1000 km]`, `f, g` in
`+/-0.01`. The validation report prints the peak excursions so they can be sized
from a run.

### The defect budget

`problem.max_accumulated_defect_km` bounds the *accumulated* dynamics error over
the whole trajectory, and `problem.defect_tolerance()` turns it into the
per-interval `eps` by dividing by the interval count.

That division matters. The defect is systematically one-signed, so it accumulates
linearly rather than as a random walk — at one point in development the measured
205 km drift was exactly `173 intervals x 1.19 km`. A tolerance stated per
interval therefore says nothing about the trajectory until it is multiplied by
the number of intervals, and the two differ by two orders of magnitude here.

The budget is what the perigee shortfall is made of. Minimum time drives the node
perigee exactly onto its constraint, so the propagated trajectory lands short by
almost precisely the accumulated defect: at 20 km of budget both methods missed
by 0.5-3 km, and at 3 km both validate.

### Initial guess

The guess propagates the true nonlinear dynamics through each arc in turn and
hands its endpoint to the next arc (`guess.x_start: "previous"` with
`type: propagation`), so it is a genuine continuous trajectory with essentially
zero dynamics defect, rather than a straight line. The steering angle ramps
between the configured endpoint primers *and winds by `-2*pi` per revolution*, so
the guess already has the structure the solution needs; see the Formulation
section for why that matters.

## Verification

Two independent checks, because they catch different failures.

`_propagation_error` re-integrates every node interval from the optimiser's own
node values with a finer RK4 (16 substeps against the solver's 6) and compares
with the next node. It is reported nondimensionally alongside a per-state
breakdown, so a failure says *which* state is responsible.

`_chained_propagation` re-integrates the **whole trajectory end to end** from the
exact initial state, using each method's own control model and never resetting to
the optimiser's nodes. This is the check that decides feasibility, and the
validation report judges the graveyard requirement on the orbit it produces
rather than on the final node.

The distinction is not academic. Restarting at every node means per-interval
errors never add up, so that check cannot see a trajectory that drifts — and
because minimum time drives the final node exactly onto the perigee constraint,
the node values always look perfect. Judging feasibility from them is asking the
optimiser to mark its own work: at one point a solve reported the graveyard
reached while the propagated orbit was 205 km short.

## Observations about trajopt

Found while building this example. The first group has been **fixed in the
package** — see [Changes to trajopt itself](#changes-to-trajopt-itself) for the
diffs and the reasoning; the rest are recorded so they can be judged on their own
merits. `FINDINGS.md` carries the full account with measurements.

### Fixed

1. **Convergence never checked `initial_state`.** `convergence.py` enumerated
   five constraint categories by name and `initial_state` matched none of them,
   so a solve could report convergence while starting 100 km from its stated
   initial condition — and it did: an apparently excellent result (+275 km of
   perigee) collapsed to +32 km once the constraint was enforced. Now every
   constraint carrying a virtual buffer must be satisfied.

2. **The pseudospectral mesh could not move a segment's initial epoch.**
   `create_free_final_time_constraints` related interior node times to the
   endpoints as `dt[k] = offset + tau[k]*dt[N-1]`, dropping the
   `(1 - tau[k])*dt[0]` term and forcing `dt[0] == 0` unconditionally. Harmless
   for one segment; in a multi-phase chain it froze every interior boundary at
   its guess longitude, and the shadow-terminator equality was violated by
   1.3e-02. With the term restored it is 1.1e-08, two orders better than multiple
   shooting.

3. **The PS defect was not comparable with the shooting defect.** Collocation
   states a residual on `dz/dtau`, multiple shooting one on the state change
   across an interval, and the two differ by roughly the interval count for the
   same trajectory error — yet they shared a tolerance. The PS defect is now
   weighted by the Radau quadrature weights, which puts it back in interval-error
   units.

4. **`flags.eps_state` was dead for `dev.sqp`.** It took both `eps_dyn` and
   `eps_state` from the dynamics constraint's `penalty_state.eps` and never read
   the flag, so tightening the defect tolerance also tightened the *step*
   criterion to something unreachable. They answer different questions and now
   have different settings.

5. **`nondim` is multiplicative only.** `Nondim` builds `d2nd`/`nd2d` as diagonal
   scale matrices with no offset, so a state confined to a narrow interval far
   from zero cannot be scaled by that interval's amplitude. Not changed in the
   package — the example instead carries `p - r_GEO` so the state is centred on
   zero. An affine map `(x - lower)/(upper - lower)` would remove the need.

### Open

6. **Multiple shooting degrades under mesh refinement.** At 12 revolutions the
   defect grew 1.3e-01 -> 3.1e-01 -> 3.9e-01 as the mesh went 10 -> 16 -> 20
   nodes per arc, and the trajectory got worse with it. The initial guess is
   dynamically exact to 3e-14, so this is not a discretisation limit; it points
   at SCP step control. Pseudospectral does the opposite and improves with size.

7. **Infeasibility surfaces as a dynamics defect, not a terminal violation.**
   The terminal cones (`scp_final_convex_inequality`) are hard while the dynamics
   are buffered, so when the target is out of reach the only slack in the problem
   is the dynamics and every infeasible run looks like a numerical failure.
   Buffering the cones was tried and is worse — minimum time then simply violates
   them — so the asymmetry is right, but it makes diagnosis hard.

8. **Scalability.** CVXPY constraint formatting attempted a 76 GiB allocation
   around 35 revolutions, and per-solve cost grows superlinearly (13 s at 5
   revolutions, 259 s at 30). This is what keeps the paper's `A/m = 1` sizing out
   of reach here.

9. **`final_time` bounds assume the segment starts at zero.** In
   `constraint_types.final_time`, `dt_min = lower / (N - 1)` and
   `dt_max = upper / (N - 1)` bound the *absolute* final time, then get applied as
   per-interval spacing bounds. Right for a first segment starting at t = 0,
   wrong for any later segment in a chain. This example works around it with a
   `final_nonconvex_inequality` on the longitude.

10. **The PS branch ignores `zoh_dilation`.** The `ms` branch applies
    `s_k == s_{k+1}` only when the flag is set; the `ps` branch always does. It
    is in fact required for consistency with the affine PS mesh, so the flag is
    simply inert there.

11. **`hp_segments` is a method-wide flag but a per-segment constraint.** It must
    divide `num_nodes - 1` for *every* segment, so a multi-phase problem with
    different node counts per phase type can only use common divisors. The
    shipped `autoscvx-ps.yaml` defaults to `hp_segments: 10`, which fails for most
    node counts; `1` is the only value that never constrains the mesh.

12. **`compute_legendre(..., use_spartan=False)` returns NaN at `x = ±1`.** The
    scipy branch evaluates `N (x P_n - P_{n-1}) / (x^2 - 1)`, which is 0/0 at the
    endpoints — and `±1` are always in the node set. Latent only because
    `USE_SPARTAN = True` is the default.

## Modelling note

The paper's eclipse-specific virtual control is unnecessary here. It exists
because physical time is a state while true longitude is the independent
variable, and the high-fidelity shadow dynamics were otherwise uncontrolled. In
this implementation `dt/dL` comes directly from the two-body modified-equinoctial
equations, so eclipsed arcs have well-defined dynamics without an artificial
control.

With the Sun in the equatorial plane and in-plane steering, the normal
acceleration is identically zero, so `h` and `k` stay at zero and the problem is
planar. They are kept in the state for generality.

The Sun line rotates about +Z at `sun_angular_rate_deg_per_day` (0.9856, one turn
per year), which makes the dynamics non-autonomous. trajopt needs nothing special
for this: physical time is carried as state `x[5]`, so a moving Sun is a
dependence on an existing state and the Jacobians simply pick up a column that
was previously zero. `phase_boundaries` places the nominal eclipse at
`2*pi*i + Omega*t_i` so the shadow follows the Sun; without that the nominal
boundaries drift out of `terminator_window_rad` after a few revolutions. Set the
rate to `0.0` to recover a fixed Sun.

## Changes to trajopt itself

Everything above runs against a modified `src/trajopt`. These are the package
changes this example forced, all of them bugs rather than accommodations —
`git diff src/` shows 93 insertions across 7 files. `FINDINGS.md` carries the
measurements.

Each fix is applied to **both** `dev/sqp` and `dev/scvx`, which carry the
affected code verbatim.

### 1. Convergence ignored most constraints

`methods/common/convergence.py`

`bool_feas1` was built from five constraint categories named by hand —
`dynamics`, `final_state`, `*nonconvex_inequality*`, `*nonconvex_equality*`,
`*continuity*`. Any other constraint carrying a virtual buffer was never checked,
and `initial_state` is one of them, so a solve could report convergence while
starting 100 km from its stated initial condition.

It did exactly that here. An apparently excellent result — perigee `+275 km`,
eccentricity `0.0021`, both comfortably inside the requirement — collapsed to
`+32 km` once the constraint was actually enforced. The whole transfer had been
bought by starting above GEO.

```python
- bool_feas1 = bool_term and bool_vb_ineq and bool_vb_eq and bool_vb_dyn and bool_cont
+ bool_feas1 = all(cnstr.is_feasible for cnstr in constraints)
```

The enumeration was replaced rather than extended with a sixth name: naming
kinds is what let the gap open. A `boundary` entry was added to the `chk` report
so the residual is visible in the iteration table.

### 2. The pseudospectral mesh could not move a segment's initial epoch

`methods/dev/{sqp,scvx}/scp_segment.py`

`create_free_final_time_constraints` related interior node times to the
endpoints as

```text
dt[k] = ps_t_offset[k] + tau[k] * dt[N-1]
```

which drops the `(1 - tau[k]) * dt[0]` term the full relation
`t_k = (1 - tau_k) t_0 + tau_k t_{N-1}` needs, and compensated by forcing
`dt[0] == 0` unconditionally in PS mode.

For a single segment that is harmless. In a multi-phase chain it is not: each
segment's start is frozen, time continuity forces its predecessor's end to be
frozen too, and the effect cascades until every interior boundary is pinned at
its initial-guess longitude. Only the last segment's endpoint stays free.

The shadow-terminator equality was then violated by **1.3e-02** — with the
boundary longitude frozen, the only way to satisfy `r(L)|sin L| = R_Earth` is to
bend the orbit radius at that longitude, which is backwards. With the term
restored it is **1.1e-08**, two orders of magnitude better than multiple
shooting manages.

`dt[0] == 0` is now applied only when the segment has no predecessor, matching
what the `ms` branch already did.

### 3. The PS defect was not comparable with the shooting defect

`methods/dev/{sqp,scvx}/scp_constraints/scp_constraint_types.py`, `scp_constraint.py`

Collocation states a residual on `dz/dtau`; multiple shooting states one on the
state change across an interval. For the same trajectory error the two differ by
roughly the interval count — a factor of ~35 at 36 nodes — yet they were tested
against the same `eps`, and shared the same penalty autotune targets.

The Radau quadrature weights were being computed and discarded. They are now
kept: halved onto `tau in [0, 1]` they sum to 1 exactly, like the shooting
interval widths, so weighting the collocation residual by them puts it back into
interval-error units.

This changes what is *reported and tested*, not what is optimised. A new
`residual_scale` / `scaled_vb` pair on `SCPConstraint` (default `1.0`) carries
the weighting into `is_feasible` and `vb_ratio`, which is necessary because
under `flag_conv: 1` the feasibility test *is* the convergence test.

### 4. `flags.eps_state` was dead for `dev.sqp`

`methods/dev/sqp/scp_segment.py`

`autoscvx.yaml` sets `flags.eps_state`, but `dev/sqp` took both `eps_dyn` and
`eps_state` from the dynamics constraint's `penalty_state.eps` and never read
the flag. `dev/scvx` read it correctly.

They answer different questions — how closely the dynamics must hold, versus how
small a step means the iteration has stopped moving — and conflating them means
tightening the defect tolerance also demands an unreachable step. At the
tolerances used here that alone would have sent every run to the iteration cap.

### Not changed, deliberately

**`nondim` is multiplicative only.** `Nondim` builds `d2nd`/`nd2d` as diagonal
scale matrices with no offset, so a state confined to a narrow interval far from
zero cannot be given a scale that resolves its motion. An affine map
`(x - lower) / (upper - lower)` would fix it for every example at once. This one
instead carries `p - r_GEO` so the state is centred on zero — see
[Scaling](#scaling) — which is a change to the example, not the package.

**The terminal cones stay hard.** `scp_final_convex_inequality` has no virtual
buffer while the dynamics do, so an unreachable target surfaces as a dynamics
defect rather than as a terminal violation, which makes diagnosis hard. Giving
the cones buffers was tried and is worse: minimum time then simply violates
them. The asymmetry is doing useful work.
