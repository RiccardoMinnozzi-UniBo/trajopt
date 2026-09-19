# Multi-phase solar-sail GEO disposal

This example implements the first problem of Kelly and Bevilacqua, *"Geostationary
debris mitigation using minimum time solar sail trajectories with eclipse
constraints"* (2021): move a sail-equipped spacecraft from circular GEO to a
graveyard orbit with

- eccentricity no greater than `0.003`, and
- perigee radius at least `r_GEO + 250 km`,

in minimum time, using solar radiation pressure alone.

Only central Earth gravity and SRP are retained. The Sun is fixed in inertial
space and the Earth casts a cylindrical shadow, so there are no dates,
ephemerides, third bodies or geopotential terms. Each sunlit and eclipsed arc is
a separate trajopt segment, tied to its neighbours by state, control and
longitude continuity.

## Running it

```bash
python main.py
```

Edit `config.yaml`, run this, get the minimum-time solution. The revolution count
cannot be optimised inside a single SCP solve — it sets the number of phases — so
`main.py` searches over it on every invocation, starting from the guess in
`problem.revolutions`. Change the sail, the discretisation, the terminal
requirement or anything else and it still lands on the minimum-time solution;
a poor guess only costs extra solves, never the right answer.

Set `problem.revolution_search.enabled: false` to skip the search and solve once
at the guess, which is quicker while iterating on something else.

Both sail sizings validate with **one shared problem definition**; only
`problem.revolutions` differs between them.

These are the minimum-time solutions the search converges on, from any starting
guess:

| | `A/m = 10` (accelerated) | `A/m = 1` (paper sizing) |
| --- | --- | --- |
| `revolutions` (minimum) | 1 | 3 |
| phases | 3 | 7 |
| elapsed time | 1.226 days | 2.673 days |
| final eccentricity | 0.001785 (cap 0.003) | 0.000422 |
| final perigee radius | r_GEO + 250.00 km | r_GEO + 250.00 km |
| shortest phase | positive | 1.157 h |
| max dynamics defect | 2.7e-05 | 2.0e-05 |
| independent re-propagation error | 2.7e-05 | 2.0e-05 |
| SCP iterations | 601 (see below) | 116, converged |
| validated | yes | yes |

The perigee requirement is active in both cases — the sail arrives at the
graveyard orbit with nothing to spare, which is what a minimum-time solution
should look like.

`A/m = 1` is the only difference between the two columns: set
`area_to_mass_m2_kg: 1.0` and let the search find the count.

The `A/m = 10` case passes every validation check, including an independent
re-integration of the dynamics, but does not set trajopt's own convergence flag:
the dynamics defect settles well inside tolerance while the state-step criterion
keeps oscillating, so it runs to the iteration cap. The returned trajectory is
sound; it is the stopping rule that does not trigger.

At `A/m = 10` the minimum is `revolutions = 1`, which is the floor of
`revolution_search.minimum` — so the search reports it as the smallest *tried*
rather than the smallest possible, since nothing below it was shown to fail.

Results land in `results/`. Figures are always written — an unconverged run is
exactly when you want to look at the trajectory — while the rotating GIF is only
produced for a solution that passes validation.

To redraw a saved solution without re-solving:

```python
from visualization import render_saved_solution
render_saved_solution("results/solution.npz", "results")
```

### Discretisation: multiple shooting or pseudospectral

```bash
python main.py                  # RK4 multiple shooting (config.yaml)
python main.py config-ps.yaml   # flipped Legendre-Radau collocation
```

`config-ps.yaml` inherits `config.yaml` and overrides only what genuinely belongs
to the discretisation, so the two cannot drift apart. The same `main.py` and the
same revolution search drive both, and both reach a validated solution:

| | multiple shooting | pseudospectral |
| --- | --- | --- |
| revolutions found | 3 | 4 |
| elapsed time | 2.673 days | 4.497 days |
| revolutions flown | 2.66 | 4.47 |
| final perigee | r_GEO + 250.00 km | r_GEO + 250.00 km |
| max dynamics defect | 2.0e-05 | 1.6e-05 |
| SCP iterations | 116 | 160 |
| validated | yes | yes |

The two find substantially different trajectories, which is expected: multiple
shooting holds the control first-order between nodes, while collocation carries
it at Radau nodes with no such assumption. Part of the gap is also structural —
see observation 3 below, which freezes the interior phase boundaries under PS and
so removes freedom the shooting formulation has. `results/` and `results-ps/`
keep the two sets of outputs side by side.

The **problem definition is identical** for both: `config-ps.yaml` has no
`problem:` block at all, so the shadow-terminator treatment and everything else
is inherited verbatim. There is exactly one place where the physics is written
down, and the two configurations cannot disagree about it.

That does cost the pseudospectral run something, and it is worth being explicit
about why. Measured at N = 3, with the terminator equality on and off:

| | terminator on | terminator off |
| --- | --- | --- |
| multiple shooting | feasible, 2.673 days | infeasible, defect 2.0e-04 |
| pseudospectral | infeasible, defect 3.0e-03 | infeasible, defect 6.4e-04 |

Shooting needs the equality at its minimum count. PS fails N = 3 either way, so
its higher revolution count is a property of the discretisation rather than of
this setting — but it does carry a larger residual on the equality, for the
reason in observation 3: with the boundary longitude frozen, the only way left
to satisfy `r(L) |sin L| = R_Earth` is to bend the orbit radius at that
longitude, which is backwards. The geometry should locate the boundary, not
deform the trajectory. Restoring the missing term in trajopt's PS mesh relation
would remove the compromise.

`hp_segments: 1` in the PS config is deliberate: it is the only value that places
no divisibility constraint on the node counts, which is what lets both methods
share them.

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
brackets the minimum from wherever the starting guess lands:

- guess **feasible** → step down until a count fails. The last feasible count is
  the minimum, and that failure is what proves it.
- guess **infeasible** → step up until a count succeeds. The first feasible count
  is the minimum.

Both rest on the same monotonicity: more revolutions means more time to reach the
graveyard orbit, so feasibility, once gained, is kept. A good guess costs one
extra solve to confirm minimality; a poor one just costs more steps.

If the walk runs off the bottom of `revolution_search.minimum` without a failure,
the search says so — the true minimum may be lower, and the bound is what stopped
it.

Each attempt owns a compiled CVXPY subproblem that can run to gigabytes, so the
search keeps only the best one alive — retaining them all exhausts memory after a
handful of solves.

### Revolution counts for the two sail sizings

Everything in the problem definition is shared between sail sizings; only
`revolutions` changes, and the elapsed-time bound is derived from it rather than
written out so that stays true.

| `area_to_mass_m2_kg` | `revolutions` | phases | transfer |
| --- | --- | --- | --- |
| 10 (accelerated demonstration) | 2 | 5 | ≈1.55 days |
| 1 (paper sizing) | 4 | 9 | ≈3.54 days |

For any other sail, run the search once and write the count it reports into
`problem.revolutions`.

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
(`src/trajopt/nondim.py`), so the amplitude of an Eq. 44 interval is the correct
scale only for a state centred on zero — `f`, `g`, `h`, `k` and the elapsed time.
The semi-latus rectum is not: it sits at ~42164 km inside an interval only a few
hundred km wide, so dividing by the amplitude leaves `p_nd ≈ 60`. Every absolute
nondimensional quantity in the solver — trust region, penalty weights,
convergence tolerances — is then measured against a variable sixty times larger
than the motion being resolved, and the dynamics defect stalls two orders of
magnitude above tolerance no matter how the penalties are tuned. Intervals that
do not contain zero are therefore scaled by their magnitude instead.

This is the single change that made the problem converge: with it the solve
drops from hitting the iteration cap at a defect of ~3e-3 to converging in a few
dozen iterations at ~7e-5.

The bounds in `state_bounds` must *bracket the whole trajectory*, not just its
endpoints. The sail pumps eccentricity on the way out and the optimiser then
brings it back under the graveyard cap, so the intermediate excursion in `f` and
`g` is several times the terminal value. A bound that clips it does not show up
as a visibly active constraint; it shows up as a dynamics defect. The validation
report prints the peak values so you can tighten them deliberately.

### Initial guess

The guess propagates the true nonlinear dynamics through each arc in turn and
hands its endpoint to the next arc (`guess.x_start: "previous"` with
`type: propagation`), so it is a genuine continuous trajectory with essentially
zero dynamics defect, rather than a straight line. The steering angle is ramped
between the paper's Equation (53)-(54) endpoint primers, radial at the start and
transverse at the end.

## Verification

`solution.py` re-integrates every node interval from the optimiser's own node
values with a finer RK4 (16 substeps against the solver's 6) and compares with
the next node. This is independent of the solver's discretisation and is the
sharpest available check that the returned trajectory is real. It is reported
nondimensionally alongside a per-state defect breakdown, so a failure tells you
*which* state is responsible.

## Observations about trajopt

Found while building this example. Nothing in the package was modified; these
are recorded so they can be judged on their own merits.

1. **`nondim` is multiplicative only.** `Nondim` builds `d2nd`/`nd2d` as diagonal
   scale matrices with no offset, so a state confined to a narrow interval far
   from zero cannot be scaled by that interval's amplitude. An affine map
   `(x - lower) / (upper - lower)` would let bound-derived scaling be used
   directly, which is what the reference's Eq. 44 assumes. See the Scaling
   section above for what this cost here.

2. **`final_time` bounds assume the segment starts at zero.** In
   `constraint_types.final_time`, `dt_min = lower / (N - 1)` and
   `dt_max = upper / (N - 1)`, where `lower`/`upper` bound the *absolute* final
   time. `scp_final_time` then applies those as per-interval spacing bounds. For
   the first segment of a trajectory that starts at t = 0 this is right, but for
   any later segment in a multi-phase chain it is not: a segment spanning
   L ∈ [94.4, 100.7] with 26 nodes gets a per-interval bound of ~[3.8, 4.0] rad
   against a true spacing of ~0.24 rad, which is infeasible. Using the segment
   *duration* rather than the absolute final time for the `dt` bounds would fix
   it. This example works around it by bounding arc endpoints with a
   `final_nonconvex_inequality` on the longitude instead.

3. **The pseudospectral mesh cannot move a segment's initial epoch.**
   `scp_segment.create_free_final_time_constraints` relates the interior node
   times to the endpoints as `dt[k] = ps_t_offset[k] + tau[k] * dt[N-1]`, which
   drops the `(1 - tau[k]) * dt[0]` term the full relation needs. It compensates
   by forcing `dt[0] == 0` unconditionally in PS mode. For a single segment that
   is fine. In a multi-phase chain it is not: each segment's start is frozen, so
   time continuity forces its predecessor's end to be frozen too, and the effect
   cascades until every interior boundary is pinned at its initial-guess value.
   Only the last segment's endpoint stays free. Restoring the missing term would
   let PS handle a free initial epoch and lift the restriction.

4. **The PS branch ignores `zoh_dilation`.** The `ms` branch applies
   `s_k == s_{k+1}` only when the flag is set; the `ps` branch always applies it.

5. **`hp_segments` is a method-wide flag but a per-segment constraint.** It must
   divide `num_nodes - 1` for *every* segment, so a multi-phase problem with
   different node counts per phase type can only use common divisors. The
   shipped `autoscvx-ps.yaml` defaults to `hp_segments: 10`, which fails for most
   node counts; `1` (one global polynomial per arc) is the only value that never
   constrains the mesh.

6. **`compute_legendre(..., use_spartan=False)` returns NaN at `x = ±1`.** The
   scipy branch evaluates `N (x P_n - P_{n-1}) / (x^2 - 1)`, which is 0/0 at the
   endpoints — and `±1` are always in the node set. Latent only because
   `USE_SPARTAN = True` is the default; the recursive branch is correct there.

7. **`flags.eps_state` appears to be dead for `dev.sqp`.** `autoscvx.yaml` sets
   `flags.eps_state: [0.001]`, but `dev/sqp/scp_segment.py` takes both `eps_dyn`
   and `eps_state` from the dynamics constraint's `penalty_state.eps` (i.e.
   `penalty.default.eps`, 1e-4) and never reads the flag. `dev/scvx` does read
   it. Worth either honouring it in `dev.sqp` or dropping it from the config.

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
