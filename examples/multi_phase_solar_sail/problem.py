"""Build the alternating sunlit/eclipse segments for the GEO disposal problem.

Problem structure (Kelly and Bevilacqua 2021, Section 5.1)
----------------------------------------------------------
State   x = [p, f, g, h, k, t_elapsed]   (MEOE plus physical time)
Control u = [theta]                      (primer steering angle, rad)
Independent variable: true longitude L (rad), not physical time.

Each revolution crosses the Earth shadow once, so the trajectory is a chain of
alternating sunlit and eclipse arcs.  Every arc is one trajopt segment; the arcs
are tied together by state, control and time continuity.

The role of ``revolutions``
---------------------------
``revolutions`` is a *structural guess*, not a boundary condition.  It fixes how
many phases the problem has and it seeds the initial guess, but no arc endpoint
is held at a constant longitude:

  * every interior arc ends where the spacecraft actually crosses the shadow
    terminator, imposed as the nonconvex equality ``shadow_boundary = 0``, so the
    boundaries move as the orbit evolves;
  * the final arc ends wherever the minimum-time solution wants it to, within
    roughly one revolution either side of the nominal endpoint.

The right value of ``revolutions`` is therefore found by re-solving with one
revolution fewer until the problem stops being feasible, which is the outer loop
of the paper's Algorithm 1.  See :mod:`search`.

A window of ``terminator_window_rad`` around each nominal boundary keeps an arc
from collapsing or inverting.  The nominal eclipse arc is about 0.304 rad, so a
window of 0.05 rad still leaves every arc at least 0.2 rad long.

Simplifications relative to the reference
-----------------------------------------
* Only Earth gravity (no lunar/solar third-body perturbations).
* Fixed inertial Sun direction (no time-varying ephemeris).

Scaling
-------
Every state is scaled by the amplitude of its allowed interval (paper Eq. 44),
and the longitude by the nominal swept longitude, so all nondimensional
variables and their derivatives are O(1).
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np

from trajopt.trajectory import Trajectory
from trajopt.utils import config_loader
from trajopt.utils import tools
from trajopt.utils.tools import AttrDict, recursive_attrdict

_SOURCE = str(Path(__file__).with_name("solar_sail.py").resolve())
_STATE_NAMES = ("p", "f", "g", "h", "k", "elapsed_time")

# Shortest arc we ever allow, as a guard against degenerate phases.
_MIN_ARC_RAD = 0.05


def max_flown_revolutions(settings: AttrDict) -> float:
    """Most revolutions the trajectory can cover, given the final-arc freedom."""
    return float(settings.revolutions) + float(
        settings.final_longitude_freedom_rad
    ) / (2.0 * np.pi)


def _state_box(settings: AttrDict) -> tuple[np.ndarray, np.ndarray]:
    """Lower and upper state bounds, in dimensional units (paper Eq. 44).

    The orbital-element bounds are set by the terminal orbit, which does not
    depend on the sail, so the same numbers serve any area-to-mass ratio.  The
    elapsed-time bound is the one exception: it has to grow with the revolution
    count, so it is derived from it rather than written out.  That keeps the
    revolution count the only thing that changes between sail sizings, and keeps
    the time state nondimensionalised to roughly [0, 1] in every case.
    """
    bounds = settings.state_bounds
    constants = settings.constants

    semi_latus = float(constants.geo_radius_km) + np.asarray(
        bounds.semi_latus_offset_km, dtype=float
    )
    eccentricity = np.asarray(bounds.eccentricity_component, dtype=float)
    inclination = np.asarray(bounds.inclination_component, dtype=float)
    elapsed_upper = (
        float(bounds.days_per_revolution_bound)
        * max_flown_revolutions(settings)
        * float(constants.day_s)
    )

    lower = np.array([semi_latus[0], eccentricity[0], eccentricity[0],
                      inclination[0], inclination[0], 0.0])
    upper = np.array([semi_latus[1], eccentricity[1], eccentricity[1],
                      inclination[1], inclination[1], elapsed_upper])
    return lower, upper


def _state_scales(lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """Scaling factor for each state, from its allowed interval (paper Eq. 44).

    The amplitude of the interval is the right scale only for a state that is
    centred near zero.  trajopt nondimensionalises by division alone, with no
    offset, so for an interval that does not contain zero the amplitude is the
    wrong choice: the semi-latus rectum sits at ~42164 km inside an interval
    only ~700 km wide, and dividing by 700 would leave p_nd ~ 60.  Every
    absolute nondimensional quantity in the solver -- trust region, penalties,
    convergence tolerances -- would then be measured against a variable sixty
    times larger than the motion being resolved, and the dynamics defect stalls
    around 1e-3 as a result.

    So: amplitude for intervals straddling zero (f, g, h, k, elapsed time),
    magnitude for those that do not (p).
    """
    amplitude = upper - lower
    magnitude = np.abs(0.5 * (lower + upper))
    return np.where(lower * upper > 0.0, magnitude, amplitude)


def _initial_state(settings: AttrDict) -> np.ndarray:
    """Circular equatorial GEO: p = r_GEO, f = g = h = k = 0, t = 0."""
    return np.array([float(settings.constants.geo_radius_km), 0.0, 0.0, 0.0, 0.0, 0.0])


def _steering_guess(progress: float, settings: AttrDict) -> float:
    """Steering angle ramped between the paper's Eq. 53-54 endpoint primers
    (radial at the start, transverse at the end)."""
    start = float(settings.guess.steering_angle_start_rad)
    stop = float(settings.guess.steering_angle_stop_rad)
    return (1.0 - progress) * start + progress * stop


def half_shadow_angle(settings: AttrDict) -> float:
    """Half-angle subtended by the cylindrical Earth shadow at GEO."""
    return float(
        np.arcsin(settings.constants.earth_radius_km / settings.constants.geo_radius_km)
    )


def phase_boundaries(settings: AttrDict) -> list[tuple[float, float, bool]]:
    """Nominal ``(longitude_start, longitude_stop, is_sunlit)`` for every arc.

    These are the *structural* boundaries: they set the number of phases and
    seed the initial guess.  The solver is free to move them (see the module
    docstring).  The Sun is fixed along ``-sun_direction_eci``, so the shadow is
    centred on integer multiples of 2*pi and starting at L0 = pi puts the
    spacecraft in full sunlight at the initial time, as in the reference.
    """
    n_rev = int(settings.revolutions)
    if n_rev < 1:
        raise ValueError(f"revolutions must be at least 1, got {n_rev}")

    start = float(settings.initial_longitude_rad)
    stop = start + 2.0 * np.pi * n_rev
    half_shadow = half_shadow_angle(settings)

    crossings = []
    for revolution in range(1, n_rev + 1):
        centre = 2.0 * np.pi * revolution
        crossings.extend((centre - half_shadow, centre + half_shadow))
    crossings = [L for L in crossings if start + _MIN_ARC_RAD < L < stop - _MIN_ARC_RAD]

    edges = [start, *crossings, stop]
    phases = []
    for arc_start, arc_stop in zip(edges[:-1], edges[1:]):
        midpoint = 0.5 * (arc_start + arc_stop)
        offset = abs(((midpoint + np.pi) % (2.0 * np.pi)) - np.pi)
        phases.append((arc_start, arc_stop, offset > half_shadow))
    return phases


def _build_segment(settings: AttrDict, sunlit: bool) -> AttrDict:
    """Return the parts of a segment configuration that every arc shares."""
    constants = settings.constants
    lower, upper = _state_box(settings)
    scales = _state_scales(lower, upper)

    params = {
        **dict(constants),
        **dict(settings.sail),
        "sun_direction_eci": list(settings.sun_direction_eci),
        # 1 in sunlight, 0 in eclipse; multiplies the whole sail acceleration.
        "illumination": 1.0 if sunlit else 0.0,
        "final_eccentricity_max": float(settings.graveyard.eccentricity_max),
        "final_perigee_radius_km": float(
            constants.geo_radius_km + settings.graveyard.perigee_altitude_km
        ),
    }

    return recursive_attrdict({
        "num_nodes": int(settings.sunlit_nodes if sunlit else settings.eclipse_nodes),
        "fcns": {
            "dynamics": f"{_SOURCE}:dynamics",
            "shadow_boundary": f"{_SOURCE}:shadow_boundary",
            "phase_end_longitude": f"{_SOURCE}:phase_end_longitude",
            "terminal_eccentricity_cone": f"{_SOURCE}:terminal_eccentricity_cone",
            "terminal_perigee_cone": f"{_SOURCE}:terminal_perigee_cone",
        },
        "params": params,
        "state": {
            name: {"idx": [index], "scale": float(scales[index])}
            for index, name in enumerate(_STATE_NAMES)
        },
        "control": {"steering_angle": {"idx": [0], "scale": 1.0}},
        "time": {"scale": 2.0 * np.pi * float(settings.revolutions)},
        "constraints": {
            "dynamics": {"type": "dynamics", "fcn": "fcns.dynamics"},
            "state_limits": {
                "type": "state_limits",
                "lower": lower.tolist(),
                "upper": upper.tolist(),
            },
            # A generous box on the steering angle; it only stops the angle from
            # winding away by multiples of 2*pi, which would be a null direction.
            "steering_limits": {
                "type": "control_limits",
                "lower": [-2.0 * np.pi],
                "upper": [2.0 * np.pi],
            },
            # A sail cannot slew instantaneously.  Without this the minimum-time
            # optimum is bang-bang: the primer flips by ~180 deg between adjacent
            # nodes, which no mesh can represent, the SCP linearisation stays
            # poor and the dynamics defect only falls like 1/N.  The limit is on
            # d(theta)/dL, steering radians per radian of orbital travel.
            "steering_rate_limit": {
                "type": "control_rate_limit",
                "value": [float(settings.max_steering_rate_rad_per_rad)],
            },
        },
        "costs": {},
        "outputs": {},
        "guess": {"type": "propagation"},
    })


def _add_minimum_time_cost(segment: AttrDict, settings: AttrDict) -> None:
    """Attach the minimum-time objective to the final segment.

    Two forms, because they fail in opposite ways:

    ``longitude``
        trajopt's ``min_time``, acting on the independent variable.  It cannot
        be gamed -- the longitude grid carries no virtual-buffer slack -- but it
        is *indifferent* to how far the orbit is over-raised, because raising
        the orbit past the requirement costs no extra longitude.  That leaves a
        flat set of near-optimal solutions for the solver to drift within.

    ``elapsed_time``
        a cost on the elapsed-time state.  Not indifferent: a higher orbit has a
        longer period, so over-raising costs time directly.  Historically this
        form was reducible by pushing elapsed time *backwards* through the
        dynamics' virtual buffer, which produced negative-duration phases; that
        was a symptom of badly scaled states and does not occur once the scaling
        is right, but it is the riskier of the two.

    ``both``
        the ungameable longitude cost, plus the elapsed-time cost at a small
        weight purely to break the tie within the flat set.
    """
    form = str(settings.minimum_time_objective)
    weight = float(settings.elapsed_time_cost_weight)

    if form in ("longitude", "both"):
        segment.costs.minimum_longitude = recursive_attrdict({"type": "min_time"})
    if form in ("elapsed_time", "both"):
        segment.costs.minimum_elapsed_time = recursive_attrdict(
            {"type": "final_state", "idx": 5,
             "w": 1.0 if form == "elapsed_time" else weight}
        )
    if form not in ("longitude", "elapsed_time", "both"):
        raise ValueError(
            f"minimum_time_objective must be 'longitude', 'elapsed_time' or "
            f"'both', got '{form}'"
        )


def build_trajectory_config(settings: AttrDict) -> AttrDict:
    """Expand the compact problem settings into the full multi-segment config."""
    phases = phase_boundaries(settings)
    x0 = _initial_state(settings)
    initial_longitude = phases[0][0]
    total_longitude = phases[-1][1] - initial_longitude
    window = float(settings.terminator_window_rad)
    freedom = float(settings.final_longitude_freedom_rad)

    segments = AttrDict()
    previous_name = None

    for index, (arc_start, arc_stop, sunlit) in enumerate(phases, start=1):
        name = f"phase_{index:03d}_{'sunlit' if sunlit else 'eclipse'}"
        segment = _build_segment(settings, sunlit)
        is_last = index == len(phases)

        # The guess propagates the true dynamics through each arc in turn and
        # hands its endpoint to the next arc, so it is a genuine trajectory:
        # continuous, and with essentially zero dynamics defect.
        progress_start = (arc_start - initial_longitude) / total_longitude
        progress_stop = (arc_stop - initial_longitude) / total_longitude
        segment.guess.update({
            "t_start": float(arc_start),
            "t_stop": float(arc_stop),
            "x_start": x0.tolist() if previous_name is None else "previous",
            "u_start": [_steering_guess(progress_start, settings)],
            "u_stop": [_steering_guess(progress_stop, settings)],
        })

        if previous_name is None:
            segment.constraints.initial_state = recursive_attrdict(
                {"type": "initial_state", "value": x0.tolist()}
            )
            segment.constraints.initial_longitude = recursive_attrdict(
                {"type": "initial_time", "value": float(arc_start)}
            )
        else:
            segment.constraints.state_continuity = recursive_attrdict(
                {"type": "state_continuity", "segment": previous_name}
            )
            segment.constraints.longitude_continuity = recursive_attrdict(
                {"type": "time_continuity", "segment": previous_name}
            )
            # The sail attitude is physically continuous, and this also pins the
            # steering angle inside eclipse arcs, where it has no effect on the
            # dynamics and would otherwise be a free null direction.
            segment.constraints.control_continuity = recursive_attrdict(
                {"type": "control_continuity", "segment": previous_name}
            )

        if is_last:
            # The trajectory may end anywhere inside the final sunlit arc: this
            # is the freedom the minimum-time cost needs.  The upper bound is
            # capped at the next shadow entry, because past that point the
            # trajectory would fly through an eclipse that has no phase to model
            # it and the sail would keep thrusting in the dark.
            next_shadow_entry = arc_stop + np.pi - half_shadow_angle(settings)
            end_lower = max(arc_start + _MIN_ARC_RAD, arc_stop - freedom)
            end_upper = min(arc_stop + freedom, next_shadow_entry - _MIN_ARC_RAD)
        else:
            # An interior arc ends on the shadow terminator.  The equality below
            # places it physically; the window is only a guard that keeps the arc
            # from collapsing or inverting.
            #
            # The equality can be switched off: over this transfer the orbit
            # radius changes by well under 1 %, so the true terminator moves by
            # less than 0.11 deg and the window alone already places the boundary
            # to far better than the discretisation.  Turning it off removes the
            # problem's only nonconvex equality.
            if bool(settings.enforce_shadow_terminator):
                segment.constraints.ends_on_terminator = recursive_attrdict({
                    "type": "final_nonconvex_equality",
                    "fcn": "fcns.shadow_boundary",
                    "scale": [1.0],
                })
            end_lower, end_upper = arc_stop - window, arc_stop + window

        segment.constraints.arc_end_window = recursive_attrdict({
            "type": "final_nonconvex_inequality",
            "fcn": "fcns.phase_end_longitude",
            "lower": np.array([float(end_lower)]),
            "upper": np.array([float(end_upper)]),
            "scale": [2.0 * np.pi],
        })

        if is_last:
            # Graveyard orbit per the IADC guidelines: near-circular, safely
            # above the geostationary belt.  Both are exact second-order cone
            # constraints, so they enter the CVXPY subproblem directly.
            segment.constraints.final_eccentricity = recursive_attrdict(
                {"type": "final_convex_inequality",
                 "fcn": "fcns.terminal_eccentricity_cone"}
            )
            segment.constraints.final_perigee = recursive_attrdict(
                {"type": "final_convex_inequality",
                 "fcn": "fcns.terminal_perigee_cone"}
            )
            _add_minimum_time_cost(segment, settings)

        segments[name] = segment
        previous_name = name

    return recursive_attrdict({"segments": segments})


class SolarSailProblem:
    """Builds and holds the trajopt method for one value of ``revolutions``.

    Typical use::

        problem = SolarSailProblem(CONFIG_FILE)
        converged = problem.solve()

    To sweep the structural guess without touching the YAML::

        problem = SolarSailProblem(CONFIG_FILE, {"revolutions": 18})

    Parameters
    ----------
    config_path:
        Path to the YAML configuration file.
    problem_overrides:
        Optional ``problem.*`` settings applied after loading the YAML.
    method_overrides:
        Optional ``method.*`` settings, deep-merged after loading the YAML, for
        sweeping solver options without editing the file.
    """

    def __init__(
        self,
        config_path: str | Path,
        problem_overrides: dict | None = None,
        method_overrides: dict | None = None,
    ) -> None:
        self.config_path = Path(config_path).resolve()
        self.config = config_loader.load_trajopt_config(str(self.config_path))
        self.settings = deepcopy(self.config.problem)

        # Deep-merged so a nested override such as
        # {"graveyard": {"perigee_altitude_km": 500.0}} keeps its siblings, and
        # routed through recursive_attrdict so nested values stay attribute-
        # accessible the way the rest of this module expects.
        if problem_overrides:
            self.settings = recursive_attrdict(
                tools.deep_merge(dict(self.settings), problem_overrides)
            )

        if method_overrides:
            self.config.method = recursive_attrdict(
                tools.deep_merge(dict(self.config.method), method_overrides)
            )

        self.config.trajectory = build_trajectory_config(self.settings)
        self.trajectory = Trajectory(self.config.trajectory)
        method_class = config_loader.resolve_scp_method_class(self.config.method)
        self.method = method_class(self.config.method, self.trajectory)

    @property
    def revolutions(self) -> int:
        return int(self.settings.revolutions)

    @property
    def segments(self):
        return self.method.scp_trajectory.scp_segments

    def solve(self) -> bool:
        """Run SCP until convergence or the iteration budget is exhausted."""
        self.method.solve()
        return bool(self.method._converged)
