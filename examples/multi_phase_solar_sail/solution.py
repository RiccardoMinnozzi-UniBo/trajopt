"""Extract, verify, save and reload a solar-sail solution."""

from __future__ import annotations

import functools
from dataclasses import dataclass, field
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from trajopt.methods.common import integrators, pseudospectral
from trajopt.utils.tools import recursive_attrdict

import solar_sail

_SECONDS_PER_HOUR = 3600.0
_SECONDS_PER_DAY = 86400.0


@dataclass
class SolarSailSolution:
    """Dimensional trajectory data, ready for analysis or plotting."""

    longitude_rad: np.ndarray
    state: np.ndarray            # (n, 6) = p, f, g, h, k, elapsed_time
    steering_angle_rad: np.ndarray
    cone_angle_rad: np.ndarray
    position_km: np.ndarray
    illuminated: np.ndarray
    phase_index: np.ndarray

    phase_start_rad: np.ndarray
    phase_stop_rad: np.ndarray
    phase_sunlit: np.ndarray
    phase_duration_h: np.ndarray

    earth_radius_km: float
    geo_radius_km: float
    target_perigee_km: float
    target_eccentricity: float
    revolutions: int
    area_to_mass_m2_kg: float
    solar_pressure_n_m2: float
    reflectivity: float
    sun_direction_eci: np.ndarray

    solver_converged: bool
    solver_status: str
    iterations: int
    max_state_jump: float
    max_longitude_jump_rad: float
    max_dynamics_defect_nd: float
    max_propagation_error_nd: float

    #: Tolerance the solve was actually held to, and the budget it came from.
    defect_tolerance_nd: float = 1.0e-4
    defect_budget_km: float = np.inf

    #: End-to-end check: the trajectory re-integrated from the true initial
    #: state, never resetting to the optimizer's nodes.  ``accumulated_*`` is
    #: what the solver claims minus what the dynamics actually deliver -- the
    #: orbit raise that came from defect rather than from the sail.
    true_final_perigee_km: float = np.nan
    true_final_eccentricity: float = np.nan
    accumulated_defect_km: float = np.inf

    defect_by_state: np.ndarray = field(default_factory=lambda: np.zeros(6))
    defect_history: np.ndarray = field(default_factory=lambda: np.empty(0))
    step_history: np.ndarray = field(default_factory=lambda: np.empty(0))
    cost_history: np.ndarray = field(default_factory=lambda: np.empty(0))

    @property
    def elapsed_days(self) -> np.ndarray:
        return self.state[:, 5] / _SECONDS_PER_DAY

    @property
    def eccentricity(self) -> np.ndarray:
        return np.hypot(self.state[:, 1], self.state[:, 2])

    @property
    def semi_latus_km(self) -> np.ndarray:
        """Absolute semi-latus rectum; the state carries it offset from r_GEO."""
        return self.state[:, 0] + self.geo_radius_km

    @property
    def perigee_radius_km(self) -> np.ndarray:
        return self.semi_latus_km / (1.0 + self.eccentricity)

    @property
    def apogee_radius_km(self) -> np.ndarray:
        return self.semi_latus_km / (1.0 - self.eccentricity)

    @property
    def radius_km(self) -> np.ndarray:
        """Orbit radius at each node."""
        return np.linalg.norm(self.position_km, axis=1)

    @property
    def acceleration_mm_s2(self) -> np.ndarray:
        """Sail acceleration magnitude, zero inside eclipse.

        For a sail at cone angle ``alpha`` from the light direction,
        ``|a| = (P A/m) cos(alpha) sqrt((1-eps)^2 + 4 eps cos^2(alpha))``.
        """
        cosine = np.cos(self.cone_angle_rad)
        pressure = self.solar_pressure_n_m2 * self.area_to_mass_m2_kg  # m/s^2
        epsilon = self.reflectivity
        magnitude = pressure * cosine * np.sqrt(
            (1.0 - epsilon) ** 2 + 4.0 * epsilon * cosine**2
        )
        return 1.0e3 * np.where(self.illuminated, magnitude, 0.0)


# ---------------------------------------------------------------------------
# geometry helpers (numpy, vectorised over all nodes)
# ---------------------------------------------------------------------------

def _orbital_frame(state: np.ndarray, longitude: np.ndarray):
    """Radial and transverse unit vectors in ECI, one row per node."""
    h, k = state[:, 3], state[:, 4]
    s_squared = 1.0 + h**2 + k**2
    ex = np.column_stack((1.0 - k**2 + h**2, 2.0 * h * k, -2.0 * k)) / s_squared[:, None]
    ey = np.column_stack((2.0 * h * k, 1.0 + k**2 - h**2, 2.0 * h)) / s_squared[:, None]
    cosine, sine = np.cos(longitude)[:, None], np.sin(longitude)[:, None]
    return cosine * ex + sine * ey, -sine * ex + cosine * ey


def _positions(state: np.ndarray, longitude: np.ndarray,
               geo_radius_km: float) -> np.ndarray:
    radial, _ = _orbital_frame(state, longitude)
    q = 1.0 + state[:, 1] * np.cos(longitude) + state[:, 2] * np.sin(longitude)
    semi_latus = state[:, 0] + geo_radius_km
    return (semi_latus / q)[:, None] * radial


def _cone_angles(state, longitude, steering, sun_direction) -> np.ndarray:
    """Sail cone angle (rad) from the light direction, per node."""
    radial, transverse = _orbital_frame(state, longitude)
    primer = np.cos(steering)[:, None] * radial + np.sin(steering)[:, None] * transverse
    sunward = -np.asarray(sun_direction, float)
    sunward = sunward / np.linalg.norm(sunward)
    cos_gamma = np.clip(primer @ sunward, -1.0 + 1e-12, 1.0 - 1e-12)
    sin_gamma = np.sqrt(np.maximum(0.0, 1.0 - cos_gamma**2))
    return np.arctan2(np.sqrt(cos_gamma**2 + 8.0) - 3.0 * cos_gamma, 4.0 * sin_gamma)


# ---------------------------------------------------------------------------
# independent verification: re-integrate the dynamics with the optimal control
# ---------------------------------------------------------------------------

_SUBSTEPS = 16


@functools.lru_cache(maxsize=4)
def _interval_propagator(illuminated: bool, params_key: tuple):
    """Build (and cache) a jitted RK4 propagator for one illumination state.

    ``params`` is captured in the closure rather than passed as an argument:
    trajopt's AttrDict is not a registered JAX pytree, so it must not cross a
    ``jit`` boundary.
    """
    params = recursive_attrdict(dict(params_key))

    @jax.jit
    def propagate(x, u_start, u_stop, start, stop):
        step = (stop - start) / _SUBSTEPS

        def deriv(state, longitude):
            alpha = (longitude - start) / (stop - start)
            u = jnp.atleast_1d((1.0 - alpha) * u_start + alpha * u_stop)
            return solar_sail.dynamics(state, u, longitude, params, None)

        def stage(carry, _):
            x, L = carry
            k1 = deriv(x, L)
            k2 = deriv(x + 0.5 * step * k1, L + 0.5 * step)
            k3 = deriv(x + 0.5 * step * k2, L + 0.5 * step)
            k4 = deriv(x + step * k3, L + step)
            return (x + (step / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4), L + step), None

        (x_end, _), _ = jax.lax.scan(stage, (x, start), jnp.arange(_SUBSTEPS))
        return x_end

    del illuminated  # already encoded in params_key
    return propagate


def _propagation_error(problem) -> float:
    """Largest node-to-node defect recomputed independently of the solver.

    Every node interval is re-integrated from the optimiser's own node values
    with a finer RK4 (16 substeps against the solver's 6) and compared with the
    next node.  Reported nondimensionally, so it is directly comparable with the
    solver's 1e-4 dynamics tolerance.
    """
    worst = 0.0
    for segment in problem.segments.values():
        data = segment.current_iter_data
        x_nd, L_nd, _, u_nd, _ = segment.index_map.unpack_znu(data.z_opt, data.nu_opt)
        scales = np.asarray(segment.nondim.state_scales)
        state = np.asarray(x_nd) * scales
        longitude = np.asarray(L_nd).reshape(-1) * segment.nondim.time_scale
        control = np.asarray(u_nd) * segment.nondim.control_scales

        params_key = tuple(sorted(
            (key, tuple(value) if isinstance(value, list) else value)
            for key, value in dict(segment.params).items()
        ))
        propagate = _interval_propagator(bool(segment.params.illumination), params_key)

        for node in range(len(longitude) - 1):
            end = propagate(
                jnp.asarray(state[node]),
                control[node, 0], control[node + 1, 0],
                longitude[node], longitude[node + 1],
            )
            error = np.abs(np.asarray(end) - state[node + 1]) / scales
            worst = max(worst, float(np.max(error)))
    return worst


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------

def phase_timing_report(problem, title: str) -> None:
    """Print a dimensional per-phase breakdown of the current iterate."""
    rows = []
    for number, (name, segment) in enumerate(problem.segments.items(), start=1):
        data = segment.current_iter_data
        x_nd, L_nd, _, _, _ = segment.index_map.unpack_znu(data.z_opt, data.nu_opt)
        state = np.asarray(x_nd) * segment.nondim.state_scales
        longitude = np.asarray(L_nd).reshape(-1) * segment.nondim.time_scale
        rows.append({
            "number": number,
            "mode": "sunlit" if bool(segment.params.illumination) else "eclipse",
            "arc_deg": float(np.rad2deg(longitude[-1] - longitude[0])),
            "hours": float((state[-1, 5] - state[0, 5]) / _SECONDS_PER_HOUR),
            "end_day": float(state[-1, 5] / _SECONDS_PER_DAY),
        })

    total = sum(row["hours"] for row in rows)
    eclipse = [row for row in rows if row["mode"] == "eclipse"]

    print(f"\n{title} -- phase timing")
    print("   #  mode      arc [deg]  duration [h]  share [%]  end [day]")
    for row in rows:
        share = 100.0 * row["hours"] / total if total else np.nan
        print(f"  {row['number']:3d}  {row['mode']:<8}{row['arc_deg']:10.3f}"
              f"{row['hours']:14.4f}{share:11.3f}{row['end_day']:11.5f}")

    eclipse_hours = sum(row["hours"] for row in eclipse)
    print(f"  total {total / 24.0:.5f} days;  eclipse {eclipse_hours:.3f} h "
          f"({100.0 * eclipse_hours / total if total else np.nan:.3f} %)")
    negative = [row for row in rows if row["hours"] <= 0.0]
    if negative:
        print(f"  WARNING: {len(negative)} phase(s) with non-positive duration: "
              f"{[row['number'] for row in negative]}")


_STATE_NAMES = ("dp", "f", "g", "h", "k", "t_elapsed")


def defect_breakdown(problem) -> list[tuple[str, str, float]]:
    """Worst dynamics defect per state, with the segment it occurs in.

    Returned nondimensionally, so the numbers are directly comparable with the
    solver's 1e-4 tolerance.  Knowing *which* state fails says a lot: a defect
    in p/f/g means the optimiser is asking the orbit to do more than the sail
    can deliver, whereas one in t_elapsed points at the objective fighting the
    dynamics.
    """
    worst = {name: (0.0, "-") for name in _STATE_NAMES}
    for name, segment in problem.segments.items():
        defect = segment.current_iter_data.get("defect", None)
        if defect is None:
            continue
        state_defect = np.abs(np.asarray(defect))[:, segment.index_map.indices.z.state]
        for index, state_name in enumerate(_STATE_NAMES):
            value = float(np.max(state_defect[:, index]))
            if value > worst[state_name][0]:
                worst[state_name] = (value, name)
    return [(state, segment, value)
            for state, (value, segment) in worst.items()]


def _chained_propagation(problem):
    """Re-integrate the whole trajectory from the true initial state.

    ``_propagation_error`` restarts at every node, so per-interval defects never
    add up and it cannot see a trajectory that drifts.  Here each phase is
    propagated from where the *previous* phase actually ended -- never resetting
    to the optimizer's nodes -- with the control model belonging to the
    discretization in use (Lagrange through the Radau nodes for collocation,
    first-order hold for multiple shooting).

    This is the check that decides whether the answer is real: it answers "is
    this a trajectory the modelled sail could fly", which per-interval defects
    do not.

    Returns ``(true_perigee_km, true_eccentricity, accumulated_defect_km)``.
    """
    flags = problem.config.method.flags
    discretize = flags.get("discretize", "ms")
    hp = int(flags.get("hp_segments", 1))

    z_end = None
    for segment in problem.segments.values():
        data = segment.current_iter_data
        z_opt = np.asarray(data.z_opt)
        nu_opt = np.asarray(data.nu_opt)
        n_nodes = z_opt.shape[0]

        dynamics = next(c for c in segment.segment.constraints.values()
                        if c.type == "dynamics").fcn_znu

        if discretize == "ps":
            operator = (pseudospectral.flipped_radau_hp_operator(n_nodes - 1, hp) if hp > 1
                        else pseudospectral.flipped_radau_differential_operator(n_nodes - 1))
            tau = (operator[1] + 1.0) / 2.0
        else:
            tau = np.linspace(0.0, 1.0, n_nodes)

        z_start = z_opt[0] if z_end is None else z_end
        _, z_dense, _ = integrators.propagate_trajectory(
            np.vstack([z_start, z_opt[1:]]), tau, nu_opt, dynamics, segment.params,
            discretize=discretize, n_steps=2000, hp_segments=hp,
        )
        z_end = z_dense[-1]

    last = list(problem.segments.values())[-1]
    scales = np.asarray(last.nondim.state_scales)
    idx = last.index_map.indices.z.state
    claimed = np.asarray(last.current_iter_data.z_opt[-1, idx]) * scales
    true = np.asarray(z_end[idx]) * scales

    # the state carries p - r_GEO, so restore the absolute value before
    # forming the perigee radius
    geo = float(list(problem.segments.values())[0].params.geo_radius_km)
    e_true = float(np.hypot(true[1], true[2]))
    return (float((true[0] + geo) / (1.0 + e_true)), e_true,
            float(claimed[0] - true[0]))


def _solver_history(problem):
    """Per-iteration max defect, max state step and cost, across all segments."""
    segments = list(problem.segments.values())
    count = min(len(segment.iter_data_list) for segment in segments)
    defect, step, cost = [], [], []
    for index in range(count):
        entries = [segment.iter_data_list[index] for segment in segments]
        # Early iterates carry no defect yet; record NaN so the plot breaks the
        # line there instead of drawing a spurious zero.
        values = [float(np.max(np.abs(entry.defect)))
                  for entry in entries if entry.get("defect", None) is not None]
        defect.append(max(values) if values else np.nan)
        checks = [float(entry.chk.dz) for entry in entries
                  if entry.get("chk", None) is not None]
        step.append(max(checks) if checks else np.nan)
        cost.append(sum(float(entry.get("cost", 0.0)) for entry in entries))
    return np.array(defect), np.array(step), np.array(cost)


def extract_solution(problem) -> SolarSailSolution:
    """Collect dimensional node values from every phase into one record."""
    states, longitudes, controls = [], [], []
    illuminated, phase_indices = [], []
    state_jumps, longitude_jumps = [], []
    phase_start, phase_stop, phase_sunlit, phase_hours = [], [], [], []
    defects = []
    previous_state = previous_longitude = None
    iterations = 0

    for index, segment in enumerate(problem.segments.values()):
        data = segment.current_iter_data
        x_nd, L_nd, _, u_nd, _ = segment.index_map.unpack_znu(data.z_opt, data.nu_opt)
        state = np.asarray(x_nd) * segment.nondim.state_scales
        longitude = np.asarray(L_nd).reshape(-1) * segment.nondim.time_scale
        control = np.asarray(u_nd) * segment.nondim.control_scales

        phase_start.append(longitude[0])
        phase_stop.append(longitude[-1])
        phase_sunlit.append(bool(segment.params.illumination))
        phase_hours.append((state[-1, 5] - state[0, 5]) / _SECONDS_PER_HOUR)

        trimmed_state, trimmed_longitude, trimmed_control = state, longitude, control
        if previous_state is not None:
            state_jumps.append(float(np.max(np.abs(state[0] - previous_state))))
            longitude_jumps.append(float(abs(longitude[0] - previous_longitude)))
            trimmed_state = state[1:]
            trimmed_longitude = longitude[1:]
            trimmed_control = control[1:]

        states.append(trimmed_state)
        longitudes.append(trimmed_longitude)
        controls.append(trimmed_control[:, 0])
        illuminated.append(np.full(len(trimmed_longitude), bool(segment.params.illumination)))
        phase_indices.append(np.full(len(trimmed_longitude), index, dtype=int))

        previous_state, previous_longitude = state[-1], longitude[-1]
        iterations = max(iterations, int(data.iter_num))
        if data.get("defect", None) is not None:
            defects.append(float(np.max(np.abs(data.defect))))

    state = np.concatenate(states)
    longitude = np.concatenate(longitudes)
    steering = np.concatenate(controls)
    settings = problem.settings
    sun_direction = np.asarray(settings.sun_direction_eci, dtype=float)
    defect_history, step_history, cost_history = _solver_history(problem)
    true_perigee, true_eccentricity, accumulated = _chained_propagation(problem)
    eps_nd = float(np.atleast_1d(
        list(problem.segments.values())[0].constraints.dynamics.penalty_state.eps)[0])

    return SolarSailSolution(
        longitude_rad=longitude,
        state=state,
        steering_angle_rad=steering,
        cone_angle_rad=_cone_angles(state, longitude, steering, sun_direction),
        position_km=_positions(state, longitude,
                               float(settings.constants.geo_radius_km)),
        illuminated=np.concatenate(illuminated),
        phase_index=np.concatenate(phase_indices),
        phase_start_rad=np.array(phase_start),
        phase_stop_rad=np.array(phase_stop),
        phase_sunlit=np.array(phase_sunlit),
        phase_duration_h=np.array(phase_hours),
        earth_radius_km=float(settings.constants.earth_radius_km),
        geo_radius_km=float(settings.constants.geo_radius_km),
        target_perigee_km=float(
            settings.constants.geo_radius_km + settings.graveyard.perigee_altitude_km
        ),
        target_eccentricity=float(settings.graveyard.eccentricity_max),
        revolutions=int(settings.revolutions),
        area_to_mass_m2_kg=float(settings.sail.area_to_mass_m2_kg),
        solar_pressure_n_m2=float(settings.sail.solar_pressure_n_m2),
        reflectivity=float(settings.sail.reflectivity),
        sun_direction_eci=sun_direction,
        solver_converged=bool(problem.method._converged),
        solver_status=str(problem.method.cp_subproblem.status),
        iterations=iterations,
        max_state_jump=max(state_jumps, default=0.0),
        max_longitude_jump_rad=max(longitude_jumps, default=0.0),
        max_dynamics_defect_nd=max(defects, default=np.inf),
        max_propagation_error_nd=_propagation_error(problem),
        defect_by_state=np.array([value for _, _, value in defect_breakdown(problem)]),
        defect_history=defect_history,
        step_history=step_history,
        cost_history=cost_history,
        defect_tolerance_nd=eps_nd,
        defect_budget_km=float(settings.max_accumulated_defect_km),
        true_final_perigee_km=true_perigee,
        true_final_eccentricity=true_eccentricity,
        accumulated_defect_km=accumulated,
    )


def validation_report(solution: SolarSailSolution) -> dict:
    """Independent checks on the returned trajectory.

    The graveyard requirement is judged on the *propagated* orbit, not on the
    node values the optimizer reports.  Those differ by the accumulated defect,
    and on this problem that difference can be most of the transfer: a solve
    whose nodes sit exactly on the target can propagate to an orbit that never
    reaches it.  Asking only whether the nodes satisfy the constraint is asking
    the optimizer to mark its own work.
    """
    final_perigee = float(solution.perigee_radius_km[-1])
    final_eccentricity = float(solution.eccentricity[-1])
    shortest_phase = float(np.min(solution.phase_duration_h))

    # Small slack absorbs the solver's own terminal tolerance.
    perigee_met = final_perigee >= solution.target_perigee_km - 0.1
    eccentricity_met = final_eccentricity <= solution.target_eccentricity + 1.0e-6
    graveyard_met = perigee_met and eccentricity_met

    # the same two tests, on the trajectory the dynamics actually produce
    true_perigee_met = solution.true_final_perigee_km >= solution.target_perigee_km - 0.1
    true_eccentricity_met = (solution.true_final_eccentricity
                             <= solution.target_eccentricity + 1.0e-6)
    true_graveyard_met = true_perigee_met and true_eccentricity_met

    within_budget = abs(solution.accumulated_defect_km) <= solution.defect_budget_km
    phases_positive = shortest_phase > 0.0

    feasible = (
        solution.solver_converged
        and true_graveyard_met
        and within_budget
        and phases_positive
        and solution.max_state_jump <= 1.0e-4
        and solution.max_longitude_jump_rad <= 1.0e-8
        and solution.max_dynamics_defect_nd <= solution.defect_tolerance_nd
        and solution.max_propagation_error_nd <= 1.0e-3
    )
    return {
        "true_final_perigee_km": solution.true_final_perigee_km,
        "true_final_eccentricity": solution.true_final_eccentricity,
        "true_graveyard_reached": true_graveyard_met,
        "accumulated_defect_km": solution.accumulated_defect_km,
        "defect_budget_km": solution.defect_budget_km,
        "within_defect_budget": within_budget,
        "defect_tolerance_nd": solution.defect_tolerance_nd,
        "revolutions": solution.revolutions,
        "solver_converged": solution.solver_converged,
        "solver_status": solution.solver_status,
        "iterations": solution.iterations,
        "elapsed_days": float(solution.elapsed_days[-1]),
        "swept_longitude_rev": float(
            (solution.longitude_rad[-1] - solution.longitude_rad[0]) / (2.0 * np.pi)
        ),
        "final_eccentricity": final_eccentricity,
        "target_eccentricity": solution.target_eccentricity,
        "eccentricity_met": eccentricity_met,
        "final_perigee_km": final_perigee,
        "target_perigee_km": solution.target_perigee_km,
        "perigee_met": perigee_met,
        "graveyard_reached": graveyard_met,
        "shortest_phase_h": shortest_phase,
        "all_phase_durations_positive": phases_positive,
        "max_state_jump": solution.max_state_jump,
        "max_longitude_jump_rad": solution.max_longitude_jump_rad,
        "max_dynamics_defect_nd": solution.max_dynamics_defect_nd,
        "max_propagation_error_nd": solution.max_propagation_error_nd,
        "defect_by_state": dict(zip(_STATE_NAMES, solution.defect_by_state.tolist())),
        "peak_semi_latus_offset_km": float(
            np.max(np.abs(solution.state[:, 0]))
        ),
        "peak_eccentricity_component": float(np.max(np.abs(solution.state[:, 1:3]))),
        "peak_eccentricity": float(np.max(solution.eccentricity)),
        "numerically_feasible": feasible,
    }


def print_validation_report(report: dict) -> None:
    """Print the validation dictionary as an aligned, readable block."""
    print("\nvalidation")
    print(f"  revolutions (guess)       : {report['revolutions']}"
          f"   flown: {report['swept_longitude_rev']:.4f}")
    print(f"  solver converged          : {report['solver_converged']}"
          f"  ({report['solver_status']}, {report['iterations']} iterations)")
    print(f"  elapsed time              : {report['elapsed_days']:.5f} days")
    print(f"  final eccentricity        : {report['final_eccentricity']:.6f}"
          f"  (target <= {report['target_eccentricity']:.6f})  "
          f"met: {report['eccentricity_met']}")
    print(f"  final perigee radius      : {report['final_perigee_km']:.3f} km"
          f"  (target >= {report['target_perigee_km']:.3f})  "
          f"met: {report['perigee_met']}")
    print(f"  graveyard reached (nodes) : {report['graveyard_reached']}")
    print(f"  --- propagated from the true initial state ---")
    print(f"  true final perigee radius : {report['true_final_perigee_km']:.3f} km"
          f"  (target >= {report['target_perigee_km']:.3f})"
          f"  shortfall {report['target_perigee_km'] - report['true_final_perigee_km']:+.3f} km")
    print(f"  true final eccentricity   : {report['true_final_eccentricity']:.6f}")
    print(f"  TRUE graveyard reached    : {report['true_graveyard_reached']}")
    print(f"  accumulated defect        : {report['accumulated_defect_km']:.3f} km"
          f"  (budget {report['defect_budget_km']:.1f})"
          f"  within: {report['within_defect_budget']}")
    print(f"  per-interval tolerance    : {report['defect_tolerance_nd']:.3e}")
    print(f"  shortest phase            : {report['shortest_phase_h']:.4f} h"
          f"  (all positive: {report['all_phase_durations_positive']})")
    print(f"  max state jump            : {report['max_state_jump']:.3e}")
    print(f"  max longitude jump        : {report['max_longitude_jump_rad']:.3e} rad")
    print(f"  max dynamics defect       : {report['max_dynamics_defect_nd']:.3e}")
    print(f"  max re-propagation error  : {report['max_propagation_error_nd']:.3e}")
    breakdown = "  ".join(f"{name}={value:.1e}"
                          for name, value in report["defect_by_state"].items())
    print(f"  defect by state           : {breakdown}")
    print(f"  peak |p - r_GEO|          : {report['peak_semi_latus_offset_km']:.1f} km")
    print(f"  peak |f|,|g| / peak e     : "
          f"{report['peak_eccentricity_component']:.6f} / "
          f"{report['peak_eccentricity']:.6f}")
    print(f"  NUMERICALLY FEASIBLE      : {report['numerically_feasible']}")


# ---------------------------------------------------------------------------
# persistence
# ---------------------------------------------------------------------------

_ARRAY_FIELDS = (
    "longitude_rad", "state", "steering_angle_rad", "cone_angle_rad", "position_km",
    "illuminated", "phase_index", "phase_start_rad", "phase_stop_rad", "phase_sunlit",
    "phase_duration_h", "sun_direction_eci", "defect_by_state", "defect_history",
    "step_history", "cost_history",
)
_SCALAR_FIELDS = (
    "earth_radius_km", "geo_radius_km", "target_perigee_km", "target_eccentricity",
    "revolutions", "area_to_mass_m2_kg", "solar_pressure_n_m2", "reflectivity",
    "solver_converged", "solver_status", "iterations", "max_state_jump",
    "max_longitude_jump_rad", "max_dynamics_defect_nd", "max_propagation_error_nd",
)


def save_solution(solution: SolarSailSolution, filename: str | Path) -> Path:
    """Save a compact, dependency-free NumPy result file."""
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    payload = {name: getattr(solution, name)
               for name in (*_ARRAY_FIELDS, *_SCALAR_FIELDS)}
    np.savez_compressed(filename, **payload)
    return filename


def load_solution(filename: str | Path) -> SolarSailSolution:
    """Load a result previously written by :func:`save_solution`."""
    casts = {"illuminated": bool, "phase_sunlit": bool, "phase_index": int,
             "solver_converged": bool, "solver_status": str, "iterations": int,
             "revolutions": int}
    with np.load(filename) as data:
        values = {}
        for name in _ARRAY_FIELDS:
            array = data[name]
            values[name] = array.astype(casts[name]) if name in casts else array
        for name in _SCALAR_FIELDS:
            cast = casts.get(name, float)
            values[name] = cast(data[name])
    return SolarSailSolution(**values)
