import sys
from pathlib import Path

import jax
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import BarycentricInterpolator

from trajopt.common.persistence import load_result
from trajopt.common.tools import AttrDict

sys.path.insert(0, "mission")
import solar_sail

jax.config.update("jax_enable_x64", True)

CONFIG = "p_ms"   # a folder in results/

plt.rcParams.update({"font.size": 9, "figure.figsize": (6.5, 5.5), "figure.constrained_layout.use": True,
                     "axes.grid": True, "grid.alpha": 0.3, "lines.linewidth": 1.2, "savefig.dpi": 300})

DAY = 86400.0
folder = Path("results") / CONFIG
stages = {path.stem.split("_", 1)[1]: load_result(path) for path in sorted(folder.glob("[0-9]_*.json"))}
solution = list(stages.values())[-1]
guess = list(stages.values())[0].initial_trajectory
arcs = solution.trajectory
params = {name: phase.config.params for name, phase in solution.problem.phases.items()}
sail = params["final_arc"]
eclipses = [name for name in arcs if not params[name].illumination]

lines = []


def report(text=""):
    print(text)
    lines.append(text)


def save(fig, name):
    fig.savefig(folder / f"{name}.png")
    fig.savefig(folder / f"{name}.pdf")


def orbit(x):
    """Perigee and apogee altitude above GEO [km], and eccentricity."""
    p = x[:, 0] + sail.geo_radius_km
    e = np.hypot(x[:, 1], x[:, 2])
    return p / (1 + e) - sail.geo_radius_km, p / (1 - e) - sail.geo_radius_km, e


def terminator_miss(x, longitude):
    """Distance of an arc end from the edge of the Earth shadow [km]."""
    residual = float(solar_sail.shadow_boundary(x, None, longitude, sail)[0])
    return sail.earth_radius_km * (np.sqrt(1 + residual) - 1)


def steering_law(arc, mesh=None):
    """Steering angle against longitude, as the discretization of the arc defines it (no mesh: straight lines)."""
    nodes = arc.sample(arc.native_tau)
    if mesh is None or mesh["discretize"] == "ms":
        return lambda longitude: np.interp(longitude, nodes.t, nodes.u[:, 0])
    order = (len(nodes.t) - 1) // mesh["segments"]   # one polynomial per segment
    pieces = [BarycentricInterpolator(nodes.t[i:i + order + 1], nodes.u[i:i + order + 1, 0])
              for i in range(0, len(nodes.t) - 1, order)]
    segment = lambda longitude: min((np.searchsorted(nodes.t, longitude, side="right") - 1) // order, len(pieces) - 1)
    return np.vectorize(lambda longitude: pieces[segment(longitude)](longitude))


@jax.jit
def dynamics(x, u, longitude, illumination):
    return solar_sail.dynamics(x, u, longitude, AttrDict(sail, illumination=illumination), None)


def fly(name, steering, x, span):
    """Integrate one arc open loop over a longitude span."""
    return solve_ivp(
        lambda longitude, x: dynamics(x, np.atleast_1d(steering(longitude)), longitude, params[name].illumination),
        span, x, method="DOP853", rtol=1e-10, atol=1e-10, dense_output=True).sol


def flight(trajectory, laws):
    """Fly the steering laws through all arcs, starting from the first node only."""
    flown, x = {}, next(iter(trajectory.values())).x[0]
    for name, arc in trajectory.items():
        flown[name] = fly(name, laws[name], x, arc.t[[0, -1]])
        x = flown[name](arc.t[-1])
    return flown


def node_defects(trajectory, laws):
    """How far the nodes are from the dynamics: every node minus the previous node flown forward, summed
    (semi-latus rectum [km], elapsed time [s])."""
    total = np.zeros(2)
    for name, arc in trajectory.items():
        nodes = arc.sample(arc.native_tau)
        for i in np.flatnonzero(np.diff(nodes.t) > 0):
            end = fly(name, laws[name], nodes.x[i], nodes.t[i:i + 2])(nodes.t[i + 1])
            total += np.abs(end - nodes.x[i + 1])[[0, 5]]
    return total


cone_angle = jax.vmap(lambda x, u, longitude: solar_sail.sail_normal_from_primer(x, u, longitude, sail)[1])


def shade_eclipses(axes):
    for ax in axes:
        for name in eclipses:
            ax.axvspan(*(arcs[name].x[[0, -1], 5] / DAY), color="0.85", linewidth=0)


# ---------------------------------------------------------------- solver performance

def history(stage, pick, combine):
    """One value per iteration of a stage, combined over the arcs."""
    return combine([[pick(step) for step in phase["scalar_history"]] for phase in stage.phase_data.values()], axis=0)


report(f"{CONFIG}: " + " -> ".join(stages))
report("\nSOLVER")
report(f"{'stage':<7}{'status':<15}{'iter':>5}{'nodes':>7}{'total [s]':>11}{'warm-up':>9}"
       f"{'s/iter':>8}{'linearize':>11}{'parse':>8}{'solve':>8}{'transfer [d]':>14}")
for name, stage in stages.items():
    t = stage.timings
    nodes = sum(len(phase["mesh"]["tau"]) for phase in stage.phase_data.values())
    days = stage.trajectory["final_arc"].x[-1, 5] / DAY
    report(f"{name:<7}{stage.status:<15}{stage.iterations:>5}{nodes:>7}{t['total_s']:>11.1f}{t['warmup_s']:>9.1f}"
           f"{(t['total_s'] - t['warmup_s']) / stage.iterations:>8.2f}{t['local_model_s']:>11.1f}"
           f"{t['canonicalization_s']:>8.1f}{t['numerical_s']:>8.1f}{days:>14.4f}")
if len(stages) > 1:
    report(f"{'total':<22}{sum(s.iterations for s in stages.values()):>5}{'':>7}"
           f"{sum(s.timings['total_s'] for s in stages.values()):>11.1f}"
           f"{sum(s.timings['warmup_s'] for s in stages.values()):>9.1f}")
report("(warm-up: compilation, inside total; linearize / parse / solve: local model, problem canonicalization, convex solver)")

final_longitude, cost, penalty = [], [], []
ratios = {"cost change": [], "dynamics buffer": [], "terminator buffer": []}
for stage in stages.values():
    final_longitude.append([step["t_final"] for step in stage.phase_data["final_arc"]["scalar_history"]])
    cost.append(history(stage, lambda step: step["cost"], np.sum))
    penalty.append(history(stage, lambda step: step["penalty_cost"], np.sum))
    for label, key in zip(ratios, ["dcost", "dynamics", "nonconvex_equality"]):
        ratios[label].append(history(stage, lambda step: step["chk"][key], np.max))
final_longitude = np.concatenate(final_longitude)
iterations = 1 + np.arange(len(final_longitude))
stage_ends = np.cumsum([stage.iterations for stage in stages.values()])

fig, (top, middle, bottom) = plt.subplots(3, 1, sharex=True)
top.plot(iterations, final_longitude, "k.-", markersize=3)
middle.plot(iterations, np.concatenate(cost), label="subproblem cost (objective + penalty + proximal)")
middle.plot(iterations, np.concatenate(penalty), label="penalty cost")
middle.set_yscale("log", nonpositive="mask")
middle.legend()
for label, values in ratios.items():
    bottom.semilogy(iterations, np.concatenate(values), label=label)
bottom.axhline(1.0, color="k", linestyle="--", linewidth=0.8)
for start, end, name in zip(np.r_[0, stage_ends[:-1]], stage_ends, stages):
    top.text((start + end) / 2 + 0.5, 1.02, name, transform=top.get_xaxis_transform(), ha="center")
    for ax in (top, middle, bottom):
        ax.axvline(end + 0.5, color="gray", linestyle=":")
top.set_ylabel("final longitude [rad]")
middle.set_ylabel("cost")
bottom.set_ylabel("value / tolerance")
bottom.set_xlabel("iteration")
bottom.legend()
save(fig, "convergence")


# ---------------------------------------------------------------- the initial guess

guess_laws = {name: steering_law(arc) for name, arc in guess.items()}
guess_flown = flight(guess, guess_laws)
guess_end = guess_flown["final_arc"](guess["final_arc"].t[-1])[None]
guess_defects = node_defects(guess, guess_laws)

report("\nINITIAL GUESS")
report(f"{'transfer time [days]':<44}{guess['final_arc'].x[-1, 5] / DAY:>12.4f}")
report(f"{'summed node defect, semi-latus rectum [km]':<44}{guess_defects[0]:>12.4g}")
report(f"{'summed node defect, elapsed time [s]':<44}{guess_defects[1]:>12.4g}")
report(f"{'its steering flown: final perigee [km]':<44}{orbit(guess_end)[0][0]:>12.4f}   "
       f"(target {sail.perigee_altitude_km:g})")
report(f"{'its steering flown: final eccentricity':<44}{orbit(guess_end)[2][0]:>12.4g}   "
       f"(target <= {sail.eccentricity_max:g})")


# ---------------------------------------------------------------- validation: fly the control open loop

laws = {name: steering_law(arc, solution.phase_data[name]["mesh"]) for name, arc in arcs.items()}
flown = flight(arcs, laws)

nodes_x = np.vstack([arc.x for arc in arcs.values()])
flown_x = np.vstack([flown[name](arc.t).T for name, arc in arcs.items()])
interior = list(arcs)[:-1]

rows = {
    "transfer time [days]":       (nodes_x[-1, 5] / DAY, flown_x[-1, 5] / DAY, ""),
    "perigee above GEO [km]":     (orbit(nodes_x)[0][-1], orbit(flown_x)[0][-1], f">= {sail.perigee_altitude_km:g}"),
    "eccentricity":               (orbit(nodes_x)[2][-1], orbit(flown_x)[2][-1], f"<= {sail.eccentricity_max:g}"),
    "worst terminator miss [km]": (max(abs(terminator_miss(arcs[name].x[-1], arcs[name].t[-1])) for name in interior),
                                   max(abs(terminator_miss(flown[name](arcs[name].t[-1]), arcs[name].t[-1]))
                                       for name in interior), ""),
}
report("\nVALIDATION (flown: the solver's steering integrated open loop from the first node)")
report(f"{'':<28}{'solver':>12}{'flown':>12}   required")
for label, (solver, open_loop, required) in rows.items():
    report(f"{label:<28}{solver:>12.7g}{open_loop:>12.7g}   {required}")

defect = max(np.max(np.abs(solution.diagnostics[name]["dynamics_defect_nd"])
                    / solution.phase_data[name]["constraints"]["dynamics"]["penalties"]["eps"]) for name in arcs)
steering_rate = 0.0
for name in arcs:
    longitude = np.linspace(*arcs[name].t[[0, -1]], 400)
    steering_rate = max(steering_rate, np.abs(np.gradient(laws[name](longitude), longitude)).max())
report(f"\n{'largest dynamics defect / tolerance':<44}{defect:>12.3f}")
report(f"{'largest steering rate [rad/rad]':<44}{steering_rate:>12.3f}   "
       f"(limit {sail.max_steering_rate_rad_per_rad:g} at the nodes)")

fig, axes = plt.subplots(3, 1, sharex=True, figsize=(6.5, 4.5))
axes[0].plot(nodes_x[:, 5] / DAY, orbit(flown_x)[0] - orbit(nodes_x)[0], "k")
axes[1].plot(nodes_x[:, 5] / DAY, orbit(flown_x)[2] - orbit(nodes_x)[2], "k")
axes[2].plot(nodes_x[:, 5] / DAY, flown_x[:, 5] - nodes_x[:, 5], "k")
shade_eclipses(axes)
axes[0].set_ylabel("perigee [km]")
axes[1].set_ylabel("eccentricity")
axes[2].set_ylabel("elapsed time [s]")
axes[2].set_xlabel("time [days]")
axes[0].set_title("flown open loop minus solver nodes", fontsize=9)
save(fig, "open_loop_error")


# ---------------------------------------------------------------- the trajectory

fig, (top, middle, bottom) = plt.subplots(3, 1, sharex=True)
for name in arcs:
    longitude = np.linspace(*arcs[name].t[[0, -1]], 50)
    fine = flown[name](longitude).T
    perigee, apogee, eccentricity = orbit(fine)
    top.plot(fine[:, 5] / DAY, perigee, "C0")
    top.plot(fine[:, 5] / DAY, apogee, "C1")
    middle.plot(fine[:, 5] / DAY, eccentricity, "C0")
    if params[name].illumination:
        angle = cone_angle(fine, laws[name](longitude)[:, None], longitude)
        bottom.plot(fine[:, 5] / DAY, np.degrees(angle), "C0")
shade_eclipses((top, middle, bottom))
top.axhline(sail.perigee_altitude_km, color="k", linestyle="--", linewidth=0.8)
middle.axhline(sail.eccentricity_max, color="k", linestyle="--", linewidth=0.8)
top.legend(["perigee", "apogee"])
top.set_ylabel("altitude above GEO [km]")
middle.set_ylabel("eccentricity")
bottom.set_ylabel("sail cone angle [deg]")
bottom.set_xlabel("time [days]   (flown open loop; gray: eclipse; dashed: targets)")
save(fig, "trajectory")


# ---------------------------------------------------------------- from the initial guess to the solution, stage by stage

fig, (top, bottom) = plt.subplots(2, 1, sharex=True, figsize=(6.5, 4.0))
curves = {"initial guess": guess} | {name: stage.trajectory for name, stage in stages.items()}
for k, (label, trajectory) in enumerate(curves.items()):
    style = dict(color="0.5", linestyle="--") if k == 0 else dict(color=f"C{k - 1}")
    for i, arc in enumerate(trajectory.values()):
        nodes = arc.sample(arc.native_tau)
        perigee, _, eccentricity = orbit(nodes.x)
        top.plot(nodes.x[:, 5] / DAY, perigee, label=label if i == 0 else None, **style)
        bottom.plot(nodes.x[:, 5] / DAY, eccentricity, **style)
top.legend()
top.set_ylabel("perigee above GEO [km]")
bottom.set_ylabel("eccentricity")
bottom.set_xlabel("time [days]   (solver nodes of each stage)")
save(fig, "stages")

(folder / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
plt.show()
