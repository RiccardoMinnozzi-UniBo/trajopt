"""Analyses across saved sweep cases, to understand what the first stage of a pipeline hands over.

Reads results/sweep/ only (run sweep.py first); figures go to results/analysis/.
Pick the analyses in ANALYSES; the cases each one compares are set in its own section.
"""
import sys
from pathlib import Path

import jax
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

from trajopt.common.persistence import load_result
from trajopt.common.tools import AttrDict

from solve import control_polynomial

sys.path.insert(0, "mission")
import solar_sail

jax.config.update("jax_enable_x64", True)

ANALYSES = ["iterations", "handover_defects", "handover_controls", "arc_end"]

plt.rcParams.update({"font.size": 9, "figure.figsize": (6.5, 5.5), "figure.constrained_layout.use": True,
                     "axes.grid": True, "grid.alpha": 0.3, "lines.linewidth": 1.2, "savefig.dpi": 300})

SWEEP = Path("results") / "sweep"
OUT = Path("results") / "analysis"
OUT.mkdir(parents=True, exist_ok=True)
COLOURS = {"p_ms": "C0", "msc_ms": "C3", "p": "C1", "p_ms_hp": "C2"}


def case_name(algorithm, weight, rate="on", budget=30):
    name = f"{algorithm}-straight_line-ramp-rate_{rate}-w_{weight:g}"
    return name if "_" not in algorithm else f"{name}-cost_10000-budget_{budget:g}"


def stages(name):
    """The saved stage results of a case, in order."""
    return [load_result(path) for path in sorted((SWEEP / name).glob("[0-9]_*.json"))]


def save(fig, name):
    fig.savefig(OUT / f"{name}.png")
    fig.savefig(OUT / f"{name}.pdf")


def steering_law(result, name):
    """The steering an arc of a stage solved for: one polynomial per segment for p and hp, straight lines for ms."""
    arc = result.trajectory[name]
    mesh = result.phase_data[name]["mesh"]
    if mesh["discretize"] == "ps":
        polynomial = control_polynomial(arc, mesh["segments"])
        return lambda longitude: polynomial((longitude - arc.t[0]) / (arc.t[-1] - arc.t[0]))[:, 0]
    nodes = arc.sample(arc.native_tau)
    return lambda longitude: np.interp(longitude, nodes.t, nodes.u[:, 0])


RHS = {}


def dynamics(params):
    """Compiled dynamics of a sunlit or an eclipse arc."""
    if params.illumination not in RHS:
        RHS[params.illumination] = jax.jit(lambda x, u, L: solar_sail.dynamics(x, u, L, AttrDict(params), None))
    return RHS[params.illumination]


def interval_defects(params, t, x, law):
    """Every node flown to the next one under a steering law: largest normalized defect per interval."""
    scales = np.array([1000.0, 0.02, 0.02, 0.2, 0.2, params.max_elapsed_s])
    f = dynamics(params)
    rhs = lambda L, y: np.asarray(f(y, np.atleast_1d(law(L)), L))
    out = []
    for i in range(len(t) - 1):
        end = solve_ivp(rhs, t[i:i + 2], x[i], method="DOP853", rtol=1e-11, atol=1e-11).y[:, -1]
        out.append(np.abs((x[i + 1] - end) / scales).max())
    return np.array(out)


# ---------------------------------------------------------------- iteration histories: how the first stage settles

def iterations():
    """End longitude of the final arc and the largest dynamics defect, iteration by iteration, across the stages
    of p_ms and msc_ms at a few cost weights. The window of the final arc's end longitude spans pi."""
    weights = [1.0, 10.0, 30.0]
    fig, axes = plt.subplots(2, len(weights), sharex="col", sharey="row", figsize=(10, 5))
    for column, weight in zip(axes.T, weights):
        for algorithm in ["p_ms", "msc_ms"]:
            results, colour = stages(case_name(algorithm, weight)), COLOURS[algorithm]
            window = results[0].problem.phases["final_arc"].config.constraints.arc_end_window
            longitude = np.concatenate([[s["t_final"] for s in r.phase_data["final_arc"]["scalar_history"]]
                                        for r in results]) - float(window.lower[0])
            defect = np.concatenate([np.max([[s["chk"]["dynamics"] for s in phase["scalar_history"]]
                                             for phase in r.phase_data.values()], axis=0) for r in results])
            iteration = 1 + np.arange(len(longitude))
            status = " -> ".join(f"{r.status.replace('not_converged', 'not conv.')}" for r in results)
            column[0].plot(iteration, longitude, ".-", color=colour, markersize=3, label=f"{algorithm}: {status}")
            column[1].semilogy(iteration, defect, color=colour)
            for ax in column:
                ax.axvline(results[0].iterations + 0.5, color=colour, linestyle=":")
        column[0].axhline(0.0, color="k", linestyle="--", linewidth=0.8)
        column[0].axhline(np.pi, color="k", linestyle="--", linewidth=0.8)
        column[1].axhline(1.0, color="k", linestyle="--", linewidth=0.8)
        column[0].set_title(f"w = {weight:g}", fontsize=9)
        column[0].legend(fontsize=7)
        column[1].set_xscale("log")
        column[1].set_xlabel("iteration   (dotted: hand-over to ms)")
    axes[0, 0].set_ylabel("final-arc end longitude\nfrom window start [rad]")
    axes[1, 0].set_ylabel("largest dynamics defect / tolerance")
    fig.suptitle("ramp control guess, rate limit on, first stage at eps_cost 1e4 and 30 km", fontsize=9)
    save(fig, "iterations")


# ---------------------------------------------------------------- the dynamics defect ms starts from

def handover_defects():
    """The guess the ms stage starts from, flown interval by interval as ms reads it (straight-line control
    between its nodes): defect over the ms tolerance, along the transfer."""
    weight = 30.0
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    for algorithm in ["p_ms", "msc_ms"]:
        results = stages(case_name(algorithm, weight))
        ms = results[1]
        longitude, ratio = [], []
        for name, guess in ms.initial_trajectory.items():
            phase = ms.problem.phases[name].config
            nodes = guess.sample(guess.native_tau)
            law = lambda L, nodes=nodes: np.interp(L, nodes.t, nodes.u[:, 0])
            longitude.append(nodes.t[1:])
            ratio.append(interval_defects(phase.params, nodes.t, nodes.x, law) / float(phase.constraints.dynamics.eps))
        longitude, ratio = np.concatenate(longitude), np.concatenate(ratio)
        ax.semilogy(longitude, ratio, ".", color=COLOURS[algorithm], markersize=3,
                    label=f"{algorithm}: largest {ratio.max():.0f}, median {np.median(ratio):.0f}; "
                          f"ms then {ms.status.replace('_', ' ')} in {ms.iterations}")
        print(f"{algorithm:7s} hand-over defect / ms tolerance: largest {ratio.max():.0f}, median {np.median(ratio):.1f}")
    ax.axhline(1.0, color="k", linestyle="--", linewidth=0.8)
    ax.set_xlabel("true longitude [rad]")
    ax.set_ylabel("defect / ms tolerance")
    ax.set_title(f"guess handed to the ms stage, w = {weight:g}, ramp control guess, rate limit on", fontsize=9)
    ax.legend(fontsize=7)
    save(fig, "handover_defects")


# ---------------------------------------------------------------- the steering handed over

def handover_controls():
    """First revolutions: the first stage's steering (its nodes and its own interpolation), the guess handed
    to ms, and the steering ms converges to. Plotted as steering + longitude, which takes out the winding."""
    weight, revolutions = 30.0, 4
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(8, 5))
    for ax, algorithm in zip(axes, ["p_ms", "msc_ms"]):
        first, ms = stages(case_name(algorithm, weight))[:2]
        start = next(iter(first.trajectory.values())).t[0]
        for number, name in enumerate(first.trajectory):
            arc = first.trajectory[name]
            if arc.t[0] > start + revolutions * 2 * np.pi:
                break
            longitude = np.linspace(arc.t[0], arc.t[-1], 200)
            nodes = arc.sample(arc.native_tau)
            guess = ms.initial_trajectory[name].sample(np.linspace(0, 1, 200))
            final = ms.trajectory[name].sample(ms.trajectory[name].native_tau)
            labels = number == 0
            ax.plot(longitude, steering_law(first, name)(longitude) + longitude, color=COLOURS[algorithm], alpha=0.4,
                    label="first stage, own interpolation" if labels else None)
            ax.plot(nodes.t, nodes.u[:, 0] + nodes.t, ".", color=COLOURS[algorithm], markersize=4,
                    label="first stage, nodes" if labels else None)
            ax.plot(guess.t, guess.u[:, 0] + guess.t, "k--", linewidth=0.8, label="handed to ms" if labels else None)
            ax.plot(final.t, final.u[:, 0] + final.t, "k-", linewidth=0.8, label="ms solution" if labels else None)
            if not first.problem.phases[name].config.params.illumination:
                ax.axvspan(arc.t[0], arc.t[-1], color="0.85", linewidth=0)
        ax.set_ylim(0, 6)
        ax.set_ylabel("steering + longitude [rad]")
        ax.set_title(f"{algorithm}, w = {weight:g}", fontsize=9)
        ax.legend(fontsize=7, ncol=4, loc="lower right")
    axes[-1].set_xlabel("true longitude [rad]   (grey: eclipse)")
    save(fig, "handover_controls")


# ---------------------------------------------------------------- the end of a sunlit arc, across transcriptions

def arc_end():
    """The last nodes of two sunlit arcs before the shadow, in every transcription: coarse p, fine p,
    ms (after msc), and hp."""
    weight = 30.0
    sources = {"coarse p (12 nodes)": (case_name("p_ms", weight), 0), "fine p": (case_name("p", weight), 0),
               "ms after msc (12 nodes)": (case_name("msc_ms", weight), 1), "hp": (case_name("p_ms_hp", weight), 2)}
    loaded = {label: stages(name)[index] for label, (name, index) in sources.items()}
    fig, axes = plt.subplots(1, 2, sharey=True, figsize=(8, 3.5))
    for ax, name in zip(axes, ["arc_03_sunlit", "arc_21_sunlit"]):
        for colour, (label, result) in zip(["C0", "C1", "C3", "C2"], loaded.items()):
            arc = result.trajectory[name]
            longitude = np.linspace(arc.t[0], arc.t[-1], 400)
            nodes = arc.sample(arc.native_tau)
            ax.plot(longitude, steering_law(result, name)(longitude) + longitude, color=colour, label=label)
            ax.plot(nodes.t, nodes.u[:, 0] + nodes.t, ".", color=colour, markersize=4)
        end = loaded["coarse p (12 nodes)"].trajectory[name].t[-1]
        ax.axvline(end, color="0.5", linestyle="--", linewidth=0.8)
        ax.set_xlim(end - 1.5, end + 0.05)
        ax.set_ylim(1.5, 4.5)
        ax.set_title(f"end of {name}   (dashed: shadow entry)", fontsize=9)
        ax.set_xlabel("true longitude [rad]")
    axes[0].set_ylabel("steering + longitude [rad]")
    axes[0].legend(fontsize=7)
    save(fig, "arc_end")


if __name__ == "__main__":
    for analysis in ANALYSES:
        print(analysis)
        globals()[analysis]()
    plt.show()
