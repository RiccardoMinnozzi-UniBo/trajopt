"""Plots for one run, or a summary of all runs.

    python postprocessing.py results/T10_p_ms     figures of one run, saved in its folder
    python postprocessing.py                      table and sweep figure of every run in results/
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import jax
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

from trajopt.common.persistence import load_result

import problem
from main import control_law

plt.rcParams.update({"font.size": 9, "figure.constrained_layout.use": True, "axes.grid": True,
                     "grid.alpha": 0.3, "savefig.dpi": 200})

PARAMS = SimpleNamespace(mu=problem.MU)
dfdx = jax.jit(jax.jacfwd(lambda x, u: problem.dynamics(x, u, 0.0, PARAMS)))
TOL = dict(method="DOP853", rtol=1e-12, atol=1e-12)


def segments(t, x, u):
    """Integrate every node interval from its own node: the continuous path and the defect at each next node."""
    arcs = [problem.fly(x[k], (t[k], t[k + 1]), u) for k in range(len(t) - 1)]
    defects = np.array([x[k + 1] - arc.y[:, -1] for k, arc in enumerate(arcs)])
    return arcs, defects


def evaluate(arcs, t):
    """Sample a chain of integrated intervals."""
    starts = np.array([arc.t[0] for arc in arcs])
    k = np.clip(np.searchsorted(starts, t, side="right") - 1, 0, len(arcs) - 1)
    return np.array([arcs[i].sol(ti) for i, ti in zip(k, t)])


def stm(arc, u):
    """State transition matrix over one integrated interval, along that interval's path."""
    def rhs(t, phi):
        return (np.asarray(dfdx(arc.sol(t), u(np.array([t]))[0])) @ phi.reshape(4, 4)).ravel()
    return solve_ivp(rhs, (arc.t[0], arc.t[-1]), np.eye(4).ravel(), **TOL).y[:, -1].reshape(4, 4)


def effort(result):
    t = np.linspace(0.0, result.trajectory["main"].t[-1], 4001)
    return np.trapezoid(np.sum(control_law(result)(t) ** 2, axis=1), t)


def guess_defects(stage):
    """Node defects of the guess a stage started from, flying each interval with that stage's control law."""
    guess = stage.initial_trajectory["main"].sample(stage.initial_trajectory["main"].native_tau)
    return guess.t[1:], np.linalg.norm(segments(guess.t, guess.x, control_law(stage, stage.initial_trajectory))[1], axis=1)


def guess_infeasibility(stage):
    """Largest node defect or final-state miss of the guess a stage started from."""
    miss = np.linalg.norm(stage.initial_trajectory["main"].sample([1.0]).x[0] - problem.XF)
    return max(guess_defects(stage)[1].max(), miss)


def e_folds(result):
    """ln sigma_max(Phi(T, 0)) along the solution, chained over its node intervals."""
    u = control_law(result)
    nodes = result.trajectory["main"].sample(result.trajectory["main"].native_tau)
    total = np.eye(4)
    for arc in segments(nodes.t, nodes.x, u)[0]:
        total = stm(arc, u) @ total
    return np.log(np.linalg.norm(total, 2))


def load(folder):
    return [load_result(path) for path in sorted(Path(folder).glob("*.json"))]


def moon_and_points(ax):
    ax.plot(1 - problem.MU, 0, "o", color="0.4", ms=6)
    ax.add_patch(plt.Circle((1 - problem.MU, 0), problem.R_MIN, color="0.4", alpha=0.15))
    ax.plot([problem.L1, problem.L2], [0, 0], "k+", ms=8)
    ax.annotate("L1", (problem.L1, 0), textcoords="offset points", xytext=(-14, 4))
    ax.annotate("L2", (problem.L2, 0), textcoords="offset points", xytext=(5, 4))


def plot_run(folder):
    folder = Path(folder)
    stages = load(folder)
    result = stages[-1]
    T = result.trajectory["main"].t[-1]
    u = control_law(result)
    nodes = result.trajectory["main"].sample(result.trajectory["main"].native_tau)
    arcs, defects = segments(nodes.t, nodes.x, u)
    t = np.linspace(0.0, T, 2001)
    x = evaluate(arcs, t)
    rollout = problem.fly(nodes.x[0], (0.0, T), u)
    rollout_x = rollout.sol(np.minimum(t, rollout.t[-1])).T

    print(f"{folder.name}: effort {effort(result):.6f}, max defect {np.abs(defects).max():.1e}")
    for number, stage in enumerate(stages, start=1):
        print(f"  stage {number}: {stage.status} in {stage.iterations} iterations, {stage.timings['total_s']:.1f} s")

    # trajectory in the rotating frame
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for number, stage in enumerate(stages, start=1):
        guess = stage.initial_trajectory["main"].sample(num=801)
        ax.plot(guess.x[:, 0], guess.x[:, 1], ":", color=["0.5", "C2", "C4"][number - 1], label=f"stage {number} initial guess")
    ax.plot(x[:, 0], x[:, 1], "C0", label="solution")
    ax.plot(rollout_x[:, 0], rollout_x[:, 1], "C3--", lw=0.9, label="solution control flown open loop")
    k = slice(None, None, max(len(nodes.t) // 40, 1))
    ax.quiver(nodes.x[k, 0], nodes.x[k, 1], nodes.u[k, 0], nodes.u[k, 1], color="C1", width=0.003, label="thrust")
    moon_and_points(ax)
    ax.set(xlim=(0.78, 1.22), ylim=(-0.2, 0.2), aspect="equal", xlabel="x [-]", ylabel="y [-]",
           title=f"{folder.name}: L1 to L2 in {T:g} TU ({T * problem.TU_DAYS:.0f} days)")
    ax.legend(loc="lower left", fontsize=7)
    fig.savefig(folder / "trajectory.png")

    # state and control histories
    fig, axes = plt.subplots(3, 1, figsize=(6.5, 6), sharex=True)
    axes[0].plot(t, x[:, :2], label=["x", "y"])
    axes[1].plot(t, x[:, 2:], label=["vx", "vy"])
    axes[2].plot(t, u(t), label=["ux", "uy"])
    axes[2].plot(t, np.linalg.norm(u(t), axis=1), "k", lw=0.8, label="|u|")
    for ax, label in zip(axes, ["position [-]", "velocity [-]", "thrust acceleration [-]"]):
        ax.set_ylabel(label)
        ax.legend(fontsize=7, ncol=3)
    axes[2].set_xlabel("t [TU]")
    fig.savefig(folder / "histories.png")

    # trajectory on the effective potential: L1 and L2 are its saddles
    gx, gy = np.meshgrid(np.linspace(0.78, 1.22, 200), np.linspace(-0.2, 0.2, 200))
    levels = -problem.potential(np.array([problem.L1, problem.L2]), 0.0)
    clip = (levels.min() - 0.06, levels.max() + 0.05)
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(projection="3d")
    surface = -problem.potential(gx, gy)
    ax.plot_surface(gx, gy, np.where(surface < clip[0], np.nan, surface), cmap="viridis", alpha=0.55, lw=0)
    ax.plot(x[:, 0], x[:, 1], np.clip(-problem.potential(x[:, 0], x[:, 1]), *clip) + 2e-3, "C3", lw=1.5)
    ax.scatter([problem.L1, problem.L2], [0, 0], levels, color="k", s=15)
    ax.set(xlabel="x [-]", ylabel="y [-]", zlabel="-Omega [-]", zlim=clip, title="Trajectory on the effective potential")
    ax.view_init(elev=35, azim=-60)
    fig.savefig(folder / "potential_3d.png")

    # why propagation fails: error growth of the open-loop flight, full horizon vs one segment
    phis = [stm(arc, u) for arc in arcs]
    growth, total = [0.0], np.eye(4)
    for phi in phis:
        total = phi @ total
        growth.append(np.log(np.linalg.norm(total, 2)))
    segment_growth = [np.log(np.linalg.norm(phi, 2)) for phi in phis]
    fig, axes = plt.subplots(2, 1, figsize=(6.5, 6), sharex=True)
    axes[0].semilogy(t, np.linalg.norm(rollout_x[:, :2] - x[:, :2], axis=1), "C3", label="open-loop flight - solution")
    axes[0].semilogy(nodes.t, np.abs(defects).max() * np.exp(growth), "k--", lw=0.8,
                     label="max node defect x sigma_max(Phi(t, 0))")
    axes[0].set(ylabel="position error [-]", ylim=(1e-12, 10), title="Errors grow by sigma_max(Phi) along the transfer")
    axes[0].legend(fontsize=7)
    axes[1].plot(nodes.t, growth, "k", label="whole horizon: ln sigma_max(Phi(t, 0))")
    axes[1].bar(nodes.t[:-1], segment_growth, width=np.diff(nodes.t), align="edge", color="C0", alpha=0.6,
                label="one interval: ln sigma_max(Phi(t_k+1, t_k))")
    axes[1].set(xlabel="t [TU]", ylabel="e-folds of error growth")
    axes[1].legend(fontsize=7)
    fig.savefig(folder / "instability.png")

    # how far from dynamically feasible each stage started, and how each stage converged
    fig, axes = plt.subplots(2, 1, figsize=(6.5, 6))
    offset = 0
    for number, stage in enumerate(stages, start=1):
        name = stage.phase_data["main"]["mesh"]["discretize"]
        axes[0].semilogy(*guess_defects(stage), ".-", ms=3, label=f"stage {number} ({name}) initial guess")
        history = stage.phase_data["main"]["scalar_history"]
        iterations = offset + np.arange(1, len(history) + 1)
        axes[1].semilogy(iterations, [h["chk"]["dynamics"] for h in history], f"C{number - 1}", label=f"stage {number} ({name})")
        offset = iterations[-1]
    axes[0].semilogy(nodes.t[1:], np.linalg.norm(defects, axis=1), "k.-", ms=3, label="solution")
    axes[0].set(xlabel="t [TU]", ylabel="node defect [-]", ylim=(1e-10, None), title="Dynamics defect at each node (each interval flown alone)")
    axes[1].axhline(1.0, color="k", lw=0.8, ls="--")
    axes[1].set(xlabel="SCP iteration", ylabel="dynamics violation / tolerance")
    for ax in axes:
        ax.legend(fontsize=7)
    fig.savefig(folder / "convergence.png")
    print(f"  figures saved in {folder}")


def summary():
    rows = []
    for folder in sorted(Path("results").glob("T*_*")):
        stages = load(folder)
        last = stages[-1]
        ok = last.status == "converged"
        rows.append(dict(T=float(folder.name[1:].split("_")[0]), method=folder.name.split("_", 1)[1], ok=ok,
                         iterations="+".join(str(s.iterations) for s in stages), ms_iterations=last.iterations,
                         time=sum(s.timings["total_s"] for s in stages), guess=guess_infeasibility(last),
                         effort=effort(last) if ok else np.nan, e_folds=e_folds(last) if ok else np.nan))
    rows.sort(key=lambda row: (row["T"], row["method"]))
    print(f"{'T':>4} {'method':>12} {'converged':>9} {'iterations':>10} {'time [s]':>8} {'ms guess infeas.':>16} {'effort':>9} {'e-folds':>7}")
    for r in rows:
        print(f"{r['T']:4g} {r['method']:>12} {str(r['ok']):>9} {r['iterations']:>10} {r['time']:8.1f} "
              f"{r['guess']:16.1e} {r['effort']:9.6f} {r['e_folds']:7.1f}")

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))
    for number, method in enumerate(["ms", "p_rollout_ms", "p_ms", "msc", "p_msc"]):
        mine = [row for row in rows if row["method"] == method]
        style = "--" if method.endswith("msc") else "-"
        for ax, key in zip(axes, ["guess", "ms_iterations", "time"]):
            ax.plot([r["T"] for r in mine], [r[key] for r in mine], f"C{number}{style}", alpha=0.4)
            ax.plot([r["T"] for r in mine if r["ok"]], [r[key] for r in mine if r["ok"]], f"C{number}o", label=method)
            ax.plot([r["T"] for r in mine if not r["ok"]], [r[key] for r in mine if not r["ok"]], f"C{number}x", ms=8)
    axes[0].set(xlabel="T [TU]", ylabel="ms initial guess: max node defect\nor final-state miss [-]", yscale="log")
    axes[1].set(xlabel="T [TU]", ylabel="ms iterations (dashed: 1 node per TU)")
    axes[2].set(xlabel="T [TU]", ylabel="wall time, all stages [s]", yscale="log")
    axes[0].legend(title="x: not converged", fontsize=7)
    fig.savefig("results/sweep.png")


if __name__ == "__main__":
    plot_run(sys.argv[1]) if len(sys.argv) > 1 else summary()
