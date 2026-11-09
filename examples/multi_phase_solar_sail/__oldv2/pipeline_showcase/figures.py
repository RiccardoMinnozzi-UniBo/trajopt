"""Slide-ready figures of the showcase pipeline, drawn from ``results/``.

    python figures.py

Writes ``figures/1_meshes.png`` ... ``figures/6_handoff.png``, each answering one
question about the p -> ms -> hp pipeline:

1. meshes      what each stage discretises one sunlit + one eclipse arc into
2. convergence how the transfer time and the dynamics defect evolve, stage by stage
3. time        where the wall time goes in each stage
4. accuracy    transfer time and re-propagated accuracy of each stage's answer
5. solution    the final orbit raise, eccentricity and sail steering
6. handoff     the steering profile as it is refined by each stage
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from solution import load_solution  # noqa: E402

RESULTS = HERE / "results"
FIGURES = HERE / "figures"
DAY = 86400.0

# categorical slots 1-3 of the validated default palette, in fixed order
STAGE_COLOR = {"p": "#2a78d6", "ms": "#eb6834", "hp": "#1baf7a"}
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#fcfcfb"
ECLIPSE = "#ecebe7"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.size": 12, "axes.titlesize": 14, "axes.titleweight": "bold", "axes.labelsize": 12,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "legend.frameon": False, "lines.linewidth": 2.0, "figure.dpi": 100,
})


def stage_title(stage: dict) -> str:
    m = stage["mesh"]
    if stage["mode"] == "ps" and m["sunlit_hp_segments"] == 1:
        return f"p  (Radau, one polynomial per arc, {m['sunlit_nodes']}/{m['eclipse_nodes']} nodes)"
    if stage["mode"] == "ps":
        return (f"hp  (Radau, {m['sunlit_hp_segments']}/{m['eclipse_hp_segments']} segments per arc, "
                f"{m['sunlit_nodes']}/{m['eclipse_nodes']} nodes)")
    return f"ms  (multiple shooting, {m['sunlit_nodes']}/{m['eclipse_nodes']} nodes, {stage['nsub']} RK4 steps each)"


def tolerance_text(stage: dict) -> str:
    return f"eps_cost {stage['eps_cost']:.0e},  defect budget {stage['defect_budget_km']:g} km"


def load():
    summary = json.loads((RESULTS / "summary.json").read_text(encoding="utf-8"))
    stages = []
    for stage in summary["stages"]:
        name = stage["name"]
        sol = RESULTS / f"{name}_solution.npz"
        it = RESULTS / f"{name}_iterates.npz"
        stage["solution"] = load_solution(sol) if sol.exists() else None
        if it.exists():
            with np.load(it) as data:
                stage["iterates"] = {k: data[k] for k in data.files}
            stage["meta"] = json.loads(str(stage["iterates"].pop("meta")))
        stages.append(stage)
    return summary, stages


def save(fig, name: str) -> Path:
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / name
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  {path}")
    return path


def _phase_count(stage) -> int:
    return len(stage["meta"]["phases"])


# --------------------------------------------------------------------------- 1
def fig_meshes(stages) -> Path:
    """Nodes of the first sunlit arc and the first eclipse arc, per stage."""
    fig, axes = plt.subplots(len(stages), 1, figsize=(12, 1.7 * len(stages) + 0.6), sharex=True)
    for ax, stage in zip(np.atleast_1d(axes), stages):
        color = STAGE_COLOR[stage["name"]]
        it = stage["iterates"]
        L0 = None
        for index in (0, 1):  # phase_001 sunlit, phase_002 eclipse
            L = it[f"phase{index:02d}__longitude"][-1]
            L0 = L[0] if L0 is None else L0
            x = np.degrees(L - L0)
            if index == 1:
                ax.axvspan(x[0], x[-1], color=ECLIPSE, zorder=0, lw=0)
            # hp segment boundaries
            segs = stage["meta"]["phases"][index]["segments"]
            if stage["mode"] == "ps" and segs > 1:
                per = (len(x) - 1) // segs
                for b in x[::per]:
                    ax.axvline(b, color=color, alpha=0.35, lw=1.0, zorder=1)
            # ms: the RK4 substeps between nodes
            if stage["mode"] != "ps" and stage["nsub"] > 1:
                sub = np.concatenate([np.linspace(a, b, stage["nsub"] + 1)[1:-1] for a, b in zip(x[:-1], x[1:])])
                ax.plot(sub, np.zeros_like(sub), "|", color=color, ms=9, mew=1.2, alpha=0.6, zorder=2)
            ax.plot(x, np.zeros_like(x), "o", color=color, ms=7, mec=SURFACE, mew=1.5, zorder=3)
        ax.set_yticks([])
        ax.grid(False)
        ax.spines["left"].set_visible(False)
        ax.set_ylim(-1, 1)
        ax.set_title(stage_title(stage), loc="left", fontsize=12.5, color=INK)
    ax = np.atleast_1d(axes)[-1]
    ax.set_xlabel("orbital longitude from arc start (deg)    [shaded: eclipse arc]")
    fig.suptitle("What each stage discretises: one sunlit arc and one eclipse arc", x=0.01, ha="left",
                 fontsize=15, fontweight="bold", color=INK)
    handles = [plt.Line2D([], [], ls="", marker="o", color=INK2, ms=7, label="optimisation node"),
               plt.Line2D([], [], ls="", marker="|", color=INK2, ms=9, label="RK4 step (ms)"),
               plt.Line2D([], [], color=INK2, alpha=0.4, label="hp segment boundary")]
    fig.legend(handles=handles, loc="upper right", ncol=3, bbox_to_anchor=(0.99, 1.0))
    fig.tight_layout()
    return save(fig, "1_meshes.png")


# --------------------------------------------------------------------------- 2
def _history(stage):
    it, n = stage["iterates"], _phase_count(stage)
    days = it[f"phase{n - 1:02d}__state"][:, -1, 5] / DAY
    defect = np.max([np.nanmax(np.abs(it[f"phase{i:02d}__defect"]), axis=(1, 2)) for i in range(n)], axis=0)
    eps = stage["meta"]["phases"][0]["eps_dyn_state"][0]
    return days, defect, eps


def fig_convergence(stages) -> Path:
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    start = 0
    final_days = None
    for stage in stages:
        if "iterates" not in stage:
            continue
        name, color = stage["name"], STAGE_COLOR[stage["name"]]
        days, defect, eps = _history(stage)
        k = np.arange(start, start + len(days))
        top.plot(k, days, "-o", color=color, ms=4, label=name)
        bottom.semilogy(k, defect, "-o", color=color, ms=4, label=name)
        bottom.hlines(eps, k[0], k[-1], colors=color, linestyles="--", lw=1.6)
        for ax in (top, bottom):
            ax.axvline(start - 0.5, color=MUTED, lw=1, alpha=0.6)
        top.text(k[0] + 0.3, 1.0, f"{name}\n{tolerance_text(stage).replace(',  ', chr(10))}",
                 transform=top.get_xaxis_transform(), va="top", fontsize=10, color=INK2)
        start += len(days)
        final_days = days[-1]
    top.set_ylabel("transfer time (days)")
    if final_days is not None:
        lo = final_days - 0.15
        top.set_ylim(lo, max(lo + 0.3, top.get_ylim()[1]))
    bottom.set_ylabel("max dynamics defect\n(per interval, scaled)")
    bottom.set_xlabel("SCP iteration (all stages in sequence)")
    bottom.plot([], [], "--", color=INK2, lw=1.6, label="stage tolerance")
    bottom.legend(loc="upper right", ncol=4)
    fig.suptitle("Convergence across the pipeline", x=0.01, ha="left", fontsize=15,
                 fontweight="bold", color=INK)
    fig.tight_layout()
    return save(fig, "2_convergence.png")


# --------------------------------------------------------------------------- 3
def fig_time(stages) -> Path:
    fig, ax = plt.subplots(figsize=(12, 3.8))
    parts = (("compile_s", "JAX compile + first CVXPY setup", "#86b6ef"),
             ("solve_s", "SCP iterations", "#1c5cab"),
             ("other_s", "build, validate, hand-over", "#d6d5d0"))
    names = [s["name"] for s in stages][::-1]
    total_all = 0.0
    for row, stage in enumerate(stages[::-1]):
        t = stage["timings"]
        values = {"compile_s": t.get("compile_s", 0.0), "solve_s": t.get("solve_s", 0.0)}
        values["other_s"] = max(t.get("total_s", 0.0) - sum(values.values()), 0.0)
        left = 0.0
        for key, label, color in parts:
            ax.barh(row, values[key], left=left, color=color, height=0.6, edgecolor=SURFACE, lw=2,
                    label=label if row == 0 else None)
            left += values[key]
        total_all += left
        ax.text(left + 8, row, f"{left:.0f} s  ·  {stage['iterations']} iterations  ·  "
                f"{values['solve_s'] / max(stage['iterations'], 1):.1f} s/iteration",
                va="center", fontsize=11, color=INK)
    ax.set_yticks(range(len(names)), names, fontsize=13)
    ax.set_xlabel("wall time (s)")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, ax.get_xlim()[1] * 1.45)
    ax.legend(loc="lower right", ncol=1)
    ax.set_title(f"Where the time goes  (total {total_all / 60:.1f} min)", loc="left", fontsize=15, color=INK)
    fig.tight_layout()
    return save(fig, "3_time.png")


# --------------------------------------------------------------------------- 4
def fig_accuracy(stages) -> Path:
    done = [s for s in stages if s.get("report")]
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.2))
    y = np.arange(len(done))[::-1]
    final = done[-1]["report"]["elapsed_days"]
    for yi, stage in zip(y, done):
        r, color = stage["report"], STAGE_COLOR[stage["name"]]
        left.plot(r["elapsed_days"], yi, "o", color=color, ms=11)
        delta = (r["elapsed_days"] - final) * 24 * 60
        left.text(r["elapsed_days"], yi + 0.22, f"{r['elapsed_days']:.4f} d" +
                  ("" if stage is done[-1] else f"  ({delta:+.1f} min)"), ha="center", fontsize=11, color=INK)
        acc = abs(r["accumulated_defect_km"])
        right.barh(yi, max(acc, 1e-4), color=color, height=0.55)
        verdict = "validated" if r["numerically_feasible"] else "not validated"
        right.text(max(acc, 1e-4) * 1.3, yi, f"{acc:.3g} km  ·  {verdict}", va="center", fontsize=11, color=INK)
    for ax in (left, right):
        ax.set_yticks(y, [s["name"] for s in done], fontsize=13)
        ax.grid(axis="y", visible=False)
        ax.set_ylim(-0.6, len(done) - 0.3)
    left.set_xlabel("transfer time (days)")
    span = max(abs(s["report"]["elapsed_days"] - final) for s in done) or 0.01
    left.set_xlim(final - 1.6 * span, final + 1.6 * span)
    left.set_title("Answer of each stage", loc="left", color=INK)
    right.set_xscale("log")
    right.axvline(3.0, color=INK2, ls="--", lw=1.6)
    right.text(3.0, len(done) - 0.45, " 3 km budget", fontsize=10, color=INK2, va="bottom")
    right.set_xlim(right.get_xlim()[0], 3.0 * 60)
    right.set_xlabel("orbit-raise error after independent re-propagation (km)")
    right.set_title("How accurate it is", loc="left", color=INK)
    fig.tight_layout()
    return save(fig, "4_accuracy.png")


# --------------------------------------------------------------------------- 5
def _shade_eclipses(ax, days, lit):
    edges = np.flatnonzero(np.diff(lit.astype(int)))
    starts = [0] if not lit[0] else []
    starts += [i + 1 for i in edges if lit[i] and not lit[i + 1]]
    stops = [i for i in edges if not lit[i] and lit[i + 1]] + ([len(lit) - 1] if not lit[-1] else [])
    for a, b in zip(starts, stops):
        ax.axvspan(days[a], days[b], color=ECLIPSE, lw=0, zorder=0)


def fig_solution(stages) -> Path:
    s = stages[-1]["solution"]
    color = STAGE_COLOR[stages[-1]["name"]]
    days = s.dense_elapsed_days
    fig, axes = plt.subplots(3, 1, figsize=(12, 7.5), sharex=True)
    rows = ((s.dense_perigee_radius_km - s.geo_radius_km, "perigee above GEO (km)",
             s.target_perigee_km - s.geo_radius_km, "graveyard requirement"),
            (s.dense_eccentricity, "eccentricity", s.target_eccentricity, "eccentricity cap"),
            (np.degrees(s.dense_cone_angle_rad), "sail cone angle (deg)", None, None))
    for ax, (values, label, target, tlabel) in zip(axes, rows):
        _shade_eclipses(ax, days, s.dense_illuminated)
        ax.plot(days, values, color=color, lw=1.4)
        if target is not None:
            ax.axhline(target, color=INK2, ls="--", lw=1.4)
            ax.text(days[0], target, f" {tlabel}", va="bottom", fontsize=10, color=INK2)
        ax.set_ylabel(label)
    axes[-1].set_xlabel("elapsed time (days)    [shaded: eclipses]")
    fig.suptitle(f"Final solution ({stages[-1]['name']}): GEO to graveyard in "
                 f"{stages[-1]['report']['elapsed_days']:.3f} days, {s.revolutions} revolutions",
                 x=0.01, ha="left", fontsize=15, fontweight="bold", color=INK)
    fig.tight_layout()
    return save(fig, "5_solution.png")


# --------------------------------------------------------------------------- 6
def fig_handoff(stages, window_days: float = 2.0) -> Path:
    fig, ax = plt.subplots(figsize=(12, 4.5))
    shaded = False
    for stage in stages:
        s = stage.get("solution")
        if s is None:
            continue
        days, cone = s.dense_elapsed_days, np.degrees(s.dense_cone_angle_rad)
        keep = days <= window_days
        if not shaded:
            _shade_eclipses(ax, days[keep], s.dense_illuminated[keep])
            shaded = True
        ax.plot(days[keep], cone[keep], color=STAGE_COLOR[stage["name"]], lw=1.8, label=stage["name"])
        # nodes, so the mesh density per stage is visible
        nk = s.elapsed_days <= window_days
        ax.plot(s.elapsed_days[nk], np.degrees(s.cone_angle_rad[nk]), "o", ms=4,
                color=STAGE_COLOR[stage["name"]], mec=SURFACE, mew=0.8)
    ax.set_xlabel("elapsed time (days)    [shaded: eclipse]")
    ax.set_ylabel("sail cone angle (deg)")
    ax.legend(loc="upper right", ncol=3, title="stage (dots: nodes)")
    ax.set_title(f"Hand-over: the steering law as each stage refines it (first {window_days:g} days)",
                 loc="left", color=INK, fontsize=15)
    fig.tight_layout()
    return save(fig, "6_handoff.png")


def main() -> None:
    summary, stages = load()
    print(f"{summary['config']}: N = {summary['revolutions']}, completed = {summary['completed']}")
    fig_meshes([s for s in stages if "iterates" in s])
    fig_convergence(stages)
    fig_time(stages)
    fig_accuracy(stages)
    if stages[-1].get("solution") is not None:
        fig_solution(stages)
    fig_handoff(stages)


if __name__ == "__main__":
    main()
