"""Four plain slide figures of the p -> ms -> hp pipeline, one message each.

    python simple_figures.py

Reads ``results/data/p_ms_hp__{p,ms,hp}.npz`` (written by ``solve.py p_ms_hp``)
and writes ``results/figures_simple/``:

1_stages.png         what each stage puts on one sunlit and one eclipse arc
2_transfer_time.png  the answer (transfer time) at every iteration, stage by stage
3_accuracy.png       the dynamics error against the tolerance, stage by stage
4_cost.png           time and iterations spent in each stage
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = HERE / "results" / "data"
OUT = HERE / "results" / "figures_simple"
DAY = 86400.0

STAGES = ("p", "ms", "hp")
COLOR = {"p": "#2a78d6", "ms": "#eb6834", "hp": "#1baf7a"}   # fixed categorical order
NAME = {"p": "p\ncoarse collocation", "ms": "ms\nmultiple shooting", "hp": "hp\nfine collocation"}
INK, INK2, MUTED, GRID, SURFACE, SHADE = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#ffffff", "#ecebe7"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.size": 15, "axes.labelsize": 15, "axes.titlesize": 18, "axes.titleweight": "bold",
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "axes.axisbelow": True, "legend.frameon": False,
})


def load(stage: str) -> dict:
    with np.load(DATA / f"p_ms_hp__{stage}.npz") as data:
        out = {key: data[key] for key in data.files}
    out["meta"] = json.loads(str(out["meta"]))
    return out


def transfer_days(d) -> np.ndarray:
    last = len(d["meta"]["phases"]) - 1
    return d[f"phase{last:02d}__state"][:, -1, 5] / DAY


def error_ratio(d) -> np.ndarray:
    """Worst dynamics defect over the whole trajectory, divided by its tolerance."""
    phases = d["meta"]["phases"]
    worst = np.max([np.nanmax(np.abs(d[f"phase{i:02d}__defect"]), axis=(1, 2)) for i in range(len(phases))],
                   axis=0)
    return worst / phases[0]["eps_dyn_state"][0]


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  {OUT / name}")


def stage_bands(ax, spans, top_labels=True):
    for stage, (a, b) in spans.items():
        if stage != "ms":
            ax.axvspan(a - 0.5, b + 0.5, color=SHADE, lw=0, zorder=0)
        if top_labels:
            ax.text((a + b) / 2, 1.02, NAME[stage].split("\n")[0], transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom", fontsize=17, fontweight="bold", color=COLOR[stage])


def series(stages):
    """Iteration index of every stage when the stages are laid end to end."""
    spans, start = {}, 0
    for s in STAGES:
        n = len(transfer_days(stages[s]))
        spans[s] = (start, start + n - 1)
        start += n
    return spans


# --------------------------------------------------------------------------- 1
def fig_stages(stages):
    fig, axes = plt.subplots(3, 1, figsize=(13, 5.6), sharex=True)
    for ax, s in zip(axes, STAGES):
        d = stages[s]
        L0 = d["phase00__longitude"][-1][0]
        for i in (0, 1):
            x = np.degrees(d[f"phase{i:02d}__longitude"][-1] - L0)
            if i == 1:
                ax.axvspan(x[0], x[-1], color="#3b3b3b", alpha=0.18, lw=0)
            nsub = int(d["meta"]["method_config"]["hyperparams"]["discretize"].get("nsub") or 0)
            if s == "ms" and nsub > 1:
                sub = np.concatenate([np.linspace(a, b, nsub + 1)[1:-1] for a, b in zip(x[:-1], x[1:])])
                ax.plot(sub, np.zeros_like(sub), "|", color=COLOR[s], ms=12, mew=1.4, alpha=0.6)
            ax.plot(x, np.zeros_like(x), "o", color=COLOR[s], ms=9, mec="white", mew=1.5)
            segs = d["meta"]["phases"][i]["segments"]
            if segs > 1:
                for b in x[::(len(x) - 1) // segs]:
                    ax.axvline(b, color=COLOR[s], lw=1.2, alpha=0.5)
        n_total = sum(int(p["n_nodes"]) for p in d["meta"]["phases"])
        ax.text(-0.01, 0.5, NAME[s], transform=ax.transAxes, ha="right", va="center",
                fontsize=15, color=COLOR[s], fontweight="bold")
        ax.text(1.01, 0.5, f"{n_total} nodes\nin total", transform=ax.transAxes, ha="left", va="center",
                fontsize=13, color=INK2)
        ax.set_yticks([]); ax.grid(False); ax.set_ylim(-1, 1)
        ax.spines["left"].set_visible(False)
    axes[0].text(0.995, 1.04, "eclipse", transform=axes[0].transAxes, ha="right", fontsize=13, color=INK2)
    axes[-1].set_xlabel("one sunlit arc + one eclipse arc  (orbital longitude, deg)")
    fig.suptitle("Three meshes for the same trajectory\n"
                 "dots = optimisation nodes,  ticks = RK4 steps (ms),  lines = hp segments",
                 x=0.02, ha="left", fontsize=17, fontweight="bold", color=INK)
    fig.tight_layout()
    save(fig, "1_stages.png")


# --------------------------------------------------------------------------- 2
def fig_transfer_time(stages):
    spans = series(stages)
    fig, ax = plt.subplots(figsize=(13, 5.6))
    stage_bands(ax, spans)
    lo, hi = np.inf, -np.inf
    for s in STAGES:
        a, b = spans[s]
        days, ok = transfer_days(stages[s]), error_ratio(stages[s]) <= 1.0
        k = np.arange(a, b + 1)
        ax.plot(k, days, "-", color=COLOR[s], lw=2.5)
        ax.plot(k[~ok], days[~ok], "o", color=COLOR[s], ms=7, mfc="white", mew=1.8)
        ax.plot(k[ok], days[ok], "o", color=COLOR[s], ms=11, mec="white", mew=1.5)
        ax.annotate(f"{days[-1]:.3f} d", (b, days[-1]), xytext=(0, 14), textcoords="offset points",
                    ha="center", fontsize=14, color=INK, fontweight="bold")
        lo, hi = min(lo, days[1:].min()), max(hi, days.max())
    ax.set_ylim(lo - 0.05, hi + 0.05)
    ax.set_xlim(-1, spans["hp"][1] + 1.5)
    ax.set_xlabel("SCP iteration")
    ax.set_ylabel("transfer time (days)")
    ax.plot([], [], "o", color=INK2, ms=7, mfc="white", mew=1.8, label="dynamics not yet satisfied")
    ax.plot([], [], "o", color=INK2, ms=11, mec="white", label="valid trajectory")
    ax.legend(loc="lower right", fontsize=13)
    ax.set_title("Transfer time during the solve: each stage ends on a valid answer", loc="left", pad=36)
    fig.tight_layout()
    save(fig, "2_transfer_time.png")


# --------------------------------------------------------------------------- 3
def fig_accuracy(stages):
    spans = series(stages)
    fig, ax = plt.subplots(figsize=(13, 5.6))
    stage_bands(ax, spans)
    for s in STAGES:
        a, b = spans[s]
        ax.semilogy(np.arange(a, b + 1), error_ratio(stages[s]), "-o", color=COLOR[s], lw=2.5, ms=6)
    ax.axhline(1.0, color=INK, ls="--", lw=1.8)
    ax.text(spans["hp"][1] + 1.2, 1.0, "tolerance", va="bottom", ha="right", fontsize=14, color=INK)
    ax.axhspan(ax.get_ylim()[0], 1.0, color="#1baf7a", alpha=0.06, lw=0, zorder=0)
    for s in ("ms", "hp"):
        a, _ = spans[s]
        y = error_ratio(stages[s])[0]
        ax.annotate("hand-over\n(new mesh)", (a, y), xytext=(28, 8), textcoords="offset points",
                    fontsize=12, color=INK2, arrowprops=dict(arrowstyle="->", color=INK2, lw=1.2))
    ax.set_xlim(-1, spans["hp"][1] + 1.5)
    ax.set_xlabel("SCP iteration")
    ax.set_ylabel("dynamics error / tolerance")
    ax.set_title("Below the dashed line the trajectory obeys the dynamics", loc="left", pad=36)
    fig.tight_layout()
    save(fig, "3_accuracy.png")


# --------------------------------------------------------------------------- 4
def fig_cost(stages):
    fig, ax = plt.subplots(figsize=(13, 4.2))
    rows = list(STAGES)[::-1]
    total = 0.0
    for y, s in enumerate(rows):
        t = stages[s]["meta"]["timings"]
        compile_s, solve_s = t.get("compile_s", 0.0), t.get("solve_s", 0.0)
        other = max(t["total_s"] - compile_s - solve_s, 0.0)
        ax.barh(y, compile_s, color=COLOR[s], alpha=0.35, height=0.6, edgecolor="white", lw=2)
        ax.barh(y, solve_s + other, left=compile_s, color=COLOR[s], height=0.6, edgecolor="white", lw=2)
        total += t["total_s"]
        ax.text(t["total_s"] + 6, y, f"{t['total_s']:.0f} s,  {stages[s]['meta']['iterations']} iterations",
                va="center", fontsize=14, color=INK)
    ax.set_yticks(range(len(rows)), [NAME[s].replace("\n", "  ") for s in rows], fontsize=14)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, ax.get_xlim()[1] * 1.35)
    ax.set_xlabel("wall time (s)")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=INK2, alpha=0.35, label="one-off setup (compile)"),
                       Patch(color=INK2, label="iterations")], loc="lower right", fontsize=13)
    ax.set_title(f"Cost of each stage   (total {total / 60:.1f} min)", loc="left")
    fig.tight_layout()
    save(fig, "4_cost.png")


# --------------------------------------------------------------------------- 5
def fig_results(stages):
    """What the pipeline delivers after each stage: transfer time and dynamics error."""
    rows = [(f"after {s}", s, transfer_days(stages[s])[-1], error_ratio(stages[s])[-1]) for s in STAGES]
    fig, (left, right) = plt.subplots(1, 2, figsize=(13, 4.2), gridspec_kw={"width_ratios": [1.2, 1]})
    ys = np.arange(len(rows))[::-1]
    for y, (_, s, days, err) in zip(ys, rows):
        left.plot(days, y, "o", color=COLOR[s], ms=15, zorder=3)
        left.annotate(f"{days:.4f} d", (days, y), xytext=(16, 0), textcoords="offset points",
                      ha="left", va="center", fontsize=14, color=INK, fontweight="bold")
        right.barh(y, err, color=COLOR[s], height=0.55)
        right.text(err * 0.88, y, f"{err:.2f}", ha="right", va="center", fontsize=14, color="white",
                   fontweight="bold")
    labels = [rows[0][0]]
    for (_, _, d0, _), (label, _, d1, _) in zip(rows[:-1], rows[1:]):
        minutes = (d1 - d0) * 24 * 60
        labels.append(label + "\n" + ("±0.0 min" if abs(minutes) < 0.05 else f"{minutes:+.1f} min"))
    for ax in (left, right):
        ax.set_yticks(ys, labels, fontsize=15)
        ax.grid(axis="y", visible=False)
        ax.set_ylim(-0.75, len(rows) - 0.4)
    lo, hi = min(r[2] for r in rows), max(r[2] for r in rows)
    pad = max(hi - lo, 0.01)
    left.set_xlim(lo - 0.15 * pad, hi + 0.55 * pad)
    left.set_xlabel("transfer time (days)")
    left.set_title("Transfer time", loc="left")
    right.set_xscale("log")
    right.set_xlim(0.05, 3)
    right.axvline(1.0, color=INK, ls="--", lw=1.8)
    right.text(1.05, len(rows) - 0.55, "tolerance", fontsize=13, color=INK, va="center")
    right.set_yticklabels([])
    right.set_xlabel("dynamics error / tolerance")
    right.set_title("Accuracy  (below 1 = valid)", loc="left")
    fig.suptitle(f"What each stage delivers   (GEO to graveyard, {stages['hp']['meta']['revolutions']} revolutions)",
                 x=0.01, ha="left", fontsize=18, fontweight="bold", color=INK)
    fig.tight_layout()
    save(fig, "5_results.png")


def main():
    stages = {s: load(s) for s in STAGES}
    fig_stages(stages)
    fig_transfer_time(stages)
    fig_accuracy(stages)
    fig_cost(stages)
    fig_results(stages)
    fig_solution()



# --------------------------------------------------------------------------- 6
def fig_solution(path=HERE.parent / "results" / "p_to_ms_to_hp" / "solution.npz"):
    """The pipeline's answer: the orbit raised from GEO to the graveyard orbit."""
    import sys
    sys.path.insert(0, str(HERE.parent))
    from solution import load_solution
    s = load_solution(path)
    days = s.dense_elapsed_days
    color = COLOR["hp"]
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(13, 6.4), sharex=True)
    rp = s.dense_perigee_radius_km - s.geo_radius_km
    target = s.target_perigee_km - s.geo_radius_km
    top.plot(days, rp, color=color, lw=2.2)
    top.axhline(target, color=INK, ls="--", lw=1.6)
    top.text(0.2, target + 8, f"graveyard: perigee {target:.0f} km above GEO", fontsize=13, color=INK, va="bottom")
    top.set_ylabel("perigee above GEO (km)")
    top.set_ylim(min(rp.min(), 0) - 15, target + 60)
    bottom.plot(days, s.dense_eccentricity, color=color, lw=1.4)
    bottom.axhline(s.target_eccentricity, color=INK, ls="--", lw=1.6)
    bottom.text(0.2, s.target_eccentricity * 1.04, f"eccentricity cap {s.target_eccentricity:g}", fontsize=13,
                color=INK, va="bottom")
    bottom.set_ylabel("eccentricity")
    bottom.set_ylim(0, s.target_eccentricity * 1.3)
    bottom.set_xlabel("time (days)")
    bottom.set_xlim(0, days[-1] * 1.01)
    top.plot(days[-1], rp[-1], "o", color=color, ms=10, zorder=3)
    top.annotate(f"reached in {days[-1]:.2f} days\n({s.revolutions} revolutions)", (days[-1], rp[-1]),
                 xytext=(-30, -95), textcoords="offset points", ha="right", va="top", fontsize=14,
                 arrowprops=dict(arrowstyle="-", color=INK2, lw=1.2, shrinkB=8),
                 color=INK, fontweight="bold")
    fig.suptitle("Pipeline result: solar sail raises the orbit from GEO to the graveyard orbit",
                 x=0.01, ha="left", fontsize=18, fontweight="bold", color=INK)
    fig.tight_layout()
    save(fig, "6_solution.png")


if __name__ == "__main__":
    main()
