"""Summary figures of results/sweep/, from the case.json of every case. Figures go to results/sweep/."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.size": 9, "figure.figsize": (6.5, 5.5), "figure.constrained_layout.use": True,
                     "axes.grid": True, "grid.alpha": 0.3, "lines.linewidth": 1.2, "savefig.dpi": 300})

SWEEP = Path("results") / "sweep"
RATE_LIMIT = True   # the formulation to summarize
ALGORITHMS = ["p", "ms", "p_ms", "msc_ms", "p_hp", "p_ms_hp"]
WEIGHTS = [1.0, 30.0]
DEFAULT = dict(p_eps_cost=1.0e4, p_budget_km=30.0)   # first-stage setting shared by every pipeline

cases = [json.loads(file.read_text()) for file in sorted(SWEEP.glob("*/case.json"))]
cases = [case for case in cases if case["rate_limit"] == RATE_LIMIT]
for case in cases:
    stages = case["stages"]
    case["converged"] = len(stages) == len(case["algorithm"].split("_")) and stages[-1]["converged"]
    case["iterations"] = sum(stage.get("iterations", 0) for stage in stages)
    case["time_s"] = sum(stage.get("time_s", 0.0) for stage in stages)
    case["transfer_days"] = stages[-1].get("transfer_days")


def save(fig, name):
    fig.savefig(SWEEP / f"{name}.png")
    fig.savefig(SWEEP / f"{name}.pdf")


def default_case(algorithm, controls, weight):
    """The case of an algorithm at the shared first-stage setting, with the weights carried over."""
    for case in cases:
        if (case["algorithm"] == algorithm and case["controls"] == controls and case["cost_weight"] == weight
                and case.get("carry_weights", True)
                and all(case.get(key, value) == value for key, value in DEFAULT.items())):
            return case


def marker(case, colour):
    """Filled marker when the case converged, hollow when it did not."""
    return dict(color=colour, marker="o", markerfacecolor=colour if case["converged"] else "none", linestyle="none")


# ---------------------------------------------------------------- how often each algorithm converges

fig, axes = plt.subplots(1, 2, sharey=True, figsize=(8, 3.2))
x = np.arange(len(ALGORITHMS))
for ax, controls in zip(axes, ["ramp", "zero"]):
    for k, weight in enumerate(WEIGHTS):
        rates, counts = [], []
        for algorithm in ALGORITHMS:
            group = [c for c in cases if c["algorithm"] == algorithm and c["controls"] == controls
                     and c["cost_weight"] == weight]
            rates.append(np.mean([c["converged"] for c in group]) if group else 0.0)
            counts.append(len(group))
        bars = ax.bar(x + (k - 0.5) * 0.35, rates, width=0.35, label=f"w = {weight:g}")
        for bar, count in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, str(count), ha="center", fontsize=7)
    ax.set_xticks(x, ALGORITHMS)
    ax.set_title(f"control guess: {controls}", fontsize=9)
axes[0].set_ylabel("fraction of cases converged")
axes[0].set_ylim(0, 1.15)
axes[0].legend()
fig.suptitle(f"rate limit {'on' if RATE_LIMIT else 'off'}   (number above a bar: cases in the group)", fontsize=9)
save(fig, "sweep_convergence")


# ---------------------------------------------------------------- first-stage settings of p_ms and msc_ms

fig, axes = plt.subplots(1, 2, sharey=True, figsize=(8, 3.5))
for ax, weight in zip(axes, WEIGHTS):
    for colour, algorithm in zip(["C0", "C3"], ["p_ms", "msc_ms"]):
        for style, eps_cost in zip(["-", "--"], [1.0e4, 1.0e-3]):
            group = sorted([c for c in cases if c["algorithm"] == algorithm and c["controls"] == "ramp"
                            and c["cost_weight"] == weight and c.get("carry_weights", True)
                            and c["p_eps_cost"] == eps_cost], key=lambda c: c["p_budget_km"])
            ax.plot([c["p_budget_km"] for c in group], [c["iterations"] for c in group], style, color=colour,
                    label=f"{algorithm}, first-stage eps_cost {eps_cost:g}")
            for c in group:
                ax.plot(c["p_budget_km"], c["iterations"], **marker(c, colour))
        for c in cases:
            if c["algorithm"] == algorithm and c["controls"] == "ramp" and c["cost_weight"] == weight \
                    and not c.get("carry_weights", True):
                ax.plot(c["p_budget_km"], c["iterations"], "x", color=colour, markersize=8)
    ax.set_xscale("log")
    ax.set_title(f"w = {weight:g}", fontsize=9)
    ax.set_xlabel("first-stage defect budget [km]")
axes[0].set_ylabel("iterations, all stages")
axes[0].legend(fontsize=7)
fig.suptitle("ramp control guess   (filled: converged; hollow: not; x: weights not carried over)", fontsize=9)
save(fig, "sweep_first_stage")


# ---------------------------------------------------------------- iterations and time per stage, shared setting

fig, (top, bottom) = plt.subplots(2, 1, sharex=True, figsize=(6.5, 5.0))
labels, colours = [], {"p": "C0", "ms": "C3", "msc": "C1", "hp": "C2"}
for i, (algorithm, weight) in enumerate([(a, w) for w in WEIGHTS for a in ALGORITHMS]):
    case = default_case(algorithm, "ramp", weight)
    labels.append(f"{algorithm}\nw = {weight:g}")
    iterations, seconds = 0, 0.0
    for stage in case["stages"]:
        hatch = None if stage.get("converged") else "///"
        top.bar(i, stage.get("iterations", 0), bottom=iterations, color=colours[stage["stage"]], hatch=hatch,
                edgecolor="k", linewidth=0.5)
        bottom.bar(i, stage.get("time_s", 0.0), bottom=seconds, color=colours[stage["stage"]], hatch=hatch,
                   edgecolor="k", linewidth=0.5)
        iterations += stage.get("iterations", 0)
        seconds += stage.get("time_s", 0.0)
for stage, colour in colours.items():
    top.bar(0, 0, color=colour, label=stage)
top.legend(ncol=4)
top.set_ylabel("iterations")
bottom.set_ylabel("wall time [s]")
bottom.set_xticks(range(len(labels)), labels, fontsize=7)
top.set_title("ramp control guess, first stage at eps_cost 1e4 and 30 km   (hatched: stage not converged)",
              fontsize=9)
save(fig, "sweep_stages")


# ---------------------------------------------------------------- which solution each case ends on

fig, ax = plt.subplots(figsize=(6.5, 3.2))
for k, algorithm in enumerate(ALGORITHMS):
    for weight, colour in zip(WEIGHTS, ["C0", "C1"]):
        group = [c for c in cases if c["algorithm"] == algorithm and c["controls"] == "ramp"
                 and c["cost_weight"] == weight and c["transfer_days"] is not None]
        offsets = np.linspace(-0.15, 0.15, len(group)) if len(group) > 1 else [0.0]
        for offset, c in zip(offsets, group):
            ax.plot(k + offset + (0.2 if weight == WEIGHTS[1] else -0.2), c["transfer_days"], **marker(c, colour))
for weight, colour in zip(WEIGHTS, ["C0", "C1"]):
    ax.plot([], [], "o", color=colour, label=f"w = {weight:g}")
ax.set_xticks(range(len(ALGORITHMS)), ALGORITHMS)
ax.set_ylabel("transfer time [days]")
ax.set_title("ramp control guess, every first-stage setting   (filled: converged; hollow: not)", fontsize=9)
ax.legend()
save(fig, "sweep_transfer_time")

# ---------------------------------------------------------------- p_ms and msc_ms against the cost weight

fig, (top, bottom) = plt.subplots(2, 1, sharex=True, figsize=(6.5, 5.0))
for colour, algorithm in zip(["C0", "C3"], ["p_ms", "msc_ms"]):
    group = sorted([c for c in cases if c["algorithm"] == algorithm and c["controls"] == "ramp"
                    and c.get("carry_weights", True) and all(c.get(k, v) == v for k, v in DEFAULT.items())],
                   key=lambda c: c["cost_weight"])
    weights = [c["cost_weight"] for c in group]
    top.plot(weights, [c["iterations"] for c in group], color=colour, label=f"{algorithm}, all stages")
    top.plot(weights, [c["stages"][0].get("iterations", 0) for c in group], "--", color=colour,
             label=f"{algorithm}, first stage")
    bottom.plot(weights, [c["transfer_days"] for c in group], color=colour, label=algorithm)
    for c in group:
        top.plot(c["cost_weight"], c["iterations"], **marker(c, colour))
        bottom.plot(c["cost_weight"], c["transfer_days"], **marker(c, colour))
top.set_xscale("log")
top.set_ylabel("iterations")
top.legend(fontsize=7)
bottom.set_xlabel("cost weight w")
bottom.set_ylabel("transfer time [days]")
top.set_title("ramp control guess, first stage at eps_cost 1e4 and 30 km   (filled: converged; hollow: not)",
              fontsize=9)
save(fig, "sweep_cost_weight")

for algorithm in ALGORITHMS:
    group = [c for c in cases if c["algorithm"] == algorithm]
    print(f"{algorithm:8s} {sum(c['converged'] for c in group):3d} / {len(group):3d} converged")
plt.show()
