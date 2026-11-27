"""Summary tables of results/study/study.csv, written to results/study/summary.md, and the figures of
results/study/figures/ (png and pdf).

    python study_tables.py           tables and figures from the existing results
    python study_tables.py --run     first solve (sweep.py) the cases of the tables that have no result yet
    python study_tables.py --fresh   all of them from scratch: the existing results/study is renamed first
    --term                           with --run or --fresh: also the autoscvx-term cases (~800, slow to fail)

A full solve takes about two and a half days on 20 cores and 63 GB.
"""
import csv
import json
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

HERE = Path(__file__).resolve().parent
STUDY = HERE / "results" / "study"
BLOCKS = ["A", "B", "J", "C", "D", "F", "G", "T", "K", "I", "L", "M", "N", "O"]   # every block the tables use
if "--fresh" in sys.argv and STUDY.exists():
    STUDY.rename(STUDY.with_name(time.strftime("study_%Y%m%d_%H%M")))
if "--run" in sys.argv or "--fresh" in sys.argv:
    term = [] if "--term" in sys.argv else ["--autoscvx"]
    subprocess.run([sys.executable, str(HERE / "sweep.py"), *BLOCKS, *term], cwd=HERE, check=True)

DAY = 86400.0
LOOSE = 1.0e4
rows = list(csv.DictReader(open(STUDY / "study.csv")))
for r in rows:
    for key in ["w", "tr", "eps_feas", "eps_cost", "first_feas", "first_cost", "transfer_days", "time_s",
                "peak_gb", "perigee_short_km"]:
        r[key] = float(r[key]) if r[key] else None
    r["iterations"] = int(r["iterations"]) if r["iterations"] else None
    r["ok"] = r["converged"] == "True"
    r["start"] = r["guess"][:2] + (r["seed"] or "")   # st, pr, st0..st9, pr0..pr9
    r["stage_its"] = [int(s.split()[-1]) for s in r["per_stage"].split(" | ") if s.split()[-1].isdigit()]

FAMILIES = {"p13": "p13", "hp13": "hp2x7", "hp25": "hp4x7", "ms": "ms12x6", "p13-ms": "p13-ms12x6",
            "hp13-ms": "hp2x7-ms12x6", "msc-ms": "ms12x6-ms12x6", "hp13-hp25": "hp2x7-hp4x7",
            "ms-hp25": "ms12x6-hp4x7", "hp13-ms-hp25": "hp2x7-ms12x6-hp4x7"}
REF = min(r["transfer_days"] for r in rows if r["blocks"] == "J")   # fine meshes at 1e-8, both agree to 1 s


def nodes(stage):
    kind, a, b = re.fullmatch(r"(hp|p|ms)(\d+)(?:x(\d+))?", stage).groups()
    a, b = int(a), int(b or 0)
    return {"p": a, "hp": a * (b - 1) + 1, "ms": a + 1}[kind]


def describe(stage):
    kind, a, b = re.fullmatch(r"(hp|p|ms)(\d+)(?:x(\d+))?", stage).groups()
    if kind == "p":
        return f"p {a}"
    if kind == "hp":
        return f"hp {a}x{b}"
    return f"ms {a}x{b} ({int(a) * int(b)} RK4)"


def short(stages):   # for the figures
    return " → ".join(describe(s).split(" (")[0] for s in stages.split("-"))


def pipeline(stages):
    return " → ".join(f"{describe(s)} [{nodes(s)}]" for s in stages.split("-"))


def default(r, family=None):
    """Nominal settings: w 30, tr 10, eps 1e-6, 21 revs, autoscvx, nominal first stage, weights carried."""
    stages = r["stages"].split("-")
    multi = len(stages) > 1 and r["family"] != "ms-hp25"
    first = (r["first_feas"], r["first_cost"]) == ((1e-4, LOOSE) if multi else (None, None))
    return (r["w"] == 30 and r["tr"] == 10 and r["eps_feas"] == 1e-6 and r["eps_cost"] == 1e-6 and r["revs"] == "21"
            and set(r["methods"]) == {"A"} and r["repeat"] == "0" and not r["first_iter"] and r["carry"] == "True"
            and first and (family is None or r["stages"] == FAMILIES[family]))


def unique(group):
    return list({r["start"]: r for r in group}.values())


def tally(group):
    return sum(r["ok"] for r in group), len(group)


def count(group):
    return "{}/{}".format(*tally(group)) if group else "-"


def med(group, key):
    values = [r[key] for r in group if r["ok"] and r[key] is not None]
    return statistics.median(values) if values else None


def median(group, key, fmt="{:.0f}"):
    value = med(group, key)
    return "-" if value is None else fmt.format(value)


def seconds_above(group):   # transfer time above the reference, s
    values = [(r["transfer_days"] - REF) * DAY for r in group if r["ok"]]
    return statistics.median(values) if values else None


def gap(group):
    value = seconds_above(group)
    return "-" if value is None else f"{value:+.0f}"


def table(header, body):
    return ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + ["| " + " | ".join(map(str, b)) + " |" for b in body]


out = [f"# Solar sail study: summary of {len(rows)} cases", "",
       f"Reference transfer: {REF:.5f} d (fine pipelines at eps 1e-8, block J). Δt is the median transfer time above it,",
       "over the converged runs. Perigee miss: open-loop flight of the solution's controls, km short of the target",
       "perigee (negative: past it). Times are solve time with the JIT warm-up excluded; the sweep ran up to 16",
       "cases at once, so only the block I column is free of load. Mesh in brackets: nodes per sunlit arc",
       "(p: N; hp SxN: S(N-1)+1; ms IxK: I+1 nodes, K RK4 steps per interval). Unless stated: autoscvx, w 30,",
       "tr 10, eps_feas = eps_cost = 1e-6, 21 revolutions, iter_max 300 per stage, first stage of a pipeline at",
       "feas 1e-4 with no cost check, penalty weights carried. Starts: st/pr = straight-line / propagated guess,",
       "unperturbed; 22 starts = both, unperturbed and 10 random perturbations each.", ""]

# ---------------------------------------------------------------- 1. single stage, mesh size (blocks A, B)
out += ["## 1. Single stage against mesh size (blocks A, B; unperturbed st and pr)", ""]
body, meshes = [], []
for kind in ["p", "hp", "ms"]:
    stages = {r["stages"] for r in rows if set(r["blocks"].split()) & {"A", "B"}
              and re.fullmatch(rf"{kind}\d+(x\d+)?", r["stages"])}
    for s in sorted(stages, key=lambda s: (nodes(s), s)):
        group = [r for r in rows if r["stages"] == s and r["methods"] == "A" and not r["seed"] and r["w"] == 30
                 and r["tr"] == 10 and r["revs"] == "21" and r["repeat"] == "0"]
        tight = [r for r in group if r["eps_feas"] == 1e-7 and r["eps_cost"] == 1e-7]
        nominal = unique([r for r in group if r["eps_feas"] == 1e-6 and r["eps_cost"] == 1e-6])
        body.append([describe(s), nodes(s), count(nominal), count(unique(tight)), median(nominal, "iterations"),
                     median(nominal, "time_s"), gap(nominal), median(nominal, "perigee_short_km", "{:.3f}"),
                     median(nominal, "peak_gb", "{:.1f}")])
        meshes.append(dict(kind=kind, nodes=nodes(s), time=med(nominal, "time_s"), gb=med(nominal, "peak_gb"),
                           miss=med(nominal, "perigee_short_km")))
out += table(["stage", "nodes/arc", "conv 1e-6", "conv 1e-7", "iters", "time s", "Δt s", "perigee miss km", "peak GB"], body)
out += ["", "p stops at 25 nodes (dense blocks); every row at 1e-6 has 2 runs (st, pr).", ""]

# ---------------------------------------------------------------- 2. pipelines at nominal settings
out += ["## 2. Discretizations and pipelines at the nominal settings (blocks C, F, I)", ""]
body, speed = [], {}
for family, stages in FAMILIES.items():
    group = unique([r for r in rows if default(r, family) and not r["family"].startswith("T")])
    st = [r for r in group if r["guess"] == "straight_line"]
    pr = [r for r in group if r["guess"] == "propagation"]
    timing = [r for r in rows if r["stages"] == stages and r["repeat"] != "0" and set(r["methods"]) == {"A"}]
    its = [r["stage_its"] for r in group if r["ok"]]
    per_stage = " + ".join(f"{statistics.median(k):.0f}" for k in zip(*its)) if its else "-"
    body.append([family, pipeline(stages), count(st), count(pr), per_stage, median(timing, "time_s"),
                 median(group, "time_s"), gap(group), median(group, "perigee_short_km", "{:.3f}"),
                 median(group, "peak_gb", "{:.1f}")])
    stage_times = [json.loads((STUDY / r["name"] / "case.json").read_text())["stages"] for r in timing]
    speed[family] = dict(stage_times=[statistics.median(s[k]["time_s"] for s in stage_times)
                                      for k in range(len(stages.split("-")))] if stage_times else [],
                         gap=seconds_above(group), miss=med(group, "perigee_short_km"))
out += table(["pipeline", "stages [nodes/arc]", "conv st (11)", "conv pr (11)", "iters per stage", "time s alone (I)",
              "time s (sweep)", "Δt s", "perigee miss km", "peak GB"], body)
out += ["", "Iterations and times: medians over the converged starts; block I: unperturbed st, 3 repeats run alone.", ""]

# ---------------------------------------------------------------- 3. initial guess of the final stage
out += ["## 3. Initial guess of the final stage (blocks C, D, F; unperturbed st and pr)", "",
        "The final stage's iterations and convergence by where its guess comes from. 'first-stage tolerances':",
        "the 4 first-stage settings of block D (feas 1e-5, 1e-4, 1e-3 without cost check; 1e-4 with cost 1e-6),", ""]
body, guesses = [], {"ms12x6": [], "hp4x7": []}
for final in guesses:
    for source in ["straight_line", "propagation"]:
        group = [r for r in rows if r["stages"] == final and default(r) and not r["seed"] and r["guess"] == source]
        body.append([describe(final), source.replace("_", " "), "-", count(group),
                     median(group, "iterations"), "-", gap(group)])
        guesses[final].append(("guess", source.replace("_", " ").replace("propagation", "propagated"),
                               med(group, "iterations")))
    firsts = sorted({r["stages"].rsplit("-", 1)[0] for r in rows
                     if r["stages"].endswith("-" + final) and r["stages"].count("-") == 1})
    for first in sorted(firsts, key=lambda s: (s[0], nodes(s))):
        stages = f"{first}-{final}"
        group = [r for r in rows if r["stages"] == stages and set(r["methods"]) == {"A"} and not r["seed"]
                 and r["w"] == 30 and r["tr"] == 10 and r["revs"] == "21" and r["eps_feas"] == 1e-6
                 and r["eps_cost"] == 1e-6 and r["repeat"] == "0" and r["carry"] == "True" and not r["first_iter"]]
        nominal = [r for r in group if default(r)]
        sweep = [r for r in group if (r["first_feas"], r["first_cost"]) in
                 [(1e-5, LOOSE), (1e-4, LOOSE), (1e-3, LOOSE), (1e-4, 1e-6)]]
        final_its = [r["stage_its"][-1] for r in nominal if r["ok"]]
        first_its = [r["stage_its"][0] for r in nominal if r["ok"]]
        body.append([describe(final), f"{describe(first)} [{nodes(first)}] solution",
                     f"{statistics.median(first_its):.0f}" if first_its else "-", count(nominal),
                     f"{statistics.median(final_its):.0f}" if final_its else "-", count(sweep), gap(nominal)])
        guesses[final].append((re.match(r"hp|p|ms", first).group(), f"{short(first)} solution",
                               statistics.median(final_its) if final_its else None))
out += table(["final stage", "guess", "first-stage iters", "conv", "final-stage iters",
              "conv over first-stage tolerances", "Δt s"], body)
out += ["", "ms12x6 → hp4x7 is a polish: its first stage is a converged ms solve (tolerances: final, 1e-5, 1e-4 loose, 1e-4/1e-6).", ""]

# ---------------------------------------------------------------- 4. robustness to hyperparameters
out += ["## 4. Robustness: converged runs over each hyperparameter set", "",
        "starts: 22 starts (F); w x tr: cost weight {1,3,10,30,100} x trust region {1,10,100}, st (G);",
        "tolerances: eps_feas x eps_cost in {1e-5,1e-6,1e-7}, st (T); first stage: block D variants, st and pr;",
        "instances: 7 and 14 revolutions, st and pr (K).", ""]
body, robust = [], {}
for family, stages in FAMILIES.items():
    base = [r for r in rows if r["stages"] == stages and set(r["methods"]) == {"A"} and r["repeat"] == "0"]
    starts = unique([r for r in base if default(r)])
    grid = [r for r in base if not r["seed"] and r["guess"] == "straight_line" and r["revs"] == "21"
            and r["eps_feas"] == 1e-6 and r["eps_cost"] == 1e-6 and r["carry"] == "True" and not r["first_iter"]
            and (r["first_feas"], r["first_cost"]) == ((1e-4, LOOSE) if "-" in stages and family != "ms-hp25" else (None, None))]
    wtr = list({(r["w"], r["tr"]): r for r in grid}.values())
    tol = list({(r["eps_feas"], r["eps_cost"]): r for r in base if not r["seed"] and r["guess"] == "straight_line"
                and r["w"] == 30 and r["tr"] == 10 and r["revs"] == "21" and r["carry"] == "True"
                and not r["first_iter"] and "T" in r["blocks"].split()}.values())
    d = [r for r in rows if r["family"] == family and "D" in r["blocks"].split() and set(r["methods"]) == {"A"}]
    inst = [r for r in base if r["revs"] in ["7", "14"] and default(dict(r, revs="21"))]
    w30 = [r for r in wtr if r["w"] == 30]
    body.append([family, pipeline(stages), count(starts), count(wtr), count(w30), count(tol), count(d), count(inst)])
    robust[family] = dict(starts=starts, wtr=wtr, tol=tol)
out += table(["pipeline", "stages [nodes/arc]", "starts", "w x tr", "(w 30 only)", "tolerances", "first stage",
              "instances"], body)
out += [""]

# ---------------------------------------------------------------- 5. autoscvx-term
out += ["## 5. autoscvx-term (hard dynamics, penalty on the final state only)", ""]
term = [r for r in rows if "T" in r["methods"]]
body = []
for r in term:
    body.append([describe(r["stages"]), r["guess"][:2], f"{r['eps_feas']:.0e}", r["final_status"], r["iterations"],
                 f"{r['time_s']:.0f}"])
out += table(["stage", "start", "eps", "status", "iters", "time s"], body) if term else ["No autoscvx-term case solved."]
out += ["", f"Only {len(term)} of the planned term cases ran; the rest were left out of the overnight runs after",
        "these failed. With the dynamics hard the first subproblem is infeasible unless the terminal cones and arc-end",
        "windows are softened too; with them softened the solver still ends without a usable trajectory after 20-30",
        "iterations (~10 s per iteration, against ~0.5 s for autoscvx).", ""]

# ---------------------------------------------------------------- 6. first stage as a cheap first pass
out += ["## 6. First stage as a cheap first pass (blocks L-O; 22 starts, no cost check, weights carried)", ""]
feas = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
body, first_pass = [], {}
for family in ["msc-ms", "hp13-ms", "hp13-hp25", "ms-hp25"]:
    for carry in ["True", "False"]:
        cells = []
        for f in feas:
            group = unique([r for r in rows if r["stages"] == FAMILIES[family] and r["first_feas"] == f
                            and r["first_cost"] == LOOSE and r["carry"] == carry and not r["first_iter"]
                            and r["eps_feas"] == 1e-6 and r["eps_cost"] == 1e-6 and r["w"] == 30 and r["tr"] == 10 and r["revs"] == "21"
                            and set(r["methods"]) == {"A"} and r["repeat"] == "0"])
            cells.append(f"{count(group)} ({median(group, 'time_s')} s)" if len(group) >= 20 else "-")
            if len(group) >= 20:
                first_pass[family, carry, f] = group
        body.append([family, "yes" if carry == "True" else "no"] + cells)
out += table(["pipeline", "carry W", *[f"first feas {f:.0e}" for f in feas]], body)
out += ["", "Converged / starts (median solve time of the converged). First-stage cost tolerance 1e-6 instead of no",
        "check and iteration caps 10/30 never change the counts; hp first stages capped at 10 iterations drop to 6/22.", ""]

(STUDY / "summary.md").write_text("\n".join(out), encoding="utf-8")
print("\n".join(out))

# ---------------------------------------------------------------- figures

FIGURES = STUDY / "figures"
FIGURES.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.alpha": 0.3, "axes.axisbelow": True, "figure.constrained_layout.use": True,
                     "savefig.dpi": 200})
BLUE, ORANGE, AQUA, YELLOW, GRAY, LIGHT = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#8c8c8c", "#e6e6e6"
KIND = {"p": BLUE, "hp": ORANGE, "ms": AQUA}
OFFSET = {("hp13-ms-hp25", 0): (-8, 0, "right"), ("hp13-ms-hp25", 1): (0, -11, "center"), ("ms-hp25", 1): (0, 11, "center"),   # label placement
          ("msc-ms", 1): (0, 11, "center"), ("hp13-ms", 1): (0, -11, "center")}
NAMES = [short(stages) for stages in FAMILIES.values()]
y = np.arange(len(FAMILIES))


def save(fig, name):
    fig.savefig(FIGURES / f"{name}.png")
    fig.savefig(FIGURES / f"{name}.pdf")
    plt.close(fig)


# starts x pipelines: which starts converge
STARTS = ["st"] + [f"st{k}" for k in range(10)] + ["pr"] + [f"pr{k}" for k in range(10)]
fig, ax = plt.subplots(figsize=(11, 4.6))
cells = np.array([[next((r["ok"] for r in robust[f]["starts"] if r["start"] == s), np.nan) for s in STARTS]
                  for f in FAMILIES], dtype=float)
ax.imshow(cells, cmap=ListedColormap([LIGHT, BLUE]), aspect="auto", vmin=0, vmax=1)
ax.set_xticks(range(len(STARTS)), STARTS, rotation=90, fontsize=9)
ax.set_yticks(y, [f"{n}  ({tally(robust[f]['starts'])[0]})" for n, f in zip(NAMES, FAMILIES)])
ax.set_xticks(np.arange(len(STARTS)) - 0.5, minor=True)
ax.set_yticks(y - 0.5, minor=True)
ax.grid(which="minor", color="white", linewidth=2)
ax.grid(which="major", visible=False)
ax.tick_params(which="minor", length=0)
ax.axvline(10.5, color="k", linewidth=1)
ax.set_xlabel("start: straight-line (st) or propagated (pr) guess, unperturbed and 10 random perturbations")
ax.set_title("Converged (blue) per start and pipeline, nominal settings; in brackets: converged starts of 22", fontsize=11)
save(fig, "starts")


def grid_map(ax, group, xkey, ykey, xs, ys, nominal):
    cells = {(r[xkey], r[ykey]): r for r in group}
    for i, yv in enumerate(ys):
        for j, xv in enumerate(xs):
            r = cells.get((xv, yv))
            if r is None:
                continue
            ax.add_patch(plt.Rectangle((j - 0.45, i - 0.45), 0.9, 0.9, color=BLUE if r["ok"] else LIGHT))
            ax.text(j, i, r["iterations"] if r["ok"] else "", ha="center", va="center", fontsize=8, color="white")
    j, i = xs.index(nominal[0]), ys.index(nominal[1])
    ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="k", linewidth=1.5))
    ax.set_xlim(-0.5, len(xs) - 0.5)
    ax.set_ylim(-0.5, len(ys) - 0.5)
    ax.grid(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0)


def maps(key, xkey, ykey, xs, ys, nominal, xlabel, ylabel, fmt, title, name):
    fig, axes = plt.subplots(2, 5, figsize=(13, 5.6), sharex=True, sharey=True)
    for ax, family, label in zip(axes.flat, FAMILIES, NAMES):
        grid_map(ax, robust[family][key], xkey, ykey, xs, ys, nominal)
        ax.set_title(f"{label}  ({tally(robust[family][key])[0]}/{len(xs) * len(ys)})", fontsize=9)
    for ax in axes[:, 0]:
        ax.set_yticks(range(len(ys)), [fmt(v) for v in ys])
        ax.set_ylabel(ylabel)
    for ax in axes[1]:
        ax.set_xticks(range(len(xs)), [fmt(v) for v in xs])
        ax.set_xlabel(xlabel)
    fig.suptitle(title)
    save(fig, name)


maps("wtr", "w", "tr", [1.0, 3.0, 10.0, 30.0, 100.0], [1.0, 10.0, 100.0], (30.0, 10.0), "cost weight w",
     "trust region tr (larger: looser)", lambda v: f"{v:g}", "Converged (blue, with iterations) over cost weight and trust region; "
     "straight-line start, eps 1e-6, framed: nominal", "weight_trust_region")
maps("tol", "eps_cost", "eps_feas", [1e-5, 1e-6, 1e-7], [1e-5, 1e-6, 1e-7], (1e-6, 1e-6), "eps_cost", "eps_feas",
     lambda v: f"{v:.0e}", "Converged (blue, with iterations) over the final tolerances; straight-line start, w 30, "
     "tr 10, framed: nominal", "tolerances")

# solve time against optimality and flight accuracy of each pipeline
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True)
for family, name in zip(FAMILIES, NAMES):
    s = speed[family]
    colour = KIND[re.match(r"hp|p|ms", FAMILIES[family].split("-")[-1]).group()]
    for k, (ax, value) in enumerate(zip(axes, [s["gap"], abs(s["miss"]) * 1e3])):
        ax.plot(sum(s["stage_times"]), value, "o", color=colour, markersize=8)
        dx, dy, ha = OFFSET.get((family, k), (6, 0, "left"))
        ax.annotate(name, (sum(s["stage_times"]), value), xytext=(dx, dy), textcoords="offset points", fontsize=8,
                    va="center", ha=ha)
axes[0].set_ylabel("transfer time above the reference (s)")
axes[1].set_ylabel("perigee miss, open-loop flight (m)")
axes[1].set_yscale("log")
for ax in axes:
    ax.set_xlabel("solve time, run alone (s)")
    ax.set_xlim(55, 175)
for kind in ["p", "hp", "ms"]:
    axes[0].plot([], [], "o", color=KIND[kind], label=f"final stage {kind}")
axes[0].legend(frameon=False, fontsize=9, loc="lower left")
fig.suptitle("Pipelines at the nominal settings (medians over the converged starts of 22)")
save(fig, "time_optimality")

# convergence of a few pipelines from the unperturbed straight-line start
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
for family, colour in zip(["hp25", "hp13-hp25", "ms-hp25"], [ORANGE, YELLOW, BLUE]):   # ms alone = first stage of ms-hp25
    r = next(r for r in rows if default(r, family) and r["start"] == "st")
    done = 0
    for k, stage in enumerate(json.loads((STUDY / r["name"] / "case.json").read_text())["stages"]):
        h = stage["history"]
        its = done + np.arange(1, len(h["cost"]) + 1)
        days = np.array(h["cost"]) * stage["transfer_days"] / h["cost"][-1]
        axes[0].plot(its, h["dynamics"], color=colour, linewidth=1.5, label=short(r["stages"]) if k == 0 else None)
        gap_s = np.where(np.array(h["dynamics"]) < 1, (days - REF) * DAY, np.nan)   # feasible iterates only
        axes[1].plot(its, gap_s, color=colour, linewidth=1.5)
        if k:
            for ax, value in zip(axes, [h["dynamics"][0], gap_s[0]]):
                ax.plot(its[0], value, "o", color=colour, markersize=6)
        done = its[-1]
axes[0].axhline(1, color=GRAY, linestyle=":", linewidth=1.5)
axes[0].set_yscale("log")
axes[0].set_ylabel("largest dynamics buffer / tolerance")
axes[1].set_ylabel("transfer time above the reference (s)")
axes[1].set_title("iterates within the dynamics tolerance", fontsize=10)
axes[1].set_ylim(-50, 1200)
axes[1].axhline(0, color=GRAY, linestyle=":", linewidth=1.5)
for ax in axes:
    ax.set_xlabel("iteration (all stages, dot: start of the next stage)")
axes[0].legend(frameon=False, fontsize=9)
fig.suptitle("Convergence from the unperturbed straight-line start (ms 12x6 alone: the first 119 iterations of the blue line)")
save(fig, "convergence")

# single stages against mesh size
fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
for kind, marker, face in [("ms", "^", AQUA), ("hp", "s", ORANGE), ("p", "o", "none")]:   # p hollow: it shares sizes with hp
    done = [m for m in meshes if m["kind"] == kind and m["time"] is not None]
    for ax, key in zip(axes, ["miss", "time", "gb"]):
        ax.plot([m["nodes"] for m in done], [abs(m[key]) * (1e3 if key == "miss" else 1) for m in done], marker,
                color=KIND[kind], markerfacecolor=face, markersize=11 if kind == "p" else 8,
                markeredgewidth=2 if kind == "p" else 1, alpha=0.8, label=kind, linestyle="none")
axes[0].set_yscale("log")
axes[0].set_ylabel("perigee miss, open-loop flight (m)")
axes[1].set_ylabel("solve time, other cases running (s)")
axes[2].set_ylabel("peak memory (GB)")
for ax in axes:
    ax.set_xlabel("nodes per sunlit arc")
axes[0].legend(frameon=False)
fig.suptitle("Single stage against mesh size (eps 1e-6, medians over the st and pr starts)")
save(fig, "mesh_size")

# first stage as a cheap first pass
fig, ax = plt.subplots(figsize=(7, 4.5))
lines = {}   # pipelines with the same counts share a line
for family in ["msc-ms", "ms-hp25", "hp13-ms", "hp13-hp25"]:
    lines.setdefault(tuple(tally(first_pass[family, "True", f])[0] for f in feas), []).append(short(FAMILIES[family]))
for (counts, names), colour in zip(lines.items(), [BLUE, ORANGE, AQUA, YELLOW]):
    ax.plot(feas, counts, "o-", color=colour, linewidth=2, markersize=7, label=" and ".join(names))
single = tally(robust["ms"]["starts"])[0]
ax.axhline(single, color=GRAY, linestyle=":", linewidth=1.5)
ax.text(1.2e-2, single + 0.3, "ms 12x6 alone", color=GRAY, ha="right", fontsize=9)
ax.set_xscale("log")
ax.set_ylim(0, 22)
ax.set_yticks([0, 5, 10, 15, 20, 22])
ax.set_xlabel("feasibility tolerance of the first stage")
ax.set_ylabel("converged starts (of 22)")
ax.legend(frameon=False, fontsize=9, loc="lower left")
ax.set_title("Loosening the first stage (final eps 1e-6, weights carried)", fontsize=11)
save(fig, "first_pass")

# why the same starts fail: penalty weight and dynamics defect of the first stage
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
for r in first_pass["msc-ms", "True", 1e-6]:
    history = json.loads((STUDY / r["name"] / "case.json").read_text())["stages"][0]["history"]
    colour = BLUE if r["ok"] else ORANGE
    axes[0].plot(history["weight"], color=colour, linewidth=1, alpha=0.7)
    axes[1].plot(history["dynamics"], color=colour, linewidth=1, alpha=0.7)
axes[0].set_ylabel("largest penalty weight")
axes[1].set_ylabel("largest dynamics buffer / tolerance")
axes[1].axhline(1, color=GRAY, linestyle=":", linewidth=1.5)
for ax in axes:
    ax.set_yscale("log")
    ax.set_xlabel("iteration of the first stage")
axes[0].plot([], [], color=BLUE, label="pipeline converged")
axes[0].plot([], [], color=ORANGE, label="pipeline failed")
axes[0].legend(frameon=False, fontsize=9)
fig.suptitle("First stage (ms 12x6, feasibility 1e-6) of ms 12x6 → ms 12x6, from the 22 starts")
save(fig, "weight_growth")
print(f"figures in {FIGURES}")
