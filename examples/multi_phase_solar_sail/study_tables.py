"""Summary tables of results/study/study.csv, written to results/study/summary.md.

    python study_tables.py           tables from the existing results
    python study_tables.py --run     first solve (sweep.py) the cases of the tables that have no result yet
    python study_tables.py --fresh   all of them from scratch: the existing results/study is renamed first
    --term                           with --run or --fresh: also the autoscvx-term cases (~800, slow to fail)

A full solve takes about two and a half days on 20 cores and 63 GB.
"""
import csv
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

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


def count(group):
    return f"{sum(r['ok'] for r in group)}/{len(group)}" if group else "-"


def median(group, key, fmt="{:.0f}"):
    values = [r[key] for r in group if r["ok"] and r[key] is not None]
    return fmt.format(statistics.median(values)) if values else "-"


def gap(group):   # transfer time above the reference, s
    values = [(r["transfer_days"] - REF) * DAY for r in group if r["ok"]]
    return f"{statistics.median(values):+.0f}" if values else "-"


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
body = []
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
out += table(["stage", "nodes/arc", "conv 1e-6", "conv 1e-7", "iters", "time s", "Δt s", "perigee miss km", "peak GB"], body)
out += ["", "p stops at 25 nodes (dense blocks); every row at 1e-6 has 2 runs (st, pr).", ""]

# ---------------------------------------------------------------- 2. pipelines at nominal settings
out += ["## 2. Discretizations and pipelines at the nominal settings (blocks C, F, I)", ""]
body = []
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
out += table(["pipeline", "stages [nodes/arc]", "conv st (11)", "conv pr (11)", "iters per stage", "time s alone (I)",
              "time s (sweep)", "Δt s", "perigee miss km", "peak GB"], body)
out += ["", "Iterations and times: medians over the converged starts; block I: unperturbed st, 3 repeats run alone.", ""]

# ---------------------------------------------------------------- 3. initial guess of the final stage
out += ["## 3. Initial guess of the final stage (blocks C, D, F; unperturbed st and pr)", "",
        "The final stage's iterations and convergence by where its guess comes from. 'first-stage tolerances':",
        "the 4 first-stage settings of block D (feas 1e-5, 1e-4, 1e-3 without cost check; 1e-4 with cost 1e-6),", ""]
body = []
for final in ["ms12x6", "hp4x7"]:
    for source in ["straight_line", "propagation"]:
        group = [r for r in rows if r["stages"] == final and default(r) and not r["seed"] and r["guess"] == source]
        body.append([describe(final), source.replace("_", " "), "-", count(group),
                     median(group, "iterations"), "-", gap(group)])
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
out += table(["final stage", "guess", "first-stage iters", "conv", "final-stage iters",
              "conv over first-stage tolerances", "Δt s"], body)
out += ["", "ms12x6 → hp4x7 is a polish: its first stage is a converged ms solve (tolerances: final, 1e-5, 1e-4 loose, 1e-4/1e-6).", ""]

# ---------------------------------------------------------------- 4. robustness to hyperparameters
out += ["## 4. Robustness: converged runs over each hyperparameter set", "",
        "starts: 22 starts (F); w x tr: cost weight {1,3,10,30,100} x trust region {1,10,100}, st (G);",
        "tolerances: eps_feas x eps_cost in {1e-5,1e-6,1e-7}, st (T); first stage: block D variants, st and pr;",
        "instances: 7 and 14 revolutions, st and pr (K).", ""]
body = []
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
body = []
for family in ["msc-ms", "hp13-ms", "hp13-hp25", "ms-hp25"]:
    for carry in ["True", "False"]:
        cells = []
        for f in feas:
            group = unique([r for r in rows if r["stages"] == FAMILIES[family] and r["first_feas"] == f
                            and r["first_cost"] == LOOSE and r["carry"] == carry and not r["first_iter"]
                            and r["eps_feas"] == 1e-6 and r["eps_cost"] == 1e-6 and r["w"] == 30 and r["tr"] == 10 and r["revs"] == "21"
                            and set(r["methods"]) == {"A"} and r["repeat"] == "0"])
            cells.append(f"{count(group)} ({median(group, 'time_s')} s)" if len(group) >= 20 else "-")
        body.append([family, "yes" if carry == "True" else "no"] + cells)
out += table(["pipeline", "carry W", *[f"first feas {f:.0e}" for f in feas]], body)
out += ["", "Converged / starts (median solve time of the converged). First-stage cost tolerance 1e-6 instead of no",
        "check and iteration caps 10/30 never change the counts; hp first stages capped at 10 iterations drop to 6/22.", ""]

(STUDY / "summary.md").write_text("\n".join(out), encoding="utf-8")
print("\n".join(out))
