"""The pseudospectral against multiple-shooting study: every case of the blocks below.

    python sweep.py            all blocks
    python sweep.py A B J      only these blocks
    python sweep.py --list     number of cases per block, nothing solved
    python sweep.py --autoscvx only the cases without autoscvx-term

One folder per case in results/study/ (case.json, log.txt, the stage configs, and the full stage results for
SAVE_BLOCKS), and one row per case in results/study/study.csv. Cases that already have a case.json are skipped,
so an interrupted sweep resumes where it stopped. A case is defined by its settings, so a case asked for by two
blocks is solved once. Cases run in a fixed random order, so a slow drift of the machine does not line up with
any factor; the timing repeats (block I) run alone, one at a time, after everything else.

Stages are written as ms<I>x<nsub> (I intervals per sunlit arc, nsub RK4 steps per interval), p<N> (one
segment of N nodes per sunlit arc) and hp<S>x<N> (S segments of N nodes). Eclipse arcs keep a fixed small mesh.
Methods are one letter per stage: A autoscvx, T autoscvx-term (hard dynamics, penalty on the final state).

Tolerances: eps_feas is the dynamics defect allowed per interval on the reference ms mesh (scaled units);
other meshes get eps_feas * intervals_ref / intervals, so every mesh has the same total defect budget.
The nonconvex constraints (terminator, arc-end window) get eps_feas unscaled. eps_cost is the change of
cost per unit cost weight (1e-6 is about 2 s of transfer time). The first stage of a pipeline uses
first_feas and first_cost ("loose" stops it once feasible); None means the final tolerances. first_iter
caps the iterations of the first stage (None: 300); a capped stage hands over whatever it reached.
Every stage: iter_max 300, flag_conv 2 (cost change and nonlinear feasibility), rate limit on, ramp steering.

Blocks L-O test the first stage as a cheap first pass: autoscvx multiplies a penalty weight by ~3 every
iteration until its buffer is below ~0.01-0.03 eps, so a tight eps lets the weights grow for many iterations
on a poor guess. Each first-stage setting runs on the unperturbed and the 10 perturbed guesses of both
families (22 starts), and the history keeps the largest weight per iteration. "carry" starts a stage from the
penalty weights the previous stage ended with (resampled to the new mesh); without it they start at 1.
"""
import csv
import ctypes
import itertools
import json
import random
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

import jax
import numpy as np
import yaml
from scipy.integrate import solve_ivp
from scipy.interpolate import BarycentricInterpolator

from trajopt.common.config_reader import load_trajopt_config
from trajopt.common.persistence import save_result
from trajopt.common.tools import AttrDict
from trajopt.formulations.stable.specs import normalize
from trajopt.methods.stable.scp.scp_method import SCPMethod

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "mission"))
import solar_sail  # noqa: E402

from solve import as_guess, carry_weights  # noqa: E402

jax.config.update("jax_enable_x64", True)

SWEEP = HERE / "results" / "study"
MEMORY_GB = 95         # sum of the estimated peaks of the cases running at once (short peaks rarely coincide)
MAX_RUNNING = 16
TIMEOUT_S = 7200       # the same for every case
SAVE_BLOCKS = {"C", "J"}   # full stage results (~15 MB each) only for these blocks; case.json has the rest
DAY = 86400.0
LOOSE = 1.0e4          # first-stage eps_cost that stops it once feasible
# autoscvx-term softens only the final state; here the targets are convex cones and the arc-end windows,
# and with these hard as well as the dynamics the first subproblem is infeasible
TERM_PENALTIES = dict(final_convex_inequality=dict(vb="standard"), final_nonconvex_inequality=dict(vb="standard"))

# ---------------------------------------------------------------- stages and pipelines

P13, HP13, HP25, MS = ("ps", 1, 13), ("ps", 2, 7), ("ps", 4, 7), ("ms", 12, 6)
PIPELINES = {          # the set every robustness block runs
    "p13": [P13], "hp13": [HP13], "hp25": [HP25], "ms": [MS],
    "p13-ms": [P13, MS], "hp13-ms": [HP13, MS], "msc-ms": [MS, MS], "hp13-hp25": [HP13, HP25],
    "ms-hp25": [MS, HP25], "hp13-ms-hp25": [HP13, MS, HP25],
}
POLISH = {"ms-hp25"}   # pipelines whose first stage is a converged solve, not a cheap start

PS_MESHES = [(1, 7), (2, 4),                      # (segments, nodes per segment): equal sizes per line
             (1, 13), (2, 7), (4, 4),
             (1, 25), (4, 7), (8, 4),             # p stops at 25 nodes
             (4, 10), (6, 7), (12, 4),
             (4, 13), (8, 7), (16, 4)]
MS_VALUES = [1, 2, 3, 6, 12, 24]
MS_MESHES = [(i, n) for i, n in itertools.product(MS_VALUES, MS_VALUES) if i * n >= 12]   # (intervals, nsub)
FIRST_STAGES = {P13: [("ps", 1, 7), P13, ("ps", 1, 19)],
                HP13: [("ps", 2, 4), HP13, ("ps", 3, 7)],
                MS: [("ms", 3, 24), ("ms", 6, 12), MS]}    # same RK4 steps per arc, fewer shooting intervals
FIRST_TOLERANCES = [(1e-5, LOOSE), (1e-4, LOOSE), (1e-3, LOOSE), (1e-4, 1e-6)]
POLISH_TOLERANCES = [(None, None), (1e-5, LOOSE), (1e-4, LOOSE), (1e-4, 1e-6)]
GUESSES = ["straight_line", "propagation"]
SEEDS = range(10)
WEIGHTS, TRUST_STEPS = [1.0, 3.0, 10.0, 30.0, 100.0], [1.0, 10.0, 100.0]
TOLERANCES = [1e-5, 1e-6, 1e-7]
INSTANCES = [7, 14]
FIRST_FEAS = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]   # first-pass grid (blocks L-O)
FIRST_COSTS = [1e-6, 1e-4, 1e-2, 1.0, LOOSE]   # the scaled cost is ~0.8 w: from 1.0 up there is no cost check
FIRST_ITERS = [10, 30]
# 1e-1 only on the cheap same-mesh pipeline: it can accept a collapsed transfer as feasible
FIRST_GRID = ([(f, c, None) for f, c in itertools.product(FIRST_FEAS + [1e-1], FIRST_COSTS)]
              + [(f, LOOSE, n) for f in [1e-6, 1e-3] for n in FIRST_ITERS])
# both cost extremes at every feasibility level, and the iteration caps: run with and without carried weights
FIRST_CARRY = ([(f, c, None) for f in FIRST_FEAS for c in [1e-6, LOOSE]] + [(1e-6, LOOSE, n) for n in FIRST_ITERS])
FIRST_CROSS = FIRST_CARRY + [(1e-3, c, None) for c in [1e-4, 1e-2]]
FIRST_LOOSE = [(f, LOOSE, None) for f in FIRST_FEAS]


def nominal(family, methods="A", **changes):
    stages = PIPELINES[family]
    case = dict(family=family, stages=stages, methods=methods * len(stages), guess="straight_line", seed=None,
                revs=21, w=30.0, tr=10.0, eps_feas=1e-6, eps_cost=1e-6, first_feas=None, first_cost=None,
                first_iter=None, carry=True, repeat=0)
    if len(stages) > 1 and family not in POLISH:
        case.update(first_feas=1e-4, first_cost=LOOSE)
    case.update(changes)
    return case


def single(stage, methods, **changes):
    return dict(nominal("ms", methods, **changes), family=label(stage), stages=[stage])


def both(make):
    """A case maker over both methods and both guesses."""
    return [case for methods, guess in itertools.product("AT", GUESSES) for case in make(methods, guess)]


# ---------------------------------------------------------------- blocks

def block_A():   # pseudospectral meshes: p against hp at equal size, nominal and tight tolerances
    return both(lambda m, g: [single(("ps", s, n), m, guess=g, eps_feas=tol, eps_cost=tol)
                              for s, n in PS_MESHES for tol in [1e-6, 1e-7]])


def block_B():   # ms meshes: intervals against RK4 steps, down to one shooting interval per arc
    return both(lambda m, g: [single(("ms", i, n), m, guess=g, eps_feas=tol, eps_cost=tol)
                              for i, n in MS_MESHES for tol in [1e-6, 1e-7]])


def block_J():   # reference transfer time: fine meshes, tight tolerances
    return [dict(nominal(family, m, eps_feas=1e-8, eps_cost=1e-8),
                 stages=PIPELINES[family][:-1] + [last])
            for family, last in [("hp13-ms-hp25", ("ps", 8, 7)), ("msc-ms", ("ms", 24, 24))] for m in "AT"]


def block_C():   # every pipeline at the nominal settings
    return both(lambda m, g: [nominal(family, m, guess=g) for family in PIPELINES])


def block_D():   # first stage: mesh, tolerances, weights carried or not
    def make(m, g):
        out = []
        for family, stages in PIPELINES.items():
            if len(stages) == 1:
                continue
            out.append(nominal(family, m, guess=g, carry=False))
            if family in POLISH:
                out += [nominal(family, m, guess=g, first_feas=f, first_cost=c) for f, c in POLISH_TOLERANCES]
                continue
            for first, (f, c) in itertools.product(FIRST_STAGES[stages[0]], FIRST_TOLERANCES):
                out.append(dict(nominal(family, m, guess=g, first_feas=f, first_cost=c), stages=[first] + stages[1:]))
        return out
    return both(make)


def block_E():   # cost of the hp polish: finer final hp
    return both(lambda m, g: [dict(nominal(family, m, guess=g), stages=PIPELINES[family][:-1] + [last])
                              for family in ["ms-hp25", "hp13-ms-hp25"] for last in [("ps", 6, 7), ("ps", 8, 7)]])


def block_F():   # poor guesses: the same random perturbations for every configuration
    return both(lambda m, g: [nominal(family, m, guess=g, seed=seed) for family in PIPELINES for seed in SEEDS])


def block_G():   # tuning: cost weight against trust region, every configuration on the same grid
    return [nominal(family, m, w=w, tr=tr)
            for family in PIPELINES for m in "AT" for w, tr in itertools.product(WEIGHTS, TRUST_STEPS)]


def block_T():   # tolerances: feasibility against cost
    return [nominal(family, m, eps_feas=f, eps_cost=c)
            for family in PIPELINES for m in "AT" for f, c in itertools.product(TOLERANCES, TOLERANCES)]


def block_H():   # autoscvx and autoscvx-term mixed in one pipeline
    return [dict(nominal(family, guess=g), methods=methods)
            for family, mixes in [("hp13-ms", ["AT", "TA"]), ("ms-hp25", ["AT", "TA"]),
                                  ("hp13-ms-hp25", ["ATA", "TAT"])]
            for methods in mixes for g in GUESSES]


def block_K():   # other problem instances: 7 and 14 revolutions
    return both(lambda m, g: [nominal(family, m, guess=g, revs=revs) for family in PIPELINES for revs in INSTANCES])


def block_I():   # timing: repeats run alone, one at a time
    return [nominal(family, m, repeat=k) for family in PIPELINES for m in "AT" for k in [1, 2, 3]]


def starts(family, **changes):
    """The 22 starts: unperturbed and seeds 0-9, from both guesses."""
    return [nominal(family, guess=g, seed=s, **changes) for g in GUESSES for s in [None, *SEEDS]]


def first_pass(family, settings, **changes):
    return [case for f, c, n in settings for case in starts(family, first_feas=f, first_cost=c, first_iter=n, **changes)]


def block_L():   # first pass on the same ms mesh: full grid of feasibility, cost and iteration cap
    return starts("ms") + first_pass("msc-ms", FIRST_GRID)


def block_M():   # first pass before a mesh or discretization change
    return starts("hp25") + [case for family in ["hp13-ms", "hp13-hp25", "ms-hp25"]
                             for case in first_pass(family, FIRST_CROSS)]


def block_N():   # first pass without carrying its weights (the carried counterparts are in L and M):
    # same mesh, ps to ms, ms to ps
    return [case for family in ["msc-ms", "hp13-ms", "ms-hp25"] for case in first_pass(family, FIRST_CARRY, carry=False)]


def block_O():   # first pass before tight final tolerances, weights carried or not
    return (starts("ms", eps_feas=1e-7, eps_cost=1e-7)
            + [case for carry in [True, False]
               for case in first_pass("msc-ms", FIRST_LOOSE, eps_feas=1e-7, eps_cost=1e-7, carry=carry)])


BLOCKS = dict(A=block_A, B=block_B, J=block_J, C=block_C, D=block_D, E=block_E,
              F=block_F, G=block_G, T=block_T, H=block_H, K=block_K, I=block_I,
              L=block_L, M=block_M, N=block_N, O=block_O)


def label(stage):
    kind, a, b = stage
    return f"ms{a}x{b}" if kind == "ms" else (f"p{b}" if a == 1 else f"hp{a}x{b}")


def case_name(case):
    parts = ["-".join(label(stage) for stage in case["stages"]), case["methods"],
             case["guess"][:2] + ("" if case["seed"] is None else f"s{case['seed']}"), f"r{case['revs']}",
             f"w{case['w']:g}", f"tr{case['tr']:g}", f"f{case['eps_feas']:g}", f"c{case['eps_cost']:g}"]
    if len(case["stages"]) > 1:
        first = "same" if case["first_feas"] is None else (
            f"{case['first_feas']:g}-" + ("loose" if case["first_cost"] == LOOSE else f"{case['first_cost']:g}"))
        if case["first_iter"]:
            first += f"-it{case['first_iter']}"
        parts += [f"first{first}", "carry" if case["carry"] else "nocarry"]
    if case["repeat"]:
        parts.append(f"rep{case['repeat']}")
    return "_".join(parts)


def cases():
    out = {}
    for block, make in BLOCKS.items():
        for case in make():
            name = case_name(case)
            case = out.setdefault(name, dict(case, name=name, blocks=[]))
            if block not in case["blocks"]:
                case["blocks"].append(block)
    return out


def ps_nodes(case):
    return max((a * (b - 1) + 1 for kind, a, b in case["stages"] if kind == "ps"), default=0)


def heavy(case):
    return ps_nodes(case) >= 25


def size(stage):
    """What sets the memory of a stage: sunlit nodes for ps, sunlit intervals for ms (not nsub)."""
    kind, a, b = stage
    return (kind, a * (b - 1) + 1 if kind == "ps" else a)


def memory_estimate(case, seen):
    """GB: per stage, the largest peak seen for a single-stage case of that size, else a guess
    (ps: 18 GB at 25 nodes, ms: 8.5 GB at 12 intervals, both growing with the square); the largest stage counts."""
    guesses = []
    for stage in case["stages"]:
        kind, n = size(stage)
        prior = 18.0 * (n / 25.0) ** 2 if kind == "ps" else 8.5 * (n / 12.0) ** 2
        guesses.append(seen.get((kind, n), max(4.0, prior)) + 0.5)
    return max(guesses)


def learn(seen, case, file):
    if file.exists() and len(case["stages"]) == 1:
        peak = max((stage.get("peak_gb", 0.0) for stage in json.loads(file.read_text())["stages"]), default=0.0)
        if peak:
            seen[size(case["stages"][0])] = max(seen.get(size(case["stages"][0]), 0.0), peak)


# ---------------------------------------------------------------- one stage

def mesh_nodes(stage, eclipse):
    kind, a, b = stage
    if kind == "ms":
        return (max(1, a // 6) if eclipse else a) + 1
    return 4 if eclipse else a * (b - 1) + 1


def intervals(stage, phases):
    return sum(mesh_nodes(stage, "eclipse" in name) - 1 for name in phases)


def configure(case, number):
    """The config of one stage: written to the case folder, then changed in memory."""
    stage, first = case["stages"][number], number == 0 and len(case["stages"]) > 1
    kind, a, b = stage
    library = "autoscvx-term" if case["methods"][number] == "T" else "autoscvx"
    inherit = [f"trajopt/methods/stable/configs/first_order/{library}.yaml"]
    if kind == "ps":
        inherit.append("trajopt/methods/common/configs/discretize/pseudospectral_p.yaml")
    problem = HERE / "mission" / ("problem.yaml" if case["revs"] == 21 else f"problem_rev{case['revs']}.yaml")
    hyperparams = dict(convergence=dict(flag_conv=2, iter_max=300),
                       penalties=dict(final_nonconvex_equality=dict(vb="standard"),
                                      final_nonconvex_inequality=dict(vb="none")))
    if library == "autoscvx-term":
        hyperparams["penalties"].update(TERM_PENALTIES)
    file = SWEEP / case["name"] / f"{number + 1}_config.yaml"
    file.write_text(yaml.safe_dump(dict(trajectory=dict(inherit=problem.as_posix()),
                                        method=dict(inherit=inherit, hyperparams=hyperparams))))
    config = load_trajopt_config(str(file))
    phases, hyper = config.trajectory.phases, config.method.hyperparams

    eps_feas, eps_cost = case["eps_feas"], case["eps_cost"]
    if first and case["first_feas"] is not None:
        eps_feas, eps_cost = case["first_feas"], case["first_cost"]
    eps_dynamics = eps_feas * intervals(MS, phases) / intervals(stage, phases)
    for name, phase in phases.items():
        phase.num_nodes = mesh_nodes(stage, "eclipse" in name)
        phase.hp_segments = a if kind == "ps" and "eclipse" not in name else 1
        for constraint in phase.constraints.values():
            if constraint.type == "dynamics":
                constraint.eps = eps_dynamics
            elif "nonconvex" in constraint.type:
                constraint.eps = eps_feas
    if kind == "ms":
        hyper.discretize.nsub = b
    hyper.convergence.eps_cost = eps_cost * case["w"]
    if first and case["first_iter"]:
        hyper.convergence.iter_max = case["first_iter"]
    hyper.trust_region.tr_step = AttrDict(z=case["tr"], nu=case["tr"] / 5.0)
    phases.final_arc.costs.minimum_elapsed_time.w = case["w"]

    if number == 0:
        guess = config.method.fcns.initial_guess.default
        guess.guess = (guess.guess.rsplit(":", 1)[0] + ":straight_line" if case["guess"] == "straight_line"
                       else f"{HERE / 'mission' / 'propagated_guess.py'}:propagation")
        if case["seed"] is not None:
            perturb(phases, case["seed"], case["guess"])
    return config


def perturb(phases, seed, guess):
    """Random errors in the guess, drawn the same way for every configuration and guess family:
    elapsed-time rate x0.5..x2 (straight line only), one steering offset of up to pi/2,
    and an independent steering offset of 0.3 rad (1 sigma) on every arc."""
    rng = np.random.default_rng(seed)
    rate = np.exp(rng.uniform(np.log(0.5), np.log(2.0)))
    offset = rng.uniform(-0.5 * np.pi, 0.5 * np.pi)
    noise = rng.normal(0.0, 0.3, len(phases))
    for phase, error in zip(phases.values(), noise):
        phase.guess.u_start = [float(phase.guess.u_start[0]) + offset + error]
        phase.guess.u_stop = [float(phase.guess.u_stop[0]) + offset + error]
        if guess == "straight_line":
            phase.guess.x_stop = [*phase.guess.x_stop[:5], float(phase.guess.x_stop[5]) * rate]


# ---------------------------------------------------------------- independent check of a solution

def steering_law(arc, mesh):
    """Steering against longitude as the discretization defines it: linear for ms, one polynomial per ps segment."""
    nodes = arc.sample(arc.native_tau)
    if mesh["discretize"] == "ms":
        return lambda longitude: np.interp(longitude, nodes.t, nodes.u[:, 0])
    order = (len(nodes.t) - 1) // mesh["segments"]
    pieces = [BarycentricInterpolator(nodes.t[i:i + order + 1], nodes.u[i:i + order + 1, 0])
              for i in range(0, len(nodes.t) - 1, order)]
    return lambda longitude: pieces[min((np.searchsorted(nodes.t, longitude, side="right") - 1) // order,
                                        len(pieces) - 1)](longitude)


def flight(result):
    """Fly the steering open loop through every arc from the initial state (DOP853, 1e-10) and compare:
    worst arc-end error against the solution, worst terminator miss, and the final orbit against its targets."""
    params = {name: phase.config.params for name, phase in result.problem.phases.items()}
    sail = params["final_arc"]
    rhs = jax.jit(lambda x, u, longitude, light: solar_sail.dynamics(
        x, u, longitude, AttrDict(sail, illumination=light), None))
    arcs = result.trajectory
    x = next(iter(arcs.values())).x[0]
    end_p_km = end_time_s = terminator_km = 0.0
    for name, arc in arcs.items():
        law, light = steering_law(arc, result.phase_data[name]["mesh"]), params[name].illumination
        x = solve_ivp(lambda longitude, x: np.asarray(rhs(x, np.atleast_1d(law(longitude)), longitude, light)),
                      arc.t[[0, -1]], x, method="DOP853", rtol=1e-10, atol=1e-10).y[:, -1]
        end_p_km = max(end_p_km, abs(x[0] - arc.x[-1, 0]))
        end_time_s = max(end_time_s, abs(x[5] - arc.x[-1, 5]))
        if name != "final_arc":
            residual = float(solar_sail.shadow_boundary(x, None, arc.t[-1], sail)[0])
            terminator_km = max(terminator_km, abs(sail.earth_radius_km * (np.sqrt(1 + residual) - 1)))
    p, e = x[0] + sail.geo_radius_km, np.hypot(x[1], x[2])
    return dict(flown_days=x[5] / DAY, end_p_km=end_p_km, end_time_s=end_time_s, terminator_km=terminator_km,
                perigee_short_km=sail.final_perigee_radius_km - p / (1 + e),
                eccentricity_excess=e - sail.eccentricity_max)


def peak_memory_gb():
    if sys.platform != "win32":
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6

    class Counters(ctypes.Structure):
        _fields_ = [("cb", ctypes.c_ulong), ("faults", ctypes.c_ulong)] + [
            (field, ctypes.c_size_t) for field in ["peak", "now", "a", "b", "c", "d", "e", "f"]]
    counters = Counters(cb=ctypes.sizeof(Counters))
    kernel32 = ctypes.windll.kernel32
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    kernel32.K32GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong]
    kernel32.K32GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb)
    return counters.peak / 1e9


def log_weights(solver):
    """Largest penalty weight of every phase at every iteration (the library keeps only the first and last)."""
    traces = []
    for phase in solver.phases.values():
        trace, record = [], phase.record_iter_data

        def logged(record=record, trace=trace, phase=phase):
            record()
            trace.append(max((float(np.max(W)) for W in phase.current_iter_data.get("W", {}).values()), default=0.0))
        phase.record_iter_data = logged
        traces.append(trace)
    return traces


def history(result, traces):
    """Per iteration: cost of the final arc, the worst dynamics defect over tolerance and the largest penalty weight."""
    phases = [data["scalar_history"] for data in result.phase_data.values() if "scalar_history" in data]
    return dict(cost=[float(row["cost"]) for row in result.phase_data["final_arc"]["scalar_history"]],
                dynamics=[max(float(rows[i]["chk"]["dynamics"]) for rows in phases if i < len(rows))
                          for i in range(len(phases[0]))],
                weight=[max(trace[i] for trace in traces if i < len(trace)) for i in range(max(map(len, traces)))])


# ---------------------------------------------------------------- one case

def solve_case(case):
    """Solve the stages one after the other and write case.json."""
    folder = SWEEP / case["name"]
    for old_file in [*folder.glob("[0-9]_*.json"), *folder.glob("[0-9]_*.npz")]:
        old_file.unlink()

    stages, result = [], None
    for number, stage in enumerate(case["stages"]):
        record = dict(stage=label(stage), method=case["methods"][number])
        start_wall = time.perf_counter()
        try:
            config = configure(case, number)
            start = next(iter(config.trajectory.phases.values())).params.initial_longitude_rad
            problem, method = normalize(config)
            solver = SCPMethod(problem=problem, method=method,
                               initial_guess=None if result is None else as_guess(result, start))
            if result is not None and case["carry"]:
                try:
                    carry_weights(result, solver)
                except Exception:
                    traceback.print_exc()
                    record["carry"] = "failed"
            traces = log_weights(solver)
            result = solver.solve()
        except Exception:
            traceback.print_exc()
            stages.append(dict(record, status="error", converged=False, wall_s=time.perf_counter() - start_wall))
            break

        record.update(
            status=result.status, reason=result.reason, converged=bool(result.converged), iterations=result.iterations,
            wall_s=time.perf_counter() - start_wall, time_s=result.timings["total_s"],
            warmup_s=result.timings["warmup_s"], peak_gb=peak_memory_gb(),
            intervals=intervals(stage, config.trajectory.phases),
            transfer_days=None if result.trajectory is None else result.trajectory["final_arc"].x[-1, 5] / DAY)
        try:
            record["history"] = history(result, traces)
            if result.trajectory is not None:
                record["flight"] = flight(result)
        except Exception:
            traceback.print_exc()
        if SAVE_BLOCKS & set(case["blocks"]):
            save_result(result, folder / f"{number + 1}_{label(stage)}")
        stages.append(record)
        print(f"{label(stage)}: {result.status} in {result.iterations} iterations", flush=True)
        if result.trajectory is None:
            break

    (folder / "case.json").write_text(json.dumps(dict(case, mesh=case["stages"], stages=stages), indent=1))


def solve_in_own_process(case):
    """A fresh Python for every case, so a crash, a hang or a memory leak stays inside that case."""
    folder = SWEEP / case["name"]
    folder.mkdir(parents=True, exist_ok=True)
    start = time.time()
    with open(folder / "log.txt", "w") as log:
        try:
            subprocess.run([sys.executable, __file__, "--case", case["name"]], stdout=log,
                           stderr=subprocess.STDOUT, timeout=TIMEOUT_S, cwd=HERE)
        except subprocess.TimeoutExpired:
            (folder / "case.json").write_text(json.dumps(
                dict(case, stages=[dict(stage="-", status="timeout", converged=False)]), indent=1))
    status = "done" if (folder / "case.json").exists() else "crashed"
    print(f"{time.strftime('%H:%M')}  {status:7s} {time.time() - start:6.0f} s  {case['name']}", flush=True)


def write_table(all_cases):
    """One row per case: its settings, how the last stage ended and its independent check."""
    rows = []
    for name, case in all_cases.items():
        file = SWEEP / name / "case.json"
        if not file.exists():
            continue
        stages = json.loads(file.read_text())["stages"]
        last = stages[-1]
        rows.append(dict(
            name=name, blocks=" ".join(case["blocks"]), family=case["family"],
            stages="-".join(label(stage) for stage in case["stages"]),
            **{key: case[key] for key in ["methods", "guess", "seed", "revs", "w", "tr", "eps_feas", "eps_cost",
                                          "first_feas", "first_cost", "first_iter", "carry", "repeat"]},
            converged=last["converged"] and len(stages) == len(case["stages"]), final_status=last["status"],
            transfer_days=last.get("transfer_days"),
            iterations=sum(stage.get("iterations", 0) for stage in stages),
            wall_s=sum(stage.get("wall_s", 0.0) for stage in stages),
            time_s=sum(stage.get("time_s", 0.0) for stage in stages),
            warmup_s=sum(stage.get("warmup_s", 0.0) for stage in stages),
            peak_gb=max(stage.get("peak_gb", 0.0) for stage in stages),
            intervals=last.get("intervals"),
            **last.get("flight", {}),
            per_stage=" | ".join(f"{s['stage']} {s.get('method', '-')} {s['status']} {s.get('iterations', '')}" for s in stages)))
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with open(SWEEP / "study.csv", "w", newline="") as file:
        writer = csv.DictWriter(file, columns)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    all_cases = cases()
    if sys.argv[1:2] == ["--case"]:   # started by solve_in_own_process
        solve_case(all_cases[sys.argv[2]])
        sys.exit()

    blocks = [arg for arg in sys.argv[1:] if arg in BLOCKS] or list(BLOCKS)
    chosen = {name: case for name, case in all_cases.items() if set(case["blocks"]) & set(blocks)
              and not ("--autoscvx" in sys.argv and "T" in case["methods"])}
    todo = [case for name, case in chosen.items() if not (SWEEP / name / "case.json").exists()]
    for block in blocks:
        members = [case for case in chosen.values() if block in case["blocks"]]
        print(f"block {block}: {len(members):4d} cases, {sum(case['blocks'][0] == block for case in members):4d} "
              f"not in an earlier block, {sum(map(heavy, members)):4d} heavy")
    print(f"{len(chosen)} cases, {sum(map(heavy, chosen.values()))} heavy, {len(todo)} still to solve")
    if "--list" in sys.argv:
        sys.exit()

    if sys.platform == "win32":   # keep the machine from sleeping while this process runs
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
    SWEEP.mkdir(parents=True, exist_ok=True)
    # blocks in the order asked for, cases in random order inside a block; ps stages of 37+ sunlit nodes
    # (30-50 GB, they run nearly alone) after everything else
    random.Random(0).shuffle(todo)
    todo.sort(key=lambda case: (ps_nodes(case) >= 37,
                                min(blocks.index(block) for block in case["blocks"] if block in blocks)))
    table_lock = threading.Lock()

    def solve_and_tabulate(case):
        solve_in_own_process(case)
        with table_lock:   # the table stays current, so a sweep stopped early still has one
            try:
                write_table(all_cases)
            except OSError as error:   # e.g. study.csv open in Excel: the next case retries
                print(f"table not written: {error}", flush=True)

    seen = {}   # largest peak memory per stage size, from the single-stage cases done so far
    for file in SWEEP.glob("*/case.json"):
        if file.parent.name in all_cases:
            learn(seen, all_cases[file.parent.name], file)
    free = dict(gb=MEMORY_GB, running=0)
    ready = threading.Condition()

    def run(case, need):
        try:
            solve_and_tabulate(case)
        finally:
            with ready:
                free["gb"] += need
                free["running"] -= 1
                learn(seen, case, SWEEP / case["name"] / "case.json")
                ready.notify_all()

    threads = []
    for case in todo:   # in order: a large case waits for room rather than being overtaken forever
        if case["repeat"]:
            continue
        need = min(memory_estimate(case, seen), MEMORY_GB)
        with ready:
            ready.wait_for(lambda: free["gb"] >= need and free["running"] < MAX_RUNNING)
            free["gb"] -= need
            free["running"] += 1
        threads.append(threading.Thread(target=run, args=(case, need)))
        threads[-1].start()
    for thread in threads:
        thread.join()
    for case in todo:
        if case["repeat"]:
            solve_and_tabulate(case)
    print(f"table in {SWEEP / 'study.csv'}")
