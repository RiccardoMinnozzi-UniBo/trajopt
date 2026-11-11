"""Solve every combination of the settings below.

One folder per case in results/sweep/ (stage results as solve.py saves them, case.json, log.txt),
and one row per case in results/sweep/sweep.csv. Cases that already have a case.json are skipped,
so an interrupted sweep resumes where it stopped; delete a case folder to solve it again.
Nothing on disk is modified: every setting is changed on the loaded config.
"""
import csv
import itertools
import json
import subprocess
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import jax

from trajopt.common.config_reader import load_trajopt_config
from trajopt.common.persistence import save_result
from trajopt.formulations.stable.specs import normalize
from trajopt.methods.stable.scp.scp_method import SCPMethod

from solve import as_guess, carry_weights

jax.config.update("jax_enable_x64", True)

ALGORITHMS = ["p", "ms", "p_ms", "p_ms_hp"]
STATE_GUESSES = ["straight_line"]     # "straight_line" and/or "propagation"; first stage of every algorithm
CONTROL_GUESSES = ["ramp", "zero"]
RATE_LIMITS = [True, False]           # steering rate constraint on or off, in every stage

# the p stage of p_ms and p_ms_hp only (p and ms alone keep methods/p.yaml and methods/ms.yaml)
P_EPS_COST = [1.0e-3, 1.0e4]
P_BUDGET_KM = [3.0, 30.0, 300.0]      # dynamics defect allowed over all intervals, in km of semi-latus rectum
P_BUFFERS = [(0.0, 0.0), (0.3, 0.0)]  # (perigee above the target [km], eccentricity below the limit)

PARALLEL = 2          # cases solved at the same time: fine p and hp take ~7 GB each
SAVE_RESULTS = True   # full results for postprocess.py (~10-20 MB per stage); False keeps only case.json
SWEEP = Path("results") / "sweep"
P_SCALE_KM = 1000.0   # scale of the semi-latus rectum, as in make_problem.py


def cases():
    for algorithm, states, controls, rate_limit in itertools.product(
            ALGORITHMS, STATE_GUESSES, CONTROL_GUESSES, RATE_LIMITS):
        name = f"{algorithm}-{states}-{controls}-rate_{'on' if rate_limit else 'off'}"
        if "_" not in algorithm:
            yield dict(name=name, algorithm=algorithm, states=states, controls=controls, rate_limit=rate_limit)
            continue
        for eps_cost, budget_km, buffer in itertools.product(P_EPS_COST, P_BUDGET_KM, P_BUFFERS):
            yield dict(name=f"{name}-cost_{eps_cost:g}-budget_{budget_km:g}-buffer_{buffer[0]:g}_{buffer[1]:g}",
                       algorithm=algorithm, states=states, controls=controls, rate_limit=rate_limit,
                       p_eps_cost=eps_cost, p_budget_km=budget_km, p_buffer=buffer)


def configure(case, stage, first):
    """The stage config of methods/, changed in memory as the case asks."""
    algorithm = case["algorithm"]
    config = load_trajopt_config(f"methods/{algorithm}/{stage}.yaml" if "_" in algorithm else f"methods/{algorithm}.yaml")
    phases = config.trajectory.phases

    if not case["rate_limit"]:
        for phase in phases.values():
            del phase.constraints["steering_rate_limit"]

    if first:
        guess = config.method.fcns.initial_guess.default
        guess.guess = guess.guess.rsplit(":", 1)[0] + ":" + case["states"]
        if case["controls"] == "zero":
            for phase in phases.values():
                phase.guess.u_start, phase.guess.u_stop = [0.0], [0.0]

    if first and "_" in algorithm:   # the p stage of a pipeline
        config.method.hyperparams.convergence.eps_cost = case["p_eps_cost"]
        intervals = sum(int(phase.num_nodes) - 1 for phase in phases.values())
        for phase in phases.values():
            phase.constraints.dynamics.eps = case["p_budget_km"] / P_SCALE_KM / intervals
        perigee_km, eccentricity = case["p_buffer"]
        phases.final_arc.params.final_perigee_radius_km += perigee_km
        phases.final_arc.params.eccentricity_max -= eccentricity
    return config


def solve_case(case):
    """Solve the stages of one case one after the other, as solve.py does, and write case.json."""
    folder = SWEEP / case["name"]
    for old_file in [*folder.glob("[0-9]_*.json"), *folder.glob("[0-9]_*.npz")]:
        old_file.unlink()

    stages, result = [], None
    for number, stage in enumerate(case["algorithm"].split("_"), start=1):
        try:
            config = configure(case, stage, first=result is None)
            start = next(iter(config.trajectory.phases.values())).params.initial_longitude_rad
            problem, method = normalize(config)
            solver = SCPMethod(problem=problem, method=method,
                               initial_guess=None if result is None else as_guess(result, start))
            if result is not None:
                carry_weights(result, solver)
            result = solver.solve()
        except Exception:
            traceback.print_exc()
            stages.append(dict(stage=stage, status="error", converged=False))
            break

        if SAVE_RESULTS:
            save_result(result, folder / f"{number}_{stage}")
        stages.append(dict(
            stage=stage, status=result.status, converged=bool(result.converged), iterations=result.iterations,
            time_s=result.timings["total_s"], warmup_s=result.timings["warmup_s"],
            transfer_days=None if result.trajectory is None else result.trajectory["final_arc"].x[-1, 5] / 86400.0))
        print(f"{stage}: {result.status} in {result.iterations} iterations", flush=True)
        if result.trajectory is None:
            break

    (folder / "case.json").write_text(json.dumps(dict(case, stages=stages), indent=2))


def solve_in_own_process(name):
    """A fresh Python for every case, so a crash or a memory leak stays inside that case."""
    folder = SWEEP / name
    folder.mkdir(parents=True, exist_ok=True)
    with open(folder / "log.txt", "w") as log:
        subprocess.run([sys.executable, __file__, name], stdout=log, stderr=subprocess.STDOUT)
    print(f"done  {name}", flush=True)


def write_table(names):
    """One row per case: its settings and how the last stage ended."""
    rows = []
    for name in names:
        file = SWEEP / name / "case.json"
        record = json.loads(file.read_text()) if file.exists() else dict(name=name, stages=[])
        stages = record.pop("stages")
        last = stages[-1] if stages else dict(status="crashed", converged=False)
        rows.append(dict(
            record,
            converged=last["converged"] and len(stages) == len(record.get("algorithm", "").split("_")),
            final_status=last["status"],
            transfer_days=last.get("transfer_days"),
            iterations=sum(stage.get("iterations", 0) for stage in stages),
            time_s=sum(stage.get("time_s", 0.0) for stage in stages),
            warmup_s=sum(stage.get("warmup_s", 0.0) for stage in stages),
            stages=" | ".join(f"{stage['stage']} {stage['status']} {stage.get('iterations', '')}" for stage in stages)))
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with open(SWEEP / "sweep.csv", "w", newline="") as file:
        writer = csv.DictWriter(file, columns)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    all_cases = {case["name"]: case for case in cases()}
    if len(sys.argv) > 1:   # started by solve_in_own_process
        solve_case(all_cases[sys.argv[1]])
    else:
        SWEEP.mkdir(parents=True, exist_ok=True)
        todo = [name for name in all_cases if not (SWEEP / name / "case.json").exists()]
        print(f"{len(all_cases)} cases, {len(todo)} still to solve, {PARALLEL} at a time", flush=True)
        with ThreadPoolExecutor(PARALLEL) as pool:
            list(pool.map(solve_in_own_process, todo))
        write_table(all_cases)
        print(f"table in {SWEEP / 'sweep.csv'}")
