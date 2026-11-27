"""Solve the L1 -> L2 transfer with one method and save each stage to results/T<T>_<method>/.

    python main.py p_ms 8       method: p, ms, msc (ms on a coarse grid), p_ms, p_msc (ms or msc from the p solution)
                                or p_rollout_ms (ms from the p control flown open loop)
"""
import sys
from pathlib import Path

import numpy as np
from scipy.interpolate import BarycentricInterpolator

from trajopt.common.config_reader import load_trajopt_config
from trajopt.common.persistence import save_result
from trajopt.common.tools import deep_merge, recursive_attrdict
from trajopt.common.trajectory import TrajectoryData
from trajopt.formulations.stable.specs import normalize
from trajopt.methods.stable.scp.scp_method import SCPMethod

import problem


def control_law(result, trajectory=None):
    """u(t) as the method defines it: one polynomial for p, linear between nodes for ms (the saved one is always linear)."""
    arc = (trajectory or result.trajectory)["main"]
    nodes = arc.sample(arc.native_tau)
    if result.phase_data["main"]["mesh"]["discretize"] == "ps":   # the first Radau node is not collocated
        return BarycentricInterpolator(nodes.t[1:], nodes.u[1:])
    return lambda t: np.column_stack([np.interp(t, nodes.t, u) for u in nodes.u.T])


def as_guess(result):
    samples = result.trajectory["main"].sample(num=2001)
    return TrajectoryData.from_samples(t=samples.t, x=samples.x, u=control_law(result)(samples.t))


def rollout(result, T):
    """The previous control flown open loop from L1: dynamically feasible by construction, as far as the instability allows."""
    control = control_law(result)
    flight = problem.fly(problem.X0, (0.0, T), control)
    t = np.linspace(0.0, T, 2001)
    return TrajectoryData.from_samples(t=t, x=flight.sol(np.minimum(t, flight.t[-1])).T, u=control(t))


def carry_weights(result, solver):
    """Start the penalty weights where the previous stage left them, resampled to the new nodes."""
    for name, old in result.phase_data["main"]["iter_data_list"][-1]["W"].items():
        penalties = solver.phases["main"].constraints[name].penalties
        old_rows, new_rows = np.linspace(0, 1, len(old)), np.linspace(0, 1, len(penalties.W))
        penalties.W = np.exp(np.column_stack([np.interp(new_rows, old_rows, np.log(w)) for w in np.asarray(old).T]))


if __name__ == "__main__":
    method, T = sys.argv[1], float(sys.argv[2])
    folder = Path("results") / f"T{T:g}_{method}"
    folder.mkdir(parents=True, exist_ok=True)

    guess, result = problem.initial_guess(T), None
    for number, stage in enumerate(method.split("_"), start=1):
        if stage == "rollout":
            guess, result = rollout(result, T), None
            continue
        config = load_trajopt_config(f"methods/{stage}.yaml")
        config.trajectory = recursive_attrdict(deep_merge(problem.trajectory(T), config.trajectory))
        config.trajectory.num_nodes = round(config.nodes_per_time_unit * T) + 1
        physical, numerical = normalize(config)
        solver = SCPMethod(problem=physical, method=numerical, initial_guess={"main": guess})
        if result is not None:
            carry_weights(result, solver)
        result = solver.solve()
        save_result(result, folder / f"{number}_{stage}")
        print(f"{stage}: {result.status} in {result.iterations} iterations, {result.timings['total_s']:.1f} s")
        if result.trajectory is None:
            break
        guess = as_guess(result)
