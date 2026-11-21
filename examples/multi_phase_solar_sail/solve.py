import dataclasses
import functools
from pathlib import Path

import jax
import numpy as np
from scipy.interpolate import BarycentricInterpolator

from trajopt.common.config_reader import load_trajopt_config
from trajopt.common.persistence import save_result
from trajopt.common.trajectory import TrajectoryFunction
from trajopt.formulations.stable.specs import normalize
from trajopt.methods.stable.scp.scp_method import SCPMethod

jax.config.update("jax_enable_x64", True)

CONFIG = "ms"   # "p" or "ms" (methods/<name>.yaml), or "p_ms_hp" (methods/p_ms_hp/<stage>.yaml, in that order)


def control_polynomial(arc, segments):
    """The control between the nodes of a pseudospectral arc: one polynomial per segment.
    The saved trajectory interpolates it linearly, which is not the control the stage solved for."""
    tau, u = arc.native_tau, arc.sample(arc.native_tau).u
    order = (len(tau) - 1) // segments
    pieces = [BarycentricInterpolator(tau[i:i + order + 1], u[i:i + order + 1]) for i in range(0, len(tau) - 1, order)]

    def evaluate(query):
        query = np.atleast_1d(query)
        segment = np.minimum(np.searchsorted(tau, query, side="right") - 1, len(tau) - 2) // order
        out = np.empty((len(query), u.shape[1]))
        for number, piece in enumerate(pieces):
            out[segment == number] = piece(query[segment == number])
        return out
    return evaluate


def resample(arc, control, shift, tau):
    samples = arc.sample(tau)
    return dataclasses.replace(samples, t=samples.t - shift, u=samples.u if control is None else control(tau))


def as_guess(result, start):
    """The solution of one stage as the initial guess of the next."""
    guess = {}
    for number, (name, arc) in enumerate(result.trajectory.items()):
        arc = arc.without_aux()
        mesh = result.phase_data[name]["mesh"]
        control = control_polynomial(arc, mesh["segments"]) if mesh["discretize"] == "ps" else None
        shift = arc.t[0] - start if number == 0 else 0.0   # the solver leaves the first longitude ~1e-9 off
        guess[name] = arc if control is None and number else TrajectoryFunction(
            functools.partial(resample, arc, control, shift), arc.native_tau)
    return guess


def carry_weights(result, solver):
    """Start the penalty weights from where the previous stage left them, resampled to the new mesh."""
    for name, phase in solver.phases.items():
        for constraint, old in result.phase_data[name]["iter_data_list"][-1]["W"].items():
            penalties = phase.constraints[constraint].penalties
            old_rows = np.linspace(0.0, 1.0, len(old))
            new_rows = np.linspace(0.0, 1.0, len(penalties.W))
            penalties.W = np.exp(np.column_stack(
                [np.interp(new_rows, old_rows, np.log(column)) for column in np.asarray(old).T]))


if __name__ == "__main__":   # sweep.py imports the two functions above
    folder = Path("results") / CONFIG
    folder.mkdir(parents=True, exist_ok=True)
    for old_file in folder.glob("*"):
        old_file.unlink()

    result = None
    for number, stage in enumerate(CONFIG.split("_"), start=1):
        stage_file = f"methods/{CONFIG}/{stage}.yaml" if "_" in CONFIG else f"methods/{CONFIG}.yaml"
        config = load_trajopt_config(stage_file)
        start = next(iter(config.trajectory.phases.values())).params.initial_longitude_rad
        problem, method = normalize(config)
        solver = SCPMethod(problem=problem, method=method,
                           initial_guess=None if result is None else as_guess(result, start))
        if result is not None:
            carry_weights(result, solver)

        result = solver.solve()
        save_result(result, folder / f"{number}_{stage}")
        print(f"{stage}: {result.status} in {result.iterations} iterations")
        if result.trajectory is None:
            break

    print(f"saved to {folder}")
