import dataclasses
from pathlib import Path

import jax
import numpy as np

from trajopt.common.config_reader import load_trajopt_config
from trajopt.common.persistence import save_result
from trajopt.common.trajectory import TrajectoryFunction
from trajopt.formulations.stable.specs import normalize
from trajopt.methods.stable.scp.scp_method import SCPMethod

jax.config.update("jax_enable_x64", True)

CONFIG = "ms"   # "p" or "ms" (methods/<name>.yaml), or "p_ms_hp" (methods/p_ms_hp/<stage>.yaml, in that order)


def as_guess(result, start):
    """The solution of one stage as the initial guess of the next."""
    guess = {name: arc.without_aux() for name, arc in result.trajectory.items()}
    # the solver leaves the first longitude ~1e-9 off, the next stage wants it exact
    name, arc = next(iter(guess.items()))
    shift = arc.t[0] - start
    guess[name] = TrajectoryFunction(
        lambda tau: dataclasses.replace(arc.sample(tau), t=arc.sample(tau).t - shift), arc.native_tau)
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
