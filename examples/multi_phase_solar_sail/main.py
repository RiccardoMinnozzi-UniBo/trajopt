"""Solve the multi-phase solar-sail GEO disposal problem end to end.

Edit ``config.yaml``, run this, get the minimum-time solution.

The revolution count cannot be optimised inside a single SCP solve -- it sets the
number of phases -- so this script searches over it, starting from the guess in
``problem.revolutions`` and bracketing the minimum from there (see ``search.py``).
That means changing the sail, the discretisation or anything else still lands on
the minimum-time solution without working out the right count by hand; a poor
guess only costs extra solves.

Set ``problem.revolution_search.enabled: false`` to skip the search and solve
once at the guess, which is quicker when iterating on something else.
"""

import sys
from pathlib import Path

from trajopt.utils import config_loader

from search import search_minimum_revolutions, solve_once
from solution import (
    phase_timing_report,
    print_validation_report,
    save_solution,
)
from visualization import render_all, render_rotating_gif

EXAMPLE_DIRECTORY = Path(__file__).resolve().parent
DEFAULT_CONFIG = EXAMPLE_DIRECTORY / "config-ps.yaml"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def main(config_file: str | Path = DEFAULT_CONFIG) -> None:
    # `python main.py config-ps.yaml` runs the pseudospectral variant, which is
    # the same problem under a different discretisation.
    CONFIG_FILE = Path(config_file)
    if not CONFIG_FILE.is_absolute():
        CONFIG_FILE = EXAMPLE_DIRECTORY / CONFIG_FILE

    # Read the config without building the problem: constructing one compiles
    # the JAX kernels, which is far too expensive just to look up a flag.
    config = config_loader.load_trajopt_config(str(CONFIG_FILE))
    print(f"config: {CONFIG_FILE.name}  "
          f"(discretisation: {config.method.flags.discretize})")
    settings = config.problem
    search = settings.revolution_search
    guess = int(settings.revolutions)

    if bool(search.enabled):
        result = search_minimum_revolutions(
            CONFIG_FILE,
            start=guess,
            minimum=int(search.minimum),
            maximum=int(search.maximum),
            max_solves=int(search.max_solves),
        )
        if not result.found:
            # Nothing validated; still save and draw the last attempt so the
            # failure can be looked at rather than just reported.
            print("\nNo feasible revolution count: reporting the last attempt.")
            attempt = result.attempts[-1]
            problem = None
        else:
            attempt, problem = result.best, result.problem
        solution, report = attempt.solution, attempt.report
        title = f"Minimum-time solution (N = {attempt.revolutions})"
    else:
        attempt, problem = solve_once(CONFIG_FILE, guess, quiet=True)
        solution, report = attempt.solution, attempt.report
        title = f"Solution at the guess (N = {attempt.revolutions})"

    if problem is not None:
        phase_timing_report(problem, title)
    print_validation_report(report)

    output = EXAMPLE_DIRECTORY / config.output.directory
    files = [save_solution(solution, output / config.output.solution_file)]

    # Figures are always produced: an unconverged run is exactly when you most
    # want to look at the trajectory.  The animation is expensive, so it is
    # reserved for solutions that pass validation.
    files.extend(render_all(solution, output, report))

    if report["numerically_feasible"]:
        files.append(render_rotating_gif(
            solution,
            output / config.output.animation_file,
            int(config.output.animation_frames),
        ))
    else:
        print("\nValidation failed: skipping the animation. "
              "The static figures above show what the solver returned.")

    print("\noutputs")
    for path in files:
        print(f"  {path}")


if __name__ == "__main__":
    main(*sys.argv[1:2])
